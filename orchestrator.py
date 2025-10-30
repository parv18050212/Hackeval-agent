import os
import glob
import asyncio
import json
from typing import List, Optional
from dotenv import load_dotenv
from Agents.project_context import ProjectAnalysisContext
from Agents.scoring_agent import ScoringAgent, CombinedAgent
from Agents.feedback_agent import FeedbackAgent
from Agents.image_eval import WorkflowAnalysisAgent
from Agents.format_agent import FormatAgent
from utils import (
    load_document_content,
    display_consolidated_report,
    display_leaderboard,
    ALLOWED_EXTS,
    save_consolidated_reports_to_excel,
    save_leaderboard_to_excel,
    EvaluationParameters,
    load_eval_parameters_from_path,
)

async def aload_document_content(file_path: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, load_document_content, file_path)

def _expand_team_glob(pattern: str) -> List[str]:
    if not pattern:
        return []
    parts = [p.strip().strip('"').strip("'") for p in pattern.split(",") if p.strip()]
    found: List[str] = []
    for p in parts:
        found.extend(glob.glob(p, recursive=True))
    found = [f for f in found if os.path.isfile(f) and os.path.splitext(f)[1].lower() in ALLOWED_EXTS]
    return sorted(set(found))

async def process_file(
    file_path: str,
    agent_mode: str,
    semaphore: asyncio.Semaphore,
    eval_params: EvaluationParameters
):
    async with semaphore:
        print("\n" + "*" * 70)
        print(f"Processing: {file_path}")
        print("*" * 70)

        ctx = ProjectAnalysisContext(file_path)
        ctx.evaluation_parameters = eval_params.to_dict()

        try:
            # 1) Load document
            text, images_b64 = await aload_document_content(file_path)
            ctx.update_raw_content(text, images_b64, [])

            # 2) Diagram summary - FIXED
            try:
                img_agent = WorkflowAnalysisAgent()

                # Try the correct method name first
                if hasattr(img_agent, 'analyze_workflows'): # Use the correct method name
                    report = img_agent.analyze_workflows(file_path) # Use the correct method name
                # You can remove or comment out the checks for 'analyze' and 'evaluate'
                # if they are definitely not used in your image_eval.py
                # elif hasattr(img_agent, 'analyze'):
                #     report = img_agent.analyze(file_path)
                # elif hasattr(img_agent, 'evaluate'):
                #     report = img_agent.evaluate(file_path)
                else:
                    # Updated error message to reflect the expected method
                    print(f"  -> Diagram summary skipped: No suitable method 'analyze_workflows' found on WorkflowAnalysisAgent")
                    report = None

                if report:
                    # Handle different return types - FIXED
                    if hasattr(report, "model_dump"):
                        ctx.update_workflow_report(report.model_dump())
                    elif hasattr(report, "dict"):
                        ctx.update_workflow_report(report.dict())
                    elif isinstance(report, dict):
                        ctx.update_workflow_report(report)
                    else:
                        print(f"  -> Diagram summary returned unknown type: {type(report)}")
            except Exception as e:
                print(f"  -> Diagram summary skipped: {e}")

            # 3) Scoring + feedback
            try:
                if agent_mode == "combined":
                    comb = CombinedAgent(eval_params=eval_params)
                    await comb.run(ctx)
                else:
                    scorer = ScoringAgent(eval_params=eval_params)
                    await scorer.run(ctx)
                    fb = FeedbackAgent(eval_params=eval_params)
                    await fb.run(ctx)
            except Exception as e:
                ctx.set_error(f"Unhandled error: {type(e).__name__}: {e}")
                print(f"  -> Scoring/Feedback error: {e}")
                import traceback
                traceback.print_exc()

            # 4) Format & Design (5 marks)
            try:
                fmt = FormatAgent()
                res = await fmt.aevaluate(file_path, text_hint=ctx.raw_text or "")
                
                format_key = "Format of the Presentation"
                if format_key in eval_params.criteria:
                    ctx.scores[format_key] = int(res.get("score", 0))
                    ctx.update_format_results(ctx.scores[format_key], res.get("notes", []))

                if res.get("notes"):
                    existing = (ctx.feedback or {}).get("suggestions", "")
                    add = "\n".join(f"- {n}" for n in res["notes"])
                    ctx.update_feedback_results({
                        **(ctx.feedback or {}),
                        "suggestions": (existing + ("\n" if existing else "") + add)
                    })
            except Exception as e:
                print(f"  -> Format eval skipped: {e}")

        except Exception as e:
            ctx.set_error(f"Unhandled error: {type(e).__name__}: {e}")
            print(f"  -> Fatal error: {e}")
            import traceback
            traceback.print_exc()

        display_consolidated_report(ctx, eval_params)
        return ctx

async def main():
    load_dotenv()
    TEAM_GLOB = os.getenv("TEAM_GLOB", "").strip()
    EVAL_PARAMS_FILE = os.getenv("EVAL_PARAMS_FILE", "").strip()
    
    files = _expand_team_glob(TEAM_GLOB) if TEAM_GLOB else []
    if not files:
        print("No input files found. Set TEAM_GLOB.")
        return

    # Load evaluation parameters from multi-format file path
    eval_params = load_eval_parameters_from_path(EVAL_PARAMS_FILE)

    print("\n" + "=" * 70)
    print("Using Evaluation Parameters:")
    for criterion, values in eval_params.criteria.items():
        print(f"  - {criterion} (Weight: {values['weight']}, Max Score: {values['max_score']})")
    if eval_params.rubric:
        print("Rubric: (first 200 chars)")
        print((eval_params.rubric or "")[:200])
    print("=" * 70)

    agent_mode = os.getenv("AGENT_MODE", "separate").lower().strip()
    semaphore = asyncio.Semaphore(int(os.getenv("CONCURRENCY", "2")))

    tasks = [asyncio.create_task(process_file(fp, agent_mode, semaphore, eval_params)) for fp in files]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    contexts = [r for r in results if r is not None]
    
    if contexts:
        display_leaderboard(contexts, eval_params)
        save_consolidated_reports_to_excel(contexts, "consolidated_reports.xlsx", eval_params)
        save_leaderboard_to_excel(contexts, "leaderboard.xlsx", eval_params)

if __name__ == "__main__":
    asyncio.run(main())