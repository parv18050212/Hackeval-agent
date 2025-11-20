import os
import re
import asyncio
import hashlib
import random
from typing import List, Optional, Dict
# Qdrant integration
from qdrant_integration import ingest_text, search

from pydantic import BaseModel, field_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from openai import RateLimitError  # ✅ new OpenAI client errors live at top-level

from utils import get_text_limiter, EvaluationParameters, _lower, _hit_count, _strength


# ===================== Data Models =====================
class WorkflowStep(BaseModel):
    idx: int
    text: str



class ScoringOutput(BaseModel):
    team_name: str
    scores: Dict[str, int]
    summary: str

    def store_scoring_embedding(self, scoring_text: str, payload: dict = None):
        """Store scoring embedding in Qdrant."""
        return ingest_text(scoring_text, payload)

    def search_similar_scoring(self, query: str, top_k: int = 5):
        """Search for similar scoring outputs in Qdrant."""
        return search(query, top_k)

class WorkflowOutput(BaseModel):
    overall: Optional[str] = None
    overall_summary: Optional[str] = None
    steps: List[WorkflowStep] = []

    def normalized(self) -> Dict:
        """Ensure consistent key naming for downstream code."""
        return {
            "overall": self.overall or self.overall_summary or "",
            "steps": [s.model_dump() for s in self.steps],
        }


class FeedbackOnly(BaseModel):
    # ✅ Accept lists natively (Option A), but also accept strings via a validator
    positive: List[str] = []
    criticism: List[str] = []
    technical: List[str] = []
    suggestions: List[str] = []

    @field_validator("positive", "criticism", "technical", "suggestions", mode="before")
    @classmethod
    def _coerce_to_list(cls, v):
        """Allow either a string or list from the LLM and coerce to List[str]."""
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v]
        # string or any other scalar
        return [str(v).strip()]


class CombinedOutput(BaseModel):
    team_name: str
    scores: Dict[str, int]
    summary: str
    workflow_analysis: Optional[WorkflowOutput]
    feedback: FeedbackOnly


# ===================== Evidence heuristics =====================
KW = {
    "problem": ["problem statement", "pain point", "root cause", "target user", "persona", "use case", "scope", "constraints"],
    "user": ["user", "ux", "ui", "accessibility", "inclusive", "interview", "survey", "usability", "user journey", "persona", "a11y"],
    "innovation": ["novel", "innovation", "patent", "state-of-the-art", "first-of-its-kind", "unique", "transformer", "gan", "diffusion"],
    "technical": ["prototype", "mvp", "poc", "demo", "architecture", "api", "docker", "kubernetes", "aws", "azure", "gcp", "latency",
                  "throughput", "accuracy", "precision", "recall", "f1", "roc", "sdk", "firmware", "microcontroller", "fpga"],
    "market": ["tam", "sam", "som", "go-to-market", "gtm", "pricing", "revenue", "business model", "unit economics", "ltv", "cac", "roi", "monetization"],
    "impact": ["sdg", "co2", "emission", "waste", "lives", "safety", "savings", "reduce", "impact", "carbon", "sustainab"],
    "research": ["baseline", "benchmark", "competitor", "alternatives", "paper", "study", "dataset", "references", "related work"],
    "presentation": ["diagram", "flowchart", "architecture diagram", "block diagram", "pipeline", "figure", "chart", "graph", "mockup", "wireframe"],
}
NUM_PAT = re.compile(r"\b\d+(\.\d+)?\s?(%|x|k|m|b|₹|\$)\b", re.I)
CIT_PAT = re.compile(r"\b(arxiv|doi\.org|et al\.|references|citation)\b|\[\d{4}\]", re.I)


def _extract_format_score_from_context(context) -> Optional[float]:
    for attr in ["format_score", "formatting_score"]:
        v = getattr(context, attr, None)
        if isinstance(v, (int, float)):
            return float(v)
    for attr in ["format_report", "format_result", "format"]:
        obj = getattr(context, attr, None)
        if isinstance(obj, dict) and isinstance(obj.get("score"), (int, float)):
            return float(obj.get("score"))
    return None


def _diagram_count(context) -> int:
    c = 0
    if getattr(context, "workflow_report", None) and isinstance(context.workflow_report, dict):
        for a in context.workflow_report.get("image_analyses", []) or []:
            if a.get("is_diagram"):
                c += 1
    return c


def _workflow_steps_count(context) -> int:
    try:
        rep = getattr(context, "workflow_report", None) or {}
        text = (rep.get("overall_summary", "") or "") + " " + " ".join(
            (s.get("text", "") or "") for s in rep.get("image_analyses", []) or []
        )
        return len(re.findall(r"\b(step\s*\d+|→|->|⇒|then|next)\b", text.lower()))
    except Exception:
        return 0


def _page_or_slide_count(context) -> int:
    for a in ["num_slides", "slide_count", "page_count", "n_pages", "pages", "slides"]:
        v = getattr(context, a, None)
        if isinstance(v, int) and v > 0:
            return v
    meta = getattr(context, "meta", None)
    if isinstance(meta, dict):
        for k in ["page_count", "slides", "num_slides"]:
            v = meta.get(k)
            if isinstance(v, int) and v > 0:
                return v
    return 0


def _evidence_signals(doc_text: str, diagram_text: str, diag_count: int,
                      fmt_0_5: Optional[float], steps_cnt: int, page_cnt: int) -> Dict[str, int]:
    t = _lower(doc_text)
    d = _lower(diagram_text)
    signals: Dict[str, int] = {}
    signals["Problem identification & depth of understanding"] = _strength(_hit_count(t, KW["problem"]))
    signals["User-Centric Approach"] = _strength(_hit_count(t, KW["user"]))
    signals["Innovation Quotient"] = _strength(_hit_count(t, KW["innovation"]))
    signals["Technical Readiness & Prototype Potential"] = _strength(
        _hit_count(t, KW["technical"]) + (1 if diag_count > 0 else 0) + (1 if steps_cnt >= 3 else 0)
    )
    signals["Market Potential & Scalability"] = _strength(_hit_count(t, KW["market"]))
    signals["Social/ Economic/ Environmental Impact"] = _strength(_hit_count(t, KW["impact"]))
    signals["Research depth & Ecosystem Awareness"] = _strength(_hit_count(t, KW["research"]) + (1 if CIT_PAT.search(t) else 0))
    pres_hits = _hit_count(t, KW["presentation"]) + _hit_count(d, KW["presentation"])
    pres_bonus = (2 if diag_count > 0 else 0) + (1 if steps_cnt >= 3 else 0)
    signals["Presentation & Communication of Idea"] = _strength(pres_hits + pres_bonus)

    if fmt_0_5 is not None:
        if fmt_0_5 >= 4.5:
            fmt_s = 3
        elif fmt_0_5 >= 3.5:
            fmt_s = 2
        elif fmt_0_5 >= 2.0:
            fmt_s = 1
        else:
            fmt_s = 0
        signals["Format of the Presentation"] = fmt_s
    else:
        signals["Format of the Presentation"] = 1 + (1 if diag_count > 0 else 0)

    if NUM_PAT.search(t):
        for k in ["Technical Readiness & Prototype Potential", "Market Potential & Scalability", "Social/ Economic/ Environmental Impact"]:
            signals[k] = min(3, signals[k] + 1)

    if page_cnt <= 4:
        for k in ["Research depth & Ecosystem Awareness", "Market Potential & Scalability", "Problem identification & depth of understanding"]:
            signals[k] = max(0, signals[k] - 1)
    elif page_cnt >= 15:
        signals["Presentation & Communication of Idea"] = min(3, signals["Presentation & Communication of Idea"] + 1)
        signals["Research depth & Ecosystem Awareness"] = min(3, signals["Research depth & Ecosystem Awareness"] + 1)

    if diag_count == 0:
        signals["Presentation & Communication of Idea"] = max(0, signals["Presentation & Communication of Idea"] - 2)
        signals["Technical Readiness & Prototype Potential"] = max(0, signals["Technical Readiness & Prototype Potential"] - 1)

    return signals


# ===================== Helpers =====================
ScoreBand = tuple[int, int]


def _stable_team_hash(context) -> int:
    basis = f"{getattr(context, 'team_name', '')}-{getattr(context, 'file_path', '')}-{getattr(context, 'file_hash', '')}"
    if not basis.strip():
        basis = (getattr(context, "raw_text", "") or "")[:128]
    h = hashlib.md5(basis.encode("utf-8")).digest()
    return h[0]


def _target_band(signal: int) -> ScoreBand:
    if signal <= 0:
        return (2, 6)
    if signal == 1:
        return (4, 7)
    if signal == 2:
        return (6, 9)
    return (8, 10)


def _normalize_llm_scores(llm_scores: Dict[str, int], eval_params: EvaluationParameters) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in (llm_scores or {}).items():
        if k in eval_params.criteria:
            out[k] = int(v)
    return out


def _fill_missing_with_band(scores: Dict[str, int], signals: Dict[str, int], eval_params: EvaluationParameters) -> Dict[str, int]:
    out = dict(scores)
    for k in eval_params.criteria:
        if k not in out or not isinstance(out[k], int) or out[k] <= 0:
            band = _target_band(signals.get(k, 0))
            out[k] = round(sum(band) / 2)
    tens = [k for k, v in out.items() if v == 10]
    if len(tens) > 1:
        tens_sorted = sorted(tens, key=lambda x: (-signals.get(x, 0), x))
        for k in tens_sorted[1:]:
            out[k] = 9
    return out


def _spread_within_team(scores: Dict[str, int], signals: Dict[str, int], team_h: int) -> Dict[str, int]:
    out = dict(scores)
    buckets: Dict[int, List[str]] = {}
    for k, v in out.items():
        buckets.setdefault(v, []).append(k)
    for v, keys in buckets.items():
        if len(keys) <= 3:
            continue
        order = sorted(keys, key=lambda x: (-signals.get(x, 0), x))
        seq = [+2, +1, -1, -2, +3, -3, +4, -4]
        for i, k in enumerate(order[3:], start=0):
            off = seq[i % len(seq)]
            if team_h % 2 == 1:
                off = -off if i % 2 == 0 else off
            out[k] = max(1, min(10, out[k] + off))
    if len(set(out.values())) < 4:
        hi = sorted(out.keys(), key=lambda x: (-signals.get(x, 0), x))[:3]
        lo = sorted(out.keys(), key=lambda x: (signals.get(x, 0), x))[:3]
        for k in hi:
            out[k] = min(10, out[k] + 1)
        for k in lo:
            out[k] = max(1, out[k] - 1)
    return out


def _override_format_with_agent(out: Dict[str, int], fmt_0_5: Optional[float], eval_params: EvaluationParameters) -> Dict[str, int]:
    if fmt_0_5 is None:
        return out
    mapped = max(1, min(10, int(round(fmt_0_5 * 2))))
    if "Format of the Presentation" in eval_params.criteria:
        out["Format of the Presentation"] = mapped
    return out


# ===================== Base invoker =====================
class _LLMInvoker:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        self.api_key = api_key
        self.model = os.getenv("OPENAI_MODEL_TEXT", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.timeout_s = int(os.getenv("LLM_TIMEOUT_S", "90"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.limiter = get_text_limiter()

    def _llm(self) -> ChatOpenAI:
        return ChatOpenAI(model=self.model, api_key=self.api_key, temperature=0.0, model_kwargs={"top_p": 0.0})

    async def ainvoke_json(self, messages, parser: JsonOutputParser):
        last_err = None
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                await self.limiter.acquire()
                resp = await asyncio.wait_for(self._llm().ainvoke(messages), timeout=self.timeout_s)
                return parser.parse(getattr(resp, "content", ""))
            except RateLimitError as e:
                last_err = e
                wait_time = delay * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"⚠️ RateLimitError: retrying in {wait_time:.2f}s (attempt {attempt+1})")
                await asyncio.sleep(wait_time)
            except Exception as e:
                last_err = e
            finally:
                try:
                    self.limiter.release()
                except Exception:
                    pass
        raise last_err or RuntimeError("scoring llm failed")


# ===================== Scoring-only agent =====================
class ScoringAgent(_LLMInvoker):
    def __init__(self, eval_params: EvaluationParameters):
        super().__init__()
        self.eval_params = eval_params
        self.parser = JsonOutputParser(pydantic_object=ScoringOutput)

        criteria_list = ", ".join(self.eval_params.criteria.keys())
        prompt_str = f"""
You are a strict hackathon judge. Use BOTH sources with equal weight:
(A) Deck text
(B) Diagram summary extracted from images (diagrams only)

{self.eval_params.rubric}

Diagram Summary (evidence):
{{workflow_report_text}}

Evaluation:

1) Scoring & Summary:
   - Score each EXACT key (INTEGER 1-10):
     {criteria_list}.
   - Apply the following weighting guidance:
     {", ".join(str(self.eval_params.criteria[k]['weight']) + '%' for k in self.eval_params.criteria)}.
   - Provide a concise summary grounded in diagram + text evidence.
   - Do not return the same number for most criteria.

2) Workflow Analysis:
   - If diagrams exist, describe them step-by-step into an overall workflow (key = overall).
   - Else, set workflow_analysis to null.

Output: single JSON object.
Format Instructions:
{{format_instructions}}

Deck Text:
{{document_text}}
"""
        self.prompt = ChatPromptTemplate.from_template(prompt_str)

    async def run(self, context):
        print(f"--- ScoringAgent: {getattr(context, 'file_path', '<unknown>')} ---")
        prompt_text = self.prompt.format(
            workflow_report_text=getattr(context, "workflow_report_text", None) or "(no diagrams found)",
            document_text=getattr(context, "raw_text", None) or "",
            format_instructions=self.parser.get_format_instructions(),
        )
        messages = [HumanMessage(content=[{"type": "text", "text": prompt_text}])]
        parsed = await self.ainvoke_json(messages, self.parser)

        diag_count = _diagram_count(context)
        steps_cnt = _workflow_steps_count(context)
        page_cnt = _page_or_slide_count(context)
        fmt_0_5 = _extract_format_score_from_context(context)

        round1_llm = _normalize_llm_scores(parsed.get("scores", {}) or {}, self.eval_params)
        round1_llm = _override_format_with_agent(round1_llm, fmt_0_5, self.eval_params)

        signals = _evidence_signals(
            getattr(context, "raw_text", "") or "",
            getattr(context, "workflow_report_text", "") or "",
            diag_count, fmt_0_5, steps_cnt, page_cnt
        )
        round1_full = _fill_missing_with_band(round1_llm, signals, self.eval_params)
        team_h = _stable_team_hash(context)
        round1_spread = _spread_within_team(round1_full, signals, team_h)

        scores_out = round1_spread

        workflow_analysis = parsed.get("workflow_analysis")
        if isinstance(workflow_analysis, WorkflowOutput):
            workflow_analysis = workflow_analysis.normalized()
        elif isinstance(workflow_analysis, dict):
            if "overall_summary" in workflow_analysis and "overall" not in workflow_analysis:
                workflow_analysis["overall"] = workflow_analysis.pop("overall_summary")

        context.update_scoring_results(
            parsed.get("team_name", "Unknown"),
            scores_out,
            parsed.get("summary", ""),
            workflow_analysis,
        )


# ===================== Combined scoring + feedback =====================
class CombinedAgent(_LLMInvoker):
    def __init__(self, eval_params: EvaluationParameters):
        super().__init__()
        self.eval_params = eval_params
        self.parser = JsonOutputParser(pydantic_object=CombinedOutput)

        criteria_list = ", ".join(self.eval_params.criteria.keys())
        prompt_str = f"""
You are a strict hackathon judge and mentor. Use deck text + diagram summary equally.
Consider diagrams only as core visual evidence.

{self.eval_params.rubric}

Diagram Summary (evidence):
{{workflow_report_text}}

1) Scoring & Summary:
   - Score each EXACT key (INTEGER 1-10):
     {criteria_list}.
   - Apply the following weighting guidance:
     {", ".join(str(self.eval_params.criteria[k]['weight']) + '%' for k in self.eval_params.criteria)}.
   - Provide a concise evidence-grounded summary.
   - Avoid giving the same number to most criteria.

2) Workflow Analysis:
   - If diagrams exist, describe steps and overall flow (key = overall); else null.

3) Feedback:
   - Return fields positive, criticism, technical, suggestions.
   - Each must be a JSON array of strings (3–6 numbered bullets).

Output: single JSON object.
Format Instructions:
{{format_instructions}}

Deck Text:
{{document_text}}
"""
        self.prompt = ChatPromptTemplate.from_template(prompt_str)

    async def run(self, context):
        print(f"--- CombinedAgent: {getattr(context, 'file_path', '<unknown>')} ---")
        prompt_text = self.prompt.format(
            workflow_report_text=getattr(context, "workflow_report_text", None) or "(no diagrams found)",
            document_text=getattr(context, "raw_text", None) or "",
            format_instructions=self.parser.get_format_instructions(),
        )
        messages = [HumanMessage(content=[{"type": "text", "text": prompt_text}])]
        parsed = await self.ainvoke_json(messages, self.parser)

        diag_count = _diagram_count(context)
        steps_cnt = _workflow_steps_count(context)
        page_cnt = _page_or_slide_count(context)
        fmt_0_5 = _extract_format_score_from_context(context)

        round1_llm = _normalize_llm_scores(parsed.get("scores", {}) or {}, self.eval_params)
        round1_llm = _override_format_with_agent(round1_llm, fmt_0_5, self.eval_params)

        signals = _evidence_signals(
            getattr(context, "raw_text", "") or "",
            getattr(context, "workflow_report_text", "") or "",
            diag_count, fmt_0_5, steps_cnt, page_cnt
        )
        round1_full = _fill_missing_with_band(round1_llm, signals, self.eval_params)
        team_h = _stable_team_hash(context)
        round1_spread = _spread_within_team(round1_full, signals, team_h)

        scores_out = round1_spread

        workflow_analysis = parsed.get("workflow_analysis")
        if isinstance(workflow_analysis, WorkflowOutput):
            workflow_analysis = workflow_analysis.normalized()
        elif isinstance(workflow_analysis, dict):
            if "overall_summary" in workflow_analysis and "overall" not in workflow_analysis:
                workflow_analysis["overall"] = workflow_analysis.pop("overall_summary")

        feedback_obj = parsed.get("feedback", {})
        context.update_scoring_results(
            parsed.get("team_name", "Unknown"),
            scores_out,
            parsed.get("summary", ""),
            workflow_analysis,
        )
        
        # Handle feedback - JsonOutputParser returns dict, not Pydantic model
        if feedback_obj:
            if isinstance(feedback_obj, dict):
                context.update_feedback_results(feedback_obj)
            elif hasattr(feedback_obj, 'model_dump'):
                context.update_feedback_results(feedback_obj.model_dump())
            elif hasattr(feedback_obj, 'dict'):
                context.update_feedback_results(feedback_obj.dict())
            else:
                try:
                    context.update_feedback_results(dict(feedback_obj))
                except Exception:
                    print(f"  -> Warning: Could not convert feedback_obj to dict, type: {type(feedback_obj)}")
        
        print("  -> Combined scoring + feedback complete.")
