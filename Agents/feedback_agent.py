import os
import asyncio
from typing import Dict, Any, List

from pydantic import BaseModel, field_validator
from openai import OpenAI

from utils import get_text_limiter, EvaluationParameters

class FeedbackOnly(BaseModel):
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

SYSTEM = (
    "You are a concise hackathon mentor. Give candid, actionable feedback. "
    "Cite slide or diagram indices when obvious. Output JSON only."
)

class FeedbackAgent:
    def __init__(self, eval_params: EvaluationParameters):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        self.client = OpenAI(api_key=api_key)
        self.eval_params = eval_params
        self.model = os.getenv("OPENAI_MODEL_TEXT", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        self.timeout_s = int(os.getenv("LLM_TIMEOUT_S", "90"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        self.limiter = get_text_limiter()

        criteria_list = "\n".join(
            f"{i}) {k}" for i, k in enumerate(self.eval_params.criteria, start=1)
        )
        self.prompt_template = f"""
Inputs:
- Deck text (truncated)
- Judge summary
- Diagram summary

Task:
Return JSON with fields: positive, criticism, technical, suggestions.
Each field should be a JSON array of strings, with 3–6 numbered bullets per field. Be specific.
Ensure feedback spans these parameters:
{criteria_list}

Diagram summary:
{{diagram_summary}}

Judge summary:
{{judge_summary}}

Deck text excerpt:
{{deck_text}}
"""

    async def _ainvoke(self, content_text: str) -> Dict[str, Any]:
        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                await self.limiter.acquire()
                resp = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": content_text},
                    ],
                    temperature=0.0,
                )
                text = (resp.choices[0].message.content or "").strip()
                import json
                js = self._extract_first_json(text) or text
                return json.loads(js)
            except Exception as e:
                last_err = e
        raise last_err or RuntimeError("feedback llm failed")

    @staticmethod
    def _extract_first_json(text: str) -> str | None:
        if "```json" in text:
            try:
                return text.split("```json", 1)[1].split("```", 1)[0].strip()
            except Exception:
                pass
        # brace match fallback
        start = -1
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start:i+1]
        return None

    async def aevaluate(self, context) -> None:
        diag_text = (context.workflow_report_text or "").strip() or "(no diagrams detected)"
        prompt = self.prompt_template.format(
            diagram_summary=diag_text,
            judge_summary=context.scoring_summary or "(no judge summary)",
            deck_text=(context.raw_text or "")[:6000],
        )
        parsed = await self._ainvoke(prompt)
        # Validate to schema
        fb = FeedbackOnly.model_validate(parsed)
        context.update_feedback_results(fb.model_dump())

    async def run(self, context) -> None:
        await self.aevaluate(context)