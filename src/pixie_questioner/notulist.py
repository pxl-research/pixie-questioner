"""Questionnaire notulist for structured updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMClientProtocol(Protocol):
    async def ainvoke(self, messages: list[dict[str, str]]):
        ...


@dataclass
class QuestionnaireNotulist:
    llm: object
    system_prompt_template: str

    def _get_client(self) -> LLMClientProtocol:
        if hasattr(self.llm, "ainvoke"):
            return self.llm  # type: ignore[return-value]
        model = getattr(self.llm, "model", None)
        if model is None:
            raise RuntimeError("LLM client is not configured.")
        return model

    def _build_prompt(self, current_question: str, current_answer: str, summary: str) -> str:
        return self.system_prompt_template.format(
            summary=summary,
            current_question=current_question,
            current_answer=current_answer,
        )

    async def generate(
        self, current_question: str, current_answer: str, summary: str
    ) -> str | None:
        system_prompt = self._build_prompt(current_question, current_answer, summary)
        try:
            response = await self._get_client().ainvoke(
                [{"role": "system", "content": system_prompt}]
            )
        except Exception:
            return None

        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        return content
