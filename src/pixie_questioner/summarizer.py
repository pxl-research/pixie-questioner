"""Conversation summarizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


DEFAULT_SUMMARY_PROMPT = (
    "Given:\n"
    "* The current question of an AI\n"
    "* The answer of a user\n"
    "* The summary of previous information.\n"
    "Summarize everything what the user wants in 1 paragraph without losing any information.\n"
    "Formulate in a gender-neutral way.\n"
    "Internally analyze in which language all this information is. Use this language in the summary.\n"
    "\n"
    "AI current question:\n"
    "{current_question}\n"
    "User current answer:\n"
    "{current_answer}\n"
    "Summary of previous information:\n"
    "{summary}\n"
    "\n"
    "New summary of what user wants (translated to the language of all this information):"
)


class LLMClientProtocol(Protocol):
    async def ainvoke(self, messages: list[dict[str, str]]):
        ...


@dataclass
class ConversationSummarizer:
    llm: object
    system_prompt_template: str = DEFAULT_SUMMARY_PROMPT

    def _get_client(self) -> LLMClientProtocol:
        if hasattr(self.llm, "ainvoke"):
            return self.llm  # type: ignore[return-value]
        model = getattr(self.llm, "model", None)
        if model is None:
            raise RuntimeError("LLM client is not configured.")
        return model

    def _build_prompt(
        self, current_question: str, current_answer: str, summary: str
    ) -> str:
        return self.system_prompt_template.format(
            current_question=current_question,
            current_answer=current_answer,
            summary=summary,
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
