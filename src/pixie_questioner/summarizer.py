"""Conversation summarizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


DEFAULT_SUMMARY_PROMPT = (
    "Given the answers of a user and the full conversation history, reformulate what "
    "the user wants in 1 paragraph without leaving away anything the user says.\n"
    "Also look into the reference conversation history for missing details and more context.\n"
    "Then summarize without losing information.\n"
    "Internally analyze in which language the conversation is. Use this language in the summary.\n\n"
    "User answers:\n"
    "{user_answers}\n\n"
    "Given conversation as a reference:\n"
    "{conversation_history}\n\n"
    "Summary (translated to the language of the conversation history):"
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

    def _build_prompt(self, conversation_history: Sequence[str]) -> str:
        user_answers = "\n".join(
            item.strip()
            for item in conversation_history
            if item.strip().lower().startswith("user:")
        )
        conversation_history_str = "\n".join(conversation_history)
        return self.system_prompt_template.format(
            user_answers=user_answers,
            conversation_history=conversation_history_str,
        )

    async def generate(self, conversation_history: Sequence[str]) -> str | None:
        system_prompt = self._build_prompt(conversation_history)
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
