"""Chat orchestration service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Sequence

from .summarizer import ConversationSummarizer

DEFAULT_SUMMARY_TRIGGER_TOKEN = "PIXIE_SUMMARY_READY"
_COMPLETION_STRIP_CHARS = " \t\n\r\"'`.,:;!?()[]{}<>"
_LEGACY_NONE_TOKEN = "none"


class LLMClientProtocol(Protocol):
    async def ainvoke(self, messages: list[dict[str, str]]):
        ...

    def astream(self, messages: list[dict[str, str]]) -> AsyncIterator[object]:
        ...


@dataclass(frozen=True)
class ChatResult:
    text: str
    response_type: str


@dataclass
class ChatService:
    llm: object
    system_prompt: str
    completion_token: str = DEFAULT_SUMMARY_TRIGGER_TOKEN
    summarizer: ConversationSummarizer | None = None

    def _get_client(self) -> LLMClientProtocol:
        if hasattr(self.llm, "ainvoke") and hasattr(self.llm, "astream"):
            return self.llm  # type: ignore[return-value]
        model = getattr(self.llm, "model", None)
        if model is None:
            raise RuntimeError("LLM client is not configured.")
        return model

    def _build_messages(
        self, question: str, session_id: str, history: Sequence[str] | None
    ) -> list[dict[str, str]]:
        system_content = f"{self.system_prompt}\n\nSession ID: {session_id}"
        if history:
            history_block = "\n".join(f"- {item}" for item in history)
            system_content = f"{system_content}\n\nHistory:\n{history_block}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

    def _extract_content(self, response: object) -> str:
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        return content

    def _normalize_completion_candidate(self, content: str) -> str:
        return content.strip(_COMPLETION_STRIP_CHARS).casefold()

    def _contains_completion_token(self, content: str) -> bool:
        if not self.completion_token:
            return False
        return re.search(re.escape(self.completion_token), content, re.IGNORECASE) is not None

    def _is_legacy_none_signal(self, content: str) -> bool:
        normalized = self._normalize_completion_candidate(content)
        return normalized == _LEGACY_NONE_TOKEN

    def _strip_completion_token(self, content: str) -> str:
        if not self.completion_token:
            return content.strip(_COMPLETION_STRIP_CHARS)
        stripped = re.sub(re.escape(self.completion_token), "", content, flags=re.IGNORECASE)
        return stripped.strip(_COMPLETION_STRIP_CHARS)

    def _build_summary_history(
        self, question: str, history: Sequence[str] | None
    ) -> list[str]:
        summary_history = list(history) if history else []
        summary_history.append(f"User: {question}")
        return summary_history

    async def _summarize(
        self, question: str, history: Sequence[str] | None
    ) -> str | None:
        if self.summarizer is None:
            return None
        summary_history = self._build_summary_history(question, history)
        return await self.summarizer.generate(summary_history)

    async def ask(
        self, question: str, session_id: str, history: Sequence[str] | None
    ) -> ChatResult:
        messages = self._build_messages(question, session_id, history)
        response = await self._get_client().ainvoke(messages)
        content = self._extract_content(response)
        if self._contains_completion_token(content) or self._is_legacy_none_signal(content):
            summary = await self._summarize(question, history)
            if summary is not None:
                return ChatResult(text=summary, response_type="summary")
            if self._contains_completion_token(content):
                return ChatResult(
                    text=self._strip_completion_token(content),
                    response_type="answer",
                )
            return ChatResult(text="", response_type="answer")
        return ChatResult(text=content, response_type="answer")

    async def stream_answer(
        self, question: str, session_id: str, history: Sequence[str] | None
    ) -> AsyncIterator[str]:
        messages = self._build_messages(question, session_id, history)
        chunks: list[str] = []
        async for chunk in self._get_client().astream(messages):
            text = getattr(chunk, "content", None)
            if text is None:
                text = str(chunk)
            if not text:
                continue
            chunks.append(text)

        if not chunks:
            return

        full_text = "".join(chunks)
        if self._contains_completion_token(full_text) or self._is_legacy_none_signal(full_text):
            summary = await self._summarize(question, history)
            if summary is not None:
                if summary:
                    yield summary
                return
            if self._contains_completion_token(full_text):
                fallback = self._strip_completion_token(full_text)
                if fallback:
                    yield fallback
            return

        for chunk in chunks:
            yield chunk
