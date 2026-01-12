"""Chat orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Sequence

from .summarizer import ConversationSummarizer

DEFAULT_SUMMARY_TRIGGER_TOKEN = "PIXIE_SUMMARY_READY"


class LLMClientProtocol(Protocol):
    async def ainvoke(self, messages: list[dict[str, str]]):
        ...

    def astream(self, messages: list[dict[str, str]]) -> AsyncIterator[object]:
        ...


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

    def _completion_tokens(self) -> tuple[str, ...]:
        tokens = ["none"]
        if self.completion_token:
            tokens.append(self.completion_token)
        return tuple(tokens)

    def _is_completion_signal(self, content: str) -> bool:
        normalized = content.strip().casefold()
        return any(normalized == token.strip().casefold() for token in self._completion_tokens())

    def _is_completion_prefix(self, content: str) -> bool:
        normalized = content.strip().casefold()
        if not normalized:
            return True
        return any(
            token.strip().casefold().startswith(normalized)
            for token in self._completion_tokens()
        )

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
    ) -> str:
        messages = self._build_messages(question, session_id, history)
        response = await self._get_client().ainvoke(messages)
        content = self._extract_content(response)
        if self._is_completion_signal(content):
            summary = await self._summarize(question, history)
            if summary:
                return summary
        return content

    async def stream_answer(
        self, question: str, session_id: str, history: Sequence[str] | None
    ) -> AsyncIterator[str]:
        messages = self._build_messages(question, session_id, history)
        pending_chunks: list[str] = []
        pending_text = ""
        streaming_direct = False
        async for chunk in self._get_client().astream(messages):
            text = getattr(chunk, "content", None)
            if text is None:
                text = str(chunk)
            if not text:
                continue
            if streaming_direct:
                yield text
                continue
            pending_chunks.append(text)
            pending_text = f"{pending_text}{text}"
            if not self._is_completion_prefix(pending_text):
                streaming_direct = True
                for pending in pending_chunks:
                    yield pending
                pending_chunks = []
                pending_text = ""

        if pending_chunks:
            if not streaming_direct and self._is_completion_signal(pending_text):
                summary = await self._summarize(question, history)
                if summary:
                    yield summary
                else:
                    for pending in pending_chunks:
                        yield pending
            else:
                for pending in pending_chunks:
                    yield pending
