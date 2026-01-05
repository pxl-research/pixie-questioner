"""Chat orchestration service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Sequence


class LLMClientProtocol(Protocol):
    async def ainvoke(self, messages: list[dict[str, str]]):
        ...

    def astream(self, messages: list[dict[str, str]]) -> AsyncIterator[object]:
        ...


@dataclass
class ChatService:
    llm: object
    system_prompt: str

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

    async def ask(self, question: str, session_id: str, history: Sequence[str]) -> str:
        messages = self._build_messages(question, session_id, history)
        response = await self._get_client().ainvoke(messages)
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        return content

    async def stream_answer(
        self, question: str, session_id: str, history: Sequence[str]
    ) -> AsyncIterator[str]:
        messages = self._build_messages(question, session_id, history)
        async for chunk in self._get_client().astream(messages):
            text = getattr(chunk, "content", None)
            if text is None:
                text = str(chunk)
            if text:
                yield text
