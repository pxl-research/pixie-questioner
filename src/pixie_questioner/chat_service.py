"""Chat orchestration service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Sequence

from .summarizer import ConversationSummarizer

DEFAULT_SUMMARY_TRIGGER_TOKEN = "PIXIE_SUMMARY_READY"
_COMPLETION_STRIP_CHARS = " \t\n\r\"'`.,:;!?()[]{}<>"


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

    def _render_system_prompt(self, questionnaire: str | None) -> str:
        prompt = self.system_prompt
        if "{summary}" in prompt:
            prompt = prompt.replace("{summary}", questionnaire or "")
        if "{completion_token}" in prompt:
            prompt = prompt.replace("{completion_token}", self.completion_token or "")
        return prompt

    def _build_messages(
        self,
        question: str,
        session_id: str,
        history: Sequence[str] | None,
        questionnaire: str | None,
    ) -> list[dict[str, str]]:
        system_prompt = self._render_system_prompt(questionnaire)
        system_content = f"{system_prompt}\n\nSession ID: {session_id}"
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

    def _contains_completion_token(self, content: str) -> bool:
        if not self.completion_token:
            return False
        return re.search(re.escape(self.completion_token), content, re.IGNORECASE) is not None

    def _strip_completion_token(self, content: str) -> str:
        if not self.completion_token:
            return content
        stripped = re.sub(re.escape(self.completion_token), "", content, flags=re.IGNORECASE)
        return stripped.strip(_COMPLETION_STRIP_CHARS)

    async def _summarize(
        self, current_question: str | None, current_answer: str, summary: str | None
    ) -> str | None:
        if self.summarizer is None:
            return None
        return await self.summarizer.generate(
            current_question=current_question or "",
            current_answer=current_answer,
            summary=summary or "",
        )

    async def ask(
        self,
        question: str,
        session_id: str,
        history: Sequence[str] | None,
        questionnaire: str | None = None,
        current_question: str | None = None,
        summary: str | None = None,
        summarize: bool = False,
    ) -> ChatResult:
        if summarize:
            summary_text = await self._summarize(current_question, question, summary)
            return ChatResult(text=summary_text or "", response_type="summary")

        messages = self._build_messages(question, session_id, history, questionnaire)
        response = await self._get_client().ainvoke(messages)
        content = self._extract_content(response)
        if self._contains_completion_token(content):
            content = self._strip_completion_token(content)
        return ChatResult(text=content, response_type="answer")

    async def stream_answer(
        self,
        question: str,
        session_id: str,
        history: Sequence[str] | None,
        questionnaire: str | None = None,
        current_question: str | None = None,
        summary: str | None = None,
        summarize: bool = False,
    ) -> AsyncIterator[str]:
        if summarize:
            summary_text = await self._summarize(current_question, question, summary)
            if summary_text:
                yield summary_text
            return

        messages = self._build_messages(question, session_id, history, questionnaire)
        if not self.completion_token:
            async for chunk in self._get_client().astream(messages):
                text = getattr(chunk, "content", None)
                if text is None:
                    text = str(chunk)
                if text:
                    yield text
            return

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
        if self._contains_completion_token(full_text):
            cleaned = self._strip_completion_token(full_text)
            if cleaned:
                yield cleaned
            return

        for chunk in chunks:
            yield chunk
