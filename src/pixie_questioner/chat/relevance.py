"""Answer relevance classifier primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class LLMClientProtocol(Protocol):
    async def ainvoke(self, messages: list[dict[str, str]]): ...


class AnswerRelevanceClassifierProtocol(Protocol):
    async def is_on_topic(
        self,
        current_question: str,
        current_answer: str,
        questionnaire: str,
    ) -> bool: ...


@dataclass
class LLMAnswerRelevanceClassifier:
    llm: object
    system_prompt_template: str

    def _get_client(self) -> LLMClientProtocol:
        if hasattr(self.llm, "ainvoke"):
            return self.llm  # type: ignore[return-value]
        model = getattr(self.llm, "model", None)
        if model is None:
            raise RuntimeError("LLM client is not configured.")
        return model

    def _extract_content(self, response: object) -> str:
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        return content

    def _parse_relevance_label(self, content: str) -> bool | None:
        normalized = content.upper()
        labels = re.findall(r"\b(ON_TOPIC|OFF_TOPIC)\b", normalized)
        if not labels:
            return None
        if len(set(labels)) > 1:
            return None
        label = labels[0]
        if re.search(r"\b(?:NOT|NO|NIET|GEEN|PAS|NE)\s+OFF_TOPIC\b", normalized):
            return None
        if re.search(r"\b(?:NOT|NO|NIET|GEEN|PAS|NE)\s+ON_TOPIC\b", normalized):
            return None
        return label == "ON_TOPIC"

    async def is_on_topic(
        self,
        current_question: str,
        current_answer: str,
        questionnaire: str,
    ) -> bool:
        system_prompt = self.system_prompt_template.format(
            questionnaire=questionnaire,
            current_question=current_question,
            current_answer=current_answer,
        )
        response = await self._get_client().ainvoke(
            [{"role": "system", "content": system_prompt}]
        )
        content = self._extract_content(response)
        parsed_label = self._parse_relevance_label(content)
        if parsed_label is None:
            return True
        return parsed_label
