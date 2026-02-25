"""Answer relevance classifier primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

ClassificationLabel = Literal["on_topic", "off_topic", "stop"]


class LLMClientProtocol(Protocol):
    async def ainvoke(self, messages: list[dict[str, str]]): ...


class AnswerRelevanceClassifierProtocol(Protocol):
    async def classify(
        self,
        current_question: str,
        current_answer: str,
        questionnaire: str,
    ) -> ClassificationLabel: ...


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

    def _parse_classification_label(self, content: str) -> ClassificationLabel | None:
        normalized = content.upper()
        labels = re.findall(r"\b(ON_TOPIC|OFF_TOPIC|STOP)\b", normalized)
        if not labels:
            return None
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            return None
        label = labels[0]
        if re.search(
            r"\b(?:NOT|NO|NIET|GEEN|PAS|NE)\s+(?:OFF_TOPIC|ON_TOPIC|STOP)\b",
            normalized,
        ):
            return None
        if label == "ON_TOPIC":
            return "on_topic"
        if label == "OFF_TOPIC":
            return "off_topic"
        return "stop"

    async def classify(
        self,
        current_question: str,
        current_answer: str,
        questionnaire: str,
    ) -> ClassificationLabel:
        system_prompt = self.system_prompt_template.format(
            questionnaire=questionnaire,
            current_question=current_question,
            current_answer=current_answer,
        )
        response = await self._get_client().ainvoke(
            [{"role": "system", "content": system_prompt}]
        )
        content = self._extract_content(response)
        parsed_label = self._parse_classification_label(content)
        if parsed_label is None:
            return "on_topic"
        return parsed_label
