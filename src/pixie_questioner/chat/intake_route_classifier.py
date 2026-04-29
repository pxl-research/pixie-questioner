"""LLM-based intake route classification for categorized handoff flows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ..llm_client import extract_llm_content, resolve_llm_client
from .types import (
    INTAKE_ROUTE_FOOD_PAIRING,
    INTAKE_ROUTE_UNKNOWN,
)


class IntakeRouteClassifierProtocol(Protocol):
    async def classify(self, current_answer: str) -> str: ...


@dataclass
class LLMIntakeRouteClassifier:
    llm: object
    system_prompt_template: str

    def _parse_route_label(self, content: str) -> str | None:
        normalized = content.upper()
        labels = re.findall(r"\b(FOOD_PAIRING|UNKNOWN)\b", normalized)
        if not labels:
            return None
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            return None
        label = labels[0]
        if re.search(
            r"\b(?:NOT|NO|NIET|GEEN|PAS|NE)\s+" r"(?:FOOD_PAIRING|UNKNOWN)\b",
            normalized,
        ):
            return None
        if label == "FOOD_PAIRING":
            return INTAKE_ROUTE_FOOD_PAIRING
        return INTAKE_ROUTE_UNKNOWN

    async def classify(self, current_answer: str) -> str:
        system_prompt = self.system_prompt_template.format(
            current_answer=current_answer,
        )
        response = await resolve_llm_client(self.llm).ainvoke(
            [{"role": "system", "content": system_prompt}]
        )
        content = extract_llm_content(response)
        parsed_label = self._parse_route_label(content)
        if parsed_label is None:
            return INTAKE_ROUTE_UNKNOWN
        return parsed_label
