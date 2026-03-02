"""LLM-backed decision maker for question vs recommendation turns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ValidationError, model_validator

from ..llm_client import extract_llm_content, resolve_llm_client

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)


class _TurnDecisionPayload(BaseModel):
    action: Literal["question", "recommend"]
    question_text: str | None = None
    meal_mode: Literal["pairing", "drinking_only", "unknown"] = "unknown"
    query_hints: str = ""

    @model_validator(mode="after")
    def validate_question_for_question_action(self) -> "_TurnDecisionPayload":
        if self.action == "question" and not (self.question_text or "").strip():
            raise ValueError("question_text is required when action is question")
        return self


@dataclass(frozen=True)
class TurnDecision:
    action: Literal["question", "recommend"]
    question_text: str
    meal_mode: Literal["pairing", "drinking_only", "unknown"]
    query_hints: str


@dataclass
class LLMTurnController:
    llm: object
    system_prompt_template: str

    def _render_prompt(
        self,
        current_answer: str,
        working_context: str,
        turn_count: int,
        max_turns: int,
        current_meal_mode: str,
        last_question: str,
        conversation_history: str,
    ) -> str:
        prompt = self.system_prompt_template
        prompt = prompt.replace("{current_answer}", current_answer)
        prompt = prompt.replace("{working_context}", working_context)
        prompt = prompt.replace("{turn_count}", str(turn_count))
        prompt = prompt.replace("{max_turns}", str(max_turns))
        prompt = prompt.replace("{current_meal_mode}", current_meal_mode)
        prompt = prompt.replace("{last_question}", last_question)
        prompt = prompt.replace("{conversation_history}", conversation_history)
        return prompt

    def _parse_payload(self, content: str) -> _TurnDecisionPayload:
        try:
            raw_payload = json.loads(content)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_PATTERN.search(content)
            if match is None:
                raise RuntimeError("Invalid turn controller payload.")
            raw_payload = json.loads(match.group(0))
        try:
            return _TurnDecisionPayload.model_validate(raw_payload)
        except ValidationError as exc:
            raise RuntimeError("Invalid turn controller payload.") from exc

    async def decide(
        self,
        *,
        current_answer: str,
        working_context: str,
        turn_count: int,
        max_turns: int,
        current_meal_mode: str = "unknown",
        last_question: str = "",
        conversation_history: str = "",
    ) -> TurnDecision:
        prompt = self._render_prompt(
            current_answer=current_answer,
            working_context=working_context,
            turn_count=turn_count,
            max_turns=max_turns,
            current_meal_mode=current_meal_mode,
            last_question=last_question,
            conversation_history=conversation_history,
        )
        response = await resolve_llm_client(self.llm).ainvoke(
            [{"role": "system", "content": prompt}]
        )
        payload = self._parse_payload(extract_llm_content(response))

        question_text = (payload.question_text or "").strip()
        if payload.action == "question" and not question_text:
            raise RuntimeError("Turn controller returned empty question_text.")

        return TurnDecision(
            action=payload.action,
            question_text=question_text,
            meal_mode=payload.meal_mode,
            query_hints=payload.query_hints.strip(),
        )
