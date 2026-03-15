"""Cretskens chat orchestration between intake and recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .chat.types import (
    INTAKE_ROUTE_FOOD_PAIRING,
    INTAKE_ROUTE_GENERAL_WINE,
    PHASE_COMPLETED as CHAT_PHASE_COMPLETED,
    ChatResult,
)
from .wine_advisor.types import (
    MEAL_MODE_DRINKING_ONLY,
    MEAL_MODE_PAIRING,
    MEAL_MODE_UNKNOWN,
    WineAdvisorResult,
)

CRETSKENS_FLOW_KEY = "cretskens_flow"
CRETSKENS_FLOW_INTAKE = "intake_chat"
CRETSKENS_FLOW_ADVISOR = "advisor"
CRETSKENS_INTAKE_STATE_KEY = "intake_state"
CRETSKENS_ADVISOR_STATE_KEY = "advisor_state"
_LEGACY_ADVISOR_STATE_KEYS = frozenset(
    {"phase", "turn_count", "meal_mode", "working_context", "chat_history"}
)


class CretskensServiceUnavailableError(RuntimeError):
    """Raised when a required LLM-backed Cretskens step is unavailable."""


class IntakeServiceProtocol(Protocol):
    async def ask(
        self,
        current_answer: str,
        state: dict[str, Any] | None,
    ) -> ChatResult: ...


class RecommendationServiceProtocol(Protocol):
    async def ask(
        self,
        *,
        current_answer: str,
        state: dict[str, Any] | None,
    ) -> WineAdvisorResult: ...

    async def recommend_from_context(
        self,
        *,
        working_context: str,
        meal_mode: str,
    ) -> WineAdvisorResult: ...


@dataclass
class CretskensService:
    intake_service: IntakeServiceProtocol
    recommendation_service: RecommendationServiceProtocol

    async def ask(
        self,
        *,
        current_answer: str,
        state: dict[str, Any] | None,
    ) -> WineAdvisorResult:
        flow, payload = self._parse_state(state)
        if flow == CRETSKENS_FLOW_ADVISOR:
            try:
                advisor_result = await self.recommendation_service.ask(
                    current_answer=current_answer,
                    state=payload,
                )
            except Exception as exc:
                raise CretskensServiceUnavailableError(
                    "Cretskens recommendation flow unavailable."
                ) from exc
            return self._wrap_advisor_result(advisor_result)

        try:
            intake_result = await self.intake_service.ask(current_answer, payload)
        except Exception as exc:
            raise CretskensServiceUnavailableError(
                "Cretskens intake flow unavailable."
            ) from exc

        if intake_result.state.get("phase") != CHAT_PHASE_COMPLETED:
            return self._build_intake_question_result(intake_result)

        try:
            advisor_result = await self.recommendation_service.recommend_from_context(
                working_context=self._build_working_context(intake_result),
                meal_mode=self._meal_mode_for_intake_state(intake_result.state),
            )
        except Exception as exc:
            raise CretskensServiceUnavailableError(
                "Cretskens recommendation flow unavailable."
            ) from exc
        return self._wrap_advisor_result(advisor_result)

    def _parse_state(
        self,
        state: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | None]:
        if not state:
            return CRETSKENS_FLOW_INTAKE, None

        flow = state.get(CRETSKENS_FLOW_KEY)
        if flow == CRETSKENS_FLOW_ADVISOR and isinstance(
            state.get(CRETSKENS_ADVISOR_STATE_KEY), dict
        ):
            return CRETSKENS_FLOW_ADVISOR, state[CRETSKENS_ADVISOR_STATE_KEY]
        if flow == CRETSKENS_FLOW_INTAKE and isinstance(
            state.get(CRETSKENS_INTAKE_STATE_KEY), dict
        ):
            return CRETSKENS_FLOW_INTAKE, state[CRETSKENS_INTAKE_STATE_KEY]
        if _LEGACY_ADVISOR_STATE_KEYS.intersection(state):
            return CRETSKENS_FLOW_ADVISOR, state
        return CRETSKENS_FLOW_INTAKE, None

    def _build_intake_question_result(
        self,
        intake_result: ChatResult,
    ) -> WineAdvisorResult:
        return WineAdvisorResult(
            text=intake_result.text,
            response_type="question",
            state={
                CRETSKENS_FLOW_KEY: CRETSKENS_FLOW_INTAKE,
                CRETSKENS_INTAKE_STATE_KEY: intake_result.state,
            },
            recommendations=None,
        )

    def _wrap_advisor_result(
        self,
        advisor_result: WineAdvisorResult,
    ) -> WineAdvisorResult:
        return WineAdvisorResult(
            text=advisor_result.text,
            response_type=advisor_result.response_type,
            state={
                CRETSKENS_FLOW_KEY: CRETSKENS_FLOW_ADVISOR,
                CRETSKENS_ADVISOR_STATE_KEY: advisor_result.state,
            },
            recommendations=advisor_result.recommendations,
        )

    def _meal_mode_for_intake_state(self, state: dict[str, Any]) -> str:
        intake_route = state.get("intake_route")
        if intake_route == INTAKE_ROUTE_FOOD_PAIRING:
            return MEAL_MODE_PAIRING
        if intake_route == INTAKE_ROUTE_GENERAL_WINE:
            return MEAL_MODE_DRINKING_ONLY
        return MEAL_MODE_UNKNOWN

    def _build_working_context(self, intake_result: ChatResult) -> str:
        parts: list[str] = []
        summary = intake_result.state.get("summary")
        if isinstance(summary, str) and summary.strip():
            parts.append(f"Summary:\n{summary.strip()}")
        questionnaire = intake_result.state.get("questionnaire")
        if isinstance(questionnaire, str) and questionnaire.strip():
            parts.append(f"Questionnaire:\n{questionnaire.strip()}")
        handoff_categories = intake_result.state.get("handoff_categories_v1")
        if isinstance(handoff_categories, str) and handoff_categories.strip():
            parts.append(f"Handoff categories:\n{handoff_categories.strip()}")
        if not parts:
            raise RuntimeError(
                "Completed intake state is missing recommendation context."
            )
        return "\n\n".join(parts)
