"""Cretskens chat orchestration between intake and recommendations."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
CRETSKENS_FEEDBACK_STATE_KEY = "feedback_state"
CRETSKENS_FEEDBACK_SUMMARY_TEXT_KEY = "feedback_summary_text"
_LEGACY_ADVISOR_STATE_KEYS = frozenset(
    {"phase", "turn_count", "meal_mode", "working_context", "chat_history"}
)
_ADVISOR_PHASE_COMPLETED = "completed"

_FEEDBACK_STEP_EASE_OF_USE = "ease_of_use"
_FEEDBACK_STEP_REUSE = "reuse"
_FEEDBACK_STEP_SATISFACTION = "satisfaction"
_FEEDBACK_STEP_DONE = "done"
_FEEDBACK_STEPS = (
    _FEEDBACK_STEP_EASE_OF_USE,
    _FEEDBACK_STEP_REUSE,
    _FEEDBACK_STEP_SATISFACTION,
    _FEEDBACK_STEP_DONE,
)

_FEEDBACK_QUESTION_EASE_OF_USE = "Hoe makkelijk vond je de assistent in gebruik? 1-5"
_FEEDBACK_QUESTION_REUSE = "Zou je me nog eens gebruiken? ja/nee"
_FEEDBACK_QUESTION_SATISFACTION = (
    "In welke mate ben je tevreden met de wijnsuggesties? 1-5"
)
_FEEDBACK_THANKS = "Bedankt voor je feedback."

_RATING_QUICK_REPLIES: list[dict[str, str]] = [
    {"label": "1", "value": "1"},
    {"label": "2", "value": "2"},
    {"label": "3", "value": "3"},
    {"label": "4", "value": "4"},
    {"label": "5", "value": "5"},
]
_YES_NO_QUICK_REPLIES: list[dict[str, str]] = [
    {"label": "ja", "value": "ja"},
    {"label": "nee", "value": "nee"},
]


@dataclass(frozen=True)
class CretskensFeedbackState:
    step: str = _FEEDBACK_STEP_EASE_OF_USE
    prompt_issued: bool = False
    ease_of_use: str = ""
    reuse: str = ""
    suggestions_satisfaction: str = ""
    summary_text: str = ""


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
        flow, payload, feedback_state = self._parse_state(state)
        if flow == CRETSKENS_FLOW_ADVISOR:
            if self._is_feedback_active(payload, feedback_state):
                return self._build_feedback_result(
                    payload or {},
                    feedback_state or CretskensFeedbackState(),
                    current_answer,
                )
            try:
                advisor_result = await self.recommendation_service.ask(
                    current_answer=current_answer,
                    state=payload,
                )
            except Exception as exc:
                raise CretskensServiceUnavailableError(
                    "Cretskens recommendation flow unavailable."
                ) from exc
            return self._wrap_advisor_result(
                advisor_result,
                feedback_state=feedback_state,
            )

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
        return self._wrap_advisor_result(advisor_result, feedback_state=None)

    def _parse_state(
        self,
        state: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any] | None, CretskensFeedbackState | None]:
        if not state:
            return CRETSKENS_FLOW_INTAKE, None, None

        flow = state.get(CRETSKENS_FLOW_KEY)
        feedback_state = self._parse_feedback_state(
            state.get(CRETSKENS_FEEDBACK_STATE_KEY)
        )
        if flow == CRETSKENS_FLOW_ADVISOR and isinstance(
            state.get(CRETSKENS_ADVISOR_STATE_KEY), dict
        ):
            advisor_state = state[CRETSKENS_ADVISOR_STATE_KEY]
            summary_text = self._feedback_summary_text_from_advisor_state(advisor_state)
            if feedback_state is None and self._advisor_is_completed(advisor_state):
                feedback_state = CretskensFeedbackState(summary_text=summary_text)
            elif (
                feedback_state is not None
                and not feedback_state.summary_text.strip()
                and self._advisor_is_completed(advisor_state)
            ):
                feedback_state = replace(feedback_state, summary_text=summary_text)
            return CRETSKENS_FLOW_ADVISOR, advisor_state, feedback_state
        if flow == CRETSKENS_FLOW_INTAKE and isinstance(
            state.get(CRETSKENS_INTAKE_STATE_KEY), dict
        ):
            return CRETSKENS_FLOW_INTAKE, state[CRETSKENS_INTAKE_STATE_KEY], None
        if _LEGACY_ADVISOR_STATE_KEYS.intersection(state):
            summary_text = self._feedback_summary_text_from_advisor_state(state)
            if feedback_state is None and self._advisor_is_completed(state):
                feedback_state = CretskensFeedbackState(summary_text=summary_text)
            elif (
                feedback_state is not None
                and not feedback_state.summary_text.strip()
                and self._advisor_is_completed(state)
            ):
                feedback_state = replace(feedback_state, summary_text=summary_text)
            return CRETSKENS_FLOW_ADVISOR, state, feedback_state
        return CRETSKENS_FLOW_INTAKE, None, None

    def _parse_feedback_state(self, raw_value: Any) -> CretskensFeedbackState | None:
        if not isinstance(raw_value, dict):
            return None
        step = raw_value.get("step")
        if not isinstance(step, str) or step not in _FEEDBACK_STEPS:
            step = _FEEDBACK_STEP_EASE_OF_USE
        prompt_issued_raw = raw_value.get("prompt_issued")
        prompt_issued = False
        if isinstance(prompt_issued_raw, bool):
            prompt_issued = prompt_issued_raw
        elif isinstance(prompt_issued_raw, str):
            prompt_issued = prompt_issued_raw.strip().casefold() in {
                "true",
                "1",
                "yes",
            }
        ease_of_use = raw_value.get("ease_of_use", "")
        reuse = raw_value.get("reuse", "")
        suggestions_satisfaction = raw_value.get("suggestions_satisfaction", "")
        summary_text = raw_value.get("summary_text", "")
        return CretskensFeedbackState(
            step=step,
            prompt_issued=prompt_issued,
            ease_of_use=ease_of_use if isinstance(ease_of_use, str) else "",
            reuse=reuse if isinstance(reuse, str) else "",
            suggestions_satisfaction=(
                suggestions_satisfaction
                if isinstance(suggestions_satisfaction, str)
                else ""
            ),
            summary_text=summary_text if isinstance(summary_text, str) else "",
        )

    def _advisor_is_completed(self, advisor_state: dict[str, Any]) -> bool:
        phase = advisor_state.get("phase")
        if phase != _ADVISOR_PHASE_COMPLETED:
            return False
        selected = advisor_state.get("selected_barcodes")
        return isinstance(selected, list) and len(selected) > 0

    def _is_feedback_active(
        self,
        advisor_state: dict[str, Any] | None,
        feedback_state: CretskensFeedbackState | None,
    ) -> bool:
        if advisor_state is None or not self._advisor_is_completed(advisor_state):
            return False
        if feedback_state is None:
            return True
        return feedback_state.step != _FEEDBACK_STEP_DONE

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
        *,
        feedback_state: CretskensFeedbackState | None,
    ) -> WineAdvisorResult:
        advisor_state = dict(advisor_result.state)
        if advisor_result.response_type == "recommendations":
            advisor_state[CRETSKENS_FEEDBACK_SUMMARY_TEXT_KEY] = advisor_result.text
        next_feedback_state = feedback_state
        follow_up_question: str | None = None
        follow_up_quick_replies: list[dict[str, str]] | None = None
        if (
            advisor_result.response_type == "recommendations"
            and self._advisor_is_completed(advisor_state)
            and next_feedback_state is None
        ):
            next_feedback_state = CretskensFeedbackState(
                summary_text=self._feedback_summary_text_from_advisor_state(
                    advisor_state
                ),
                prompt_issued=True,
            )
            follow_up_question = _FEEDBACK_QUESTION_EASE_OF_USE
            follow_up_quick_replies = _RATING_QUICK_REPLIES
        wrapped_state: dict[str, Any] = {
            CRETSKENS_FLOW_KEY: CRETSKENS_FLOW_ADVISOR,
            CRETSKENS_ADVISOR_STATE_KEY: advisor_state,
        }
        if next_feedback_state is not None:
            wrapped_state[CRETSKENS_FEEDBACK_STATE_KEY] = (
                self._serialize_feedback_state(next_feedback_state)
            )
        return WineAdvisorResult(
            text=advisor_result.text,
            response_type=advisor_result.response_type,
            state=wrapped_state,
            recommendations=advisor_result.recommendations,
            quick_replies=(
                follow_up_quick_replies
                if follow_up_question is not None
                else advisor_result.quick_replies
            ),
            follow_up_question=follow_up_question,
        )

    def _serialize_feedback_state(
        self,
        feedback_state: CretskensFeedbackState,
    ) -> dict[str, Any]:
        return {
            "step": feedback_state.step,
            "prompt_issued": feedback_state.prompt_issued,
            "ease_of_use": feedback_state.ease_of_use,
            "reuse": feedback_state.reuse,
            "suggestions_satisfaction": feedback_state.suggestions_satisfaction,
            "summary_text": feedback_state.summary_text,
        }

    def _build_feedback_result(
        self,
        advisor_state: dict[str, Any],
        feedback_state: CretskensFeedbackState,
        current_answer: str,
    ) -> WineAdvisorResult:
        if not feedback_state.prompt_issued:
            return self._build_feedback_question_result(
                advisor_state,
                replace(feedback_state, prompt_issued=True),
            )

        normalized_answer = current_answer.strip().casefold()
        if feedback_state.step == _FEEDBACK_STEP_EASE_OF_USE:
            if normalized_answer not in {"1", "2", "3", "4", "5"}:
                return self._build_feedback_question_result(
                    advisor_state, feedback_state
                )
            return self._build_feedback_question_result(
                advisor_state,
                replace(
                    feedback_state,
                    step=_FEEDBACK_STEP_REUSE,
                    prompt_issued=True,
                    ease_of_use=normalized_answer,
                ),
            )

        if feedback_state.step == _FEEDBACK_STEP_REUSE:
            if normalized_answer not in {"ja", "nee"}:
                return self._build_feedback_question_result(
                    advisor_state, feedback_state
                )
            return self._build_feedback_question_result(
                advisor_state,
                replace(
                    feedback_state,
                    step=_FEEDBACK_STEP_SATISFACTION,
                    prompt_issued=True,
                    reuse=normalized_answer,
                ),
            )

        if feedback_state.step == _FEEDBACK_STEP_SATISFACTION:
            if normalized_answer not in {"1", "2", "3", "4", "5"}:
                return self._build_feedback_question_result(
                    advisor_state, feedback_state
                )
            return WineAdvisorResult(
                text=_FEEDBACK_THANKS,
                response_type="question",
                state={
                    CRETSKENS_FLOW_KEY: CRETSKENS_FLOW_ADVISOR,
                    CRETSKENS_ADVISOR_STATE_KEY: advisor_state,
                    CRETSKENS_FEEDBACK_STATE_KEY: self._serialize_feedback_state(
                        replace(
                            feedback_state,
                            step=_FEEDBACK_STEP_DONE,
                            prompt_issued=True,
                            suggestions_satisfaction=normalized_answer,
                        )
                    ),
                },
                recommendations=None,
                quick_replies=None,
            )

        return self._build_feedback_question_result(advisor_state, feedback_state)

    def _build_feedback_question_result(
        self,
        advisor_state: dict[str, Any],
        feedback_state: CretskensFeedbackState,
    ) -> WineAdvisorResult:
        if feedback_state.step == _FEEDBACK_STEP_EASE_OF_USE:
            text = _FEEDBACK_QUESTION_EASE_OF_USE
            quick_replies = _RATING_QUICK_REPLIES
        elif feedback_state.step == _FEEDBACK_STEP_REUSE:
            text = _FEEDBACK_QUESTION_REUSE
            quick_replies = _YES_NO_QUICK_REPLIES
        else:
            text = _FEEDBACK_QUESTION_SATISFACTION
            quick_replies = _RATING_QUICK_REPLIES
        return WineAdvisorResult(
            text=text,
            response_type="question",
            state={
                CRETSKENS_FLOW_KEY: CRETSKENS_FLOW_ADVISOR,
                CRETSKENS_ADVISOR_STATE_KEY: advisor_state,
                CRETSKENS_FEEDBACK_STATE_KEY: self._serialize_feedback_state(
                    feedback_state
                ),
            },
            recommendations=None,
            quick_replies=quick_replies,
        )

    def _extract_summary_text(self, advisor_state: dict[str, Any]) -> str:
        working_context = advisor_state.get("working_context")
        if not isinstance(working_context, str):
            return ""
        normalized_context = " ".join(working_context.split())
        if not normalized_context:
            return ""
        lower_context = normalized_context.casefold()
        summary_marker = "summary:"
        start_index = lower_context.find(summary_marker)
        if start_index < 0:
            return ""
        summary_text = normalized_context[start_index + len(summary_marker) :].strip()
        if not summary_text:
            return ""

        lower_summary_text = summary_text.casefold()
        split_markers = ("questionnaire:", "handoff categories:")
        split_indices = [
            lower_summary_text.find(marker)
            for marker in split_markers
            if lower_summary_text.find(marker) >= 0
        ]
        if split_indices:
            summary_text = summary_text[: min(split_indices)].strip()
        return summary_text

    def _feedback_summary_text_from_advisor_state(
        self, advisor_state: dict[str, Any]
    ) -> str:
        summary_text = advisor_state.get(CRETSKENS_FEEDBACK_SUMMARY_TEXT_KEY)
        if isinstance(summary_text, str) and summary_text.strip():
            return summary_text.strip()
        return self._extract_summary_text(advisor_state)

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
