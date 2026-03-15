"""State parsing, normalization and serialization helpers."""

from __future__ import annotations

import re

from .language import SUPPORTED_RESPONSE_LANGUAGES, detect_language
from .types import (
    FEEDBACK_PENDING,
    INTAKE_ROUTE_UNKNOWN,
    VALID_INTAKE_ROUTES,
    PHASE_AWAITING_FEEDBACK,
    PHASE_COMPLETED,
    PHASE_INTAKE,
    PHASE_TERMINATED,
    VALID_FEEDBACK_STATUSES,
    VALID_PHASES,
    ChatState,
    INITIAL_QUESTION,
)


def extract_questionnaire_indices(questionnaire: str) -> list[int]:
    indices = set()
    for match in re.finditer(r"(?m)^\s*(\d+)\.", questionnaire):
        indices.add(int(match.group(1)))
    return sorted(indices)


def is_valid_questionnaire(
    questionnaire: str,
    questionnaire_indices: list[int],
) -> bool:
    for index in questionnaire_indices:
        if not re.search(rf"(?m)^\s*{index}\.", questionnaire):
            return False
    return True


def _extract_field_value(questionnaire: str, index: int) -> str:
    """Return the value portion of a numbered field line, or '' if absent."""
    match = re.search(rf"(?m)^\s*{index}\.\s*[^:\n]*:[^\S\n]*(.*?)$", questionnaire)
    if match:
        return match.group(1).strip()
    return ""


def _is_blank_value(value: str) -> bool:
    """Return True when *value* is empty or bracket-only (e.g. ``[]``)."""
    stripped = value.strip().strip("[]").strip()
    return not stripped


def _prevent_field_regression(
    current_questionnaire: str,
    updated_questionnaire: str,
    questionnaire_indices: list[int],
) -> str:
    """Restore any filled field that the proposed update blanked."""
    result = updated_questionnaire
    for index in questionnaire_indices:
        current_value = _extract_field_value(current_questionnaire, index)
        if _is_blank_value(current_value):
            continue
        updated_value = _extract_field_value(result, index)
        if _is_blank_value(updated_value):
            pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*).*$")
            result = pattern.sub(
                lambda m, v=current_value: f"{m.group(1)}{v}", result, count=1
            )
    return result


def merge_partial_questionnaire_update(
    current_questionnaire: str,
    proposed_questionnaire: str,
    questionnaire_indices: list[int],
) -> str | None:
    if is_valid_questionnaire(proposed_questionnaire, questionnaire_indices):
        return _prevent_field_regression(
            current_questionnaire, proposed_questionnaire, questionnaire_indices
        )

    allowed_indices = set(questionnaire_indices)
    proposed_values: dict[int, str] = {}
    for match in re.finditer(r"(?m)^\s*(\d+)\.\s*(.+)$", proposed_questionnaire):
        index = int(match.group(1))
        if index not in allowed_indices:
            continue
        field_text = match.group(2).strip()
        if ":" not in field_text:
            continue
        value = field_text.split(":", maxsplit=1)[1].strip()
        if _is_blank_value(value):
            continue
        proposed_values[index] = value

    if not proposed_values:
        return None

    merged = current_questionnaire
    merged_any = False
    for index in questionnaire_indices:
        value = proposed_values.get(index)
        if value is None:
            continue
        pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*).*$")
        merged, count = pattern.subn(lambda m: f"{m.group(1)}{value}", merged, count=1)
        if count:
            merged_any = True

    if not merged_any:
        return None
    if not is_valid_questionnaire(merged, questionnaire_indices):
        return None
    return merged


def _read_state_string(
    state: dict[str, str],
    field_name: str,
    default: str,
) -> str:
    value = state.get(field_name, default)
    if not isinstance(value, str):
        raise RuntimeError(f"State field '{field_name}' must be a string.")
    return value


def _normalize_phase(phase: str) -> str:
    if phase in VALID_PHASES:
        return phase
    return PHASE_INTAKE


def _normalize_feedback_status(feedback_status: str) -> str:
    if feedback_status in VALID_FEEDBACK_STATUSES:
        return feedback_status
    return FEEDBACK_PENDING


def _normalize_intake_route(intake_route: str) -> str:
    if intake_route in VALID_INTAKE_ROUTES:
        return intake_route
    return INTAKE_ROUTE_UNKNOWN


def _parse_boundary_violation_count(boundary_violation_count_raw: str) -> int:
    try:
        boundary_violation_count = int(boundary_violation_count_raw)
    except ValueError:
        return 0
    return max(boundary_violation_count, 0)


def _parse_state_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes"}


def _normalize_questionnaire(
    questionnaire: str,
    questionnaire_template: str,
) -> str:
    if questionnaire.strip():
        return questionnaire
    return questionnaire_template


def _infer_legacy_conversation_language(
    summary_text: str,
    default_language: str,
) -> str:
    if not summary_text.strip():
        return default_language
    return detect_language(
        summary_text,
        supported_language_set=set(SUPPORTED_RESPONSE_LANGUAGES),
        default_language_code=default_language,
    )


def _normalize_phase_context(
    phase: str,
    feedback_status: str,
    summary: str,
    final_summary: str,
    conversation_language: str,
    post_summary_language: str,
    default_language: str,
) -> tuple[str, str, str, str, str]:
    normalized_final_summary = final_summary if final_summary.strip() else summary
    normalized_conversation_language = (
        conversation_language.strip() or post_summary_language.strip()
    )
    if not normalized_conversation_language:
        if phase == PHASE_INTAKE:
            normalized_conversation_language = default_language
        else:
            normalized_conversation_language = _infer_legacy_conversation_language(
                normalized_final_summary,
                default_language,
            )
    if phase == PHASE_INTAKE:
        return (
            phase,
            FEEDBACK_PENDING,
            "",
            normalized_conversation_language,
            "",
        )

    if phase == PHASE_TERMINATED:
        return (
            phase,
            FEEDBACK_PENDING,
            "",
            normalized_conversation_language,
            normalized_conversation_language,
        )

    normalized_phase = phase
    if phase == PHASE_AWAITING_FEEDBACK and feedback_status != FEEDBACK_PENDING:
        normalized_phase = PHASE_COMPLETED

    return (
        normalized_phase,
        feedback_status,
        normalized_final_summary,
        normalized_conversation_language,
        normalized_conversation_language,
    )


def parse_state(
    state: dict[str, str] | None,
    questionnaire_template: str,
    default_language: str,
) -> ChatState:
    if not state:
        return ChatState(
            questionnaire=questionnaire_template,
            conversation_language=default_language,
        )

    questionnaire = _normalize_questionnaire(
        _read_state_string(state, "questionnaire", ""),
        questionnaire_template,
    )
    summary = _read_state_string(state, "summary", "")
    last_question = _read_state_string(state, "last_question", INITIAL_QUESTION)
    conversation_language = _read_state_string(state, "conversation_language", "")
    recent_user_turns = _read_state_string(state, "recent_user_turns", "")
    phase = _normalize_phase(_read_state_string(state, "phase", PHASE_INTAKE))
    feedback_status = _normalize_feedback_status(
        _read_state_string(state, "feedback_status", FEEDBACK_PENDING)
    )
    final_summary = _read_state_string(state, "final_summary", "")
    post_summary_language = _read_state_string(state, "post_summary_language", "")
    boundary_violation_count = _parse_boundary_violation_count(
        _read_state_string(state, "boundary_violation_count", "0")
    )
    handoff_categories_v1 = _read_state_string(state, "handoff_categories_v1", "")
    intake_route = _normalize_intake_route(
        _read_state_string(state, "intake_route", INTAKE_ROUTE_UNKNOWN)
    )
    dish_follow_up_count = _parse_boundary_violation_count(
        _read_state_string(state, "dish_follow_up_count", "0")
    )
    wine_follow_up_count = _parse_boundary_violation_count(
        _read_state_string(state, "wine_follow_up_count", "0")
    )
    routing_question_asked = _parse_state_bool(
        _read_state_string(state, "routing_question_asked", "false")
    )
    (
        phase,
        feedback_status,
        final_summary,
        conversation_language,
        post_summary_language,
    ) = _normalize_phase_context(
        phase=phase,
        feedback_status=feedback_status,
        summary=summary,
        final_summary=final_summary,
        conversation_language=conversation_language,
        post_summary_language=post_summary_language,
        default_language=default_language,
    )
    return ChatState(
        questionnaire=questionnaire,
        summary=summary,
        last_question=last_question,
        conversation_language=conversation_language,
        recent_user_turns=recent_user_turns,
        phase=phase,
        feedback_status=feedback_status,
        final_summary=final_summary,
        post_summary_language=post_summary_language,
        boundary_violation_count=boundary_violation_count,
        handoff_categories_v1=handoff_categories_v1,
        intake_route=intake_route,
        dish_follow_up_count=dish_follow_up_count,
        wine_follow_up_count=wine_follow_up_count,
        routing_question_asked=routing_question_asked,
    )


def serialize_state(state: ChatState) -> dict[str, str]:
    return {
        "questionnaire": state.questionnaire,
        "summary": state.summary,
        "last_question": state.last_question,
        "conversation_language": state.conversation_language,
        "recent_user_turns": state.recent_user_turns,
        "phase": state.phase,
        "feedback_status": state.feedback_status,
        "final_summary": state.final_summary,
        "post_summary_language": (
            state.post_summary_language or state.conversation_language
        ),
        "boundary_violation_count": str(state.boundary_violation_count),
        "handoff_categories_v1": state.handoff_categories_v1,
        "intake_route": state.intake_route,
        "dish_follow_up_count": str(state.dish_follow_up_count),
        "wine_follow_up_count": str(state.wine_follow_up_count),
        "routing_question_asked": ("true" if state.routing_question_asked else "false"),
    }
