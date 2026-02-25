"""State parsing, normalization and serialization helpers."""

from __future__ import annotations

import re
from typing import Callable

from .types import (
    FEEDBACK_PENDING,
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


def merge_partial_questionnaire_update(
    current_questionnaire: str,
    proposed_questionnaire: str,
    questionnaire_indices: list[int],
) -> str | None:
    if is_valid_questionnaire(proposed_questionnaire, questionnaire_indices):
        return proposed_questionnaire

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
        if not value:
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
        pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:\s*).*$")
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


def _parse_boundary_violation_count(boundary_violation_count_raw: str) -> int:
    try:
        boundary_violation_count = int(boundary_violation_count_raw)
    except ValueError:
        return 0
    return max(boundary_violation_count, 0)


def _normalize_questionnaire(
    questionnaire: str,
    questionnaire_template: str,
) -> str:
    if questionnaire.strip():
        return questionnaire
    return questionnaire_template


def _normalize_phase_context(
    phase: str,
    feedback_status: str,
    summary: str,
    final_summary: str,
    post_summary_language: str,
    detect_language: Callable[[str], str],
) -> tuple[str, str, str, str]:
    if phase == PHASE_INTAKE:
        return phase, FEEDBACK_PENDING, "", ""

    if phase == PHASE_TERMINATED:
        return phase, FEEDBACK_PENDING, "", post_summary_language

    normalized_final_summary = final_summary if final_summary.strip() else summary
    normalized_post_summary_language = (
        post_summary_language
        if post_summary_language.strip()
        else detect_language(normalized_final_summary)
    )
    normalized_phase = phase
    if phase == PHASE_AWAITING_FEEDBACK and feedback_status != FEEDBACK_PENDING:
        normalized_phase = PHASE_COMPLETED

    return (
        normalized_phase,
        feedback_status,
        normalized_final_summary,
        normalized_post_summary_language,
    )


def parse_state(
    state: dict[str, str] | None,
    questionnaire_template: str,
    detect_language: Callable[[str], str],
) -> ChatState:
    if not state:
        return ChatState(questionnaire=questionnaire_template)

    questionnaire = _normalize_questionnaire(
        _read_state_string(state, "questionnaire", ""),
        questionnaire_template,
    )
    summary = _read_state_string(state, "summary", "")
    last_question = _read_state_string(state, "last_question", INITIAL_QUESTION)
    phase = _normalize_phase(_read_state_string(state, "phase", PHASE_INTAKE))
    feedback_status = _normalize_feedback_status(
        _read_state_string(state, "feedback_status", FEEDBACK_PENDING)
    )
    final_summary = _read_state_string(state, "final_summary", "")
    post_summary_language = _read_state_string(state, "post_summary_language", "")
    boundary_violation_count = _parse_boundary_violation_count(
        _read_state_string(state, "boundary_violation_count", "0")
    )
    phase, feedback_status, final_summary, post_summary_language = (
        _normalize_phase_context(
            phase=phase,
            feedback_status=feedback_status,
            summary=summary,
            final_summary=final_summary,
            post_summary_language=post_summary_language,
            detect_language=detect_language,
        )
    )

    return ChatState(
        questionnaire=questionnaire,
        summary=summary,
        last_question=last_question,
        phase=phase,
        feedback_status=feedback_status,
        final_summary=final_summary,
        post_summary_language=post_summary_language,
        boundary_violation_count=boundary_violation_count,
    )


def serialize_state(state: ChatState) -> dict[str, str]:
    return {
        "questionnaire": state.questionnaire,
        "summary": state.summary,
        "last_question": state.last_question,
        "phase": state.phase,
        "feedback_status": state.feedback_status,
        "final_summary": state.final_summary,
        "post_summary_language": state.post_summary_language,
        "boundary_violation_count": str(state.boundary_violation_count),
    }
