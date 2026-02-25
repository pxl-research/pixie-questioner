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


def parse_state(
    state: dict[str, str] | None,
    questionnaire_template: str,
    detect_language: Callable[[str], str],
) -> ChatState:
    if not state:
        return ChatState(questionnaire=questionnaire_template)
    questionnaire = state.get("questionnaire", "")
    summary = state.get("summary", "")
    last_question = state.get("last_question", INITIAL_QUESTION)
    phase = state.get("phase", PHASE_INTAKE)
    feedback_status = state.get("feedback_status", FEEDBACK_PENDING)
    final_summary = state.get("final_summary", "")
    post_summary_language = state.get("post_summary_language", "")
    boundary_violation_count_raw = state.get("boundary_violation_count", "0")

    if not isinstance(questionnaire, str):
        raise RuntimeError("State field 'questionnaire' must be a string.")
    if not isinstance(summary, str):
        raise RuntimeError("State field 'summary' must be a string.")
    if not isinstance(last_question, str):
        raise RuntimeError("State field 'last_question' must be a string.")
    if not isinstance(phase, str):
        raise RuntimeError("State field 'phase' must be a string.")
    if not isinstance(feedback_status, str):
        raise RuntimeError("State field 'feedback_status' must be a string.")
    if not isinstance(final_summary, str):
        raise RuntimeError("State field 'final_summary' must be a string.")
    if not isinstance(post_summary_language, str):
        raise RuntimeError("State field 'post_summary_language' must be a string.")
    if not isinstance(boundary_violation_count_raw, str):
        raise RuntimeError("State field 'boundary_violation_count' must be a string.")

    try:
        boundary_violation_count = int(boundary_violation_count_raw)
    except ValueError:
        boundary_violation_count = 0
    boundary_violation_count = max(boundary_violation_count, 0)

    if not questionnaire.strip():
        questionnaire = questionnaire_template
    if phase not in VALID_PHASES:
        phase = PHASE_INTAKE
    if feedback_status not in VALID_FEEDBACK_STATUSES:
        feedback_status = FEEDBACK_PENDING

    if phase == PHASE_INTAKE:
        feedback_status = FEEDBACK_PENDING
        final_summary = ""
        post_summary_language = ""
    elif phase == PHASE_TERMINATED:
        feedback_status = FEEDBACK_PENDING
        final_summary = ""
    else:
        if not final_summary.strip():
            final_summary = summary
        if not post_summary_language.strip():
            post_summary_language = detect_language(final_summary)
        if phase == PHASE_AWAITING_FEEDBACK and feedback_status != FEEDBACK_PENDING:
            phase = PHASE_COMPLETED

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
