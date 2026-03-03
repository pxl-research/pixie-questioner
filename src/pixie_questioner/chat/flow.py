"""Pure chat flow helpers for state transitions."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .types import (
    FEEDBACK_PENDING,
    PHASE_AWAITING_FEEDBACK,
    PHASE_COMPLETED,
    PHASE_INTAKE,
    PHASE_TERMINATED,
    ChatState,
)


def to_terminated_state(state: ChatState, language: str) -> ChatState:
    return ChatState(
        questionnaire=state.questionnaire,
        summary=state.summary,
        last_question=state.last_question,
        phase=PHASE_TERMINATED,
        feedback_status=FEEDBACK_PENDING,
        final_summary="",
        post_summary_language=language,
        boundary_violation_count=state.boundary_violation_count,
        handoff_categories_v1="",
    )


def to_off_topic_redirect_state(state: ChatState) -> ChatState:
    return ChatState(
        questionnaire=state.questionnaire,
        summary=state.summary,
        last_question=state.last_question,
        phase=PHASE_INTAKE,
        feedback_status=FEEDBACK_PENDING,
        final_summary="",
        post_summary_language="",
        boundary_violation_count=state.boundary_violation_count,
        handoff_categories_v1="",
    )


def to_feedback_state(
    state: ChatState,
    updated_questionnaire: str,
    updated_summary: str,
    language: str,
) -> ChatState:
    return ChatState(
        questionnaire=updated_questionnaire,
        summary=updated_summary,
        last_question=state.last_question,
        phase=PHASE_AWAITING_FEEDBACK,
        feedback_status=FEEDBACK_PENDING,
        final_summary=updated_summary,
        post_summary_language=language,
        boundary_violation_count=state.boundary_violation_count,
        handoff_categories_v1="",
    )


def to_intake_state_with_next_question(
    state: ChatState,
    updated_questionnaire: str,
    updated_summary: str,
    next_question: str,
) -> ChatState:
    return ChatState(
        questionnaire=updated_questionnaire,
        summary=updated_summary,
        last_question=next_question,
        phase=PHASE_INTAKE,
        feedback_status=FEEDBACK_PENDING,
        boundary_violation_count=state.boundary_violation_count,
        handoff_categories_v1="",
    )


def to_categorized_handoff_completed_state(
    state: ChatState,
    updated_questionnaire: str,
    updated_summary: str,
    handoff_categories_v1: str,
) -> ChatState:
    return ChatState(
        questionnaire=updated_questionnaire,
        summary=updated_summary,
        last_question=state.last_question,
        phase=PHASE_COMPLETED,
        feedback_status=FEEDBACK_PENDING,
        final_summary="",
        post_summary_language="",
        boundary_violation_count=0,
        handoff_categories_v1=handoff_categories_v1,
    )


def mark_feedback_completed(state: ChatState, feedback_status: str) -> ChatState:
    return replace(
        state,
        phase=PHASE_COMPLETED,
        feedback_status=feedback_status,
        handoff_categories_v1="",
    )


def ensure_feedback_context(
    state: ChatState,
    detect_language: Callable[[str], str],
) -> tuple[ChatState, str, str]:
    summary_text = state.final_summary if state.final_summary.strip() else state.summary
    language = (
        state.post_summary_language
        if state.post_summary_language.strip()
        else detect_language(summary_text)
    )
    next_state = replace(
        state,
        final_summary=summary_text,
        post_summary_language=language,
    )
    return next_state, summary_text, language
