"""Outcome builders for intake turns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .flow import (
    to_categorized_handoff_completed_state,
    to_feedback_state,
    to_intake_state_with_next_question,
    to_off_topic_redirect_state,
    to_terminated_state,
)
from .questionnaire import mask_later_blank_fields
from .types import ChatResult, ChatState

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler


def handle_off_topic_intake(
    handler: ChatTurnHandler,
    current_state: ChatState,
) -> ChatResult:
    current_state.boundary_violation_count += 1
    language = (
        current_state.conversation_language or handler.language.default_language_code
    )
    if (
        current_state.boundary_violation_count
        >= handler.domain_policy.off_topic_termination_threshold
    ):
        terminated_state = to_terminated_state(current_state, language)
        return handler.answer_result(
            handler.messages.goodbye(language),
            terminated_state,
            None,
            None,
            None,
        )

    redirected_state = to_off_topic_redirect_state(current_state)
    return handler.answer_result(
        handler.messages.off_topic_redirect(
            language=language,
            current_question=current_state.last_question,
        ),
        redirected_state,
        None,
        None,
        None,
    )


async def build_completion_result(
    handler: ChatTurnHandler,
    current_state: ChatState,
    updated_questionnaire: str,
    updated_summary: str,
) -> ChatResult:
    if handler.is_categorized_handoff_mode():
        return await _build_categorized_completion_result(
            handler,
            current_state,
            updated_questionnaire,
            updated_summary,
        )

    language = (
        current_state.conversation_language or handler.language.default_language_code
    )
    feedback_state = to_feedback_state(
        state=current_state,
        updated_questionnaire=updated_questionnaire,
        updated_summary=updated_summary,
        language=language,
    )
    return handler.answer_result(
        handler.messages.handoff(language),
        feedback_state,
        updated_summary,
        handler.messages.feedback_question(language),
        handler.messages.feedback_quick_replies(language),
    )


async def build_next_question_result(
    handler: ChatTurnHandler,
    current_state: ChatState,
    updated_questionnaire: str,
    updated_summary: str,
) -> ChatResult:
    conversation_language = (
        current_state.conversation_language or handler.language.default_language_code
    )
    focused_questionnaire = mask_later_blank_fields(
        updated_questionnaire,
        handler.questionnaire_indices,
    )
    next_question = await handler.question_asker.ask(
        focused_questionnaire,
        conversation_language,
    )
    next_state = to_intake_state_with_next_question(
        state=current_state,
        updated_questionnaire=updated_questionnaire,
        updated_summary=updated_summary,
        next_question=next_question,
    )
    return handler.answer_result(next_question, next_state, None, None, None)


async def _build_categorized_completion_result(
    handler: ChatTurnHandler,
    current_state: ChatState,
    updated_questionnaire: str,
    updated_summary: str,
) -> ChatResult:
    language = (
        current_state.conversation_language or handler.language.default_language_code
    )
    handoff_categories_v1 = (
        await handler.categorized_handoff.build_handoff_categories_v1(
            updated_questionnaire
        )
    )
    completed_state = to_categorized_handoff_completed_state(
        state=current_state,
        updated_questionnaire=updated_questionnaire,
        updated_summary=updated_summary,
        handoff_categories_v1=handoff_categories_v1,
    )
    return handler.answer_result(
        handler.categorized_handoff.build_handoff_text(
            handler.messages.handoff(language),
            handoff_categories_v1,
        ),
        completed_state,
        None,
        None,
        None,
    )
