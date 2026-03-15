"""Post-intake turn handling for feedback and completed phases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .flow import (
    ensure_feedback_context,
    mark_feedback_completed,
    to_categorized_handoff_completed_state,
)
from .types import ChatResult, ChatState

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler


def handle_terminated_phase(
    handler: ChatTurnHandler,
    current_state: ChatState,
) -> ChatResult:
    language = (
        current_state.conversation_language or handler.language.default_language_code
    )
    return handler.answer_result(
        handler.messages.goodbye(language),
        current_state,
        None,
        None,
        None,
    )


def handle_awaiting_feedback_phase(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
) -> ChatResult:
    feedback_state, summary_text, language = ensure_feedback_context(
        current_state,
        default_language=handler.language.default_language_code,
    )
    feedback_response = handler.messages.normalize_feedback(current_answer)
    if feedback_response is None:
        return handler.answer_result(
            handler.messages.feedback_question(language),
            feedback_state,
            summary_text,
            None,
            handler.messages.feedback_quick_replies(language),
        )

    completed_state = mark_feedback_completed(feedback_state, feedback_response)
    return handler.summary_result(
        handler.messages.feedback_thanks(language),
        completed_state,
        summary_text,
    )


async def handle_completed_phase(
    handler: ChatTurnHandler,
    current_state: ChatState,
) -> ChatResult:
    if handler.is_categorized_handoff_mode():
        return await _build_completed_categorized_handoff_result(
            handler,
            current_state,
        )

    completed_state, summary_text, language = ensure_feedback_context(
        current_state,
        default_language=handler.language.default_language_code,
    )
    return handler.summary_result(
        handler.messages.feedback_thanks(language),
        completed_state,
        summary_text,
    )


async def _build_completed_categorized_handoff_result(
    handler: ChatTurnHandler,
    current_state: ChatState,
) -> ChatResult:
    handoff_categories_v1 = current_state.handoff_categories_v1.strip()
    if not handoff_categories_v1:
        handoff_categories_v1 = (
            await handler.categorized_handoff.build_handoff_categories_v1(
                current_state.questionnaire
            )
        )

    language = (
        current_state.conversation_language or handler.language.default_language_code
    )
    completed_state = to_categorized_handoff_completed_state(
        state=current_state,
        updated_questionnaire=current_state.questionnaire,
        updated_summary=current_state.summary,
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
