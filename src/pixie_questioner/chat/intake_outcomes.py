"""Outcome builders for intake turns."""

from __future__ import annotations

import logging

from dataclasses import replace
from typing import TYPE_CHECKING

from .flow import (
    to_categorized_handoff_completed_state,
    to_feedback_state,
    to_intake_state_with_next_question,
    to_off_topic_redirect_state,
    to_terminated_state,
)
from .intake_routing import (
    DISH_FIELD_INDICES,
    WINE_FIELD_INDICES,
    determine_next_question_track,
)
from .questionnaire import focus_questionnaire_for_route, mask_later_blank_fields
from .types import ChatResult, ChatState

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler

logger = logging.getLogger(__name__)


async def handle_off_topic_intake(
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
        goodbye_text = await _generate_off_topic_message(
            handler=handler,
            task="goodbye",
            current_question=current_state.last_question,
            current_answer="",
            language=language,
            fallback=handler.messages.goodbye(language),
        )
        return handler.answer_result(
            goodbye_text,
            terminated_state,
            None,
            None,
            None,
        )

    redirected_state = to_off_topic_redirect_state(current_state)
    redirect_text = await _generate_off_topic_message(
        handler=handler,
        task="redirect",
        current_question=current_state.last_question,
        current_answer="",
        language=language,
        fallback=handler.messages.off_topic_redirect(
            language=language,
            current_question=current_state.last_question,
        ),
    )
    return handler.answer_result(
        redirect_text,
        redirected_state,
        None,
        None,
        None,
    )


async def _generate_off_topic_message(
    handler: ChatTurnHandler,
    task: str,
    current_question: str,
    current_answer: str,
    language: str,
    fallback: str,
) -> str:
    if handler.off_topic_redirect_invoker is None:
        return fallback
    summary = (
        f"Language: {language}\nTask: {task}\nCurrent question: {current_question}"
    )
    try:
        generated = await handler.off_topic_redirect_invoker.generate(
            current_question=task,
            current_answer=current_answer,
            summary=summary,
        )
        return generated.strip() or fallback
    except Exception:
        logger.warning("Off-topic LLM invoker failed; using fallback.", exc_info=True)
        return fallback


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
    next_question_track = determine_next_question_track(
        current_state,
        updated_questionnaire,
    )
    if next_question_track is None:
        raise RuntimeError(
            "No follow-up question remains for the current intake state."
        )
    if handler.is_categorized_handoff_mode() and next_question_track == "dish":
        focused_questionnaire = mask_later_blank_fields(
            current_state.dish_questionnaire,
            DISH_FIELD_INDICES,
        )
    elif handler.is_categorized_handoff_mode() and next_question_track == "wine":
        focused_questionnaire = mask_later_blank_fields(
            current_state.wine_questionnaire,
            WINE_FIELD_INDICES,
        )
    else:
        focused_questionnaire = focus_questionnaire_for_route(
            updated_questionnaire,
            handler.questionnaire_indices,
            current_state.intake_route,
            allow_dish_questions=next_question_track != "wine",
        )
    next_question = await handler.question_asker.ask(
        focused_questionnaire,
        conversation_language,
    )
    if not next_question.strip():
        return await build_completion_result(
            handler,
            current_state,
            updated_questionnaire,
            updated_summary,
        )
    next_state = to_intake_state_with_next_question(
        state=current_state,
        updated_questionnaire=updated_questionnaire,
        updated_summary=updated_summary,
        next_question=next_question,
    )
    if next_question_track == "dish":
        next_state = replace(
            next_state,
            dish_follow_up_count=current_state.dish_follow_up_count + 1,
        )
    elif next_question_track == "wine":
        next_state = replace(
            next_state,
            wine_follow_up_count=current_state.wine_follow_up_count + 1,
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
