"""Questionnaire and summary update pipeline for intake turns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .questionnaire import ensure_initial_question_field, questionnaire_complete
from .state_codec import merge_partial_questionnaire_update
from .summary import (
    rewrite_summary_in_conversation_language,
    summary_matches_conversation_language,
)
from .types import INITIAL_QUESTION, ChatState

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler


async def apply_intake_updates(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
) -> tuple[str, str]:
    updated_questionnaire = await _update_questionnaire(
        handler,
        current_state,
        current_answer,
    )
    if handler.is_categorized_handoff_mode():
        updated_questionnaire = (
            await handler.categorized_handoff.normalize_questionnaire(
                updated_questionnaire
            )
        )
    if current_state.last_question.strip() == INITIAL_QUESTION:
        updated_questionnaire = ensure_initial_question_field(
            updated_questionnaire,
            current_answer,
        )

    updated_summary = await _update_summary(
        handler,
        current_state,
        current_answer,
        enforce_supported_language=questionnaire_complete(updated_questionnaire),
    )
    return updated_questionnaire, updated_summary


async def _update_questionnaire(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
) -> str:
    try:
        updated = await handler.notulist.generate(
            current_question=current_state.last_question,
            current_answer=current_answer,
            summary=current_state.questionnaire,
        )
    except Exception as exc:
        raise RuntimeError("Failed to update questionnaire.") from exc

    merged = merge_partial_questionnaire_update(
        current_questionnaire=current_state.questionnaire,
        proposed_questionnaire=updated,
        questionnaire_indices=handler.questionnaire_indices,
    )
    if merged is None:
        return current_state.questionnaire
    return merged


async def _update_summary(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
    *,
    enforce_supported_language: bool,
) -> str:
    try:
        generated_summary = await handler.summarizer.generate(
            current_question=current_state.last_question,
            current_answer=current_answer,
            summary=current_state.summary,
        )
    except Exception as exc:
        raise RuntimeError("Failed to summarize conversation.") from exc

    if not enforce_supported_language:
        return generated_summary

    conversation_language = current_state.conversation_language
    if summary_matches_conversation_language(
        generated_summary,
        conversation_language,
        handler.language.supported_language_set,
    ):
        return generated_summary

    rewritten_summary = await rewrite_summary_in_conversation_language(
        handler.llm,
        handler.language,
        conversation_language,
        generated_summary,
    )
    if rewritten_summary is not None:
        return rewritten_summary

    fallback_summary = current_state.summary.strip() or current_answer.strip()
    if fallback_summary:
        return fallback_summary
    return generated_summary
