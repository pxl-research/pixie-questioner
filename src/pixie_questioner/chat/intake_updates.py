"""Questionnaire and summary update pipeline for intake turns."""

from __future__ import annotations

import re

from typing import TYPE_CHECKING

from .intake_routing import WINE_FIELD_INDICES
from .intake_routing import determine_next_question_track
from .intake_routing import intake_complete_for_state
from .intake_routing import is_routing_question
from .questionnaire import ensure_initial_question_field
from .state_codec import merge_partial_questionnaire_update
from .summary import (
    rewrite_summary_in_conversation_language,
    summary_matches_conversation_language,
)
from .types import INITIAL_QUESTION, INTAKE_ROUTE_GENERAL_WINE, ChatState

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
    if (
        current_state.last_question.strip() == INITIAL_QUESTION
        and current_state.intake_route != INTAKE_ROUTE_GENERAL_WINE
    ):
        updated_questionnaire = ensure_initial_question_field(
            updated_questionnaire,
            current_answer,
        )
    updated_questionnaire = _apply_asked_field_answer_progress(
        current_state=current_state,
        updated_questionnaire=updated_questionnaire,
        current_answer=current_answer,
    )

    updated_summary = await _update_summary(
        handler,
        current_state,
        current_answer,
        enforce_supported_language=intake_complete_for_state(
            current_state,
            updated_questionnaire,
        ),
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


def _field_value(questionnaire: str, index: int) -> str | None:
    match = re.search(rf"(?m)^\s*{index}\.\s*[^:\n]*:[^\S\n]*(.*?)$", questionnaire)
    if match is None:
        return None
    return match.group(1)


def _is_blank_questionnaire_value(value: str | None) -> bool:
    if value is None:
        return True
    return not value.strip().strip("[]").strip()


def _replace_field_value(questionnaire: str, index: int, new_value: str) -> str:
    pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*).*$")
    updated, count = pattern.subn(rf"\g<1>{new_value}", questionnaire, count=1)
    if count == 0:
        return questionnaire
    return updated


def _first_blank_field_index(
    questionnaire: str,
    candidate_indices: list[int],
) -> int | None:
    for index in candidate_indices:
        if _is_blank_questionnaire_value(_field_value(questionnaire, index)):
            return index
    return None


def _resolve_asked_field_index(current_state: ChatState) -> int | None:
    if current_state.last_question.strip() == INITIAL_QUESTION:
        return None
    if is_routing_question(current_state.last_question):
        return None
    if current_state.intake_route != INTAKE_ROUTE_GENERAL_WINE:
        return None

    question_track = determine_next_question_track(
        current_state,
        current_state.questionnaire,
    )
    if question_track != "wine":
        return None
    candidate_indices = WINE_FIELD_INDICES
    return _first_blank_field_index(current_state.questionnaire, candidate_indices)


def _apply_asked_field_answer_progress(
    *,
    current_state: ChatState,
    updated_questionnaire: str,
    current_answer: str,
) -> str:
    normalized_answer = current_answer.strip()
    if not normalized_answer:
        return updated_questionnaire

    target_index = _resolve_asked_field_index(current_state)
    if target_index is None:
        return updated_questionnaire
    if not _is_blank_questionnaire_value(
        _field_value(current_state.questionnaire, target_index)
    ):
        return updated_questionnaire
    if not _is_blank_questionnaire_value(
        _field_value(updated_questionnaire, target_index)
    ):
        return updated_questionnaire
    return _replace_field_value(updated_questionnaire, target_index, normalized_answer)


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
