"""Questionnaire and summary update pipeline for intake turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .intake_routing import intake_complete_for_state
from .intake_routing import is_routing_question
from .intake_routing import resolve_intake_stage
from .intake_routing import DISH_FIELD_INDICES, WINE_FIELD_INDICES
from .questionnaire import combine_questionnaire_sections
from .questionnaire import ensure_initial_question_field
from .questionnaire import normalize_blank_like_questionnaire_values
from .questionnaire import select_questionnaire_fields
from .state_codec import merge_partial_questionnaire_update
from .summary import (
    rewrite_summary_in_conversation_language,
    summary_matches_conversation_language,
)
from .types import (
    INITIAL_QUESTION,
    INTAKE_ROUTE_GENERAL_WINE,
    ChatState,
)

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler


@dataclass(frozen=True)
class IntakeUpdateResult:
    questionnaire: str
    dish_questionnaire: str
    wine_questionnaire: str
    summary: str


async def apply_intake_updates(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
) -> IntakeUpdateResult:
    categorized_handoff_mode = handler.is_categorized_handoff_mode()
    (
        updated_questionnaire,
        dish_questionnaire,
        wine_questionnaire,
    ) = await _update_questionnaire(
        handler,
        current_state,
        current_answer,
    )
    if (
        current_state.last_question.strip() == INITIAL_QUESTION
        and current_state.intake_route != INTAKE_ROUTE_GENERAL_WINE
    ):
        updated_questionnaire = ensure_initial_question_field(
            updated_questionnaire,
            current_answer,
        )
        if categorized_handoff_mode:
            (
                dish_questionnaire,
                wine_questionnaire,
            ) = _split_stage_questionnaires(updated_questionnaire)

    updated_state = _state_with_updated_questionnaires(
        current_state,
        questionnaire=updated_questionnaire,
        dish_questionnaire=dish_questionnaire,
        wine_questionnaire=wine_questionnaire,
        resolve_stage=categorized_handoff_mode,
    )
    updated_summary = await _update_summary(
        handler,
        current_state,
        current_answer,
        enforce_supported_language=intake_complete_for_state(
            updated_state,
            updated_questionnaire,
        ),
    )
    return IntakeUpdateResult(
        questionnaire=updated_questionnaire,
        dish_questionnaire=dish_questionnaire,
        wine_questionnaire=wine_questionnaire,
        summary=updated_summary,
    )


def _split_stage_questionnaires(questionnaire: str) -> tuple[str, str]:
    return (
        select_questionnaire_fields(questionnaire, DISH_FIELD_INDICES),
        select_questionnaire_fields(questionnaire, WINE_FIELD_INDICES),
    )


def _state_with_updated_questionnaires(
    current_state: ChatState,
    *,
    questionnaire: str,
    dish_questionnaire: str,
    wine_questionnaire: str,
    resolve_stage: bool,
) -> ChatState:
    updated_state = ChatState(
        questionnaire=questionnaire,
        dish_questionnaire=dish_questionnaire,
        wine_questionnaire=wine_questionnaire,
        summary=current_state.summary,
        last_question=current_state.last_question,
        conversation_language=current_state.conversation_language,
        recent_user_turns=current_state.recent_user_turns,
        phase=current_state.phase,
        feedback_status=current_state.feedback_status,
        final_summary=current_state.final_summary,
        post_summary_language=current_state.post_summary_language,
        boundary_violation_count=current_state.boundary_violation_count,
        handoff_categories_v1=current_state.handoff_categories_v1,
        intake_route=current_state.intake_route,
        intake_stage=current_state.intake_stage,
        dish_follow_up_count=current_state.dish_follow_up_count,
        wine_follow_up_count=current_state.wine_follow_up_count,
        routing_question_asked=current_state.routing_question_asked,
    )
    if resolve_stage:
        updated_state.intake_stage = resolve_intake_stage(updated_state)
    return updated_state


def _should_skip_general_wine_routing_update(current_state: ChatState) -> bool:
    if current_state.intake_route != INTAKE_ROUTE_GENERAL_WINE:
        return False
    last_question = current_state.last_question.strip()
    return is_routing_question(last_question)


def _is_opening_general_wine_question(last_question: str) -> bool:
    stripped = last_question.strip()
    return stripped == INITIAL_QUESTION or is_routing_question(stripped)


def _is_opening_general_wine_turn(current_state: ChatState) -> bool:
    if current_state.intake_route != INTAKE_ROUTE_GENERAL_WINE:
        return False
    return _is_opening_general_wine_question(current_state.last_question)


async def _update_questionnaire(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
) -> tuple[str, str, str]:
    if not handler.is_categorized_handoff_mode():
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
            merged = current_state.questionnaire
        return merged, "", ""

    dish_questionnaire, wine_questionnaire = _split_stage_questionnaires(
        current_state.questionnaire
    )
    if current_state.dish_questionnaire.strip():
        dish_questionnaire = current_state.dish_questionnaire
    if current_state.wine_questionnaire.strip():
        wine_questionnaire = current_state.wine_questionnaire

    if _should_skip_general_wine_routing_update(current_state):
        combined = combine_questionnaire_sections(
            dish_questionnaire,
            wine_questionnaire,
        )
        if _is_opening_general_wine_turn(current_state):
            combined = normalize_blank_like_questionnaire_values(
                combined,
                WINE_FIELD_INDICES,
            )
        dish_questionnaire, wine_questionnaire = _split_stage_questionnaires(combined)
        return combined, dish_questionnaire, wine_questionnaire

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
        merged = current_state.questionnaire
    if _is_opening_general_wine_turn(current_state):
        merged = normalize_blank_like_questionnaire_values(
            merged,
            WINE_FIELD_INDICES,
        )
    dish_questionnaire, wine_questionnaire = _split_stage_questionnaires(merged)
    combined = combine_questionnaire_sections(dish_questionnaire, wine_questionnaire)
    return combined, dish_questionnaire, wine_questionnaire


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
