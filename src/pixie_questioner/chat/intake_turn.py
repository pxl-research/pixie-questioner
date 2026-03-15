"""Intake turn handling for classification, updates and next-step selection."""

from __future__ import annotations

import logging

from dataclasses import replace
from typing import TYPE_CHECKING

from .intake_routing import build_routing_question_result
from .intake_routing import intake_complete_for_state
from .intake_routing import is_routing_question
from .intake_outcomes import (
    build_completion_result,
    build_next_question_result,
    handle_off_topic_intake,
)
from .intake_updates import apply_intake_updates
from .relevance import ClassificationLabel
from .types import (
    INITIAL_QUESTION,
    INTAKE_ROUTE_FOOD_PAIRING,
    INTAKE_ROUTE_UNKNOWN,
    ChatResult,
    ChatState,
)

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler

logger = logging.getLogger(__name__)


async def handle_intake_phase(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
) -> ChatResult:
    routed_state, routing_result = await _maybe_route_cretskens_intake(
        handler,
        current_state,
        current_answer,
    )
    if routing_result is not None:
        return routing_result

    classification = await _classify_intake_answer(
        handler,
        routed_state,
        current_answer,
    )
    if classification == "off_topic":
        return handle_off_topic_intake(handler, routed_state)
    if classification == "stop":
        return await _handle_on_topic_intake(
            handler,
            routed_state,
            current_answer,
            force_complete=True,
        )
    return await _handle_on_topic_intake(handler, routed_state, current_answer)


async def _maybe_route_cretskens_intake(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
) -> tuple[ChatState, ChatResult | None]:
    if not handler.is_categorized_handoff_mode():
        return current_state, None
    if current_state.intake_route != INTAKE_ROUTE_UNKNOWN:
        return current_state, None
    if (
        current_state.last_question.strip() != INITIAL_QUESTION
        and not is_routing_question(current_state.last_question)
    ):
        return replace(current_state, intake_route=INTAKE_ROUTE_FOOD_PAIRING), None

    resolved_route = await _classify_intake_route(handler, current_answer)
    if resolved_route == INTAKE_ROUTE_UNKNOWN:
        language = (
            current_state.conversation_language
            or handler.language.default_language_code
        )
        return (
            current_state,
            build_routing_question_result(handler, current_state, language),
        )

    next_state = replace(current_state, intake_route=resolved_route)
    return next_state, None


async def _classify_intake_route(
    handler: ChatTurnHandler,
    current_answer: str,
) -> str:
    if handler.intake_route_classifier is None:
        return INTAKE_ROUTE_UNKNOWN
    try:
        return await handler.intake_route_classifier.classify(current_answer)
    except Exception:
        logger.warning(
            "Intake route classifier failed; defaulting to unknown route.",
            exc_info=True,
        )
        return INTAKE_ROUTE_UNKNOWN


async def _classify_intake_answer(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
) -> ClassificationLabel:
    if current_state.last_question.strip() == INITIAL_QUESTION:
        return "on_topic"
    if handler.answer_relevance_classifier is None:
        return "on_topic"
    try:
        return await handler.answer_relevance_classifier.classify(
            current_question=current_state.last_question,
            current_answer=current_answer,
            questionnaire=current_state.questionnaire,
        )
    except Exception:
        logger.warning(
            "Answer relevance classifier failed; defaulting to on_topic.",
            exc_info=True,
        )
        return "on_topic"


async def _handle_on_topic_intake(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
    *,
    force_complete: bool = False,
) -> ChatResult:
    current_state.boundary_violation_count = 0
    updated_questionnaire, updated_summary = await apply_intake_updates(
        handler,
        current_state,
        current_answer,
    )
    if (
        intake_complete_for_state(current_state, updated_questionnaire)
        or force_complete
    ):
        return await build_completion_result(
            handler,
            current_state,
            updated_questionnaire,
            updated_summary,
        )
    return await build_next_question_result(
        handler,
        current_state,
        updated_questionnaire,
        updated_summary,
    )
