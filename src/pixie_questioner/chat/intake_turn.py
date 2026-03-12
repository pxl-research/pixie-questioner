"""Intake turn handling for classification, updates and next-step selection."""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from .intake_outcomes import (
    build_completion_result,
    build_next_question_result,
    handle_off_topic_intake,
)
from .intake_updates import apply_intake_updates
from .questionnaire import questionnaire_complete
from .relevance import ClassificationLabel
from .types import INITIAL_QUESTION, ChatResult, ChatState

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler

logger = logging.getLogger(__name__)


async def handle_intake_phase(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str,
) -> ChatResult:
    classification = await _classify_intake_answer(
        handler,
        current_state,
        current_answer,
    )
    if classification == "off_topic":
        return handle_off_topic_intake(handler, current_state)
    if classification == "stop":
        return await _handle_on_topic_intake(
            handler,
            current_state,
            current_answer,
            force_complete=True,
        )
    return await _handle_on_topic_intake(handler, current_state, current_answer)


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
    if questionnaire_complete(updated_questionnaire) or force_complete:
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
