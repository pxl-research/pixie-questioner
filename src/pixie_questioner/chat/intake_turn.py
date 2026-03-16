"""Intake turn handling for classification, updates and next-step selection."""

from __future__ import annotations

import logging
import re

from dataclasses import replace
from typing import TYPE_CHECKING

from .categorized_handoff import (
    extract_questionnaire_field_value,
    is_blank_questionnaire_value,
)
from .intake_routing import build_routing_question_result
from .intake_routing import initial_intake_stage_for_route
from .intake_routing import intake_complete_for_state
from .intake_routing import is_routing_question
from .intake_routing import resolve_intake_stage
from .intake_routing import WINE_FIELD_INDICES
from .intake_outcomes import (
    build_completion_result,
    build_next_question_result,
    handle_off_topic_intake,
)
from .intake_updates import apply_intake_updates
from .questionnaire import first_blank_field_index
from .relevance import ClassificationLabel
from .types import (
    INITIAL_QUESTION,
    INTAKE_ROUTE_FOOD_PAIRING,
    INTAKE_ROUTE_GENERAL_WINE,
    INTAKE_ROUTE_UNKNOWN,
    INTAKE_STAGE_WINE,
    ChatResult,
    ChatState,
)

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler

logger = logging.getLogger(__name__)

_ORIGIN_FOLLOW_UP_BY_LANGUAGE = {
    "nl": (
        "Dat land of die regio hebben we momenteel niet in ons aanbod. "
        "Is een ander land of een andere regio ook goed, of wil je liever "
        "dat ik los van herkomst iets voorstel?"
    ),
    "en": (
        "We currently do not have that country or region in our selection. "
        "Would another country or region work, or would you prefer a "
        "suggestion without an origin preference?"
    ),
    "fr": (
        "Nous n'avons actuellement pas ce pays ou cette region dans notre "
        "assortiment. Un autre pays ou une autre region conviendrait-il, ou "
        "preferez-vous une suggestion sans preference d'origine ?"
    ),
}
_NO_PREFERENCE_ORIGIN_PATTERNS = (
    "geen voorkeur",
    "maakt niet uit",
    "maakt me niet uit",
    "niet gespecificeerd",
    "ik weet het niet",
    "no preference",
    "doesn't matter",
    "not specified",
    "pas de preference",
    "peu importe",
)
_ORIGIN_CUE_PATTERNS = (
    re.compile(r"\buit\s+[^\d,.!?;\n]+", flags=re.IGNORECASE),
    re.compile(r"\bfrom\s+[^\d,.!?;\n]+", flags=re.IGNORECASE),
)


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

    next_state = replace(
        current_state,
        intake_route=resolved_route,
        intake_stage=initial_intake_stage_for_route(resolved_route),
    )
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
    updates = await apply_intake_updates(
        handler,
        current_state,
        current_answer,
    )
    updated_state = replace(
        current_state,
        questionnaire=updates.questionnaire,
        dish_questionnaire=updates.dish_questionnaire,
        wine_questionnaire=updates.wine_questionnaire,
        summary=updates.summary,
    )
    if handler.is_categorized_handoff_mode():
        updated_state.intake_stage = resolve_intake_stage(updated_state)
    unsupported_origin_follow_up = _build_unsupported_origin_follow_up(
        handler,
        current_state,
        updated_state,
        current_answer,
    )
    if unsupported_origin_follow_up is not None:
        follow_up_state = replace(
            updated_state,
            last_question=unsupported_origin_follow_up,
        )
        if (
            updated_state.intake_stage == INTAKE_STAGE_WINE
            or updated_state.intake_route == INTAKE_ROUTE_GENERAL_WINE
        ):
            follow_up_state = replace(
                follow_up_state,
                wine_follow_up_count=current_state.wine_follow_up_count + 1,
            )
        return handler.answer_result(
            unsupported_origin_follow_up,
            follow_up_state,
            None,
            None,
            None,
        )
    if (
        intake_complete_for_state(updated_state, updates.questionnaire)
        or force_complete
    ):
        return await build_completion_result(
            handler,
            updated_state,
            updates.questionnaire,
            updates.summary,
        )
    return await build_next_question_result(
        handler,
        updated_state,
        updates.questionnaire,
        updates.summary,
    )


def _build_unsupported_origin_follow_up(
    handler: ChatTurnHandler,
    current_state: ChatState,
    updated_state: ChatState,
    current_answer: str,
) -> str | None:
    if not handler.is_categorized_handoff_mode():
        return None
    if (
        updated_state.intake_stage != INTAKE_STAGE_WINE
        and updated_state.intake_route != INTAKE_ROUTE_GENERAL_WINE
    ):
        return None

    origin_value = extract_questionnaire_field_value(
        updated_state.wine_questionnaire, 6
    )
    if not is_blank_questionnaire_value(origin_value):
        return None

    normalized_answer = " ".join(current_answer.casefold().split())
    if not normalized_answer:
        return None
    if any(pattern in normalized_answer for pattern in _NO_PREFERENCE_ORIGIN_PATTERNS):
        return None

    previous_blank_field = first_blank_field_index(
        current_state.wine_questionnaire,
        WINE_FIELD_INDICES,
    )
    opening_general_wine_turn = (
        current_state.intake_route == INTAKE_ROUTE_GENERAL_WINE
        and (
            current_state.last_question.strip() == INITIAL_QUESTION
            or is_routing_question(current_state.last_question)
        )
    )
    if previous_blank_field != 6 and not (
        opening_general_wine_turn and _contains_origin_cue(current_answer)
    ):
        return None

    language = (
        updated_state.conversation_language or handler.language.default_language_code
    )
    return _ORIGIN_FOLLOW_UP_BY_LANGUAGE.get(
        language, _ORIGIN_FOLLOW_UP_BY_LANGUAGE["en"]
    )


def _contains_origin_cue(current_answer: str) -> bool:
    normalized_answer = current_answer.strip()
    if not normalized_answer:
        return False
    return any(pattern.search(normalized_answer) for pattern in _ORIGIN_CUE_PATTERNS)
