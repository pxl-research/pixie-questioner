"""Cretskens-specific intake routing and question helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .questionnaire import (
    has_blank_fields,
    questionnaire_complete,
    select_questionnaire_fields,
)
from .types import (
    INTAKE_ROUTE_FOOD_PAIRING,
    INTAKE_ROUTE_GENERAL_WINE,
    INTAKE_ROUTE_UNKNOWN,
    INTAKE_STAGE_DISH,
    INTAKE_STAGE_ROUTING,
    INTAKE_STAGE_WINE,
    ChatResult,
    ChatState,
)

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler

ROUTING_QUESTION_BY_LANGUAGE = {
    "nl": "Zoek je een wijn voor bij een gerecht, of om gewoon zo te drinken?",
    "en": "Are you looking for a wine for a dish, or just to drink on its own?",
    "fr": "Cherchez-vous un vin pour accompagner un plat, ou simplement à boire seul ?",
}
MAX_DISH_FOLLOW_UPS = 3
MAX_WINE_FOLLOW_UPS = 5
DISH_FIELD_INDICES = [0, 1, 2, 3, 4]
WINE_FIELD_INDICES = [5, 6, 7, 8]


def build_routing_question(conversation_language: str) -> str:
    return ROUTING_QUESTION_BY_LANGUAGE.get(
        conversation_language,
        ROUTING_QUESTION_BY_LANGUAGE["en"],
    )


def is_routing_question(question_text: str) -> bool:
    return question_text.strip() in set(ROUTING_QUESTION_BY_LANGUAGE.values())


def build_routing_question_result(
    handler: ChatTurnHandler,
    current_state: ChatState,
    conversation_language: str,
) -> ChatResult:
    routing_question = build_routing_question(conversation_language)
    next_state = replace(
        current_state,
        last_question=routing_question,
        routing_question_asked=True,
        intake_stage=INTAKE_STAGE_ROUTING,
    )
    return handler.answer_result(routing_question, next_state, None, None, None)


def initial_intake_stage_for_route(intake_route: str) -> str:
    if intake_route == INTAKE_ROUTE_GENERAL_WINE:
        return INTAKE_STAGE_WINE
    if intake_route == INTAKE_ROUTE_FOOD_PAIRING:
        return INTAKE_STAGE_DISH
    return INTAKE_STAGE_ROUTING


def _has_blank_fields_in_questionnaire(
    questionnaire: str,
    field_indices: list[int],
) -> bool:
    focused = select_questionnaire_fields(questionnaire, field_indices)
    return has_blank_fields(focused, field_indices)


def resolve_intake_stage_for_questionnaire(
    current_state: ChatState,
    questionnaire: str,
) -> str:
    if current_state.intake_route == INTAKE_ROUTE_GENERAL_WINE:
        return INTAKE_STAGE_WINE
    if current_state.intake_route != INTAKE_ROUTE_FOOD_PAIRING:
        return INTAKE_STAGE_ROUTING
    if current_state.intake_stage == INTAKE_STAGE_WINE:
        return INTAKE_STAGE_WINE
    if current_state.dish_follow_up_count >= MAX_DISH_FOLLOW_UPS:
        return INTAKE_STAGE_WINE
    if _has_blank_fields_in_questionnaire(questionnaire, DISH_FIELD_INDICES):
        return INTAKE_STAGE_DISH
    return INTAKE_STAGE_WINE


def resolve_intake_stage(current_state: ChatState) -> str:
    return resolve_intake_stage_for_questionnaire(
        current_state,
        current_state.questionnaire,
    )


def determine_next_question_track(
    current_state: ChatState,
    questionnaire: str,
) -> str | None:
    dish_has_blanks = _has_blank_fields_in_questionnaire(
        questionnaire,
        DISH_FIELD_INDICES,
    )
    wine_has_blanks = _has_blank_fields_in_questionnaire(
        questionnaire,
        WINE_FIELD_INDICES,
    )

    if current_state.intake_route == INTAKE_ROUTE_UNKNOWN:
        if questionnaire_complete(questionnaire):
            return None
        return "generic"

    if current_state.intake_route == INTAKE_ROUTE_GENERAL_WINE:
        if current_state.wine_follow_up_count == 0:
            return "wine"
        if current_state.wine_follow_up_count < MAX_WINE_FOLLOW_UPS and wine_has_blanks:
            return "wine"
        return None

    stage = resolve_intake_stage_for_questionnaire(current_state, questionnaire)
    if (
        stage == INTAKE_STAGE_DISH
        and current_state.dish_follow_up_count < MAX_DISH_FOLLOW_UPS
        and dish_has_blanks
    ):
        return "dish"

    if stage == INTAKE_STAGE_DISH:
        if (
            current_state.dish_follow_up_count == 0
            and current_state.wine_follow_up_count == 0
        ):
            return "wine"
        if _has_blank_fields_in_questionnaire(questionnaire, [7]):  # BUDGET GUARD
            return "wine"
        return None

    if current_state.wine_follow_up_count < MAX_WINE_FOLLOW_UPS and wine_has_blanks:
        return "wine"
    if (
        current_state.dish_follow_up_count == 0
        and current_state.wine_follow_up_count == 0
    ):
        return "wine"
    if _has_blank_fields_in_questionnaire(questionnaire, [7]):  # BUDGET GUARD
        return "wine"
    return None


def intake_complete_for_state(
    current_state: ChatState,
    questionnaire: str,
) -> bool:
    if current_state.intake_route == INTAKE_ROUTE_UNKNOWN:
        return questionnaire_complete(questionnaire)
    next_track = determine_next_question_track(current_state, questionnaire)
    if next_track is not None:
        return False
    return True
