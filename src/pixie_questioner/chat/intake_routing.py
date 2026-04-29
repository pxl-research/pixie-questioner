"""Cretskens-specific intake routing and question helpers."""

from __future__ import annotations

from .questionnaire import (
    has_blank_fields,
    questionnaire_complete,
    select_questionnaire_fields,
)
from .types import (
    INTAKE_ROUTE_FOOD_PAIRING,
    INTAKE_ROUTE_UNKNOWN,
    INTAKE_STAGE_DISH,
    INTAKE_STAGE_WINE,
    ChatState,
)

MAX_DISH_FOLLOW_UPS = 3
MAX_WINE_FOLLOW_UPS = 5
DISH_FIELD_INDICES = [0, 1, 2, 3, 4]
WINE_FIELD_INDICES = [5, 6, 7, 8]


def initial_intake_stage_for_route(intake_route: str) -> str:
    if intake_route == INTAKE_ROUTE_FOOD_PAIRING:
        return INTAKE_STAGE_DISH
    return INTAKE_STAGE_DISH


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
    if current_state.intake_route != INTAKE_ROUTE_FOOD_PAIRING:
        return INTAKE_STAGE_DISH
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
