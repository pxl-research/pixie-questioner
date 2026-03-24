"""State parsing, normalization and serialization helpers."""

from __future__ import annotations

import json
import re

from .language import SUPPORTED_RESPONSE_LANGUAGES, detect_language
from .questionnaire import combine_questionnaire_sections, select_questionnaire_fields
from .types import (
    FEEDBACK_PENDING,
    INTAKE_ROUTE_FOOD_PAIRING,
    INTAKE_ROUTE_GENERAL_WINE,
    INTAKE_STAGE_DISH,
    INTAKE_STAGE_ROUTING,
    INTAKE_STAGE_WINE,
    INTAKE_ROUTE_UNKNOWN,
    VALID_INTAKE_STAGES,
    VALID_INTAKE_ROUTES,
    PHASE_AWAITING_FEEDBACK,
    PHASE_COMPLETED,
    PHASE_INTAKE,
    PHASE_TERMINATED,
    VALID_FEEDBACK_STATUSES,
    VALID_PHASES,
    ChatState,
    INITIAL_QUESTION,
)

_CRETSKENS_DISH_FIELD_INDICES = [0, 1, 2, 3, 4]
_CRETSKENS_WINE_FIELD_INDICES = [5, 6, 7, 8]


def extract_questionnaire_indices(questionnaire: str) -> list[int]:
    indices = set()
    for match in re.finditer(r"(?m)^\s*(\d+)\.", questionnaire):
        indices.add(int(match.group(1)))
    return sorted(indices)


def is_valid_questionnaire(
    questionnaire: str,
    questionnaire_indices: list[int],
) -> bool:
    for index in questionnaire_indices:
        if not re.search(rf"(?m)^\s*{index}\.", questionnaire):
            return False
    return True


def _extract_field_value(questionnaire: str, index: int) -> str:
    """Return the value portion of a numbered field line, or '' if absent."""
    match = re.search(rf"(?m)^\s*{index}\.\s*[^:\n]*:[^\S\n]*(.*?)$", questionnaire)
    if match:
        return match.group(1).strip()
    return ""


def _is_blank_value(value: str) -> bool:
    """Return True when *value* is empty or bracket-only (e.g. ``[]``)."""
    stripped = value.strip().strip("[]").strip()
    return not stripped


def _is_legacy_machine_coded_field_8_value(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict) or not parsed:
        return False
    return all(
        isinstance(key, str) and key.casefold().startswith(("smaak_", "geur_"))
        for key in parsed
    )


def _replace_field_value(questionnaire: str, index: int, value: str) -> str:
    pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*).*$")
    updated, _ = pattern.subn(
        lambda match: f"{match.group(1)}{value}", questionnaire, 1
    )
    return updated


def _sanitize_legacy_machine_coded_field_8(
    questionnaire: str,
    fallback_questionnaire: str = "",
) -> str:
    field_8_value = _extract_field_value(questionnaire, 8)
    if not _is_legacy_machine_coded_field_8_value(field_8_value):
        return questionnaire
    fallback_value = _extract_field_value(fallback_questionnaire, 8)
    replacement = "[]"
    if not (
        _is_blank_value(fallback_value)
        or _is_legacy_machine_coded_field_8_value(fallback_value)
    ):
        replacement = fallback_value
    return _replace_field_value(questionnaire, 8, replacement)


def _prevent_field_regression(
    current_questionnaire: str,
    updated_questionnaire: str,
    questionnaire_indices: list[int],
) -> str:
    """Restore any filled field that the proposed update blanked."""
    result = updated_questionnaire
    for index in questionnaire_indices:
        current_value = _extract_field_value(current_questionnaire, index)
        if _is_blank_value(current_value):
            continue
        updated_value = _extract_field_value(result, index)
        if _is_blank_value(updated_value):
            pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*).*$")
            result = pattern.sub(
                lambda m, v=current_value: f"{m.group(1)}{v}", result, count=1
            )
    return result


def merge_partial_questionnaire_update(
    current_questionnaire: str,
    proposed_questionnaire: str,
    questionnaire_indices: list[int],
) -> str | None:
    if is_valid_questionnaire(proposed_questionnaire, questionnaire_indices):
        return _prevent_field_regression(
            current_questionnaire, proposed_questionnaire, questionnaire_indices
        )

    allowed_indices = set(questionnaire_indices)
    proposed_values: dict[int, str] = {}
    for match in re.finditer(r"(?m)^\s*(\d+)\.\s*(.+)$", proposed_questionnaire):
        index = int(match.group(1))
        if index not in allowed_indices:
            continue
        field_text = match.group(2).strip()
        if ":" not in field_text:
            continue
        value = field_text.split(":", maxsplit=1)[1].strip()
        if _is_blank_value(value):
            continue
        proposed_values[index] = value

    if not proposed_values:
        return None

    merged = current_questionnaire
    merged_any = False
    for index in questionnaire_indices:
        value = proposed_values.get(index)
        if value is None:
            continue
        pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*).*$")
        merged, count = pattern.subn(lambda m: f"{m.group(1)}{value}", merged, count=1)
        if count:
            merged_any = True

    if not merged_any:
        return None
    if not is_valid_questionnaire(merged, questionnaire_indices):
        return None
    return merged


def _read_state_string(
    state: dict[str, str],
    field_name: str,
    default: str,
) -> str:
    value = state.get(field_name, default)
    if not isinstance(value, str):
        raise RuntimeError(f"State field '{field_name}' must be a string.")
    return value


def _normalize_phase(phase: str) -> str:
    if phase in VALID_PHASES:
        return phase
    return PHASE_INTAKE


def _normalize_feedback_status(feedback_status: str) -> str:
    if feedback_status in VALID_FEEDBACK_STATUSES:
        return feedback_status
    return FEEDBACK_PENDING


def _normalize_intake_route(intake_route: str) -> str:
    if intake_route in VALID_INTAKE_ROUTES:
        return intake_route
    return INTAKE_ROUTE_UNKNOWN


def _normalize_intake_stage(intake_stage: str) -> str:
    if intake_stage in VALID_INTAKE_STAGES:
        return intake_stage
    return ""


def _parse_boundary_violation_count(boundary_violation_count_raw: str) -> int:
    try:
        boundary_violation_count = int(boundary_violation_count_raw)
    except ValueError:
        return 0
    return max(boundary_violation_count, 0)


def _parse_state_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes"}


def _infer_intake_stage(
    intake_route: str,
    intake_stage: str,
    dish_follow_up_count: int,
    wine_follow_up_count: int,
) -> str:
    if intake_stage == INTAKE_STAGE_WINE:
        return INTAKE_STAGE_WINE
    if intake_route == INTAKE_ROUTE_GENERAL_WINE:
        return INTAKE_STAGE_WINE
    if intake_route == INTAKE_ROUTE_FOOD_PAIRING:
        if wine_follow_up_count > 0:
            return INTAKE_STAGE_WINE
        if intake_stage == INTAKE_STAGE_DISH:
            return INTAKE_STAGE_DISH
        if dish_follow_up_count > 0:
            return INTAKE_STAGE_DISH
        return INTAKE_STAGE_DISH
    return INTAKE_STAGE_ROUTING


def _normalize_questionnaire(
    questionnaire: str,
    questionnaire_template: str,
) -> str:
    if questionnaire.strip():
        return questionnaire
    return questionnaire_template


def _infer_legacy_conversation_language(
    summary_text: str,
    default_language: str,
) -> str:
    if not summary_text.strip():
        return default_language
    return detect_language(
        summary_text,
        supported_language_set=set(SUPPORTED_RESPONSE_LANGUAGES),
        default_language_code=default_language,
    )


def _normalize_phase_context(
    phase: str,
    feedback_status: str,
    summary: str,
    final_summary: str,
    conversation_language: str,
    post_summary_language: str,
    default_language: str,
) -> tuple[str, str, str, str, str]:
    normalized_final_summary = final_summary if final_summary.strip() else summary
    normalized_conversation_language = (
        conversation_language.strip() or post_summary_language.strip()
    )
    if not normalized_conversation_language:
        if phase == PHASE_INTAKE:
            normalized_conversation_language = default_language
        else:
            normalized_conversation_language = _infer_legacy_conversation_language(
                normalized_final_summary,
                default_language,
            )
    if phase == PHASE_INTAKE:
        return (
            phase,
            FEEDBACK_PENDING,
            "",
            normalized_conversation_language,
            "",
        )

    if phase == PHASE_TERMINATED:
        return (
            phase,
            FEEDBACK_PENDING,
            "",
            normalized_conversation_language,
            normalized_conversation_language,
        )

    normalized_phase = phase
    if phase == PHASE_AWAITING_FEEDBACK and feedback_status != FEEDBACK_PENDING:
        normalized_phase = PHASE_COMPLETED

    return (
        normalized_phase,
        feedback_status,
        normalized_final_summary,
        normalized_conversation_language,
        normalized_conversation_language,
    )


def parse_state(
    state: dict[str, str] | None,
    questionnaire_template: str,
    default_language: str,
    split_questionnaire_field_groups: tuple[list[int], list[int]] | None = None,
) -> ChatState:
    dish_questionnaire = ""
    wine_questionnaire = ""
    dish_indices: list[int] = []
    wine_indices: list[int] = []
    intake_stage = ""
    if not state:
        if split_questionnaire_field_groups is not None:
            dish_indices, wine_indices = split_questionnaire_field_groups
            intake_stage = INTAKE_STAGE_ROUTING
        questionnaire = _normalize_questionnaire("", questionnaire_template)
        if split_questionnaire_field_groups is not None:
            dish_questionnaire = select_questionnaire_fields(
                questionnaire, dish_indices
            )
            wine_questionnaire = select_questionnaire_fields(
                questionnaire, wine_indices
            )
        else:
            dish_questionnaire = ""
            wine_questionnaire = ""
        return ChatState(
            questionnaire=questionnaire,
            dish_questionnaire=dish_questionnaire,
            wine_questionnaire=wine_questionnaire,
            conversation_language=default_language,
            intake_stage=intake_stage,
        )

    questionnaire = _normalize_questionnaire(
        _read_state_string(state, "questionnaire", ""),
        questionnaire_template,
    )
    if split_questionnaire_field_groups is not None:
        dish_indices, wine_indices = split_questionnaire_field_groups
        # Sanitize the combined questionnaire first so that its field-8 value
        # serves as a clean fallback when sanitizing the raw split fields below.
        questionnaire = _sanitize_legacy_machine_coded_field_8(questionnaire)
    summary = _read_state_string(state, "summary", "")
    last_question = _read_state_string(state, "last_question", INITIAL_QUESTION)
    conversation_language = _read_state_string(state, "conversation_language", "")
    recent_user_turns = _read_state_string(state, "recent_user_turns", "")
    phase = _normalize_phase(_read_state_string(state, "phase", PHASE_INTAKE))
    feedback_status = _normalize_feedback_status(
        _read_state_string(state, "feedback_status", FEEDBACK_PENDING)
    )
    final_summary = _read_state_string(state, "final_summary", "")
    post_summary_language = _read_state_string(state, "post_summary_language", "")
    boundary_violation_count = _parse_boundary_violation_count(
        _read_state_string(state, "boundary_violation_count", "0")
    )
    handoff_categories_v1 = _read_state_string(state, "handoff_categories_v1", "")
    intake_route = _normalize_intake_route(
        _read_state_string(state, "intake_route", INTAKE_ROUTE_UNKNOWN)
    )
    dish_follow_up_count = _parse_boundary_violation_count(
        _read_state_string(state, "dish_follow_up_count", "0")
    )
    wine_follow_up_count = _parse_boundary_violation_count(
        _read_state_string(state, "wine_follow_up_count", "0")
    )
    routing_question_asked = _parse_state_bool(
        _read_state_string(state, "routing_question_asked", "false")
    )
    if split_questionnaire_field_groups is not None:
        fallback_dish_questionnaire = select_questionnaire_fields(
            questionnaire, dish_indices
        )
        fallback_wine_questionnaire = select_questionnaire_fields(
            questionnaire, wine_indices
        )
        # Sanitize machine-coded field 8 in the raw split fields.  The
        # fallbacks come from the already-sanitized combined questionnaire so
        # that a free-text field-8 value in the combined state is preserved
        # while a JSON-encoded legacy value is replaced.
        raw_dish_questionnaire = _sanitize_legacy_machine_coded_field_8(
            _read_state_string(state, "dish_questionnaire", ""),
            fallback_dish_questionnaire,
        )
        raw_wine_questionnaire = _sanitize_legacy_machine_coded_field_8(
            _read_state_string(state, "wine_questionnaire", ""),
            fallback_wine_questionnaire,
        )
        dish_questionnaire = (
            raw_dish_questionnaire
            if raw_dish_questionnaire.strip()
            else fallback_dish_questionnaire
        )
        wine_questionnaire = (
            raw_wine_questionnaire
            if raw_wine_questionnaire.strip()
            else fallback_wine_questionnaire
        )
        # Recombine the sanitized splits into the canonical questionnaire.
        # No additional field-8 sanitization is needed: both splits were
        # individually cleaned above.
        questionnaire = (
            combine_questionnaire_sections(dish_questionnaire, wine_questionnaire)
            or questionnaire
        )
        dish_questionnaire = select_questionnaire_fields(questionnaire, dish_indices)
        wine_questionnaire = select_questionnaire_fields(questionnaire, wine_indices)
        intake_stage = _infer_intake_stage(
            intake_route,
            _normalize_intake_stage(_read_state_string(state, "intake_stage", "")),
            dish_follow_up_count,
            wine_follow_up_count,
        )
    (
        phase,
        feedback_status,
        final_summary,
        conversation_language,
        post_summary_language,
    ) = _normalize_phase_context(
        phase=phase,
        feedback_status=feedback_status,
        summary=summary,
        final_summary=final_summary,
        conversation_language=conversation_language,
        post_summary_language=post_summary_language,
        default_language=default_language,
    )
    return ChatState(
        questionnaire=questionnaire,
        dish_questionnaire=dish_questionnaire,
        wine_questionnaire=wine_questionnaire,
        summary=summary,
        last_question=last_question,
        conversation_language=conversation_language,
        recent_user_turns=recent_user_turns,
        phase=phase,
        feedback_status=feedback_status,
        final_summary=final_summary,
        post_summary_language=post_summary_language,
        boundary_violation_count=boundary_violation_count,
        handoff_categories_v1=handoff_categories_v1,
        intake_route=intake_route,
        intake_stage=intake_stage,
        dish_follow_up_count=dish_follow_up_count,
        wine_follow_up_count=wine_follow_up_count,
        routing_question_asked=routing_question_asked,
    )


def serialize_state(state: ChatState) -> dict[str, str]:
    questionnaire = state.questionnaire
    if not questionnaire.strip():
        questionnaire = (
            combine_questionnaire_sections(
                state.dish_questionnaire,
                state.wine_questionnaire,
            )
            or state.questionnaire
        )

    should_emit_split_questionnaires = bool(
        state.intake_stage.strip()
        or state.dish_questionnaire.strip()
        or state.wine_questionnaire.strip()
    )
    # Re-derive split questionnaires from the canonical source.  The field
    # indices are inferred from the state's existing split fields (populated
    # during parse_state for split-aware cases) so no case-specific constants
    # are needed here.
    derived_dish_questionnaire = ""
    derived_wine_questionnaire = ""
    if should_emit_split_questionnaires:
        if state.wine_questionnaire.strip():
            wine_indices = extract_questionnaire_indices(state.wine_questionnaire)
            derived_wine_questionnaire = select_questionnaire_fields(
                questionnaire, wine_indices
            )
        if state.dish_questionnaire.strip():
            dish_indices = extract_questionnaire_indices(state.dish_questionnaire)
            derived_dish_questionnaire = select_questionnaire_fields(
                questionnaire, dish_indices
            )

    serialized = {
        "questionnaire": questionnaire,
        "summary": state.summary,
        "last_question": state.last_question,
        "conversation_language": state.conversation_language,
        "recent_user_turns": state.recent_user_turns,
        "phase": state.phase,
        "feedback_status": state.feedback_status,
        "final_summary": state.final_summary,
        "post_summary_language": (
            state.post_summary_language or state.conversation_language
        ),
        "boundary_violation_count": str(state.boundary_violation_count),
        "handoff_categories_v1": state.handoff_categories_v1,
        "intake_route": state.intake_route,
        "dish_follow_up_count": str(state.dish_follow_up_count),
        "wine_follow_up_count": str(state.wine_follow_up_count),
        "routing_question_asked": ("true" if state.routing_question_asked else "false"),
    }
    if derived_dish_questionnaire.strip():
        serialized["dish_questionnaire"] = derived_dish_questionnaire
    if derived_wine_questionnaire.strip():
        serialized["wine_questionnaire"] = derived_wine_questionnaire
    if state.intake_stage.strip():
        serialized["intake_stage"] = state.intake_stage
    return serialized
