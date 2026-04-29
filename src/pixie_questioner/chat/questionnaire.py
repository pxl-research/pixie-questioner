"""Questionnaire-specific helper functions for chat intake flows."""

from __future__ import annotations

import re

_UNSPECIFIED_FIELD_MARKERS = frozenset(
    {
        "niet gespecificeerd",
        "niet opgegeven",
        "onbekend",
        "not specified",
        "unspecified",
        "unknown",
        "non spécifié",
        "non specifie",
    }
)


def _questionnaire_lines_by_index(questionnaire: str) -> dict[int, str]:
    lines_by_index: dict[int, str] = {}
    for line in questionnaire.splitlines():
        match = re.match(r"^\s*(\d+)\.\s", line)
        if match is None:
            continue
        lines_by_index[int(match.group(1))] = line.rstrip()
    return lines_by_index


def select_questionnaire_fields(questionnaire: str, field_indices: list[int]) -> str:
    lines_by_index = _questionnaire_lines_by_index(questionnaire)
    return "\n".join(
        lines_by_index[index] for index in field_indices if index in lines_by_index
    )


def combine_questionnaire_sections(*questionnaires: str) -> str:
    combined_lines: dict[int, str] = {}
    for questionnaire in questionnaires:
        if not questionnaire.strip():
            continue
        combined_lines.update(_questionnaire_lines_by_index(questionnaire))
    return "\n".join(line for _, line in sorted(combined_lines.items()))


def questionnaire_complete(questionnaire: str) -> bool:
    return "[]" not in questionnaire


def _is_blank_field_value(value: str) -> bool:
    return not value.strip().strip("[]").strip()


def _is_unspecified_field_value(value: str) -> bool:
    normalized = value.strip().strip("[]").strip().casefold()
    return normalized in _UNSPECIFIED_FIELD_MARKERS


def _field_value(questionnaire: str, index: int) -> str | None:
    match = re.search(
        rf"(?m)^\s*{index}\.\s*[^:\n]*:[^\S\n]*(.*?)$",
        questionnaire,
    )
    if match is None:
        return None
    return match.group(1)


def has_blank_fields(questionnaire: str, field_indices: list[int]) -> bool:
    for index in field_indices:
        value = _field_value(questionnaire, index)
        if value is not None and _is_blank_field_value(value):
            return True
    return False


def first_blank_field_index(
    questionnaire: str,
    questionnaire_indices: list[int],
) -> int | None:
    for index in questionnaire_indices:
        value = _field_value(questionnaire, index)
        if value is not None and _is_blank_field_value(value):
            return index
    return None


def mask_later_blank_fields(
    questionnaire: str,
    questionnaire_indices: list[int],
) -> str:
    """Keep only the earliest blank field as ``[]`` for deterministic question flow."""
    first_blank_index = first_blank_field_index(questionnaire, questionnaire_indices)
    if first_blank_index is None:
        return questionnaire

    masked = questionnaire
    for index in questionnaire_indices:
        if index <= first_blank_index:
            continue
        pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*)(.*?)$")
        match = pattern.search(masked)
        if match is None:
            continue
        if not _is_blank_field_value(match.group(2)):
            continue
        masked = pattern.sub(r"\1[pending later]", masked, count=1)

    return masked


def _mask_specific_blank_fields(
    questionnaire: str,
    field_indices: list[int],
) -> str:
    masked = questionnaire
    for index in field_indices:
        pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*)(.*?)$")
        match = pattern.search(masked)
        if match is None or not _is_blank_field_value(match.group(2)):
            continue
        masked = pattern.sub(r"\1[pending later]", masked, count=1)
    return masked


def normalize_blank_like_questionnaire_values(
    questionnaire: str,
    field_indices: list[int],
) -> str:
    normalized_questionnaire = questionnaire
    for index in field_indices:
        pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*)(.*?)$")
        match = pattern.search(normalized_questionnaire)
        if match is None:
            continue
        if not (
            _is_blank_field_value(match.group(2))
            or _is_unspecified_field_value(match.group(2))
        ):
            continue
        normalized_questionnaire = pattern.sub(
            r"\1[]",
            normalized_questionnaire,
            count=1,
        )
    return normalized_questionnaire


def clear_later_blank_field_updates(
    *,
    questionnaire: str,
    previous_questionnaire: str,
    field_indices: list[int],
    current_field_index: int | None,
) -> str:
    if current_field_index is None:
        return questionnaire

    normalized_questionnaire = questionnaire
    for index in field_indices:
        if index <= current_field_index:
            continue
        previous_value = _field_value(previous_questionnaire, index)
        if previous_value is None or not _is_blank_field_value(previous_value):
            continue
        pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*)(.*?)$")
        match = pattern.search(normalized_questionnaire)
        if match is None or _is_blank_field_value(match.group(2)):
            continue
        normalized_questionnaire = pattern.sub(
            r"\1[]",
            normalized_questionnaire,
            count=1,
        )
    return normalized_questionnaire


def focus_questionnaire_for_route(
    questionnaire: str,
    questionnaire_indices: list[int],
    intake_route: str,
    *,
    allow_dish_questions: bool = True,
) -> str:
    del intake_route
    if allow_dish_questions:
        return mask_later_blank_fields(questionnaire, questionnaire_indices)

    masked = _mask_specific_blank_fields(questionnaire, [0, 1, 2, 3, 4])
    return mask_later_blank_fields(masked, questionnaire_indices)


def ensure_initial_question_field(
    questionnaire: str,
    current_answer: str,
) -> str:
    """Ensure field 0 is not left blank after the very first user turn."""
    normalized_answer = current_answer.strip()
    if not normalized_answer:
        return questionnaire

    pattern = re.compile(r"(?m)^(\s*0\.\s*[^:\n]*:[^\S\n]*)(.*)$")
    match = pattern.search(questionnaire)
    if match is None:
        return questionnaire
    if not _is_blank_field_value(match.group(2)):
        return questionnaire
    return pattern.sub(
        lambda field_match: f"{field_match.group(1)}{normalized_answer}",
        questionnaire,
        count=1,
    )
