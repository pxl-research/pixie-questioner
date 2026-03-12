"""Questionnaire-specific helper functions for chat intake flows."""

from __future__ import annotations

import re


def questionnaire_complete(questionnaire: str) -> bool:
    return "[]" not in questionnaire


def _is_blank_field_value(value: str) -> bool:
    return not value.strip().strip("[]").strip()


def mask_later_blank_fields(
    questionnaire: str,
    questionnaire_indices: list[int],
) -> str:
    """Keep only the earliest blank field as ``[]`` for deterministic question flow."""
    first_blank_index: int | None = None
    for index in questionnaire_indices:
        match = re.search(
            rf"(?m)^\s*{index}\.\s*[^:\n]*:[^\S\n]*(.*?)$",
            questionnaire,
        )
        if match is None:
            continue
        if _is_blank_field_value(match.group(1)):
            first_blank_index = index
            break

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
