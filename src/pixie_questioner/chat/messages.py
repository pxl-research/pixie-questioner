"""Localized chat copy and token normalization helpers."""

from __future__ import annotations

import secrets
import re

from .policy import ChatQuickReply


def get_localized_options(
    localized_values: dict[str, tuple[str, ...]],
    language: str,
) -> tuple[str, ...]:
    normalized_language = language.casefold()
    language_prefix = normalized_language.split("-")[0]
    if (
        normalized_language in localized_values
        and localized_values[normalized_language]
    ):
        return localized_values[normalized_language]
    if language_prefix in localized_values and localized_values[language_prefix]:
        return localized_values[language_prefix]
    if "en" in localized_values and localized_values["en"]:
        return localized_values["en"]
    for options in localized_values.values():
        if options:
            return options
    raise RuntimeError("Localized configuration is missing message options.")


def choose_localized_message(
    localized_values: dict[str, tuple[str, ...]],
    language: str,
) -> str:
    return secrets.choice(get_localized_options(localized_values, language))


def build_feedback_quick_replies(
    quick_replies: dict[str, tuple[ChatQuickReply, ...]],
    language: str,
) -> list[dict[str, str]]:
    normalized_language = language.casefold()
    language_prefix = normalized_language.split("-")[0]
    selected = quick_replies.get(normalized_language)
    if not selected:
        selected = quick_replies.get(language_prefix)
    if not selected:
        selected = quick_replies.get("en")
    if not selected:
        for quick_reply_options in quick_replies.values():
            if quick_reply_options:
                selected = quick_reply_options
                break
    if not selected:
        return []
    return [{"label": item.label, "value": item.value} for item in selected]


def build_off_topic_redirect(
    language: str,
    current_question: str,
    initial_question: str,
    off_topic_initial_questions: dict[str, tuple[str, ...]],
    off_topic_redirects: dict[str, tuple[str, ...]],
) -> str:
    if current_question.strip() == initial_question:
        return choose_localized_message(off_topic_initial_questions, language)
    return choose_localized_message(off_topic_redirects, language).format(
        question=current_question
    )


def normalize_feedback_response(
    current_answer: str,
    feedback_yes_tokens: frozenset[str],
    feedback_no_tokens: frozenset[str],
    feedback_yes_value: str,
    feedback_no_value: str,
) -> str | None:
    tokens = set(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", current_answer.casefold()))
    has_yes = bool(tokens & feedback_yes_tokens)
    has_no = bool(tokens & feedback_no_tokens)
    if has_yes and not has_no:
        return feedback_yes_value
    if has_no and not has_yes:
        return feedback_no_value
    return None


def simplify_technical_terms(
    text: str,
    language: str,
    term_replacements: dict[str, tuple[tuple[str, str], ...]],
) -> str:
    language_code = language.casefold().split("-")[0]
    replacements = term_replacements.get(language_code)
    if not replacements:
        return text

    simplified = text
    for source, target in replacements:
        simplified = re.sub(
            rf"\b{re.escape(source)}\b",
            lambda match: (
                target.capitalize() if match.group(0)[0].isupper() else target
            ),
            simplified,
            flags=re.IGNORECASE,
        )
    return simplified
