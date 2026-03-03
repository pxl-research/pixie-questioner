"""Localized chat copy and token normalization helpers."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Any

from .policy import ChatDomainPolicy, ChatQuickReply
from .types import FEEDBACK_NO, FEEDBACK_YES, INITIAL_QUESTION


def _resolve_for_language(
    localized_values: dict[str, Any],
    language: str,
) -> Any:
    """Find the best localized value with cascading fallback.

    Resolution order: exact match -> prefix match -> English -> first non-empty.
    Returns ``None`` when no non-empty entry exists.
    """
    normalized = language.casefold()
    prefix = normalized.split("-")[0]
    for key in (normalized, prefix, "en"):
        value = localized_values.get(key)
        if value:
            return value
    for value in localized_values.values():
        if value:
            return value
    return None


def get_localized_options(
    localized_values: dict[str, tuple[str, ...]],
    language: str,
) -> tuple[str, ...]:
    result = _resolve_for_language(localized_values, language)
    if result is None:
        raise RuntimeError("Localized configuration is missing message options.")
    return result


def choose_localized_message(
    localized_values: dict[str, tuple[str, ...]],
    language: str,
) -> str:
    return secrets.choice(get_localized_options(localized_values, language))


def build_feedback_quick_replies(
    quick_replies: dict[str, tuple[ChatQuickReply, ...]],
    language: str,
) -> list[dict[str, str]]:
    selected = _resolve_for_language(quick_replies, language)
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
            lambda match, replacement=target: (
                replacement.capitalize() if match.group(0)[0].isupper() else replacement
            ),
            simplified,
            flags=re.IGNORECASE,
        )
    return simplified


@dataclass(frozen=True)
class PolicyMessages:
    """Localized message builder backed by a domain policy.

    Wraps a ChatDomainPolicy so callers can request messages by language
    without passing individual config dicts every time.
    """

    policy: ChatDomainPolicy

    def handoff(self, language: str) -> str:
        return choose_localized_message(self.policy.handoff_lines, language)

    def feedback_question(self, language: str) -> str:
        return choose_localized_message(self.policy.feedback_questions, language)

    def feedback_quick_replies(self, language: str) -> list[dict[str, str]]:
        return build_feedback_quick_replies(
            self.policy.quick_replies, language=language
        )

    def feedback_thanks(self, language: str) -> str:
        return choose_localized_message(self.policy.feedback_thanks, language)

    def goodbye(self, language: str) -> str:
        return choose_localized_message(self.policy.goodbye_lines, language)

    def off_topic_redirect(self, language: str, current_question: str) -> str:
        return build_off_topic_redirect(
            language=language,
            current_question=current_question,
            initial_question=INITIAL_QUESTION,
            off_topic_initial_questions=self.policy.off_topic_initial_questions,
            off_topic_redirects=self.policy.off_topic_redirects,
        )

    def normalize_feedback(self, current_answer: str) -> str | None:
        return normalize_feedback_response(
            current_answer=current_answer,
            feedback_yes_tokens=self.policy.feedback_yes_tokens,
            feedback_no_tokens=self.policy.feedback_no_tokens,
            feedback_yes_value=FEEDBACK_YES,
            feedback_no_value=FEEDBACK_NO,
        )

    def simplify_terms(self, text: str, language: str) -> str:
        return simplify_technical_terms(
            text,
            language=language,
            term_replacements=self.policy.term_replacements,
        )
