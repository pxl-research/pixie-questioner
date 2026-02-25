"""Language detection and response language selection helpers."""

from __future__ import annotations

import re

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

SUPPORTED_LANGUAGE_CLOSE_MARGIN = 0.15
OTHER_LANGUAGE_STRONG_PROBABILITY = 0.85
OTHER_LANGUAGE_MIN_WORDS = 4
SUPPORTED_NON_DUTCH_STRONG_PROBABILITY = 0.85
SUPPORTED_NON_DUTCH_MIN_WORDS = 4

DetectorFactory.seed = 0


def detect_language_candidates(text: str) -> list[tuple[str, float]]:
    if not text.strip():
        return []
    try:
        detected = detect_langs(text)
    except LangDetectException:
        return []
    return [(candidate.lang, candidate.prob) for candidate in detected]


def detect_language(
    text: str,
    supported_language_set: set[str],
    default_language_code: str,
) -> str:
    candidates = detect_language_candidates(text)
    if not candidates:
        return default_language_code

    for language_code, _ in candidates:
        if language_code in supported_language_set:
            return language_code
    return default_language_code


def _select_best_supported_language(
    candidates: list[tuple[str, float]],
    supported_language_set: set[str],
    supported_language_priority: dict[str, int],
) -> tuple[str, float] | None:
    supported_candidates = [
        (language_code, probability)
        for language_code, probability in candidates
        if language_code in supported_language_set
    ]
    if not supported_candidates:
        return None
    return min(
        supported_candidates,
        key=lambda item: (
            -item[1],
            supported_language_priority[item[0]],
        ),
    )


def _select_preferred_language_from_text(
    text: str,
    supported_language_set: set[str],
    supported_language_priority: dict[str, int],
    default_language_code: str,
) -> tuple[str, bool]:
    candidates = detect_language_candidates(text)
    if not candidates:
        return default_language_code, True

    top_language, top_probability = candidates[0]
    best_supported = _select_best_supported_language(
        candidates,
        supported_language_set=supported_language_set,
        supported_language_priority=supported_language_priority,
    )
    word_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text))

    if top_language in supported_language_set:
        if top_language == default_language_code:
            return top_language, False
        if (
            top_probability >= SUPPORTED_NON_DUTCH_STRONG_PROBABILITY
            and word_count >= SUPPORTED_NON_DUTCH_MIN_WORDS
        ):
            return top_language, False
        return top_language, True

    if best_supported is not None:
        best_supported_language, best_supported_probability = best_supported
        score_gap = top_probability - best_supported_probability
        if score_gap <= SUPPORTED_LANGUAGE_CLOSE_MARGIN:
            return best_supported_language, True
        if (
            top_probability >= OTHER_LANGUAGE_STRONG_PROBABILITY
            and word_count >= OTHER_LANGUAGE_MIN_WORDS
        ):
            return top_language, False
        return best_supported_language, True

    if (
        top_probability >= OTHER_LANGUAGE_STRONG_PROBABILITY
        and word_count >= OTHER_LANGUAGE_MIN_WORDS
    ):
        return top_language, False
    return default_language_code, True


def detect_preferred_language(
    current_answer: str,
    previous_summary: str,
    supported_language_set: set[str],
    supported_language_priority: dict[str, int],
    default_language_code: str,
) -> str:
    answer_language, answer_is_uncertain = _select_preferred_language_from_text(
        current_answer,
        supported_language_set=supported_language_set,
        supported_language_priority=supported_language_priority,
        default_language_code=default_language_code,
    )
    if not answer_is_uncertain:
        return answer_language

    summary_language, summary_is_uncertain = _select_preferred_language_from_text(
        previous_summary,
        supported_language_set=supported_language_set,
        supported_language_priority=supported_language_priority,
        default_language_code=default_language_code,
    )
    if not summary_is_uncertain:
        return summary_language

    if summary_language in supported_language_set:
        return summary_language
    return default_language_code


def build_language_instruction(
    language_code: str,
    supported_language_names: dict[str, str],
    supported_language_codes: tuple[str, ...],
    default_language_code: str,
) -> str:
    default_language_name = supported_language_names.get(
        default_language_code,
        default_language_code.upper(),
    )
    if language_code in supported_language_names:
        language_target = supported_language_names[language_code]
        default_hint = f"- If uncertain, default to {default_language_name}."
    else:
        language_target = f"the language with ISO code '{language_code}'"
        fallback_targets = ", ".join(
            supported_language_names.get(code, code.upper())
            for code in supported_language_codes[:3]
        )
        default_hint = f"- If uncertain, default to one of: {fallback_targets}."
    return (
        "\n\nLanguage policy:\n"
        f"- Respond in {language_target}.\n"
        f"{default_hint}\n"
        f"- Prefer {default_language_name} when there is no strong language signal."
    )
