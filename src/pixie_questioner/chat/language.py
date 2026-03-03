"""Language detection and response language selection helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

SUPPORTED_NON_DUTCH_STRONG_PROBABILITY = 0.85
SUPPORTED_NON_DUTCH_MIN_WORDS = 4
SUPPORTED_RESPONSE_LANGUAGES = frozenset({"nl", "en", "fr"})

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
    effective_supported_languages = (
        supported_language_set & SUPPORTED_RESPONSE_LANGUAGES
    )
    candidates = detect_language_candidates(text)
    if not candidates:
        return default_language_code

    for language_code, _ in candidates:
        if language_code in effective_supported_languages:
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
    effective_supported_languages = (
        supported_language_set & SUPPORTED_RESPONSE_LANGUAGES
    )
    candidates = detect_language_candidates(text)
    if not candidates:
        return default_language_code, True

    best_supported = _select_best_supported_language(
        candidates,
        supported_language_set=effective_supported_languages,
        supported_language_priority=supported_language_priority,
    )
    if best_supported is None:
        return default_language_code, True

    best_supported_language, best_supported_probability = best_supported
    word_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text))

    if best_supported_language == default_language_code:
        return default_language_code, False
    if (
        best_supported_probability >= SUPPORTED_NON_DUTCH_STRONG_PROBABILITY
        and word_count >= SUPPORTED_NON_DUTCH_MIN_WORDS
    ):
        return best_supported_language, False
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


@dataclass
class LanguageSupport:
    """Pre-computed language detection and instruction builder.

    Binds the normalized language configuration so callers do not need
    to thread individual config values through every call.
    """

    supported_language_codes: tuple[str, ...]
    supported_language_set: set[str]
    supported_language_priority: dict[str, int]
    supported_language_names: dict[str, str]
    default_language_code: str

    def detect(self, text: str) -> str:
        return detect_language(
            text,
            supported_language_set=self.supported_language_set,
            default_language_code=self.default_language_code,
        )

    def detect_preferred(self, current_answer: str, previous_summary: str) -> str:
        return detect_preferred_language(
            current_answer=current_answer,
            previous_summary=previous_summary,
            supported_language_set=self.supported_language_set,
            supported_language_priority=self.supported_language_priority,
            default_language_code=self.default_language_code,
        )

    def build_instruction(self, language_code: str) -> str:
        return build_language_instruction(
            language_code,
            supported_language_names=self.supported_language_names,
            supported_language_codes=self.supported_language_codes,
            default_language_code=self.default_language_code,
        )


def create_language_support(
    language_codes: tuple[str, ...],
    language_names: dict[str, str],
) -> LanguageSupport:
    """Build a LanguageSupport from raw domain-policy language configuration."""
    normalized_codes = tuple(
        dict.fromkeys(
            code.strip().casefold() for code in language_codes if code.strip()
        )
    )
    if not normalized_codes:
        raise RuntimeError("language_codes must include at least one language.")

    supported_set = set(normalized_codes)
    priority = {code: idx for idx, code in enumerate(normalized_codes)}
    default = "nl" if "nl" in supported_set else normalized_codes[0]

    names: dict[str, str] = {
        code.strip().casefold(): name
        for code, name in language_names.items()
        if code.strip()
    }
    for code in normalized_codes:
        names.setdefault(code, code.upper())
    names.setdefault("en", "English")

    return LanguageSupport(
        supported_language_codes=normalized_codes,
        supported_language_set=supported_set,
        supported_language_priority=priority,
        supported_language_names=names,
        default_language_code=default,
    )
