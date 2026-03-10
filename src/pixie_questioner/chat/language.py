"""Language detection and response language selection helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

CONVERSATION_LANGUAGE_CONFIDENCE = 0.90
CONVERSATION_LANGUAGE_MIN_WORDS = 6
MAX_RECENT_USER_TURNS = 6
SUPPORTED_RESPONSE_LANGUAGES = frozenset({"nl", "en", "fr"})
_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")
_EXPLICIT_SWITCH_HINTS = frozenset(
    {
        "antwoord",
        "answer",
        "continue",
        "english",
        "francais",
        "français",
        "graag",
        "please",
        "reply",
        "respond",
        "spreek",
        "switch",
        "verder",
        "verdergaan",
    }
)
_EXPLICIT_LANGUAGE_ALIASES: dict[str, frozenset[str]] = {
    "nl": frozenset({"nederlands", "dutch", "hollands", "neerlandais"}),
    "en": frozenset({"engels", "english", "anglais"}),
    "fr": frozenset({"frans", "french", "francais", "français"}),
}

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


def _normalize_turn_text(text: str) -> str:
    return " ".join(text.split())


def _extract_words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text)


def _supported_detection_languages(supported_language_set: set[str]) -> set[str]:
    detected_languages = supported_language_set & SUPPORTED_RESPONSE_LANGUAGES
    if detected_languages:
        return detected_languages
    return supported_language_set


def _supported_language_aliases(
    supported_language_set: set[str],
    supported_language_names: dict[str, str],
) -> dict[str, frozenset[str]]:
    aliases: dict[str, frozenset[str]] = {}
    for language_code in supported_language_set:
        base_aliases = set(_EXPLICIT_LANGUAGE_ALIASES.get(language_code, frozenset()))
        configured_name = (
            supported_language_names.get(language_code, "").casefold().strip()
        )
        if configured_name:
            base_aliases.add(configured_name)
        aliases[language_code] = frozenset(alias for alias in base_aliases if alias)
    return aliases


def detect_explicit_language_switch(
    text: str,
    supported_language_set: set[str],
    supported_language_names: dict[str, str],
) -> str | None:
    normalized_text = text.casefold()
    tokens = set(_extract_words(normalized_text))
    if not tokens:
        return None
    aliases = _supported_language_aliases(
        supported_language_set=supported_language_set,
        supported_language_names=supported_language_names,
    )
    for language_code, language_aliases in aliases.items():
        if not language_aliases:
            continue
        alias_present = any(
            re.search(rf"\b{re.escape(alias)}\b", normalized_text)
            for alias in language_aliases
        )
        if not alias_present:
            continue
        if tokens & _EXPLICIT_SWITCH_HINTS:
            return language_code
    return None


def append_recent_user_turn(
    recent_user_turns: str,
    current_answer: str,
    *,
    max_turns: int = MAX_RECENT_USER_TURNS,
) -> str:
    normalized_answer = _normalize_turn_text(current_answer)
    turns = [line.strip() for line in recent_user_turns.splitlines() if line.strip()]
    if normalized_answer:
        turns.append(normalized_answer)
    if len(turns) > max_turns:
        turns = turns[-max_turns:]
    return "\n".join(turns)


def resolve_conversation_language(
    current_answer: str,
    conversation_language: str,
    recent_user_turns: str,
    supported_language_set: set[str],
    supported_language_priority: dict[str, int],
    supported_language_names: dict[str, str],
    default_language_code: str,
) -> tuple[str, str]:
    updated_recent_user_turns = append_recent_user_turn(
        recent_user_turns,
        current_answer,
    )
    explicit_language = detect_explicit_language_switch(
        current_answer,
        supported_language_set=supported_language_set,
        supported_language_names=supported_language_names,
    )
    if explicit_language is not None:
        return explicit_language, updated_recent_user_turns

    effective_supported_languages = _supported_detection_languages(
        supported_language_set
    )
    combined_text = updated_recent_user_turns.strip()
    if len(_extract_words(combined_text)) < CONVERSATION_LANGUAGE_MIN_WORDS:
        return conversation_language or default_language_code, updated_recent_user_turns

    best_supported = _select_best_supported_language(
        detect_language_candidates(combined_text),
        supported_language_set=effective_supported_languages,
        supported_language_priority=supported_language_priority,
    )
    if best_supported is None:
        return conversation_language or default_language_code, updated_recent_user_turns

    best_language, best_probability = best_supported
    if best_probability < CONVERSATION_LANGUAGE_CONFIDENCE:
        return conversation_language or default_language_code, updated_recent_user_turns
    return best_language, updated_recent_user_turns


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

    def resolve_conversation(
        self,
        current_answer: str,
        conversation_language: str,
        recent_user_turns: str,
    ) -> tuple[str, str]:
        return resolve_conversation_language(
            current_answer=current_answer,
            conversation_language=conversation_language,
            recent_user_turns=recent_user_turns,
            supported_language_set=self.supported_language_set,
            supported_language_priority=self.supported_language_priority,
            supported_language_names=self.supported_language_names,
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
