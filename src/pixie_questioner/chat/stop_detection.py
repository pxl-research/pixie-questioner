"""Stop phrase detection helpers."""

from __future__ import annotations

import re

_WORD_TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


def tokenize_words(text: str) -> tuple[str, ...]:
    return tuple(_WORD_TOKEN_PATTERN.findall(text.casefold()))


def build_stop_phrase_token_sequences(
    stop_phrases: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        phrase_tokens
        for phrase_tokens in (tokenize_words(phrase) for phrase in stop_phrases)
        if phrase_tokens
    )


def matches_stop_phrase(
    answer_tokens: tuple[str, ...],
    stop_phrase_tokens: tuple[str, ...],
    stop_context_tokens: frozenset[str],
) -> bool:
    matched_tokens = 0
    for answer_token in answer_tokens:
        if (
            matched_tokens < len(stop_phrase_tokens)
            and answer_token == stop_phrase_tokens[matched_tokens]
        ):
            matched_tokens += 1
            continue
        if answer_token not in stop_context_tokens:
            return False
    return matched_tokens == len(stop_phrase_tokens)


def user_requested_stop(
    current_answer: str,
    stop_phrase_token_sequences: tuple[tuple[str, ...], ...],
    stop_context_tokens: frozenset[str],
) -> bool:
    answer_tokens = tokenize_words(current_answer)
    if not answer_tokens:
        return False
    return any(
        matches_stop_phrase(
            answer_tokens,
            stop_phrase_tokens,
            stop_context_tokens=stop_context_tokens,
        )
        for stop_phrase_tokens in stop_phrase_token_sequences
    )
