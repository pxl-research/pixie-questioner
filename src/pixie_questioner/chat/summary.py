"""Summary language helpers for chat intake flows."""

from __future__ import annotations

import logging
import re

from ..llm_client import extract_llm_content, resolve_llm_client
from .language import LanguageSupport, detect_language_candidates

logger = logging.getLogger(__name__)

SUMMARY_LANGUAGE_MIN_WORDS = 4


def summary_matches_conversation_language(
    summary_text: str,
    conversation_language: str,
    supported_language_set: set[str],
) -> bool:
    if not summary_text.strip():
        return True
    if conversation_language not in supported_language_set:
        return True
    candidates = detect_language_candidates(summary_text)
    if not candidates:
        return True
    primary_language_code, _ = max(candidates, key=lambda candidate: candidate[1])
    if primary_language_code != conversation_language:
        word_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", summary_text))
        if word_count < SUMMARY_LANGUAGE_MIN_WORDS:
            return True
    return primary_language_code == conversation_language


async def rewrite_summary_in_conversation_language(
    llm: object,
    language: LanguageSupport,
    conversation_language: str,
    generated_summary: str,
) -> str | None:
    language_name = language.supported_language_names.get(
        conversation_language,
        conversation_language.upper(),
    )
    rewrite_prompt = (
        "Rewrite the intake summary below in "
        f"{language_name}. Keep all facts and meaning exactly. "
        "Do not add or remove details. Output one concise paragraph only.\n\n"
        "Summary:\n"
        f"{generated_summary}"
    )

    try:
        response = await resolve_llm_client(llm).ainvoke(
            [{"role": "system", "content": rewrite_prompt}]
        )
    except Exception:
        logger.warning(
            "Failed to rewrite summary in preferred language.",
            exc_info=True,
        )
        return None

    rewritten_summary = extract_llm_content(response).strip()
    if not rewritten_summary:
        return None
    if not summary_matches_conversation_language(
        rewritten_summary,
        conversation_language,
        language.supported_language_set,
    ):
        return None
    return rewritten_summary
