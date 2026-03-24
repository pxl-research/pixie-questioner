"""Shared low-level origin text normalization helpers."""

from __future__ import annotations

import re
import unicodedata

from collections.abc import Iterable


def normalize_origin_lookup_value(value: str) -> str:
    lowered = " ".join(value.strip().split()).casefold()
    decomposed = unicodedata.normalize("NFKD", lowered)
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    # Collapse hyphens, slashes, and similar punctuation to spaces so that
    # user variants like "Zuid Afrika" match the catalog "Zuid-Afrika".
    collapsed = re.sub(r"[-/''\"]+", " ", without_accents)
    return re.sub(r"\s+", " ", collapsed).strip()


def build_normalized_origin_lookup(values: Iterable[str]) -> frozenset[str]:
    return frozenset(
        normalized
        for raw in values
        if (normalized := normalize_origin_lookup_value(raw))
    )
