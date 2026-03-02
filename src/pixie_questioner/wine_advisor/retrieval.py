"""Deterministic retrieval and ranking over the in-memory wine catalog."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .catalog_models import WineCatalogRecord

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class RetrievalConstraints:
    color: str | None = None
    country: str | None = None
    region: str | None = None
    max_price_eur: float | None = None


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_PATTERN.findall(text.casefold()) if token}


def _record_corpus(record: WineCatalogRecord) -> str:
    return " ".join(
        (
            record.name,
            record.grape_description,
            record.aroma_description,
            record.taste_description,
            record.foodpair_description,
            record.country,
            record.region,
            record.color,
        )
    )


def _score_constraints(
    record: WineCatalogRecord, constraints: RetrievalConstraints
) -> float:
    score = 0.0
    if constraints.color:
        score += (
            3.0 if record.color.casefold() == constraints.color.casefold() else -1.0
        )
    if constraints.country:
        score += (
            3.0 if record.country.casefold() == constraints.country.casefold() else -1.0
        )
    if constraints.region:
        score += (
            4.0 if record.region.casefold() == constraints.region.casefold() else -1.0
        )
    if constraints.max_price_eur is not None:
        score += 2.0 if record.price_eur <= constraints.max_price_eur else -3.0
    return score


def _record_score(
    record: WineCatalogRecord,
    query_tokens: set[str],
    constraints: RetrievalConstraints,
) -> float:
    corpus_tokens = _tokenize(_record_corpus(record))
    overlap = len(query_tokens.intersection(corpus_tokens))
    return (overlap * 4.0) + _score_constraints(record, constraints)


def _select_best_record(
    existing: tuple[float, WineCatalogRecord] | None,
    score: float,
    candidate: WineCatalogRecord,
) -> tuple[float, WineCatalogRecord]:
    if existing is None:
        return score, candidate
    current_score, current = existing
    if score > current_score:
        return score, candidate
    if score < current_score:
        return existing
    if candidate.name < current.name:
        return score, candidate
    return existing


def rank_catalog_records(
    records: list[WineCatalogRecord],
    query: str,
    constraints: RetrievalConstraints | None = None,
    limit: int = 20,
) -> list[WineCatalogRecord]:
    effective_constraints = constraints or RetrievalConstraints()
    query_tokens = _tokenize(query)
    best_per_barcode: dict[str, tuple[float, WineCatalogRecord]] = {}

    for record in records:
        score = _record_score(record, query_tokens, effective_constraints)
        best_per_barcode[record.barcode] = _select_best_record(
            best_per_barcode.get(record.barcode),
            score,
            record,
        )

    ranked = sorted(
        best_per_barcode.values(),
        key=lambda item: (-item[0], item[1].barcode),
    )
    return [record for _, record in ranked[: max(limit, 0)]]
