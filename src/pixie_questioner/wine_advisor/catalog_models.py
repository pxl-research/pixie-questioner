"""Catalog row models for wine advisor retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WineCatalogRecord:
    barcode: str
    name: str
    price_eur: float
    country: str
    region: str
    color: str
    grape_description: str
    aroma_description: str
    taste_description: str
    foodpair_description: str
