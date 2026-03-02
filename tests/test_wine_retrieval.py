from __future__ import annotations

from pixie_questioner.wine_advisor.catalog_models import WineCatalogRecord
from pixie_questioner.wine_advisor.retrieval import (
    RetrievalConstraints,
    rank_catalog_records,
)


def _record(
    *,
    barcode: str,
    name: str,
    price_eur: float,
    country: str,
    region: str,
    color: str,
    grape_description: str,
    aroma_description: str,
    taste_description: str,
    foodpair_description: str,
) -> WineCatalogRecord:
    return WineCatalogRecord(
        barcode=barcode,
        name=name,
        price_eur=price_eur,
        country=country,
        region=region,
        color=color,
        grape_description=grape_description,
        aroma_description=aroma_description,
        taste_description=taste_description,
        foodpair_description=foodpair_description,
    )


def test_rank_catalog_records_prefers_foodpair_keyword_match() -> None:
    records = [
        _record(
            barcode="2",
            name="Lichte witte",
            price_eur=9.0,
            country="Frankrijk",
            region="Loire",
            color="wit",
            grape_description="Sauvignon",
            aroma_description="Citrus",
            taste_description="Fris",
            foodpair_description="Vis en schaaldieren",
        ),
        _record(
            barcode="1",
            name="Stevige rode",
            price_eur=12.0,
            country="Frankrijk",
            region="Bordeaux",
            color="rood",
            grape_description="Merlot",
            aroma_description="Donker fruit",
            taste_description="Kruidig",
            foodpair_description="Rund en gegrild vlees",
        ),
    ]

    ranked = rank_catalog_records(records, query="kruidig rund")

    assert ranked[0].barcode == "1"


def test_rank_catalog_records_boosts_constraints() -> None:
    records = [
        _record(
            barcode="10",
            name="Match wijn",
            price_eur=14.0,
            country="Frankrijk",
            region="Bordeaux",
            color="rood",
            grape_description="Merlot",
            aroma_description="Rood fruit",
            taste_description="Kruidig",
            foodpair_description="Rund",
        ),
        _record(
            barcode="20",
            name="Niet match wijn",
            price_eur=22.0,
            country="Italië",
            region="Toscane",
            color="wit",
            grape_description="Sangiovese",
            aroma_description="Bloemen",
            taste_description="Fris",
            foodpair_description="Pasta",
        ),
    ]

    ranked = rank_catalog_records(
        records,
        query="wijn bij rund",
        constraints=RetrievalConstraints(
            color="rood",
            country="Frankrijk",
            region="Bordeaux",
            max_price_eur=15.0,
        ),
    )

    assert ranked[0].barcode == "10"


def test_rank_catalog_records_returns_unique_barcodes() -> None:
    records = [
        _record(
            barcode="99",
            name="Wijn A",
            price_eur=12.0,
            country="Frankrijk",
            region="Bordeaux",
            color="rood",
            grape_description="Merlot",
            aroma_description="Rood fruit",
            taste_description="Droog",
            foodpair_description="Rund",
        ),
        _record(
            barcode="99",
            name="Wijn A duplicate",
            price_eur=12.0,
            country="Frankrijk",
            region="Bordeaux",
            color="rood",
            grape_description="Merlot",
            aroma_description="Rood fruit",
            taste_description="Droog",
            foodpair_description="Rund",
        ),
    ]

    ranked = rank_catalog_records(records, query="rund")

    assert [record.barcode for record in ranked] == ["99"]


def test_rank_catalog_records_deterministic_order_for_ties() -> None:
    records = [
        _record(
            barcode="b",
            name="Wine B",
            price_eur=10.0,
            country="Frankrijk",
            region="Bordeaux",
            color="rood",
            grape_description="Merlot",
            aroma_description="fruit",
            taste_description="droog",
            foodpair_description="rund",
        ),
        _record(
            barcode="a",
            name="Wine A",
            price_eur=10.0,
            country="Frankrijk",
            region="Bordeaux",
            color="rood",
            grape_description="Merlot",
            aroma_description="fruit",
            taste_description="droog",
            foodpair_description="rund",
        ),
    ]

    ranked = rank_catalog_records(records, query="niet-voorkomend-zoekwoord")

    assert [record.barcode for record in ranked] == ["a", "b"]
