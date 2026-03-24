# ruff: noqa: S101
from pixie_questioner.origin_normalization import (
    build_normalized_origin_lookup,
    normalize_origin_lookup_value,
)


def test_normalize_origin_lookup_value_is_accent_and_separator_tolerant() -> None:
    assert normalize_origin_lookup_value("  Zuid-Afrika  ") == "zuid afrika"
    assert normalize_origin_lookup_value("Zuid Afrika") == "zuid afrika"
    assert normalize_origin_lookup_value("Non spécifié") == "non specifie"


def test_build_normalized_origin_lookup_omits_empty_values() -> None:
    lookup = build_normalized_origin_lookup(["", "  ", "Spanje", "spain"])

    assert "spanje" in lookup
    assert "spain" in lookup
    assert "" not in lookup
