# ruff: noqa: S101,S105,S106
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

import pytest

from pixie_questioner.wine_advisor.catalog_loader import (
    REQUIRED_COLUMNS,
    load_wine_catalog,
)


def _column_label(index: int) -> str:
    label = ""
    value = index + 1
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _cell_xml(cell_ref: str, value: str | float) -> str:
    if isinstance(value, (int, float)):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    escaped = escape(str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{escaped}</t></is></c>'


def _build_xlsx(path: Path, headers: list[str], rows: list[list[str | float]]) -> None:
    sheet_rows: list[str] = []

    def _row_xml(row_index: int, values: Sequence[str | float]) -> str:
        cells = [
            _cell_xml(f"{_column_label(column_index)}{row_index}", value)
            for column_index, value in enumerate(values)
        ]
        return f'<row r="{row_index}">{"".join(cells)}</row>'

    sheet_rows.append(_row_xml(1, headers))
    for idx, row in enumerate(rows, start=2):
        sheet_rows.append(_row_xml(idx, row))

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        "</worksheet>"
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        "<workbook "
        'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="wijnen" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )

    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        "<Relationship "
        'Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )

    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        "<Relationship "
        'Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", root_rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def test_load_wine_catalog_reads_expected_fields(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.xlsx"
    _build_xlsx(
        catalog_path,
        headers=list(REQUIRED_COLUMNS),
        rows=[
            [
                "123456",
                "Testwijn",
                11.5,
                "Frankrijk",
                "Bordeaux",
                "rood",
                "Merlot",
                "Rood fruit",
                "Droog",
                "Rund",
            ]
        ],
    )

    records = load_wine_catalog(catalog_path)

    assert len(records) == 1
    first = records[0]
    assert first.barcode == "123456"
    assert first.name == "Testwijn"
    assert first.price_eur == 11.5
    assert first.country == "Frankrijk"
    assert first.region == "Bordeaux"
    assert first.color == "rood"


def test_load_wine_catalog_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.xlsx"

    with pytest.raises(RuntimeError, match="Wine catalog file not found"):
        load_wine_catalog(missing_path)


def test_load_wine_catalog_missing_required_columns_raises(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.xlsx"
    headers = [column for column in REQUIRED_COLUMNS if column != "regio"]
    _build_xlsx(
        catalog_path,
        headers=headers,
        rows=[
            ["123", "Wijn", 9.0, "Frankrijk", "rood", "Merlot", "fruit", "droog", "vis"]
        ],
    )

    with pytest.raises(RuntimeError, match="Missing required columns"):
        load_wine_catalog(catalog_path)


def test_load_wine_catalog_invalid_price_raises(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.xlsx"
    _build_xlsx(
        catalog_path,
        headers=list(REQUIRED_COLUMNS),
        rows=[
            [
                "123456",
                "Testwijn",
                "prijs onbekend",
                "Frankrijk",
                "Bordeaux",
                "rood",
                "Merlot",
                "Rood fruit",
                "Droog",
                "Rund",
            ]
        ],
    )

    with pytest.raises(RuntimeError, match="Invalid price"):
        load_wine_catalog(catalog_path)
