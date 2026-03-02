"""Load wine catalog rows from an XLSX file."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # nosec B405
import zipfile
from pathlib import Path

from .catalog_models import WineCatalogRecord

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

REQUIRED_COLUMNS = (
    "barcode",
    "naam",
    "prijs",
    "land",
    "regio",
    "kleur",
    "druiven_beschrijving",
    "geur_beschrijving",
    "smaak_beschrijving",
    "foodpair_beschrijving",
)

_COLUMN_REF_PATTERN = re.compile(r"([A-Z]+)")


def _column_index_from_ref(cell_ref: str) -> int:
    match = _COLUMN_REF_PATTERN.match(cell_ref)
    if match is None:
        return -1
    label = match.group(1)
    index = 0
    for char in label:
        index = (index * 26) + (ord(char) - 64)
    return index - 1


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(
        archive.read("xl/sharedStrings.xml")
    )  # nosec B314  # noqa: S314
    values: list[str] = []
    for shared_item in root.findall(f"{{{NS_MAIN}}}si"):
        tokens = [
            token.text or "" for token in shared_item.findall(f".//{{{NS_MAIN}}}t")
        ]
        values.append("".join(tokens))
    return values


def _resolve_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(
        archive.read("xl/workbook.xml")
    )  # nosec B314  # noqa: S314
    sheets = workbook.findall(f".//{{{NS_MAIN}}}sheet")
    if not sheets:
        raise RuntimeError("Wine catalog workbook has no sheets.")

    selected = sheets[0]
    for sheet in sheets:
        if (sheet.attrib.get("name") or "").strip().casefold() == "wijnen":
            selected = sheet
            break

    relation_id = selected.attrib.get(f"{{{NS_REL}}}id", "")
    relations = ET.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )  # nosec B314  # noqa: S314
    for relation in relations.findall(f"{{{NS_PACKAGE_REL}}}Relationship"):
        if relation.attrib.get("Id") != relation_id:
            continue
        target = relation.attrib.get("Target", "").lstrip("/")
        return f"xl/{target}"
    raise RuntimeError("Wine catalog worksheet relationship is missing.")


def _read_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(
            token.text or "" for token in cell.findall(f".//{{{NS_MAIN}}}t")
        ).strip()

    value_element = cell.find(f"{{{NS_MAIN}}}v")
    if value_element is None or value_element.text is None:
        return ""
    raw_value = value_element.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)].strip()
        except (IndexError, ValueError):
            return raw_value.strip()
    return raw_value.strip()


def _row_values(row: ET.Element, shared_strings: list[str]) -> list[str]:
    values: list[str] = []
    for cell in row.findall(f"{{{NS_MAIN}}}c"):
        cell_ref = cell.attrib.get("r", "")
        column_index = _column_index_from_ref(cell_ref)
        if column_index < 0:
            continue
        while len(values) <= column_index:
            values.append("")
        values[column_index] = _read_cell_value(cell, shared_strings)
    return values


def _build_header_index(headers: list[str]) -> dict[str, int]:
    indexed_headers = {
        header.strip().casefold(): idx for idx, header in enumerate(headers)
    }
    missing = [column for column in REQUIRED_COLUMNS if column not in indexed_headers]
    if missing:
        raise RuntimeError(f"Missing required columns: {', '.join(missing)}")
    return indexed_headers


def _price_from_cell(raw_price: str, row_number: int) -> float:
    normalized = raw_price.strip().replace(",", ".")
    try:
        return float(normalized)
    except ValueError as exc:
        raise RuntimeError(f"Invalid price at row {row_number}: {raw_price!r}") from exc


def _value_at(values: list[str], header_index: dict[str, int], key: str) -> str:
    index = header_index[key]
    if index >= len(values):
        return ""
    return values[index].strip()


def load_wine_catalog(path: Path) -> list[WineCatalogRecord]:
    resolved_path = path.expanduser()
    if not resolved_path.exists():
        raise RuntimeError(f"Wine catalog file not found: {path}")

    try:
        with zipfile.ZipFile(resolved_path) as archive:
            shared_strings = _load_shared_strings(archive)
            sheet_path = _resolve_sheet_path(archive)
            worksheet = ET.fromstring(
                archive.read(sheet_path)
            )  # nosec B314  # noqa: S314
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Invalid wine catalog file: {path}") from exc

    rows = worksheet.findall(f".//{{{NS_MAIN}}}row")
    if not rows:
        raise RuntimeError("Wine catalog worksheet has no rows.")

    header_index = _build_header_index(_row_values(rows[0], shared_strings))
    records: list[WineCatalogRecord] = []

    for row_number, row in enumerate(rows[1:], start=2):
        values = _row_values(row, shared_strings)
        barcode = _value_at(values, header_index, "barcode")
        if not barcode:
            continue
        records.append(
            WineCatalogRecord(
                barcode=barcode,
                name=_value_at(values, header_index, "naam"),
                price_eur=_price_from_cell(
                    _value_at(values, header_index, "prijs"), row_number
                ),
                country=_value_at(values, header_index, "land"),
                region=_value_at(values, header_index, "regio"),
                color=_value_at(values, header_index, "kleur"),
                grape_description=_value_at(
                    values, header_index, "druiven_beschrijving"
                ),
                aroma_description=_value_at(values, header_index, "geur_beschrijving"),
                taste_description=_value_at(values, header_index, "smaak_beschrijving"),
                foodpair_description=_value_at(
                    values, header_index, "foodpair_beschrijving"
                ),
            )
        )

    if not records:
        raise RuntimeError("Wine catalog contains no usable records.")
    return records
