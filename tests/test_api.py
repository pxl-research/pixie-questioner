# ruff: noqa: S101,S105,S106
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

from fastapi.testclient import TestClient

from pixie_questioner import config as config_module
from pixie_questioner.schemas import CHAT_CONTRACT_VERSION
from pixie_questioner.wine_advisor.types import (
    WineAdvisorResult,
    WineCatalogRecommendation,
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


def _build_catalog_xlsx(path: Path) -> None:
    headers = [
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
    ]
    rows = [
        [
            "w1",
            "Testwijn",
            10.0,
            "Frankrijk",
            "Bordeaux",
            "rood",
            "Merlot",
            "Rood fruit",
            "Droog",
            "Rund",
        ]
    ]

    def _row_xml(row_index: int, values: list[str | float]) -> str:
        cells = [
            _cell_xml(f"{_column_label(column_index)}{row_index}", value)
            for column_index, value in enumerate(values)
        ]
        return f'<row r="{row_index}">{"".join(cells)}</row>'

    sheet_rows = [_row_xml(1, headers)]
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


def _write_prompt(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _set_env(monkeypatch, tmp_path: Path) -> None:
    _write_prompt(tmp_path / "question_asker.txt", "system {summary}")
    _write_prompt(
        tmp_path / "questionnaire_notulist.txt",
        "notulist {summary} {current_question} {current_answer}",
    )
    _write_prompt(
        tmp_path / "questionnaire_template.txt",
        "0. Initial user question: []\n1. Occasion, Vibe, Setting: []",
    )
    _write_prompt(
        tmp_path / "relevance_classifier.txt",
        "Return ON_TOPIC or OFF_TOPIC for {current_question} and {current_answer}",
    )
    profile_path = _write_prompt(
        tmp_path / "domain_profile.yml",
        "\n".join(
            [
                "version: 1",
                "languages:",
                "  base: [nl, en, fr]",
                "  extra: []",
                "  names:",
                "    nl: Dutch",
                "    en: English",
                "    fr: French",
                "prompts:",
                "  question_asker_path: question_asker.txt",
                "  questionnaire_notulist_path: questionnaire_notulist.txt",
                "  questionnaire_template_path: questionnaire_template.txt",
                "  relevance_classifier_path: relevance_classifier.txt",
                "messages:",
                "  handoff:",
                '    nl: ["handoff nl"]',
                '    en: ["handoff en"]',
                '    fr: ["handoff fr"]',
                "  feedback_question:",
                '    nl: ["feedback question nl"]',
                '    en: ["feedback question en"]',
                '    fr: ["feedback question fr"]',
                "  feedback_thanks:",
                '    nl: ["feedback thanks nl"]',
                '    en: ["feedback thanks en"]',
                '    fr: ["feedback thanks fr"]',
                "  goodbye:",
                '    nl: ["goodbye nl"]',
                '    en: ["goodbye en"]',
                '    fr: ["goodbye fr"]',
                "  off_topic_redirect:",
                '    nl: ["redirect nl: {question}"]',
                '    en: ["redirect en: {question}"]',
                '    fr: ["redirect fr: {question}"]',
                "  off_topic_initial_question:",
                '    nl: ["initial nl"]',
                '    en: ["initial en"]',
                '    fr: ["initial fr"]',
                "quick_replies:",
                "  nl:",
                '    - { label: "Ja", value: "yes" }',
                '    - { label: "Nee", value: "no" }',
                "  en:",
                '    - { label: "Yes", value: "yes" }',
                '    - { label: "No", value: "no" }',
                "  fr:",
                '    - { label: "Oui", value: "yes" }',
                '    - { label: "Non", value: "no" }',
                "tokens:",
                '  feedback_yes: ["ja", "yes", "oui"]',
                '  feedback_no: ["nee", "no", "non"]',
                "behavior:",
                '  completion_token: "PIXIE_SUMMARY_READY"',
                "  off_topic_termination_threshold: 2",
            ]
        ),
    )

    catalog_path = tmp_path / "catalog.xlsx"
    _build_catalog_xlsx(catalog_path)

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_API_VERSION", "2024-01-01")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("DOMAIN_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("WINE_CATALOG_PATH", str(catalog_path))


def _load_main(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch, tmp_path)
    config_module.get_settings.cache_clear()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    sys.modules.pop("apps.api.main", None)
    import apps.api.main as main

    return main


class StubWineAdvisorService:
    def __init__(self, result: WineAdvisorResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def ask(
        self,
        *,
        current_answer: str,
        state: dict[str, object],
    ) -> WineAdvisorResult:
        await asyncio.sleep(0)
        self.calls.append((current_answer, state))
        return self.result


class StubAuditClient:
    def __init__(self) -> None:
        self.profile = "test-profile"
        self.calls: list[dict[str, object]] = []

    async def post_event(self, payload: dict[str, object]) -> None:
        await asyncio.sleep(0)
        self.calls.append(payload)


def test_health_endpoint(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    client = TestClient(main.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_question_and_audits(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    result = WineAdvisorResult(
        text="Welke wijnkleur heeft je voorkeur?",
        response_type="question",
        state={"phase": "intake", "turn_count": 1},
        recommendations=None,
    )
    advisor_stub = StubWineAdvisorService(result)
    audit_stub = StubAuditClient()
    main.wine_advisor_service = advisor_stub
    main.audit_client = audit_stub
    request_state = {"location": "Antwerpen"}

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={
            "session_id": "session-1",
            "current_answer": "Ik eet rund",
            "state": request_state,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "question"
    assert payload["text"] == "Welke wijnkleur heeft je voorkeur?"
    assert payload["conversation_id"] == "session-1"
    assert payload["state"] == {"phase": "intake", "turn_count": 1}
    assert response.headers["X-Pixie-Chat-Contract"] == CHAT_CONTRACT_VERSION
    assert advisor_stub.calls == [("Ik eet rund", request_state)]

    assert audit_stub.calls
    audit_payload = audit_stub.calls[0]
    assert set(audit_payload) == {"conversation_id", "profile", "messages", "metadata"}
    assert audit_payload["messages"] == [
        {"role": "user", "content": "Ik eet rund"},
        {
            "role": "assistant",
            "content": "Welke wijnkleur heeft je voorkeur?",
        },
    ]


def test_chat_returns_recommendations(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    result = WineAdvisorResult(
        text="Dit zijn passende opties uit de catalogus.",
        response_type="recommendations",
        state={"phase": "completed", "turn_count": 2},
        recommendations=[
            WineCatalogRecommendation(
                barcode="w1",
                name="Testwijn",
                price_eur=10.0,
                country="Frankrijk",
                region="Bordeaux",
            )
        ],
    )
    advisor_stub = StubWineAdvisorService(result)
    main.wine_advisor_service = advisor_stub
    main.audit_client = None

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={
            "session_id": "session-2",
            "current_answer": "Toon opties",
            "state": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "recommendations"
    assert payload["text"] == "Dit zijn passende opties uit de catalogus."
    assert payload["conversation_id"] == "session-2"
    assert payload["recommendations"] == [
        {
            "name": "Testwijn",
            "price_eur": 10.0,
            "country": "Frankrijk",
            "region": "Bordeaux",
        }
    ]


def test_chat_audit_omits_blank_location_metadata(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    result = WineAdvisorResult(
        text="Welke smaak heb je in gedachten?",
        response_type="question",
        state={"phase": "intake", "turn_count": 1},
        recommendations=None,
    )
    advisor_stub = StubWineAdvisorService(result)
    audit_stub = StubAuditClient()
    main.wine_advisor_service = advisor_stub
    main.audit_client = audit_stub

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={
            "session_id": "session-3",
            "current_answer": "geen idee",
            "state": {"location": "   "},
        },
    )

    assert response.status_code == 200
    assert audit_stub.calls
    audit_payload = audit_stub.calls[0]
    assert set(audit_payload) == {"conversation_id", "profile", "messages"}
    assert "metadata" not in audit_payload
