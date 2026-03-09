# ruff: noqa: S101,S105,S106
from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

from fastapi.testclient import TestClient

from pixie_questioner import config as config_module
from pixie_questioner.chat.types import ChatResult
from pixie_questioner.schemas import (
    BROOKLYN_CHAT_CONTRACT_VERSION,
    CRETSKENS_CHAT_CONTRACT_VERSION,
)
from pixie_questioner.wine_advisor.types import WineAdvisorResult


def _column_label(index: int) -> str:
    label = ""
    value = index + 1
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _cell_xml(cell_ref: str, value: str | int | float) -> str:
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

    def _row_xml(row_index: int, values: Sequence[str | int | float]) -> str:
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


def _write_cretskens_profile(tmp_path: Path) -> Path:
    _write_prompt(
        tmp_path / "turn_controller_prompt.txt",
        "You are a wine-advice intake controller.",
    )
    _write_prompt(
        tmp_path / "ranker_prompt.txt",
        "Select the best candidate barcodes for this customer.",
    )
    return _write_prompt(
        tmp_path / "advisor_profile.yml",
        "\n".join(
            [
                "version: 1",
                "prompts:",
                "  turn_controller_path: turn_controller_prompt.txt",
                "  ranker_path: ranker_prompt.txt",
                'recommendation_intro: "Leuk, ik heb een paar mooie opties voor je klaargezet."',
            ]
        ),
    )


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
    brooklyn_profile_path = _write_prompt(
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
    cretskens_profile_path = _write_cretskens_profile(tmp_path)

    catalog_path = tmp_path / "catalog.xlsx"
    _build_catalog_xlsx(catalog_path)

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_API_VERSION", "2024-01-01")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("BROOKLYN_DOMAIN_PROFILE_PATH", str(brooklyn_profile_path))
    monkeypatch.setenv("CRETSKENS_PROFILE_PATH", str(cretskens_profile_path))
    monkeypatch.setenv("CRETSKENS_WINE_CATALOG_PATH", str(catalog_path))


def _load_main(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch, tmp_path)
    config_module.get_settings.cache_clear()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    sys.modules.pop("pixie_questioner.app_runtime", None)
    sys.modules.pop("pixie_questioner.api_app", None)
    sys.modules.pop("apps.api.main", None)
    import apps.api.main as main

    return main


class StubBrooklynService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def ask(self, *, current_answer: str, state: dict[str, object]) -> ChatResult:
        await asyncio.sleep(0)
        self.calls.append((current_answer, state))
        return ChatResult(
            text="Brooklyn vraag",
            response_type="answer",
            state={"phase": "intake", "last_question": "Brooklyn vraag"},
            summary=None,
        )


class StubCretskensService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def ask(
        self,
        *,
        current_answer: str,
        state: dict[str, object],
    ) -> WineAdvisorResult:
        await asyncio.sleep(0)
        self.calls.append((current_answer, state))
        return WineAdvisorResult(
            text="Cretskens vraag",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        )


def test_case_route_brooklyn_calls_brooklyn_handler(
    monkeypatch, tmp_path: Path
) -> None:
    main = _load_main(monkeypatch, tmp_path)
    brooklyn_stub = StubBrooklynService()
    cretskens_stub = StubCretskensService()
    main.runtime.brooklyn_chat_service = brooklyn_stub
    main.runtime.cretskens_wine_advisor_service = cretskens_stub

    client = TestClient(main.app)
    response = client.post(
        "/cases/brooklyn/chat",
        json={
            "session_id": "session-1",
            "current_answer": "ik zoek een broek",
            "state": {},
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Pixie-Case"] == "brooklyn"
    assert response.headers["X-Pixie-Case-Contract"] == BROOKLYN_CHAT_CONTRACT_VERSION
    assert len(brooklyn_stub.calls) == 1
    assert len(cretskens_stub.calls) == 0


def test_case_route_cretskens_calls_wine_handler(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    brooklyn_stub = StubBrooklynService()
    cretskens_stub = StubCretskensService()
    main.runtime.brooklyn_chat_service = brooklyn_stub
    main.runtime.cretskens_wine_advisor_service = cretskens_stub

    client = TestClient(main.app)
    response = client.post(
        "/cases/cretskens/chat",
        json={
            "session_id": "session-2",
            "current_answer": "ik zoek wijn",
            "state": {},
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Pixie-Case"] == "cretskens"
    assert response.headers["X-Pixie-Case-Contract"] == CRETSKENS_CHAT_CONTRACT_VERSION
    assert len(brooklyn_stub.calls) == 0
    assert len(cretskens_stub.calls) == 1


def test_case_route_case2_alias_returns_404(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    brooklyn_stub = StubBrooklynService()
    cretskens_stub = StubCretskensService()
    main.runtime.brooklyn_chat_service = brooklyn_stub
    main.runtime.cretskens_wine_advisor_service = cretskens_stub

    client = TestClient(main.app)
    response = client.post(
        "/cases/case2/chat",
        json={
            "session_id": "session-2b",
            "current_answer": "ik zoek wijn",
            "state": {},
        },
    )

    assert response.status_code == 404
    assert len(brooklyn_stub.calls) == 0
    assert len(cretskens_stub.calls) == 0


def test_case_route_unknown_case_returns_404(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/cases/unknown/chat",
        json={
            "session_id": "session-3",
            "current_answer": "test",
            "state": {},
        },
    )

    assert response.status_code == 404


def test_legacy_chat_route_returns_404(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    brooklyn_stub = StubBrooklynService()
    cretskens_stub = StubCretskensService()
    main.runtime.brooklyn_chat_service = brooklyn_stub
    main.runtime.cretskens_wine_advisor_service = cretskens_stub

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={
            "session_id": "session-4",
            "current_answer": "ik zoek een broek",
            "state": {},
        },
    )

    assert response.status_code == 404
    assert len(brooklyn_stub.calls) == 0
    assert len(cretskens_stub.calls) == 0


def test_case_route_rejects_state_from_other_case(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    brooklyn_stub = StubBrooklynService()
    cretskens_stub = StubCretskensService()
    main.runtime.brooklyn_chat_service = brooklyn_stub
    main.runtime.cretskens_wine_advisor_service = cretskens_stub

    client = TestClient(main.app)

    cretskens_response = client.post(
        "/cases/cretskens/chat",
        json={
            "session_id": "session-5",
            "current_answer": "ik zoek wijn",
            "state": {},
        },
    )
    assert cretskens_response.status_code == 200
    cretskens_state = cretskens_response.json()["state"]

    brooklyn_response = client.post(
        "/cases/brooklyn/chat",
        json={
            "session_id": "session-6",
            "current_answer": "ik zoek een broek",
            "state": cretskens_state,
        },
    )

    assert brooklyn_response.status_code == 400
    assert "scope invalid" in brooklyn_response.json()["detail"].casefold()
    assert len(brooklyn_stub.calls) == 0
