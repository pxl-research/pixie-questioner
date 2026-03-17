# ruff: noqa: S101,S105,S106
from __future__ import annotations

import asyncio
import sys

import pytest
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape
from zipfile import ZipFile

from fastapi.testclient import TestClient

from pixie_questioner import config as config_module
from pixie_questioner.chat.types import ChatResult
from pixie_questioner.cretskens_service import (
    CretskensService,
    CretskensServiceUnavailableError,
)
from pixie_questioner.rate_limiter import SlidingWindowRateLimiter
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
        "\n".join(
            [
                "You are a wine-advice intake controller.",
                "Preference loop before recommendation",
                "budget, country/region of origin, grape variety, and style",
                "friendly in-store wine advisor",
                "Use upbeat openers only occasionally",
                "Vary how questions start across turns",
                "Mirror the customer's language (NL/EN/FR) used in recent messages.",
                'If unsure between question vs recommend, choose action "question".',
            ]
        ),
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


def _write_cretskens_domain_profile(tmp_path: Path) -> Path:
    _write_prompt(tmp_path / "cretskens_question_asker.txt", "system {summary}")
    _write_prompt(
        tmp_path / "cretskens_questionnaire_notulist.txt",
        "notulist {summary} {current_question} {current_answer}",
    )
    _write_prompt(
        tmp_path / "cretskens_questionnaire_template.txt",
        (
            "0. Dish description (free text from customer): []\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): []\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
            "3. Cooking method (BBQ or grill, stew, or none): []\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
            "5. Kleur van de wijn (wit, rood, rosé): []\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ),
    )
    _write_prompt(
        tmp_path / "cretskens_intake_route_classifier.txt",
        "Return exactly one label: FOOD_PAIRING, GENERAL_WINE, or UNKNOWN.\n{current_answer}",
    )
    return _write_prompt(
        tmp_path / "cretskens_domain_profile.yml",
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
                "  question_asker_path: cretskens_question_asker.txt",
                "  questionnaire_notulist_path: cretskens_questionnaire_notulist.txt",
                "  questionnaire_template_path: cretskens_questionnaire_template.txt",
                "  intake_route_classifier_path: cretskens_intake_route_classifier.txt",
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
                "  completion_mode: categorized_handoff",
                "  off_topic_termination_threshold: 2",
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
    cretskens_domain_profile_path = _write_cretskens_domain_profile(tmp_path)

    catalog_path = tmp_path / "catalog.xlsx"
    _build_catalog_xlsx(catalog_path)

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_API_VERSION", "2024-01-01")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("BROOKLYN_DOMAIN_PROFILE_PATH", str(brooklyn_profile_path))
    monkeypatch.setenv(
        "CRETSKENS_DOMAIN_PROFILE_PATH",
        str(cretskens_domain_profile_path),
    )
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


class StubWineAdvisorService:
    def __init__(
        self,
        result: WineAdvisorResult,
        recommend_result: WineAdvisorResult | None = None,
    ) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.recommend_result = recommend_result or result
        self.recommend_calls: list[tuple[str, str]] = []

    async def ask(
        self,
        *,
        current_answer: str,
        state: dict[str, Any] | None,
    ) -> WineAdvisorResult:
        await asyncio.sleep(0)
        self.calls.append((current_answer, state))
        return self.result

    async def recommend_from_context(
        self,
        *,
        working_context: str,
        meal_mode: str,
    ) -> WineAdvisorResult:
        await asyncio.sleep(0)
        self.recommend_calls.append((working_context, meal_mode))
        return self.recommend_result


class StubChatService:
    def __init__(self, result: ChatResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    async def ask(
        self,
        current_answer: str,
        state: dict[str, Any] | None,
    ) -> ChatResult:
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


class StubCretskensService:
    def __init__(self, result: WineAdvisorResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def ask(
        self,
        *,
        current_answer: str,
        state: dict[str, Any],
    ) -> WineAdvisorResult:
        await asyncio.sleep(0)
        self.calls.append((current_answer, state))
        return self.result


class FailingCretskensService:
    async def ask(
        self,
        *,
        current_answer: str,
        state: dict[str, Any],
    ) -> WineAdvisorResult:
        await asyncio.sleep(0)
        raise CretskensServiceUnavailableError("LLM unavailable")


class PassthroughStateSigner:
    def verify_and_strip(self, state: dict[str, object]) -> dict[str, object]:
        return state

    def sign(self, state: dict[str, object]) -> dict[str, object]:
        return state


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
    audit_stub = StubAuditClient()
    cretskens_stub = StubCretskensService(result)
    main.runtime.cretskens_service = cretskens_stub
    main.runtime.audit_client = audit_stub
    request_state = {"location": "Antwerpen"}

    client = TestClient(main.app)
    response = client.post(
        "/cases/cretskens/chat",
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
    response_state = payload["state"]
    assert response_state["phase"] == "intake"
    assert response_state["turn_count"] == 1
    assert "_sig" in response_state
    assert "_ts" in response_state
    assert response.headers["X-Pixie-Chat-Contract"] == CHAT_CONTRACT_VERSION
    assert cretskens_stub.calls == [("Ik eet rund", request_state)]

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


def test_cretskens_chat_uses_intake_service_before_wine_advisor(
    monkeypatch, tmp_path: Path
) -> None:
    main = _load_main(monkeypatch, tmp_path)
    chat_stub = StubChatService(
        ChatResult(
            text="Wat zit er vooral in de pad thai?",
            response_type="answer",
            state={
                "phase": "intake",
                "last_question": "Wat zit er vooral in de pad thai?",
            },
            summary=None,
        )
    )
    advisor_stub = StubWineAdvisorService(
        WineAdvisorResult(
            text="Welke wijnkleur heeft je voorkeur?",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        )
    )
    main.runtime.cretskens_service = CretskensService(
        intake_service=chat_stub,
        recommendation_service=advisor_stub,
    )

    client = TestClient(main.app)
    response = client.post(
        "/cases/cretskens/chat",
        json={
            "session_id": "session-pad-thai",
            "current_answer": "We eten vanavond pad thai",
            "state": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "question"
    assert payload["text"] == "Wat zit er vooral in de pad thai?"
    assert chat_stub.calls == [("We eten vanavond pad thai", None)]
    assert advisor_stub.calls == []


def test_cretskens_chat_completed_intake_returns_recommendations(
    monkeypatch, tmp_path: Path
) -> None:
    main = _load_main(monkeypatch, tmp_path)
    chat_stub = StubChatService(
        ChatResult(
            text="Dank je wel, dit geeft een mooi beeld van je gerecht.",
            response_type="answer",
            state={
                "phase": "completed",
                "summary": "Klant eet pad thai en zoekt een passende wijn.",
                "questionnaire": (
                    "0. Dish description (free text from customer): pad thai\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
                    "3. Cooking method (BBQ or grill, stew, or none): none\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
                    "5. Kleur van de wijn (wit, rood, rosé): wit\n"
                    "6. Herkomst (land en/of regio): niet gespecificeerd\n"
                    "7. Maximum budget in euro: 20\n"
                    "8. Gewenste geur/smaak (vrije tekst): fris en fruitig"
                ),
                "handoff_categories_v1": "{}",
                "intake_route": "food_pairing",
            },
            summary=None,
        )
    )
    recommendation = WineCatalogRecommendation(
        barcode="w1",
        name="Testwijn",
        country="Frankrijk",
        region="Bordeaux",
        aroma_description="Rijp rood fruit",
        taste_description="Fluweelzacht en rond",
    )
    advisor_stub = StubWineAdvisorService(
        result=WineAdvisorResult(
            text="Welke wijnkleur heeft je voorkeur?",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        ),
        recommend_result=WineAdvisorResult(
            text="Dit zijn passende opties uit de catalogus.",
            response_type="recommendations",
            state={"phase": "completed", "selected_barcodes": ["w1"]},
            recommendations=[recommendation],
        ),
    )
    main.runtime.cretskens_service = CretskensService(
        intake_service=chat_stub,
        recommendation_service=advisor_stub,
    )

    client = TestClient(main.app)
    response = client.post(
        "/cases/cretskens/chat",
        json={
            "session_id": "session-pad-thai-recommend",
            "current_answer": "We eten vanavond pad thai",
            "state": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "recommendations"
    assert payload["text"] == "Dit zijn passende opties uit de catalogus."
    assert payload["recommendations"][0]["name"] == "Testwijn"
    assert chat_stub.calls == [("We eten vanavond pad thai", None)]
    assert advisor_stub.calls == []
    assert len(advisor_stub.recommend_calls) == 1
    assert advisor_stub.recommend_calls[0][1] == "pairing"
    assert "pad thai" in advisor_stub.recommend_calls[0][0]


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
                country="Frankrijk",
                region="Bordeaux",
                aroma_description="Rijp rood fruit",
                taste_description="Fluweelzacht en rond",
            )
        ],
    )
    cretskens_stub = StubCretskensService(result)
    main.runtime.cretskens_service = cretskens_stub
    main.runtime.audit_client = None

    client = TestClient(main.app)
    response = client.post(
        "/cases/cretskens/chat",
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
            "country": "Frankrijk",
            "region": "Bordeaux",
            "aroma_description": "Rijp rood fruit",
            "taste_description": "Fluweelzacht en rond",
        }
    ]


def test_chat_returns_recommendations_with_defaults_for_blank_taste_fields(
    monkeypatch, tmp_path: Path
) -> None:
    main = _load_main(monkeypatch, tmp_path)
    result = WineAdvisorResult(
        text="Dit zijn passende opties uit de catalogus.",
        response_type="recommendations",
        state={"phase": "completed", "turn_count": 2},
        recommendations=[
            WineCatalogRecommendation(
                barcode="w1",
                name="Testwijn",
                country="Frankrijk",
                region="Bordeaux",
                aroma_description="",
                taste_description="   ",
            )
        ],
    )
    cretskens_stub = StubCretskensService(result)
    main.runtime.cretskens_service = cretskens_stub
    main.runtime.audit_client = None

    client = TestClient(main.app)
    response = client.post(
        "/cases/cretskens/chat",
        json={
            "session_id": "session-2b",
            "current_answer": "Toon opties",
            "state": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "recommendations"
    assert payload["recommendations"] == [
        {
            "name": "Testwijn",
            "country": "Frankrijk",
            "region": "Bordeaux",
            "aroma_description": "Not specified",
            "taste_description": "Not specified",
        }
    ]


def test_cretskens_chat_returns_unavailable_question_when_service_fails(
    monkeypatch, tmp_path: Path
) -> None:
    main = _load_main(monkeypatch, tmp_path)
    main.runtime.cretskens_service = FailingCretskensService()
    main.runtime.audit_client = None

    client = TestClient(main.app)
    response = client.post(
        "/cases/cretskens/chat",
        json={
            "session_id": "session-unavailable",
            "current_answer": "We eten vanavond pad thai",
            "state": {"location": "Antwerpen"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "question"
    assert payload["text"] == "Tijdelijk niet beschikbaar."
    assert payload["conversation_id"] == "session-unavailable"
    assert payload["state"]["location"] == "Antwerpen"
    assert "_sig" in payload["state"]
    assert "_ts" in payload["state"]


def test_chat_audit_omits_blank_location_metadata(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    result = WineAdvisorResult(
        text="Welke smaak heb je in gedachten?",
        response_type="question",
        state={"phase": "intake", "turn_count": 1},
        recommendations=None,
    )
    audit_stub = StubAuditClient()
    main.runtime.cretskens_service = StubCretskensService(result)
    main.runtime.audit_client = audit_stub

    client = TestClient(main.app)
    response = client.post(
        "/cases/cretskens/chat",
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


def test_chat_rejects_overlong_input(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/cases/cretskens/chat",
        json={
            "session_id": "session-too-long",
            "current_answer": "x" * (main.settings.wine_advisor_max_input_chars + 1),
            "state": {},
        },
    )

    assert response.status_code == 422


def test_chat_respects_configured_input_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WINE_ADVISOR_MAX_INPUT_CHARS", "5")
    main = _load_main(monkeypatch, tmp_path)
    advisor_stub = StubWineAdvisorService(
        WineAdvisorResult(
            text="Vraag",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        )
    )
    main.runtime.cretskens_service = StubCretskensService(advisor_stub.result)
    client = TestClient(main.app)

    response = client.post(
        "/cases/cretskens/chat",
        json={
            "session_id": "session-tight-input-limit",
            "current_answer": "abcdef",
            "state": {},
        },
    )

    assert response.status_code == 422


def test_chat_rejects_oversized_state_payload(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/cases/cretskens/chat",
        json={
            "session_id": "session-large-state",
            "current_answer": "hallo",
            "state": {
                "blob": "x" * (main.settings.wine_advisor_max_state_bytes + 256),
            },
        },
    )

    assert response.status_code == 422


def test_chat_respects_configured_state_size_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WINE_ADVISOR_MAX_STATE_BYTES", "32")
    main = _load_main(monkeypatch, tmp_path)
    advisor_stub = StubWineAdvisorService(
        WineAdvisorResult(
            text="Vraag",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        )
    )
    signer = PassthroughStateSigner()
    main.runtime.cretskens_service = StubCretskensService(advisor_stub.result)
    main.runtime.cretskens_state_signer = signer
    main.runtime.brooklyn_state_signer = signer
    client = TestClient(main.app)

    response = client.post(
        "/cases/cretskens/chat",
        json={
            "session_id": "session-tight-state-limit",
            "current_answer": "hallo",
            "state": {"blob": "x" * 40},
        },
    )

    assert response.status_code == 422


def test_legacy_chat_route_returns_404(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    client = TestClient(main.app)

    response = client.post(
        "/chat",
        json={
            "session_id": "legacy-chat-route",
            "current_answer": "hallo",
            "state": {},
        },
    )

    assert response.status_code == 404


def test_chat_ignores_x_forwarded_for_for_ip_rate_limits(
    monkeypatch, tmp_path: Path
) -> None:
    main = _load_main(monkeypatch, tmp_path)
    advisor_stub = StubWineAdvisorService(
        WineAdvisorResult(
            text="Vraag",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        )
    )
    main.runtime.cretskens_service = StubCretskensService(advisor_stub.result)
    main.runtime.ip_rate_limiter = SlidingWindowRateLimiter(
        max_requests=1, window_seconds=60
    )
    client = TestClient(main.app)

    first_response = client.post(
        "/cases/cretskens/chat",
        headers={"X-Forwarded-For": "198.51.100.10"},
        json={
            "session_id": "session-forwarded-1",
            "current_answer": "hallo",
            "state": {},
        },
    )
    second_response = client.post(
        "/cases/cretskens/chat",
        headers={"X-Forwarded-For": "203.0.113.20"},
        json={
            "session_id": "session-forwarded-2",
            "current_answer": "hallo",
            "state": {},
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 429


def test_chat_uses_x_forwarded_for_for_trusted_proxy_ip_rate_limits(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "testclient")
    main = _load_main(monkeypatch, tmp_path)
    advisor_stub = StubWineAdvisorService(
        WineAdvisorResult(
            text="Vraag",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        )
    )
    main.runtime.cretskens_service = StubCretskensService(advisor_stub.result)
    main.runtime.ip_rate_limiter = SlidingWindowRateLimiter(
        max_requests=1, window_seconds=60
    )
    client = TestClient(main.app)

    first_response = client.post(
        "/cases/cretskens/chat",
        headers={"X-Forwarded-For": "198.51.100.10"},
        json={
            "session_id": "session-trusted-forwarded-1",
            "current_answer": "hallo",
            "state": {},
        },
    )
    second_response = client.post(
        "/cases/cretskens/chat",
        headers={"X-Forwarded-For": "203.0.113.20"},
        json={
            "session_id": "session-trusted-forwarded-2",
            "current_answer": "hallo",
            "state": {},
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200


def test_chat_uses_x_forwarded_for_rightmost_non_trusted_in_multi_proxy_chain(
    monkeypatch, tmp_path: Path
) -> None:
    """In a two-proxy chain both proxies are trusted; the real client IP is the
    first non-trusted entry walking right-to-left through XFF."""
    monkeypatch.setenv("TRUSTED_PROXY_IPS", "testclient,10.0.0.1")
    main = _load_main(monkeypatch, tmp_path)
    advisor_stub = StubWineAdvisorService(
        WineAdvisorResult(
            text="Vraag",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        )
    )
    main.runtime.cretskens_service = StubCretskensService(advisor_stub.result)
    main.runtime.ip_rate_limiter = SlidingWindowRateLimiter(
        max_requests=1, window_seconds=60
    )
    client = TestClient(main.app)

    # XFF chain: spoofed_prefix, real_client, inner_proxy(10.0.0.1)
    # Walking right→left: 10.0.0.1 is trusted (skip), 198.51.100.10 is not → use it.
    first_response = client.post(
        "/cases/cretskens/chat",
        headers={"X-Forwarded-For": "1.2.3.4, 198.51.100.10, 10.0.0.1"},
        json={
            "session_id": "session-multiproxy-1",
            "current_answer": "hallo",
            "state": {},
        },
    )
    # Same real client IP → rate limited.
    second_response = client.post(
        "/cases/cretskens/chat",
        headers={"X-Forwarded-For": "9.9.9.9, 198.51.100.10, 10.0.0.1"},
        json={
            "session_id": "session-multiproxy-2",
            "current_answer": "hallo",
            "state": {},
        },
    )
    # Different real client IP → not rate limited.
    third_response = client.post(
        "/cases/cretskens/chat",
        headers={"X-Forwarded-For": "9.9.9.9, 203.0.113.20, 10.0.0.1"},
        json={
            "session_id": "session-multiproxy-3",
            "current_answer": "hallo",
            "state": {},
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert third_response.status_code == 200


def test_turn_controller_prompt_includes_preference_loop_rules(
    monkeypatch, tmp_path: Path
) -> None:
    main = _load_main(monkeypatch, tmp_path)
    prompt = main.settings.cretskens_profile.turn_controller_prompt

    assert "Preference loop before recommendation" in prompt
    assert "budget, country/region of origin, grape variety, and style" in prompt
    assert "friendly in-store wine advisor" in prompt
    assert "Use upbeat openers only occasionally" in prompt
    assert "Vary how questions start across turns" in prompt
    assert (
        "Mirror the customer's language (NL/EN/FR) used in recent messages." in prompt
    )
    assert (
        'If unsure between question vs recommend, choose action "question".' in prompt
    )


def test_build_runtime_raises_on_short_signing_key(monkeypatch, tmp_path: Path) -> None:
    """A configured STATE_SIGNING_KEY shorter than 32 chars must raise RuntimeError
    at startup so misconfigured multi-worker deployments fail fast instead of
    generating per-process ephemeral keys that produce intermittent 400s."""
    import pixie_questioner.config as cfg_mod
    from pixie_questioner.app_runtime import build_runtime

    monkeypatch.setenv("STATE_SIGNING_KEY", "too-short")
    cfg_mod.get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="STATE_SIGNING_KEY is too short"):
            build_runtime()
    finally:
        cfg_mod.get_settings.cache_clear()
