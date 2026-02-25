from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from pixie_questioner import config as config_module
from pixie_questioner.chat.types import ChatResult
from pixie_questioner.schemas import CHAT_CONTRACT_VERSION


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
                "term_replacements:",
                "  nl:",
                "    snit: pasvorm",
                "    silhouet: vorm",
                "    indigo: donkerblauw",
            ]
        ),
    )

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_API_VERSION", "2024-01-01")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("DOMAIN_PROFILE_PATH", str(profile_path))


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


class StubChatService:
    def __init__(self, result: ChatResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def ask(self, current_answer: str, state: dict[str, str]) -> ChatResult:
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


def test_chat_returns_summary_and_audits(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    state = {
        "questionnaire": "0. Initial user question: []",
        "summary": "Existing summary",
        "last_question": "Last question",
    }
    result = ChatResult(
        text="handoff",
        response_type="summary",
        state=state,
        summary="Final summary",
        follow_up_question="Would you use this assistant again? (yes/no)",
    )
    chat_stub = StubChatService(result)
    audit_stub = StubAuditClient()
    main.chat_service = chat_stub
    main.audit_client = audit_stub
    request_state = {"location": "Antwerpen"}

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={
            "session_id": "session-1",
            "current_answer": "stop",
            "state": request_state,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "summary"
    assert payload["summary"] == "Final summary"
    assert (
        payload["follow_up_question"] == "Would you use this assistant again? (yes/no)"
    )
    assert payload["conversation_id"] == "session-1"
    assert response.headers["X-Pixie-Chat-Contract"] == CHAT_CONTRACT_VERSION
    assert chat_stub.calls == [("stop", request_state)]

    assert audit_stub.calls
    audit_payload = audit_stub.calls[0]
    assert set(audit_payload) == {"conversation_id", "profile", "messages", "metadata"}
    assert audit_payload["conversation_id"] == "session-1"
    assert audit_payload["profile"] == "test-profile"
    assert audit_payload["messages"] == [
        {"role": "user", "content": "stop"},
        {"role": "assistant", "content": "handoff"},
        {"role": "summary", "content": "Final summary"},
    ]
    assert audit_payload["metadata"] == {"location": "Antwerpen"}


def test_chat_audit_omits_blank_location_metadata(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    result = ChatResult(
        text="Next question",
        response_type="answer",
        state={},
        summary=None,
    )
    chat_stub = StubChatService(result)
    audit_stub = StubAuditClient()
    main.chat_service = chat_stub
    main.audit_client = audit_stub
    request_state = {"location": "   "}

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={
            "session_id": "session-blank-location",
            "current_answer": "answer",
            "state": request_state,
        },
    )

    assert response.status_code == 200
    assert chat_stub.calls == [("answer", request_state)]
    assert audit_stub.calls
    audit_payload = audit_stub.calls[0]
    assert audit_payload["conversation_id"] == "session-blank-location"
    assert set(audit_payload) == {"conversation_id", "profile", "messages"}
    assert audit_payload["messages"] == [
        {"role": "user", "content": "answer"},
        {"role": "assistant", "content": "Next question"},
    ]
    assert "metadata" not in audit_payload


def test_chat_returns_answer(monkeypatch, tmp_path: Path) -> None:
    main = _load_main(monkeypatch, tmp_path)
    state = {
        "questionnaire": "0. Initial user question: []",
        "summary": "Existing summary",
        "last_question": "Last question",
    }
    result = ChatResult(
        text="Next question",
        response_type="answer",
        state=state,
        summary=None,
        quick_replies=[
            {"label": "Ja", "value": "yes"},
            {"label": "Nee", "value": "no"},
        ],
    )
    chat_stub = StubChatService(result)
    main.chat_service = chat_stub
    main.audit_client = None

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={
            "session_id": "session-2",
            "current_answer": "answer",
            "state": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "answer"
    assert payload["text"] == "Next question"
    assert payload["conversation_id"] == "session-2"
    assert response.headers["X-Pixie-Chat-Contract"] == CHAT_CONTRACT_VERSION
    assert "summary" not in payload
    assert "follow_up_question" not in payload
    assert payload["quick_replies"] == [
        {"label": "Ja", "value": "yes"},
        {"label": "Nee", "value": "no"},
    ]
