from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from pixie_questioner import config as config_module
from pixie_questioner.chat_service import ChatResult
from pixie_questioner.schemas import CHAT_CONTRACT_VERSION


def _write_prompt(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _set_env(monkeypatch, tmp_path: Path) -> None:
    system_path = _write_prompt(tmp_path / "system.txt", "system {summary}")
    notulist_path = _write_prompt(
        tmp_path / "notulist.txt",
        "notulist {summary} {current_question} {current_answer}",
    )
    summarizer_path = _write_prompt(
        tmp_path / "summarizer.txt",
        "summary {summary} {current_question} {current_answer}",
    )
    template_path = _write_prompt(
        tmp_path / "template.txt",
        "0. Initial user question: []\n1. Occasion, Vibe, Setting: []",
    )

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_API_VERSION", "2024-01-01")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("SYSTEM_PROMPT_PATH", str(system_path))
    monkeypatch.setenv("SUMMARIZER_PROMPT_PATH", str(summarizer_path))
    monkeypatch.setenv("QUESTIONNAIRE_NOTULIST_PROMPT_PATH", str(notulist_path))
    monkeypatch.setenv("QUESTIONNAIRE_TEMPLATE_PATH", str(template_path))


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

    client = TestClient(main.app)
    response = client.post(
        "/chat",
        json={
            "session_id": "session-1",
            "current_answer": "stop",
            "state": {},
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
    assert chat_stub.calls == [("stop", {})]

    assert audit_stub.calls
    audit_payload = audit_stub.calls[0]
    assert audit_payload["conversation_id"] == "session-1"
    assert audit_payload["profile"] == "test-profile"
    assert audit_payload["summary"] == "Final summary"


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
