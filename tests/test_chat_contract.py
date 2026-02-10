from __future__ import annotations

from pydantic import TypeAdapter

from pixie_questioner.schemas import (
    CHAT_CONTRACT_VERSION,
    ChatAnswerResponse,
    ChatRequest,
    ChatResponse,
    ChatSummaryResponse,
)


def _is_nullable_string(schema: dict[str, object]) -> bool:
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return False
    types = {
        item.get("type")
        for item in any_of
        if isinstance(item, dict) and isinstance(item.get("type"), str)
    }
    return types == {"string", "null"}


def test_chat_contract_version_is_pinned() -> None:
    assert CHAT_CONTRACT_VERSION == "chat-v1.0.0"


def test_chat_request_matches_contract_shape() -> None:
    schema = ChatRequest.model_json_schema()

    assert set(schema.get("required", [])) == {"session_id", "current_answer"}
    assert schema["properties"]["session_id"]["type"] == "string"
    assert schema["properties"]["session_id"]["minLength"] == 1
    assert schema["properties"]["current_answer"]["type"] == "string"
    assert schema["properties"]["current_answer"]["minLength"] == 1
    assert schema["properties"]["state"]["type"] == "object"
    assert schema["properties"]["state"]["additionalProperties"] == {"type": "string"}


def test_chat_answer_response_matches_contract_shape() -> None:
    schema = ChatAnswerResponse.model_json_schema()

    assert set(schema.get("required", [])) == {
        "type",
        "text",
        "conversation_id",
        "state",
    }
    assert schema["properties"]["type"]["const"] == "answer"
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["conversation_id"]["type"] == "string"
    assert schema["properties"]["state"]["type"] == "object"
    assert schema["properties"]["state"]["additionalProperties"] == {"type": "string"}
    assert _is_nullable_string(schema["properties"]["summary"])
    assert _is_nullable_string(schema["properties"]["follow_up_question"])


def test_chat_summary_response_matches_contract_shape() -> None:
    schema = ChatSummaryResponse.model_json_schema()

    assert set(schema.get("required", [])) == {
        "type",
        "text",
        "summary",
        "conversation_id",
        "state",
    }
    assert schema["properties"]["type"]["const"] == "summary"
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["summary"]["type"] == "string"
    assert schema["properties"]["conversation_id"]["type"] == "string"
    assert schema["properties"]["state"]["type"] == "object"
    assert schema["properties"]["state"]["additionalProperties"] == {"type": "string"}
    assert _is_nullable_string(schema["properties"]["follow_up_question"])


def test_chat_response_discriminator_is_stable() -> None:
    schema = TypeAdapter(ChatResponse).json_schema()

    assert schema["discriminator"]["propertyName"] == "type"
    assert schema["discriminator"]["mapping"] == {
        "answer": "#/$defs/ChatAnswerResponse",
        "summary": "#/$defs/ChatSummaryResponse",
    }
    assert schema["oneOf"] == [
        {"$ref": "#/$defs/ChatAnswerResponse"},
        {"$ref": "#/$defs/ChatSummaryResponse"},
    ]
