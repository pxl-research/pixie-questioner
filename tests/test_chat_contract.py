from __future__ import annotations

from typing import cast

from pydantic import TypeAdapter

from pixie_questioner.schemas import (
    AUDIT_EVENT_CONTRACT_VERSION,
    CHAT_CONTRACT_VERSION,
    AuditEventPayload,
    ChatAnswerResponse,
    ChatRequest,
    ChatResponse,
    ChatSummaryResponse,
)


def _is_nullable_string(schema: dict[str, object]) -> bool:
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return False
    types: set[str] = set()
    for item in any_of:
        if not isinstance(item, dict):
            continue
        item_dict = cast(dict[str, object], item)
        item_type = item_dict.get("type")
        if isinstance(item_type, str):
            types.add(item_type)
    return types == {"string", "null"}


def _is_nullable_quick_reply_array(schema: dict[str, object]) -> bool:
    any_of = schema.get("anyOf")
    if not isinstance(any_of, list):
        return False

    array_schema: dict[str, object] | None = None
    null_present = False
    for item in any_of:
        if not isinstance(item, dict):
            continue
        item_dict = cast(dict[str, object], item)
        item_type = item_dict.get("type")
        if item_type == "array":
            array_schema = item_dict
        elif item_type == "null":
            null_present = True

    if not null_present or array_schema is None:
        return False
    if array_schema.get("maxItems") != 6:
        return False
    items = array_schema.get("items")
    if not isinstance(items, dict):
        return False
    items_dict = cast(dict[str, object], items)
    return items_dict.get("$ref") == "#/$defs/QuickReplyOption"


def test_chat_contract_version_is_pinned() -> None:
    assert CHAT_CONTRACT_VERSION == "chat-v1.1.0"


def test_audit_event_contract_version_is_pinned() -> None:
    assert AUDIT_EVENT_CONTRACT_VERSION == "audit-event-v1.0.0"


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
    assert _is_nullable_quick_reply_array(schema["properties"]["quick_replies"])

    option_schema = schema["$defs"]["QuickReplyOption"]
    assert option_schema["additionalProperties"] is False
    assert set(option_schema.get("required", [])) == {"label", "value"}
    assert option_schema["properties"]["label"]["type"] == "string"
    assert option_schema["properties"]["label"]["minLength"] == 1
    assert option_schema["properties"]["value"]["type"] == "string"
    assert option_schema["properties"]["value"]["minLength"] == 1


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


def test_audit_event_payload_matches_contract_shape() -> None:
    schema = AuditEventPayload.model_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema.get("required", [])) == {
        "conversation_id",
        "profile",
        "messages",
    }
    assert schema["properties"]["conversation_id"]["type"] == "string"
    assert schema["properties"]["conversation_id"]["minLength"] == 1
    assert schema["properties"]["profile"]["type"] == "string"
    assert schema["properties"]["profile"]["minLength"] == 1
    assert schema["properties"]["messages"]["type"] == "array"
    assert schema["properties"]["messages"]["minItems"] == 1
    assert schema["properties"]["messages"]["items"] == {
        "$ref": "#/$defs/AuditEventMessage"
    }

    message_schema = schema["$defs"]["AuditEventMessage"]
    assert message_schema["additionalProperties"] is False
    assert set(message_schema.get("required", [])) == {"role", "content"}
    assert message_schema["properties"]["role"]["enum"] == [
        "user",
        "assistant",
        "summary",
    ]
    assert message_schema["properties"]["content"]["type"] == "string"
    assert message_schema["properties"]["content"]["minLength"] == 1

    metadata_schema = schema["$defs"]["AuditEventMetadata"]
    assert metadata_schema["additionalProperties"] is False
    assert set(metadata_schema.get("required", [])) == {"location"}
    assert metadata_schema["properties"]["location"]["type"] == "string"
    assert metadata_schema["properties"]["location"]["minLength"] == 1
