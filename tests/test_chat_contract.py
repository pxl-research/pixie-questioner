# ruff: noqa: S101,S105,S106
from __future__ import annotations

from pydantic import TypeAdapter

from pixie_questioner.schemas import (
    AUDIT_EVENT_CONTRACT_VERSION,
    BROOKLYN_CHAT_CONTRACT_VERSION,
    CRETSKENS_CHAT_CONTRACT_VERSION,
    CHAT_CONTRACT_VERSION,
    AuditEventPayload,
    ChatQuestionResponse,
    ChatRecommendation,
    ChatRecommendationsResponse,
    ChatRequest,
    ChatResponse,
)


def _allows_any_additional_properties(schema: dict[str, object]) -> bool:
    additional = schema.get("additionalProperties")
    if additional is True:
        return True
    if isinstance(additional, dict):
        return len(additional) == 0
    return False


def test_chat_contract_version_is_pinned() -> None:
    assert CHAT_CONTRACT_VERSION == "chat-v2.0.0"


def test_case_chat_contract_versions_are_pinned() -> None:
    assert BROOKLYN_CHAT_CONTRACT_VERSION == "chat-brooklyn-v2.0.0"
    assert CRETSKENS_CHAT_CONTRACT_VERSION == "chat-cretskens-v1.0.0"


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
    assert _allows_any_additional_properties(schema["properties"]["state"])


def test_chat_question_response_matches_contract_shape() -> None:
    schema = ChatQuestionResponse.model_json_schema()

    assert set(schema.get("required", [])) == {
        "type",
        "text",
        "conversation_id",
        "state",
    }
    assert schema["properties"]["type"]["const"] == "question"
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["conversation_id"]["type"] == "string"
    assert schema["properties"]["state"]["type"] == "object"
    assert _allows_any_additional_properties(schema["properties"]["state"])


def test_chat_recommendation_item_matches_contract_shape() -> None:
    schema = ChatRecommendation.model_json_schema()

    assert set(schema.get("required", [])) == {
        "name",
        "price_eur",
        "country",
        "region",
    }
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["price_eur"]["type"] == "number"
    assert schema["properties"]["country"]["type"] == "string"
    assert schema["properties"]["region"]["type"] == "string"


def test_chat_recommendations_response_matches_contract_shape() -> None:
    schema = ChatRecommendationsResponse.model_json_schema()

    assert set(schema.get("required", [])) == {
        "type",
        "text",
        "recommendations",
        "conversation_id",
        "state",
    }
    assert schema["properties"]["type"]["const"] == "recommendations"
    assert schema["properties"]["text"]["type"] == "string"
    assert schema["properties"]["recommendations"]["type"] == "array"
    assert schema["properties"]["recommendations"]["minItems"] == 1
    assert schema["properties"]["recommendations"]["maxItems"] == 5
    assert schema["properties"]["recommendations"]["items"] == {
        "$ref": "#/$defs/ChatRecommendation"
    }
    assert schema["properties"]["conversation_id"]["type"] == "string"
    assert schema["properties"]["state"]["type"] == "object"
    assert _allows_any_additional_properties(schema["properties"]["state"])


def test_chat_response_discriminator_is_stable() -> None:
    schema = TypeAdapter(ChatResponse).json_schema()

    assert schema["discriminator"]["propertyName"] == "type"
    assert schema["discriminator"]["mapping"] == {
        "question": "#/$defs/ChatQuestionResponse",
        "recommendations": "#/$defs/ChatRecommendationsResponse",
    }
    assert schema["oneOf"] == [
        {"$ref": "#/$defs/ChatQuestionResponse"},
        {"$ref": "#/$defs/ChatRecommendationsResponse"},
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
