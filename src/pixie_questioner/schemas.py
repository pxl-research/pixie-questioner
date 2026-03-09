"""API schemas."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CHAT_CONTRACT_VERSION = "chat-v2.0.0"
BROOKLYN_CHAT_CONTRACT_VERSION = "chat-brooklyn-v2.0.0"
CRETSKENS_CHAT_CONTRACT_VERSION = "chat-cretskens-v1.0.0"
CASE2_CHAT_CONTRACT_VERSION = CRETSKENS_CHAT_CONTRACT_VERSION
AUDIT_EVENT_CONTRACT_VERSION = "audit-event-v1.0.0"
MAX_CHAT_STATE_KEYS = 64


def _resolve_chat_limits() -> tuple[int, int]:
    from .config import get_settings

    settings = get_settings()
    return (
        max(settings.wine_advisor_max_input_chars, 1),
        max(settings.wine_advisor_max_state_bytes, 1),
    )


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    current_answer: str = Field(..., min_length=1)
    state: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_state_payload(self) -> "ChatRequest":
        max_input_chars, max_state_bytes = _resolve_chat_limits()
        if len(self.current_answer) > max_input_chars:
            raise ValueError(f"current_answer too long (max {max_input_chars} chars).")

        if len(self.state) > MAX_CHAT_STATE_KEYS:
            raise ValueError(f"state has too many keys (max {MAX_CHAT_STATE_KEYS}).")

        try:
            encoded = json.dumps(self.state, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("state contains non-serializable values.") from exc

        state_size = len(encoded.encode("utf-8"))
        if state_size > max_state_bytes:
            raise ValueError(f"state payload too large (max {max_state_bytes} bytes).")

        return self


class ChatRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    price_eur: float
    country: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)


class QuickReplyOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class ChatQuestionResponse(BaseModel):
    type: Literal["question"]
    text: str
    conversation_id: str
    state: dict[str, Any]


class ChatRecommendationsResponse(BaseModel):
    type: Literal["recommendations"]
    text: str
    recommendations: list[ChatRecommendation] = Field(min_length=1, max_length=5)
    conversation_id: str
    state: dict[str, Any]


class ChatAnswerResponse(BaseModel):
    type: Literal["answer"]
    text: str
    summary: str | None = None
    follow_up_question: str | None = None
    quick_replies: list[QuickReplyOption] | None = Field(default=None, max_length=6)
    conversation_id: str
    state: dict[str, Any]


class ChatSummaryResponse(BaseModel):
    type: Literal["summary"]
    text: str
    summary: str
    follow_up_question: str | None = None
    conversation_id: str
    state: dict[str, Any]


ChatResponse = Annotated[
    ChatQuestionResponse | ChatRecommendationsResponse,
    Field(discriminator="type"),
]


CaseChatResponse = Annotated[
    ChatQuestionResponse
    | ChatRecommendationsResponse
    | ChatAnswerResponse
    | ChatSummaryResponse,
    Field(discriminator="type"),
]


class AuditEventMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str = Field(..., min_length=1)


class AuditEventMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "summary"]
    content: str = Field(..., min_length=1)


class AuditEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(..., min_length=1)
    profile: str = Field(..., min_length=1)
    messages: list[AuditEventMessage] = Field(..., min_length=1)
    metadata: AuditEventMetadata | None = None
