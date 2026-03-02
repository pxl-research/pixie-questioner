"""API schemas."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CHAT_CONTRACT_VERSION = "chat-v2.0.0"
AUDIT_EVENT_CONTRACT_VERSION = "audit-event-v1.0.0"


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    current_answer: str = Field(..., min_length=1)
    state: dict[str, Any] = Field(default_factory=dict)


class ChatRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    price_eur: float
    country: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)


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


ChatResponse = Annotated[
    ChatQuestionResponse | ChatRecommendationsResponse,
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
