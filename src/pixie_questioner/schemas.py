"""API schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

CHAT_CONTRACT_VERSION = "chat-v1.1.0"
AUDIT_EVENT_CONTRACT_VERSION = "audit-event-v1.0.0"


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    current_answer: str = Field(..., min_length=1)
    state: dict[str, str] = Field(default_factory=dict)


class QuickReplyOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class ChatAnswerResponse(BaseModel):
    type: Literal["answer"]
    text: str
    summary: str | None = None
    follow_up_question: str | None = None
    quick_replies: list[QuickReplyOption] | None = Field(default=None, max_length=6)
    conversation_id: str
    state: dict[str, str]


class ChatSummaryResponse(BaseModel):
    type: Literal["summary"]
    text: str
    summary: str
    follow_up_question: str | None = None
    conversation_id: str
    state: dict[str, str]


ChatResponse = Annotated[
    ChatAnswerResponse | ChatSummaryResponse, Field(discriminator="type")
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
