"""API schemas."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

CHAT_CONTRACT_VERSION = "chat-v1.0.0"


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    current_answer: str = Field(..., min_length=1)
    state: dict[str, str] = Field(default_factory=dict)


class ChatAnswerResponse(BaseModel):
    type: Literal["answer"]
    text: str
    summary: str | None = None
    follow_up_question: str | None = None
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
