"""API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    current_answer: str = Field(..., min_length=1)
    state: dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    type: Literal["answer", "summary"]
    text: str
    conversation_id: str
    state: dict[str, str]
    summary: str | None = Field(default=None)
