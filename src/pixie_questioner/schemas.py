"""API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    history: list[str] = Field(default_factory=list)
    questionnaire: str | None = None
    summary: str | None = None
    current_question: str | None = None
    summarize: bool = False


class ChatResponse(BaseModel):
    type: Literal["answer", "summary"]
    text: str
    conversation_id: str
