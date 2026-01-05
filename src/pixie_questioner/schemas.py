"""API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    history: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
