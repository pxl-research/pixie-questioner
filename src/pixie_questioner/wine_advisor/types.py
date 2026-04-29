"""Shared state and record types for the wine advisor flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PHASE_INTAKE = "intake"
PHASE_COMPLETED = "completed"

MEAL_MODE_PAIRING = "pairing"

ConversationRole = Literal["user", "assistant"]
AdvisorPhase = Literal["intake", "completed"]
MealMode = Literal["pairing"]


@dataclass(frozen=True)
class ConversationTurn:
    role: ConversationRole
    text: str


@dataclass(frozen=True)
class WineCatalogRecommendation:
    barcode: str
    name: str
    country: str
    region: str
    aroma_description: str
    taste_description: str


@dataclass(frozen=True)
class WineAdvisorResult:
    text: str
    response_type: Literal["question", "recommendations"]
    state: dict[str, Any]
    recommendations: list[WineCatalogRecommendation] | None = None
    quick_replies: list[dict[str, str]] | None = None
    follow_up_question: str | None = None


@dataclass
class WineAdvisorState:
    phase: AdvisorPhase = PHASE_INTAKE
    turn_count: int = 0
    last_question: str = ""
    meal_mode: MealMode = MEAL_MODE_PAIRING
    working_context: str = ""
    chat_history: list[ConversationTurn] = field(default_factory=list)
    candidate_barcodes: list[str] = field(default_factory=list)
    selected_barcodes: list[str] = field(default_factory=list)
