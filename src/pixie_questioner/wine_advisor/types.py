"""Shared state and record types for the wine advisor flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PHASE_INTAKE = "intake"
PHASE_COMPLETED = "completed"

MEAL_MODE_PAIRING = "pairing"
MEAL_MODE_DRINKING_ONLY = "drinking_only"
MEAL_MODE_UNKNOWN = "unknown"

ConversationRole = Literal["user", "assistant"]
AdvisorPhase = Literal["intake", "completed"]
MealMode = Literal["pairing", "drinking_only", "unknown"]


@dataclass(frozen=True)
class ConversationTurn:
    role: ConversationRole
    text: str


@dataclass(frozen=True)
class WineCatalogRecommendation:
    barcode: str
    name: str
    price_eur: float
    country: str
    region: str


@dataclass(frozen=True)
class WineAdvisorResult:
    text: str
    response_type: Literal["question", "recommendations"]
    state: dict[str, Any]
    recommendations: list[WineCatalogRecommendation] | None = None


@dataclass
class WineAdvisorState:
    phase: AdvisorPhase = PHASE_INTAKE
    turn_count: int = 0
    last_question: str = ""
    meal_mode: MealMode = MEAL_MODE_UNKNOWN
    working_context: str = ""
    chat_history: list[ConversationTurn] = field(default_factory=list)
    candidate_barcodes: list[str] = field(default_factory=list)
    selected_barcodes: list[str] = field(default_factory=list)
    debug_enabled: bool = False
    debug_trace: list[dict[str, Any]] = field(default_factory=list)
