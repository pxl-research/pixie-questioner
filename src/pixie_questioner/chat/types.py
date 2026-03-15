"""Shared chat domain types and state constants."""

from __future__ import annotations

from dataclasses import dataclass

INITIAL_QUESTION = "Initial question:"

PHASE_INTAKE = "intake"
PHASE_AWAITING_FEEDBACK = "awaiting_feedback"
PHASE_COMPLETED = "completed"
PHASE_TERMINATED = "terminated"
VALID_PHASES = {
    PHASE_INTAKE,
    PHASE_AWAITING_FEEDBACK,
    PHASE_COMPLETED,
    PHASE_TERMINATED,
}

FEEDBACK_PENDING = "pending"
FEEDBACK_YES = "yes"
FEEDBACK_NO = "no"
VALID_FEEDBACK_STATUSES = {FEEDBACK_PENDING, FEEDBACK_YES, FEEDBACK_NO}

INTAKE_ROUTE_UNKNOWN = "unknown"
INTAKE_ROUTE_FOOD_PAIRING = "food_pairing"
INTAKE_ROUTE_GENERAL_WINE = "general_wine"
VALID_INTAKE_ROUTES = {
    INTAKE_ROUTE_UNKNOWN,
    INTAKE_ROUTE_FOOD_PAIRING,
    INTAKE_ROUTE_GENERAL_WINE,
}


@dataclass(frozen=True)
class ChatResult:
    text: str
    response_type: str
    state: dict[str, str]
    summary: str | None
    follow_up_question: str | None = None
    quick_replies: list[dict[str, str]] | None = None


@dataclass
class ChatState:
    questionnaire: str = ""
    summary: str = ""
    last_question: str = INITIAL_QUESTION
    conversation_language: str = ""
    recent_user_turns: str = ""
    phase: str = PHASE_INTAKE
    feedback_status: str = FEEDBACK_PENDING
    final_summary: str = ""
    post_summary_language: str = ""
    boundary_violation_count: int = 0
    handoff_categories_v1: str = ""
    intake_route: str = INTAKE_ROUTE_UNKNOWN
    dish_follow_up_count: int = 0
    wine_follow_up_count: int = 0
    routing_question_asked: bool = False
