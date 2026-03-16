"""Turn orchestration for chat intake flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..llm_client import LLMPromptInvoker
from .categorized_handoff import CategorizedHandoffBuilder
from .feedback_turn import (
    handle_awaiting_feedback_phase,
    handle_completed_phase,
    handle_terminated_phase,
)
from .intake_route_classifier import IntakeRouteClassifierProtocol
from .intake_turn import handle_intake_phase
from .language import LanguageSupport
from .messages import PolicyMessages
from .policy import ChatDomainPolicy
from .question_asker import QuestionAsker
from .relevance import AnswerRelevanceClassifierProtocol
from .types import (
    PHASE_AWAITING_FEEDBACK,
    PHASE_COMPLETED,
    PHASE_TERMINATED,
    ChatResult,
    ChatState,
)

AnswerResultBuilder = Callable[
    [str, ChatState, str | None, str | None, list[dict[str, str]] | None],
    ChatResult,
]
SummaryResultBuilder = Callable[[str, ChatState, str], ChatResult]


@dataclass
class ChatTurnHandler:
    llm: object
    notulist: LLMPromptInvoker
    summarizer: LLMPromptInvoker
    question_asker: QuestionAsker
    unsupported_origin_follow_up: LLMPromptInvoker | None
    messages: PolicyMessages
    language: LanguageSupport
    categorized_handoff: CategorizedHandoffBuilder
    domain_policy: ChatDomainPolicy
    questionnaire_indices: list[int]
    answer_result: AnswerResultBuilder
    summary_result: SummaryResultBuilder
    answer_relevance_classifier: AnswerRelevanceClassifierProtocol | None = None
    intake_route_classifier: IntakeRouteClassifierProtocol | None = None

    def is_categorized_handoff_mode(self) -> bool:
        return self.domain_policy.completion_mode == "categorized_handoff"

    async def handle(
        self,
        current_state: ChatState,
        current_answer: str,
    ) -> ChatResult:
        if current_state.phase == PHASE_TERMINATED:
            return handle_terminated_phase(self, current_state)
        if current_state.phase == PHASE_AWAITING_FEEDBACK:
            return handle_awaiting_feedback_phase(
                self,
                current_state,
                current_answer,
            )
        if current_state.phase == PHASE_COMPLETED:
            return await handle_completed_phase(
                self,
                current_state,
            )
        return await handle_intake_phase(self, current_state, current_answer)
