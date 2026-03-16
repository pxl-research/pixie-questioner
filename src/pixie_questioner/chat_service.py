"""Chat orchestration service."""

from __future__ import annotations

from dataclasses import dataclass

from .chat.categorized_handoff import CategorizedHandoffBuilder
from .chat.intake_routing import DISH_FIELD_INDICES, WINE_FIELD_INDICES
from .chat.intake_route_classifier import IntakeRouteClassifierProtocol
from .chat.language import create_language_support
from .chat.messages import PolicyMessages
from .chat.policy import ChatDomainPolicy
from .chat.question_asker import QuestionAsker
from .chat.relevance import AnswerRelevanceClassifierProtocol
from .chat.state_codec import (
    extract_questionnaire_indices,
    parse_state,
    serialize_state,
)
from .chat.turn_handler import ChatTurnHandler
from .chat.types import ChatResult, ChatState
from .llm_client import LLMPromptInvoker


@dataclass
class ChatService:
    llm: object
    question_asker_prompt: str
    notulist: LLMPromptInvoker
    summarizer: LLMPromptInvoker
    questionnaire_template: str
    domain_policy: ChatDomainPolicy
    answer_relevance_classifier: AnswerRelevanceClassifierProtocol | None = None
    intake_route_classifier: IntakeRouteClassifierProtocol | None = None
    aroma_taste_classifier: LLMPromptInvoker | None = None
    handoff_classifier: LLMPromptInvoker | None = None
    unsupported_origin_follow_up: LLMPromptInvoker | None = None

    def __post_init__(self) -> None:
        _validate_service_configuration(self)
        self._questionnaire_indices = extract_questionnaire_indices(
            self.questionnaire_template
        )
        self._split_questionnaire_field_groups = None
        if self.domain_policy.completion_mode == "categorized_handoff":
            self._split_questionnaire_field_groups = (
                DISH_FIELD_INDICES,
                WINE_FIELD_INDICES,
            )
        self._language = create_language_support(
            language_codes=self.domain_policy.language_codes,
            language_names=self.domain_policy.language_names,
        )
        self._messages = PolicyMessages(policy=self.domain_policy)
        self._question_asker = QuestionAsker(
            llm=self.llm,
            prompt_template=self.question_asker_prompt,
            completion_token=self.domain_policy.completion_token,
            language=self._language,
            messages=self._messages,
        )
        self._categorized_handoff = CategorizedHandoffBuilder(
            handoff_classifier=self.handoff_classifier,
            aroma_taste_classifier=self.aroma_taste_classifier,
        )
        self._turn_handler = ChatTurnHandler(
            llm=self.llm,
            notulist=self.notulist,
            summarizer=self.summarizer,
            question_asker=self._question_asker,
            unsupported_origin_follow_up=self.unsupported_origin_follow_up,
            messages=self._messages,
            language=self._language,
            categorized_handoff=self._categorized_handoff,
            domain_policy=self.domain_policy,
            questionnaire_indices=self._questionnaire_indices,
            answer_result=self._answer_result,
            summary_result=self._summary_result,
            answer_relevance_classifier=self.answer_relevance_classifier,
            intake_route_classifier=self.intake_route_classifier,
        )

    def _parse_state(self, state: dict[str, str] | None) -> ChatState:
        return parse_state(
            state,
            questionnaire_template=self.questionnaire_template,
            default_language=self._language.default_language_code,
            split_questionnaire_field_groups=self._split_questionnaire_field_groups,
        )

    def _answer_result(
        self,
        text: str,
        state: ChatState,
        summary: str | None = None,
        follow_up_question: str | None = None,
        quick_replies: list[dict[str, str]] | None = None,
    ) -> ChatResult:
        return ChatResult(
            text=text,
            response_type="answer",
            state=serialize_state(state),
            summary=summary,
            follow_up_question=follow_up_question,
            quick_replies=quick_replies,
        )

    def _summary_result(self, text: str, state: ChatState, summary: str) -> ChatResult:
        return ChatResult(
            text=text,
            response_type="summary",
            state=serialize_state(state),
            summary=summary,
        )

    def _update_conversation_context(
        self,
        current_state: ChatState,
        current_answer: str,
    ) -> ChatState:
        (
            current_state.conversation_language,
            current_state.recent_user_turns,
        ) = self._language.resolve_conversation(
            current_answer=current_answer,
            conversation_language=current_state.conversation_language,
            recent_user_turns=current_state.recent_user_turns,
        )
        current_state.post_summary_language = current_state.conversation_language
        return current_state

    async def ask(
        self,
        current_answer: str,
        state: dict[str, str] | None,
    ) -> ChatResult:
        current_state = self._parse_state(state)
        current_state = self._update_conversation_context(
            current_state,
            current_answer,
        )
        return await self._turn_handler.handle(current_state, current_answer)


def _validate_service_configuration(service: ChatService) -> None:
    if not service.questionnaire_template.strip():
        raise RuntimeError("Questionnaire template is required.")

    indices = extract_questionnaire_indices(service.questionnaire_template)
    if not indices:
        raise RuntimeError("Questionnaire template must contain numbered fields.")

    if service.domain_policy.off_topic_termination_threshold < 1:
        raise RuntimeError(
            "off_topic_termination_threshold must be greater than or equal to 1."
        )
