"""Chat orchestration service."""

from __future__ import annotations

from dataclasses import dataclass

from .chat.flow import (
    ensure_feedback_context,
    mark_feedback_completed,
    to_feedback_state,
    to_intake_state_with_next_question,
    to_off_topic_redirect_state,
    to_terminated_state,
)
from .chat.language import create_language_support
from .chat.messages import PolicyMessages
from .chat.policy import ChatDomainPolicy
from .chat.question_asker import QuestionAsker
from .chat.relevance import AnswerRelevanceClassifierProtocol
from .chat.state_codec import (
    extract_questionnaire_indices,
    is_valid_questionnaire,
    parse_state,
    serialize_state,
)
from .chat.stop_detection import (
    build_stop_phrase_token_sequences,
    user_requested_stop,
)
from .chat.types import (
    INITIAL_QUESTION,
    PHASE_AWAITING_FEEDBACK,
    PHASE_COMPLETED,
    PHASE_TERMINATED,
    ChatResult,
    ChatState,
)
from .notulist import QuestionnaireNotulist
from .summarizer import ConversationSummarizer


@dataclass
class ChatService:
    llm: object
    question_asker_prompt: str
    notulist: QuestionnaireNotulist
    summarizer: ConversationSummarizer
    questionnaire_template: str
    domain_policy: ChatDomainPolicy
    answer_relevance_classifier: AnswerRelevanceClassifierProtocol | None = None

    def __post_init__(self) -> None:
        if not self.questionnaire_template.strip():
            raise RuntimeError("Questionnaire template is required.")
        self._questionnaire_indices = extract_questionnaire_indices(
            self.questionnaire_template
        )
        if not self._questionnaire_indices:
            raise RuntimeError("Questionnaire template must contain numbered fields.")
        if self.domain_policy.off_topic_termination_threshold < 1:
            raise RuntimeError(
                "off_topic_termination_threshold must be greater than or equal to 1."
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
        self._stop_phrase_token_sequences = build_stop_phrase_token_sequences(
            self.domain_policy.stop_phrases
        )

    # -- Helpers ---------------------------------------------------------------

    def _user_requested_stop(self, current_answer: str) -> bool:
        return user_requested_stop(
            current_answer,
            stop_phrase_token_sequences=self._stop_phrase_token_sequences,
            stop_context_tokens=self.domain_policy.stop_context_tokens,
        )

    def _questionnaire_complete(self, questionnaire: str) -> bool:
        return "[]" not in questionnaire

    def _parse_state(self, state: dict[str, str] | None) -> ChatState:
        return parse_state(
            state,
            questionnaire_template=self.questionnaire_template,
            detect_language=self._language.detect,
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

    # -- Phase handlers --------------------------------------------------------

    def _resolve_goodbye_language(
        self, current_state: ChatState, current_answer: str
    ) -> str:
        if current_state.post_summary_language:
            return current_state.post_summary_language
        return self._language.detect_preferred(
            current_answer=current_answer,
            previous_summary=current_state.summary,
        )

    def _handle_terminated_phase(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        language = self._resolve_goodbye_language(current_state, current_answer)
        current_state.post_summary_language = language
        return self._answer_result(
            text=self._messages.goodbye(language),
            state=current_state,
        )

    def _handle_user_stop_request(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        language = self._resolve_goodbye_language(current_state, current_answer)
        terminated_state = to_terminated_state(current_state, language)
        return self._answer_result(
            text=self._messages.goodbye(language),
            state=terminated_state,
        )

    def _handle_awaiting_feedback_phase(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        feedback_state, summary_text, language = ensure_feedback_context(
            current_state,
            detect_language=self._language.detect,
        )
        feedback_response = self._messages.normalize_feedback(current_answer)
        if feedback_response is None:
            return self._answer_result(
                text=self._messages.feedback_question(language),
                state=feedback_state,
                summary=summary_text,
                quick_replies=self._messages.feedback_quick_replies(language),
            )
        completed_state = mark_feedback_completed(feedback_state, feedback_response)
        return self._summary_result(
            text=self._messages.feedback_thanks(language),
            state=completed_state,
            summary=summary_text,
        )

    def _handle_completed_phase(self, current_state: ChatState) -> ChatResult:
        completed_state, summary_text, language = ensure_feedback_context(
            current_state,
            detect_language=self._language.detect,
        )
        return self._summary_result(
            text=self._messages.feedback_thanks(language),
            state=completed_state,
            summary=summary_text,
        )

    # -- Intake phase ----------------------------------------------------------

    async def _is_on_topic_for_intake(
        self, current_state: ChatState, current_answer: str
    ) -> bool:
        if current_state.last_question.strip() == INITIAL_QUESTION:
            return True
        if self.answer_relevance_classifier is None:
            return True
        try:
            return await self.answer_relevance_classifier.is_on_topic(
                current_question=current_state.last_question,
                current_answer=current_answer,
                questionnaire=current_state.questionnaire,
            )
        except Exception:
            return True

    def _handle_off_topic_intake(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        current_state.boundary_violation_count += 1
        language = self._language.detect_preferred(
            current_answer=current_answer,
            previous_summary=current_state.summary,
        )
        if (
            current_state.boundary_violation_count
            >= self.domain_policy.off_topic_termination_threshold
        ):
            terminated_state = to_terminated_state(current_state, language)
            return self._answer_result(
                text=self._messages.goodbye(language),
                state=terminated_state,
            )
        redirected_state = to_off_topic_redirect_state(current_state)
        return self._answer_result(
            text=self._messages.off_topic_redirect(
                language=language,
                current_question=current_state.last_question,
            ),
            state=redirected_state,
        )

    async def _update_questionnaire(
        self, current_state: ChatState, current_answer: str
    ) -> str:
        try:
            updated = await self.notulist.generate(
                current_question=current_state.last_question,
                current_answer=current_answer,
                summary=current_state.questionnaire,
            )
        except Exception as exc:
            raise RuntimeError("Failed to update questionnaire.") from exc
        if updated is None:
            raise RuntimeError("Failed to update questionnaire.")
        if not is_valid_questionnaire(
            updated, questionnaire_indices=self._questionnaire_indices
        ):
            return current_state.questionnaire
        return updated

    async def _update_summary(
        self, current_state: ChatState, current_answer: str
    ) -> str:
        try:
            updated = await self.summarizer.generate(
                current_question=current_state.last_question,
                current_answer=current_answer,
                summary=current_state.summary,
            )
        except Exception as exc:
            raise RuntimeError("Failed to summarize conversation.") from exc
        if updated is None:
            raise RuntimeError("Failed to summarize conversation.")
        return updated

    async def _handle_on_topic_intake(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        current_state.boundary_violation_count = 0
        updated_questionnaire = await self._update_questionnaire(
            current_state, current_answer
        )
        updated_summary = await self._update_summary(current_state, current_answer)

        if self._questionnaire_complete(
            updated_questionnaire
        ) or self._user_requested_stop(current_answer):
            language = self._language.detect(updated_summary)
            feedback_state = to_feedback_state(
                state=current_state,
                updated_questionnaire=updated_questionnaire,
                updated_summary=updated_summary,
                language=language,
            )
            return self._answer_result(
                text=self._messages.handoff(language),
                state=feedback_state,
                summary=updated_summary,
                follow_up_question=self._messages.feedback_question(language),
                quick_replies=self._messages.feedback_quick_replies(language),
            )

        preferred_language = self._language.detect_preferred(
            current_answer=current_answer,
            previous_summary=current_state.summary,
        )
        next_question = await self._question_asker.ask(
            updated_questionnaire, preferred_language
        )
        next_state = to_intake_state_with_next_question(
            state=current_state,
            updated_questionnaire=updated_questionnaire,
            updated_summary=updated_summary,
            next_question=next_question,
        )
        return self._answer_result(text=next_question, state=next_state)

    async def _handle_intake_phase(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        is_on_topic = await self._is_on_topic_for_intake(current_state, current_answer)
        if not is_on_topic:
            return self._handle_off_topic_intake(current_state, current_answer)
        return await self._handle_on_topic_intake(current_state, current_answer)

    # -- Public entry point ----------------------------------------------------

    async def ask(
        self,
        current_answer: str,
        state: dict[str, str] | None,
    ) -> ChatResult:
        current_state = self._parse_state(state)

        # Turn routing is phase-first. This keeps the orchestration flow explicit:
        # terminated -> explicit user stop -> feedback -> completed -> intake.
        if current_state.phase == PHASE_TERMINATED:
            return self._handle_terminated_phase(current_state, current_answer)

        if self._user_requested_stop(current_answer):
            return self._handle_user_stop_request(current_state, current_answer)

        if current_state.phase == PHASE_AWAITING_FEEDBACK:
            return self._handle_awaiting_feedback_phase(current_state, current_answer)

        if current_state.phase == PHASE_COMPLETED:
            return self._handle_completed_phase(current_state)

        return await self._handle_intake_phase(current_state, current_answer)
