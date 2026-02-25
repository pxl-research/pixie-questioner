"""Chat orchestration service."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .chat.language import (
    build_language_instruction,
    detect_language,
    detect_preferred_language,
)
from .chat.messages import (
    build_feedback_quick_replies,
    build_off_topic_redirect,
    choose_localized_message,
    normalize_feedback_response,
    simplify_technical_terms,
)
from .chat.policy import ChatDomainPolicy
from .chat.relevance import (
    AnswerRelevanceClassifierProtocol,
    LLMClientProtocol,
)
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
    FEEDBACK_NO,
    FEEDBACK_PENDING,
    FEEDBACK_YES,
    INITIAL_QUESTION,
    PHASE_AWAITING_FEEDBACK,
    PHASE_COMPLETED,
    PHASE_INTAKE,
    PHASE_TERMINATED,
    ChatResult,
    ChatState,
)
from .notulist import QuestionnaireNotulist
from .summarizer import ConversationSummarizer

_COMPLETION_STRIP_CHARS = " \t\n\r\"'`.,:;!?()[]{}<>"


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

        self._completion_token = self.domain_policy.completion_token
        normalized_language_codes = tuple(
            dict.fromkeys(
                code.strip().casefold()
                for code in self.domain_policy.language_codes
                if code.strip()
            )
        )
        if not normalized_language_codes:
            raise RuntimeError(
                "domain_policy.language_codes must include at least one language."
            )

        self._supported_language_codes = normalized_language_codes
        self._supported_language_set = set(normalized_language_codes)
        self._supported_language_priority = {
            code: index for index, code in enumerate(normalized_language_codes)
        }
        self._default_language_code = (
            "nl"
            if "nl" in self._supported_language_set
            else normalized_language_codes[0]
        )
        self._supported_language_names = {
            code.strip().casefold(): name
            for code, name in self.domain_policy.language_names.items()
            if code.strip()
        }
        for language_code in self._supported_language_codes:
            self._supported_language_names.setdefault(
                language_code, language_code.upper()
            )
        self._supported_language_names.setdefault("en", "English")
        self._stop_phrase_token_sequences = build_stop_phrase_token_sequences(
            self.domain_policy.stop_phrases
        )

    def _get_client(self) -> LLMClientProtocol:
        if hasattr(self.llm, "ainvoke"):
            return self.llm  # type: ignore[return-value]
        model = getattr(self.llm, "model", None)
        if model is None:
            raise RuntimeError("LLM client is not configured.")
        return model

    def _render_question_asker_prompt(self, questionnaire: str) -> str:
        prompt = self.question_asker_prompt
        if "{summary}" in prompt:
            prompt = prompt.replace("{summary}", questionnaire)
        if "{completion_token}" in prompt:
            prompt = prompt.replace("{completion_token}", self._completion_token)
        return prompt

    def _extract_content(self, response: object) -> str:
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        return content

    def _strip_completion_token(self, content: str) -> str:
        stripped = re.sub(
            re.escape(self._completion_token), "", content, flags=re.IGNORECASE
        )
        return stripped.strip(_COMPLETION_STRIP_CHARS)

    def _detect_language(self, text: str) -> str:
        return detect_language(
            text,
            supported_language_set=self._supported_language_set,
            default_language_code=self._default_language_code,
        )

    def _detect_preferred_language(
        self, current_answer: str, previous_summary: str
    ) -> str:
        return detect_preferred_language(
            current_answer=current_answer,
            previous_summary=previous_summary,
            supported_language_set=self._supported_language_set,
            supported_language_priority=self._supported_language_priority,
            default_language_code=self._default_language_code,
        )

    def _build_language_instruction(self, language_code: str) -> str:
        return build_language_instruction(
            language_code,
            supported_language_names=self._supported_language_names,
            supported_language_codes=self._supported_language_codes,
            default_language_code=self._default_language_code,
        )

    def _build_handoff_line(self, language: str) -> str:
        return choose_localized_message(self.domain_policy.handoff_lines, language)

    def _build_feedback_question(self, language: str) -> str:
        return choose_localized_message(self.domain_policy.feedback_questions, language)

    def _build_feedback_quick_replies(self, language: str) -> list[dict[str, str]]:
        return build_feedback_quick_replies(
            self.domain_policy.quick_replies,
            language=language,
        )

    def _build_feedback_thanks(self, language: str) -> str:
        return choose_localized_message(self.domain_policy.feedback_thanks, language)

    def _build_goodbye_line(self, language: str) -> str:
        return choose_localized_message(self.domain_policy.goodbye_lines, language)

    def _build_off_topic_redirect(self, language: str, current_question: str) -> str:
        return build_off_topic_redirect(
            language=language,
            current_question=current_question,
            initial_question=INITIAL_QUESTION,
            off_topic_initial_questions=self.domain_policy.off_topic_initial_questions,
            off_topic_redirects=self.domain_policy.off_topic_redirects,
        )

    def _normalize_feedback_response(self, current_answer: str) -> str | None:
        return normalize_feedback_response(
            current_answer=current_answer,
            feedback_yes_tokens=self.domain_policy.feedback_yes_tokens,
            feedback_no_tokens=self.domain_policy.feedback_no_tokens,
            feedback_yes_value=FEEDBACK_YES,
            feedback_no_value=FEEDBACK_NO,
        )

    def _simplify_technical_terms(self, text: str, language: str) -> str:
        return simplify_technical_terms(
            text,
            language=language,
            term_replacements=self.domain_policy.term_replacements,
        )

    async def _is_on_topic_answer(
        self,
        current_question: str,
        current_answer: str,
        questionnaire: str,
    ) -> bool:
        if self.answer_relevance_classifier is None:
            return True
        try:
            return await self.answer_relevance_classifier.is_on_topic(
                current_question=current_question,
                current_answer=current_answer,
                questionnaire=questionnaire,
            )
        except Exception:
            return True

    def _should_skip_relevance_check(self, current_question: str) -> bool:
        return current_question.strip() == INITIAL_QUESTION

    def _is_valid_questionnaire(self, questionnaire: str) -> bool:
        return is_valid_questionnaire(
            questionnaire,
            questionnaire_indices=self._questionnaire_indices,
        )

    def _parse_state(self, state: dict[str, str] | None) -> ChatState:
        return parse_state(
            state,
            questionnaire_template=self.questionnaire_template,
            detect_language=self._detect_language,
        )

    def _serialize_state(self, state: ChatState) -> dict[str, str]:
        return serialize_state(state)

    def _user_requested_stop(self, current_answer: str) -> bool:
        return user_requested_stop(
            current_answer,
            stop_phrase_token_sequences=self._stop_phrase_token_sequences,
            stop_context_tokens=self.domain_policy.stop_context_tokens,
        )

    def _questionnaire_complete(self, questionnaire: str) -> bool:
        return "[]" not in questionnaire

    async def _ask_next_question(
        self, questionnaire: str, preferred_language: str
    ) -> str:
        system_prompt = self._render_question_asker_prompt(questionnaire)
        system_prompt = (
            f"{system_prompt}{self._build_language_instruction(preferred_language)}"
        )
        response = await self._get_client().ainvoke(
            [{"role": "system", "content": system_prompt}]
        )
        content = self._extract_content(response)
        if (
            self._completion_token
            and self._completion_token.casefold() in content.casefold()
        ):
            content = self._strip_completion_token(content)
        return self._simplify_technical_terms(content, preferred_language)

    async def ask(
        self,
        current_answer: str,
        state: dict[str, str] | None,
    ) -> ChatResult:
        current_state = self._parse_state(state)
        if current_state.phase == PHASE_TERMINATED:
            language = current_state.post_summary_language
            if not language:
                language = self._detect_preferred_language(
                    current_answer=current_answer,
                    previous_summary=current_state.summary,
                )
                current_state.post_summary_language = language
            return ChatResult(
                text=self._build_goodbye_line(language),
                response_type="answer",
                state=self._serialize_state(current_state),
                summary=None,
            )

        if self._user_requested_stop(current_answer):
            language = self._detect_preferred_language(
                current_answer=current_answer,
                previous_summary=current_state.summary,
            )
            terminated_state = ChatState(
                questionnaire=current_state.questionnaire,
                summary=current_state.summary,
                last_question=current_state.last_question,
                phase=PHASE_TERMINATED,
                feedback_status=FEEDBACK_PENDING,
                final_summary="",
                post_summary_language=language,
                boundary_violation_count=current_state.boundary_violation_count,
            )
            return ChatResult(
                text=self._build_goodbye_line(language),
                response_type="answer",
                state=self._serialize_state(terminated_state),
                summary=None,
            )

        if current_state.phase == PHASE_AWAITING_FEEDBACK:
            summary_text = current_state.final_summary
            if not summary_text.strip():
                summary_text = current_state.summary
                current_state.final_summary = summary_text
            language = current_state.post_summary_language
            if not language:
                language = self._detect_language(summary_text)
                current_state.post_summary_language = language
            feedback_response = self._normalize_feedback_response(current_answer)
            if feedback_response is not None:
                current_state.phase = PHASE_COMPLETED
                current_state.feedback_status = feedback_response
                return ChatResult(
                    text=self._build_feedback_thanks(language),
                    response_type="summary",
                    state=self._serialize_state(current_state),
                    summary=summary_text,
                )

            current_state.feedback_status = FEEDBACK_PENDING
            return ChatResult(
                text=self._build_feedback_question(language),
                response_type="answer",
                state=self._serialize_state(current_state),
                summary=summary_text,
                quick_replies=self._build_feedback_quick_replies(language),
            )

        if current_state.phase == PHASE_COMPLETED:
            summary_text = current_state.final_summary
            if not summary_text.strip():
                summary_text = current_state.summary
                current_state.final_summary = summary_text
            language = current_state.post_summary_language
            if not language:
                language = self._detect_language(summary_text)
                current_state.post_summary_language = language
            return ChatResult(
                text=self._build_feedback_thanks(language),
                response_type="summary",
                state=self._serialize_state(current_state),
                summary=summary_text,
            )

        if self._should_skip_relevance_check(current_state.last_question):
            is_on_topic = True
        else:
            is_on_topic = await self._is_on_topic_answer(
                current_question=current_state.last_question,
                current_answer=current_answer,
                questionnaire=current_state.questionnaire,
            )

        if not is_on_topic:
            current_state.boundary_violation_count += 1
            language = self._detect_preferred_language(
                current_answer=current_answer,
                previous_summary=current_state.summary,
            )
            if (
                current_state.boundary_violation_count
                >= self.domain_policy.off_topic_termination_threshold
            ):
                terminated_state = ChatState(
                    questionnaire=current_state.questionnaire,
                    summary=current_state.summary,
                    last_question=current_state.last_question,
                    phase=PHASE_TERMINATED,
                    feedback_status=FEEDBACK_PENDING,
                    final_summary="",
                    post_summary_language=language,
                    boundary_violation_count=current_state.boundary_violation_count,
                )
                return ChatResult(
                    text=self._build_goodbye_line(language),
                    response_type="answer",
                    state=self._serialize_state(terminated_state),
                    summary=None,
                )

            redirected_state = ChatState(
                questionnaire=current_state.questionnaire,
                summary=current_state.summary,
                last_question=current_state.last_question,
                phase=PHASE_INTAKE,
                feedback_status=FEEDBACK_PENDING,
                final_summary="",
                post_summary_language="",
                boundary_violation_count=current_state.boundary_violation_count,
            )
            return ChatResult(
                text=self._build_off_topic_redirect(
                    language=language,
                    current_question=current_state.last_question,
                ),
                response_type="answer",
                state=self._serialize_state(redirected_state),
                summary=None,
            )

        current_state.boundary_violation_count = 0
        current_question = current_state.last_question
        try:
            updated_questionnaire = await self.notulist.generate(
                current_question=current_question,
                current_answer=current_answer,
                summary=current_state.questionnaire,
            )
        except Exception as exc:
            raise RuntimeError("Failed to update questionnaire.") from exc
        if updated_questionnaire is None:
            raise RuntimeError("Failed to update questionnaire.")
        if not self._is_valid_questionnaire(updated_questionnaire):
            updated_questionnaire = current_state.questionnaire

        try:
            updated_summary = await self.summarizer.generate(
                current_question=current_question,
                current_answer=current_answer,
                summary=current_state.summary,
            )
        except Exception as exc:
            raise RuntimeError("Failed to summarize conversation.") from exc
        if updated_summary is None:
            raise RuntimeError("Failed to summarize conversation.")

        if self._questionnaire_complete(
            updated_questionnaire
        ) or self._user_requested_stop(current_answer):
            language = self._detect_language(updated_summary)
            final_state = ChatState(
                questionnaire=updated_questionnaire,
                summary=updated_summary,
                last_question=current_question,
                phase=PHASE_AWAITING_FEEDBACK,
                feedback_status=FEEDBACK_PENDING,
                final_summary=updated_summary,
                post_summary_language=language,
                boundary_violation_count=current_state.boundary_violation_count,
            )
            return ChatResult(
                text=self._build_handoff_line(language),
                response_type="answer",
                state=self._serialize_state(final_state),
                summary=updated_summary,
                follow_up_question=self._build_feedback_question(language),
                quick_replies=self._build_feedback_quick_replies(language),
            )

        preferred_language = self._detect_preferred_language(
            current_answer=current_answer,
            previous_summary=current_state.summary,
        )
        next_question = await self._ask_next_question(
            updated_questionnaire,
            preferred_language,
        )
        next_state = ChatState(
            questionnaire=updated_questionnaire,
            summary=updated_summary,
            last_question=next_question,
            phase=PHASE_INTAKE,
            feedback_status=FEEDBACK_PENDING,
            boundary_violation_count=current_state.boundary_violation_count,
        )
        return ChatResult(
            text=next_question,
            response_type="answer",
            state=self._serialize_state(next_state),
            summary=None,
        )
