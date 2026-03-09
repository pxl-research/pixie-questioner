"""Chat orchestration service."""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass

from .chat.categorized_handoff import CategorizedHandoffBuilder
from .chat.flow import (
    ensure_feedback_context,
    mark_feedback_completed,
    to_categorized_handoff_completed_state,
    to_feedback_state,
    to_intake_state_with_next_question,
    to_off_topic_redirect_state,
    to_terminated_state,
)
from .chat.language import (
    SUPPORTED_NON_DUTCH_MIN_WORDS,
    SUPPORTED_RESPONSE_LANGUAGES,
    create_language_support,
    detect_language_candidates,
)
from .chat.messages import PolicyMessages
from .chat.policy import ChatDomainPolicy
from .chat.question_asker import QuestionAsker
from .chat.relevance import AnswerRelevanceClassifierProtocol, ClassificationLabel
from .chat.state_codec import (
    extract_questionnaire_indices,
    merge_partial_questionnaire_update,
    parse_state,
    serialize_state,
)
from .chat.types import (
    INITIAL_QUESTION,
    PHASE_AWAITING_FEEDBACK,
    PHASE_COMPLETED,
    PHASE_TERMINATED,
    ChatResult,
    ChatState,
)
from .llm_client import LLMPromptInvoker, extract_llm_content, resolve_llm_client

logger = logging.getLogger(__name__)


@dataclass
class ChatService:
    llm: object
    question_asker_prompt: str
    notulist: LLMPromptInvoker
    summarizer: LLMPromptInvoker
    questionnaire_template: str
    domain_policy: ChatDomainPolicy
    answer_relevance_classifier: AnswerRelevanceClassifierProtocol | None = None
    aroma_taste_classifier: LLMPromptInvoker | None = None
    handoff_classifier: LLMPromptInvoker | None = None

    def __post_init__(self) -> None:
        _validate_service_configuration(self)
        self._questionnaire_indices = extract_questionnaire_indices(
            self.questionnaire_template
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

    def _is_categorized_handoff_mode(self) -> bool:
        return self.domain_policy.completion_mode == "categorized_handoff"

    async def ask(
        self,
        current_answer: str,
        state: dict[str, str] | None,
    ) -> ChatResult:
        current_state = self._parse_state(state)

        if current_state.phase == PHASE_TERMINATED:
            return _handle_terminated_phase(self, current_state, current_answer)
        if current_state.phase == PHASE_AWAITING_FEEDBACK:
            return _handle_awaiting_feedback_phase(self, current_state, current_answer)
        if current_state.phase == PHASE_COMPLETED:
            return await _handle_completed_phase(self, current_state, current_answer)
        return await _handle_intake_phase(self, current_state, current_answer)


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


def _questionnaire_complete(questionnaire: str) -> bool:
    return "[]" not in questionnaire


def _is_blank_field_value(value: str) -> bool:
    return not value.strip().strip("[]").strip()


def _mask_later_blank_fields(
    questionnaire: str,
    questionnaire_indices: list[int],
) -> str:
    """Keep only the earliest blank field as ``[]`` for deterministic question flow.

    The question-asker prompt instructs the model to only ask about fields that still
    contain ``[]``. By masking later blank fields, we prevent jumps back and forth in
    question order when multiple fields are still empty.
    """
    first_blank_index: int | None = None
    for index in questionnaire_indices:
        match = re.search(
            rf"(?m)^\s*{index}\.\s*[^:\n]*:[^\S\n]*(.*?)$",
            questionnaire,
        )
        if match is None:
            continue
        if _is_blank_field_value(match.group(1)):
            first_blank_index = index
            break

    if first_blank_index is None:
        return questionnaire

    masked = questionnaire
    for index in questionnaire_indices:
        if index <= first_blank_index:
            continue
        pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*)(.*?)$")
        match = pattern.search(masked)
        if match is None:
            continue
        if not _is_blank_field_value(match.group(2)):
            continue
        masked = pattern.sub(r"\1[pending later]", masked, count=1)

    return masked


def _ensure_initial_question_field(
    questionnaire: str,
    current_answer: str,
) -> str:
    """Ensure field 0 is not left blank after the very first user turn."""
    normalized_answer = current_answer.strip()
    if not normalized_answer:
        return questionnaire

    pattern = re.compile(r"(?m)^(\s*0\.\s*[^:\n]*:[^\S\n]*)(.*?)$")
    match = pattern.search(questionnaire)
    if match is None:
        return questionnaire
    if not _is_blank_field_value(match.group(2)):
        return questionnaire
    return pattern.sub(
        lambda field_match: f"{field_match.group(1)}{normalized_answer}",
        questionnaire,
        count=1,
    )


def _has_supported_summary_language_signal(
    service: ChatService,
    summary_text: str,
) -> bool:
    effective_supported_languages = (
        service._language.supported_language_set & SUPPORTED_RESPONSE_LANGUAGES
    )
    if not effective_supported_languages:
        return False
    candidates = detect_language_candidates(summary_text)
    if not candidates:
        return True
    primary_language_code, _ = max(candidates, key=lambda candidate: candidate[1])
    if primary_language_code in effective_supported_languages:
        return True
    word_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", summary_text))
    return word_count < SUPPORTED_NON_DUTCH_MIN_WORDS


async def _rewrite_summary_in_preferred_language(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
    generated_summary: str,
) -> str | None:
    preferred_language = service._language.detect_preferred(
        current_answer=current_answer,
        previous_summary=current_state.summary,
    )
    language_name = service._language.supported_language_names.get(
        preferred_language,
        preferred_language.upper(),
    )
    rewrite_prompt = (
        "Rewrite the intake summary below in "
        f"{language_name}. Keep all facts and meaning exactly. "
        "Do not add or remove details. Output one concise paragraph only.\n\n"
        "Summary:\n"
        f"{generated_summary}"
    )

    try:
        response = await resolve_llm_client(service.llm).ainvoke(
            [{"role": "system", "content": rewrite_prompt}]
        )
    except Exception:
        logger.warning(
            "Failed to rewrite summary in preferred language.",
            exc_info=True,
        )
        return None

    rewritten_summary = extract_llm_content(response).strip()
    if not rewritten_summary:
        return None
    if not _has_supported_summary_language_signal(service, rewritten_summary):
        return None
    if service._language.detect(rewritten_summary) != preferred_language:
        return None
    return rewritten_summary


def _resolve_goodbye_language(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
) -> str:
    if current_state.post_summary_language:
        return current_state.post_summary_language
    return service._language.detect_preferred(
        current_answer=current_answer,
        previous_summary=current_state.summary,
    )


def _handle_terminated_phase(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
) -> ChatResult:
    language = _resolve_goodbye_language(service, current_state, current_answer)
    current_state.post_summary_language = language
    return service._answer_result(
        text=service._messages.goodbye(language),
        state=current_state,
    )


def _handle_awaiting_feedback_phase(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
) -> ChatResult:
    feedback_state, summary_text, language = ensure_feedback_context(
        current_state,
        detect_language=service._language.detect,
    )
    feedback_response = service._messages.normalize_feedback(current_answer)
    if feedback_response is None:
        return service._answer_result(
            text=service._messages.feedback_question(language),
            state=feedback_state,
            summary=summary_text,
            quick_replies=service._messages.feedback_quick_replies(language),
        )

    completed_state = mark_feedback_completed(feedback_state, feedback_response)
    return service._summary_result(
        text=service._messages.feedback_thanks(language),
        state=completed_state,
        summary=summary_text,
    )


async def _handle_completed_phase(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
) -> ChatResult:
    if service._is_categorized_handoff_mode():
        return await _build_completed_categorized_handoff_result(
            service,
            current_state,
            current_answer,
        )

    completed_state, summary_text, language = ensure_feedback_context(
        current_state,
        detect_language=service._language.detect,
    )
    return service._summary_result(
        text=service._messages.feedback_thanks(language),
        state=completed_state,
        summary=summary_text,
    )


async def _build_completed_categorized_handoff_result(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
) -> ChatResult:
    handoff_categories_v1 = current_state.handoff_categories_v1.strip()
    if not handoff_categories_v1:
        handoff_categories_v1 = (
            await service._categorized_handoff.build_handoff_categories_v1(
                current_state.questionnaire
            )
        )

    language = service._language.detect_preferred(
        current_answer=current_answer,
        previous_summary=current_state.summary,
    )
    completed_state = to_categorized_handoff_completed_state(
        state=current_state,
        updated_questionnaire=current_state.questionnaire,
        updated_summary=current_state.summary,
        handoff_categories_v1=handoff_categories_v1,
    )
    return service._answer_result(
        text=service._categorized_handoff.build_handoff_text(
            service._messages.handoff(language),
            handoff_categories_v1,
        ),
        state=completed_state,
    )


async def _classify_intake_answer(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
) -> ClassificationLabel:
    if current_state.last_question.strip() == INITIAL_QUESTION:
        return "on_topic"
    if service.answer_relevance_classifier is None:
        return "on_topic"
    try:
        return await service.answer_relevance_classifier.classify(
            current_question=current_state.last_question,
            current_answer=current_answer,
            questionnaire=current_state.questionnaire,
        )
    except Exception:
        logger.warning(
            "Answer relevance classifier failed; defaulting to on_topic.",
            exc_info=True,
        )
        return "on_topic"


def _handle_off_topic_intake(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
) -> ChatResult:
    current_state.boundary_violation_count += 1
    language = service._language.detect_preferred(
        current_answer=current_answer,
        previous_summary=current_state.summary,
    )
    if (
        current_state.boundary_violation_count
        >= service.domain_policy.off_topic_termination_threshold
    ):
        terminated_state = to_terminated_state(current_state, language)
        return service._answer_result(
            text=service._messages.goodbye(language),
            state=terminated_state,
        )

    redirected_state = to_off_topic_redirect_state(current_state)
    return service._answer_result(
        text=service._messages.off_topic_redirect(
            language=language,
            current_question=current_state.last_question,
        ),
        state=redirected_state,
    )


async def _update_questionnaire(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
) -> str:
    try:
        updated = await service.notulist.generate(
            current_question=current_state.last_question,
            current_answer=current_answer,
            summary=current_state.questionnaire,
        )
    except Exception as exc:
        raise RuntimeError("Failed to update questionnaire.") from exc

    merged = merge_partial_questionnaire_update(
        current_questionnaire=current_state.questionnaire,
        proposed_questionnaire=updated,
        questionnaire_indices=service._questionnaire_indices,
    )
    if merged is None:
        return current_state.questionnaire
    return merged


async def _update_summary(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
    enforce_supported_language: bool,
) -> str:
    try:
        generated_summary = await service.summarizer.generate(
            current_question=current_state.last_question,
            current_answer=current_answer,
            summary=current_state.summary,
        )
    except Exception as exc:
        raise RuntimeError("Failed to summarize conversation.") from exc

    if not enforce_supported_language:
        return generated_summary

    if _has_supported_summary_language_signal(service, generated_summary):
        return generated_summary

    rewritten_summary = await _rewrite_summary_in_preferred_language(
        service,
        current_state,
        current_answer,
        generated_summary,
    )
    if rewritten_summary is not None:
        return rewritten_summary

    fallback_summary = current_state.summary.strip() or current_answer.strip()
    if fallback_summary:
        return fallback_summary
    return generated_summary


async def _apply_intake_updates(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
) -> tuple[str, str]:
    updated_questionnaire = await _update_questionnaire(
        service,
        current_state,
        current_answer,
    )
    if service._is_categorized_handoff_mode():
        updated_questionnaire = (
            await service._categorized_handoff.normalize_questionnaire(
                updated_questionnaire
            )
        )
    if current_state.last_question.strip() == INITIAL_QUESTION:
        updated_questionnaire = _ensure_initial_question_field(
            updated_questionnaire,
            current_answer,
        )

    intake_complete = _questionnaire_complete(updated_questionnaire)
    updated_summary = await _update_summary(
        service,
        current_state,
        current_answer,
        enforce_supported_language=intake_complete,
    )
    return updated_questionnaire, updated_summary


async def _build_completion_result(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
    updated_questionnaire: str,
    updated_summary: str,
) -> ChatResult:
    if service._is_categorized_handoff_mode():
        return await _build_categorized_completion_result(
            service,
            current_state,
            current_answer,
            updated_questionnaire,
            updated_summary,
        )

    language = service._language.detect(updated_summary)
    feedback_state = to_feedback_state(
        state=current_state,
        updated_questionnaire=updated_questionnaire,
        updated_summary=updated_summary,
        language=language,
    )
    return service._answer_result(
        text=service._messages.handoff(language),
        state=feedback_state,
        summary=updated_summary,
        follow_up_question=service._messages.feedback_question(language),
        quick_replies=service._messages.feedback_quick_replies(language),
    )


async def _build_categorized_completion_result(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
    updated_questionnaire: str,
    updated_summary: str,
) -> ChatResult:
    language = service._language.detect_preferred(
        current_answer=current_answer,
        previous_summary=updated_summary,
    )
    handoff_categories_v1 = (
        await service._categorized_handoff.build_handoff_categories_v1(
            updated_questionnaire
        )
    )
    completed_state = to_categorized_handoff_completed_state(
        state=current_state,
        updated_questionnaire=updated_questionnaire,
        updated_summary=updated_summary,
        handoff_categories_v1=handoff_categories_v1,
    )
    return service._answer_result(
        text=service._categorized_handoff.build_handoff_text(
            service._messages.handoff(language),
            handoff_categories_v1,
        ),
        state=completed_state,
    )


async def _build_next_question_result(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
    updated_questionnaire: str,
    updated_summary: str,
) -> ChatResult:
    preferred_language = service._language.detect_preferred(
        current_answer=current_answer,
        previous_summary=current_state.summary,
    )
    focused_questionnaire = _mask_later_blank_fields(
        updated_questionnaire,
        service._questionnaire_indices,
    )
    next_question = await service._question_asker.ask(
        focused_questionnaire,
        preferred_language,
    )
    next_state = to_intake_state_with_next_question(
        state=current_state,
        updated_questionnaire=updated_questionnaire,
        updated_summary=updated_summary,
        next_question=next_question,
    )
    return service._answer_result(text=next_question, state=next_state)


async def _handle_on_topic_intake(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
    *,
    force_complete: bool = False,
) -> ChatResult:
    current_state.boundary_violation_count = 0
    updated_questionnaire, updated_summary = await _apply_intake_updates(
        service,
        current_state,
        current_answer,
    )
    if _questionnaire_complete(updated_questionnaire) or force_complete:
        return await _build_completion_result(
            service,
            current_state,
            current_answer,
            updated_questionnaire,
            updated_summary,
        )
    return await _build_next_question_result(
        service,
        current_state,
        current_answer,
        updated_questionnaire,
        updated_summary,
    )


async def _handle_intake_phase(
    service: ChatService,
    current_state: ChatState,
    current_answer: str,
) -> ChatResult:
    classification = await _classify_intake_answer(
        service,
        current_state,
        current_answer,
    )
    if classification == "off_topic":
        return _handle_off_topic_intake(service, current_state, current_answer)
    if classification == "stop":
        return await _handle_on_topic_intake(
            service,
            current_state,
            current_answer,
            force_complete=True,
        )
    return await _handle_on_topic_intake(service, current_state, current_answer)
