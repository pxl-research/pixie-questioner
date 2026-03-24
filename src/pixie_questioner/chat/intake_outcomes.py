"""Outcome builders for intake turns."""

from __future__ import annotations

import logging

from dataclasses import replace
from typing import TYPE_CHECKING

from .flow import (
    to_categorized_handoff_completed_state,
    to_feedback_state,
    to_intake_state_with_next_question,
    to_off_topic_redirect_state,
    to_terminated_state,
)
from .intake_routing import (
    DISH_FIELD_INDICES,
    WINE_FIELD_INDICES,
    determine_next_question_track,
)
from .questionnaire import (
    first_blank_field_index,
    focus_questionnaire_for_route,
    mask_later_blank_fields,
    select_questionnaire_fields,
)
from .types import INITIAL_QUESTION, INTAKE_ROUTE_FOOD_PAIRING, ChatResult, ChatState

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler

logger = logging.getLogger(__name__)

# Pre-crafted aroma/taste questions per language.  Field 8 has fixed question
# requirements (example clusters + negation allowance) that don't vary per
# conversation, so LLM generation is unnecessary and error-prone.
_AROMA_TASTE_QUESTIONS: dict[str, str] = {
    "nl": (
        "Welke geuren of smaken spreken je aan bij wijn? "
        "Denk bijvoorbeeld aan citrus, wit of rood fruit, bloemig of kruidig "
        "qua geur, en droog, fris, fruitig of romig qua smaak. "
        "Je mag ook aangeven wat je liever niet hebt, "
        "zoals niet zoet of geen houttoetsen?"
    ),
    "en": (
        "What aromas or tastes do you enjoy in wine? "
        "For example, citrus, white or red fruit, floral, or spicy for aroma, "
        "and dry, fresh, fruity, or creamy for taste. "
        "Feel free to mention what you'd prefer to avoid, "
        "like not sweet or no oak notes?"
    ),
    "fr": (
        "Quels arômes ou saveurs appréciez-vous dans un vin ? "
        "Par exemple, agrumes, fruits blancs ou rouges, floral ou épicé "
        "pour l'arôme, et sec, frais, fruité ou crémeux pour le goût. "
        "N'hésitez pas à mentionner ce que vous préférez éviter, "
        "comme pas trop sucré ou sans notes boisées ?"
    ),
}


async def handle_off_topic_intake(
    handler: ChatTurnHandler,
    current_state: ChatState,
    current_answer: str = "",
) -> ChatResult:
    current_state.boundary_violation_count += 1
    language = (
        current_state.conversation_language or handler.language.default_language_code
    )
    if (
        current_state.boundary_violation_count
        >= handler.domain_policy.off_topic_termination_threshold
    ):
        terminated_state = to_terminated_state(current_state, language)
        goodbye_text = await _generate_off_topic_message(
            handler=handler,
            task="goodbye",
            current_question=current_state.last_question,
            current_answer=current_answer,
            language=language,
            fallback=handler.messages.goodbye(language),
        )
        return handler.answer_result(
            goodbye_text,
            terminated_state,
            None,
            None,
            None,
        )

    redirected_state = to_off_topic_redirect_state(current_state)
    redirect_text = await _generate_off_topic_message(
        handler=handler,
        task="redirect",
        current_question=current_state.last_question,
        current_answer=current_answer,
        language=language,
        fallback=handler.messages.off_topic_redirect(
            language=language,
            current_question=current_state.last_question,
        ),
    )
    return handler.answer_result(
        redirect_text,
        redirected_state,
        None,
        None,
        None,
    )


async def _generate_off_topic_message(
    handler: ChatTurnHandler,
    task: str,
    current_question: str,
    current_answer: str,
    language: str,
    fallback: str,
) -> str:
    if handler.off_topic_redirect_invoker is None:
        return fallback
    summary = (
        f"Language: {language}\nTask: {task}\nCurrent question: {current_question}"
    )
    try:
        generated = await handler.off_topic_redirect_invoker.generate(
            current_question=task,
            current_answer=current_answer,
            summary=summary,
        )
        return generated.strip() or fallback
    except Exception:
        logger.warning("Off-topic LLM invoker failed; using fallback.", exc_info=True)
        return fallback


async def build_completion_result(
    handler: ChatTurnHandler,
    current_state: ChatState,
    updated_questionnaire: str,
    updated_summary: str,
) -> ChatResult:
    if handler.is_categorized_handoff_mode():
        return await _build_categorized_completion_result(
            handler,
            current_state,
            updated_questionnaire,
            updated_summary,
        )

    language = (
        current_state.conversation_language or handler.language.default_language_code
    )
    feedback_state = to_feedback_state(
        state=current_state,
        updated_questionnaire=updated_questionnaire,
        updated_summary=updated_summary,
        language=language,
    )
    return handler.answer_result(
        handler.messages.handoff(language),
        feedback_state,
        updated_summary,
        handler.messages.feedback_question(language),
        handler.messages.feedback_quick_replies(language),
    )


async def build_next_question_result(
    handler: ChatTurnHandler,
    current_state: ChatState,
    updated_questionnaire: str,
    updated_summary: str,
) -> ChatResult:
    conversation_language = (
        current_state.conversation_language or handler.language.default_language_code
    )
    next_question_track = determine_next_question_track(
        current_state,
        updated_questionnaire,
    )
    if next_question_track is None:
        raise RuntimeError(
            "No follow-up question remains for the current intake state."
        )
    dish_questionnaire = select_questionnaire_fields(
        updated_questionnaire,
        DISH_FIELD_INDICES,
    )
    wine_questionnaire = select_questionnaire_fields(
        updated_questionnaire,
        WINE_FIELD_INDICES,
    )
    if handler.is_categorized_handoff_mode() and next_question_track == "dish":
        focused_questionnaire = mask_later_blank_fields(
            dish_questionnaire,
            DISH_FIELD_INDICES,
        )
    elif handler.is_categorized_handoff_mode() and next_question_track == "wine":
        wine_focused = mask_later_blank_fields(
            wine_questionnaire,
            WINE_FIELD_INDICES,
        )
        if current_state.intake_route == INTAKE_ROUTE_FOOD_PAIRING:
            focused_questionnaire = dish_questionnaire + "\n" + wine_focused
        else:
            focused_questionnaire = wine_focused
    else:
        focused_questionnaire = focus_questionnaire_for_route(
            updated_questionnaire,
            handler.questionnaire_indices,
            current_state.intake_route,
            allow_dish_questions=next_question_track != "wine",
        )
    last_response = current_state.last_question
    if last_response.strip() == INITIAL_QUESTION:
        last_response = ""
    # Field 8 (aroma/taste): use a static question to avoid LLM generation
    # failures caused by conflicting prompt constraints for this field.
    if (
        handler.is_categorized_handoff_mode()
        and next_question_track == "wine"
        and first_blank_field_index(wine_questionnaire, WINE_FIELD_INDICES) == 8
    ):
        lang = conversation_language.casefold().split("-")[0]
        static_question = _AROMA_TASTE_QUESTIONS.get(lang, _AROMA_TASTE_QUESTIONS["nl"])
        next_state = to_intake_state_with_next_question(
            state=current_state,
            updated_questionnaire=updated_questionnaire,
            updated_summary=updated_summary,
            next_question=static_question,
        )
        next_state = replace(
            next_state,
            wine_follow_up_count=current_state.wine_follow_up_count + 1,
        )
        return handler.answer_result(static_question, next_state, None, None, None)
    next_question = await handler.question_asker.ask(
        focused_questionnaire,
        conversation_language,
        last_response=last_response,
    )
    normalized_question = next_question.strip()
    if handler.is_categorized_handoff_mode() and not _is_question_like(
        normalized_question
    ):
        logger.warning(
            "Question asker returned a non-question follow-up; retrying once."
        )
        retry_prompt_hint = (
            "Your previous response was not a valid customer question:\n"
            f"{normalized_question}\n\n"
            "Return exactly one clear customer-facing question for the next "
            "missing field. Do not use internal field labels."
        )
        retry_question = await handler.question_asker.ask(
            focused_questionnaire,
            conversation_language=conversation_language,
            last_response=retry_prompt_hint,
        )
        normalized_question = retry_question.strip()
        if not _is_question_like(normalized_question):
            raise RuntimeError(
                "Question asker returned a non-question follow-up while intake still has "
                "unanswered fields."
            )
    if not normalized_question:
        raise RuntimeError(
            "Question asker returned an empty question while intake still has "
            "unanswered fields. This is a transient LLM failure."
        )
    next_state = to_intake_state_with_next_question(
        state=current_state,
        updated_questionnaire=updated_questionnaire,
        updated_summary=updated_summary,
        next_question=normalized_question,
    )
    if next_question_track == "dish":
        next_state = replace(
            next_state,
            dish_follow_up_count=current_state.dish_follow_up_count + 1,
        )
    elif next_question_track == "wine":
        next_state = replace(
            next_state,
            wine_follow_up_count=current_state.wine_follow_up_count + 1,
        )
    return handler.answer_result(normalized_question, next_state, None, None, None)


def _is_question_like(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return normalized.endswith(("?", "？"))


async def _build_categorized_completion_result(
    handler: ChatTurnHandler,
    current_state: ChatState,
    updated_questionnaire: str,
    updated_summary: str,
) -> ChatResult:
    language = (
        current_state.conversation_language or handler.language.default_language_code
    )
    handoff_categories_v1 = (
        await handler.categorized_handoff.build_handoff_categories_v1(
            updated_questionnaire
        )
    )
    completed_state = to_categorized_handoff_completed_state(
        state=current_state,
        updated_questionnaire=updated_questionnaire,
        updated_summary=updated_summary,
        handoff_categories_v1=handoff_categories_v1,
    )
    return handler.answer_result(
        handler.categorized_handoff.build_handoff_text(
            handler.messages.handoff(language),
            handoff_categories_v1,
        ),
        completed_state,
        None,
        None,
        None,
    )
