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
)
from .types import INITIAL_QUESTION, INTAKE_ROUTE_FOOD_PAIRING, ChatResult, ChatState

if TYPE_CHECKING:
    from .turn_handler import ChatTurnHandler

logger = logging.getLogger(__name__)

_QUESTION_PREFIXES = (
    "what",
    "which",
    "where",
    "when",
    "why",
    "how",
    "do ",
    "does ",
    "did ",
    "is ",
    "are ",
    "can ",
    "could ",
    "would ",
    "will ",
    "have ",
    "has ",
    "wat",
    "welk",
    "welke",
    "waar",
    "wanneer",
    "waarom",
    "hoe",
    "heb",
    "heeft",
    "ben ",
    "is ",
    "zijn ",
    "kan ",
    "kun ",
    "mag ",
    "zou ",
    "wil ",
    "wilt ",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "ou ",
    "où",
    "quand",
    "pourquoi",
    "comment",
    "avez",
    "voulez",
    "souhaitez",
    "pouvez",
    "est-ce",
)

_FALLBACK_QUESTIONS = {
    "nl": {
        0: "Welk gerecht of welke maaltijd wil je met wijn combineren?",
        1: "Gaat het om een aperitief, hoofdgerecht, kaasgang of dessert?",
        2: "Wat is de hoofdproteïne van het gerecht?",
        3: "Welke bereidingsmethode past het best bij het gerecht?",
        4: "Als het om kaas gaat: blauwe, zachte of harde kaas?",
        5: "Welke kleur wijn heeft je voorkeur: wit, rood of rosé?",
        6: "Heb je een voorkeur voor een land of regio waar de wijn vandaan komt?",
        7: "Wat is je maximale budget in euro?",
        8: "Welke geur of smaak zoek je in de wijn?",
    },
    "en": {
        0: "Which dish or meal would you like to pair with wine?",
        1: "Is it an aperitif, main course, cheese course, or dessert?",
        2: "What is the main protein of the dish?",
        3: "Which cooking method fits best for the dish?",
        4: "If it is a cheese course: blue, soft, or hard cheese?",
        5: "Which wine color do you prefer: white, red, or rosé?",
        6: "Do you have a preferred country or region of origin?",
        7: "What is your maximum budget in euro?",
        8: "Which aroma or taste profile are you looking for?",
    },
    "fr": {
        0: "Quel plat ou quel repas souhaitez-vous accorder avec un vin ?",
        1: "S'agit-il d'un apéritif, d'un plat principal, d'un plateau de fromages ou d'un dessert ?",
        2: "Quelle est la protéine principale du plat ?",
        3: "Quel mode de cuisson correspond le mieux au plat ?",
        4: "Si c'est un plateau de fromages : bleu, pâte molle ou pâte dure ?",
        5: "Quelle couleur de vin préférez-vous : blanc, rouge ou rosé ?",
        6: "Avez-vous une préférence pour un pays ou une région d'origine ?",
        7: "Quel est votre budget maximum en euros ?",
        8: "Quel profil d'arôme ou de goût recherchez-vous ?",
    },
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
    if handler.is_categorized_handoff_mode() and next_question_track == "dish":
        focused_questionnaire = mask_later_blank_fields(
            current_state.dish_questionnaire,
            DISH_FIELD_INDICES,
        )
    elif handler.is_categorized_handoff_mode() and next_question_track == "wine":
        wine_focused = mask_later_blank_fields(
            current_state.wine_questionnaire,
            WINE_FIELD_INDICES,
        )
        if current_state.intake_route == INTAKE_ROUTE_FOOD_PAIRING:
            focused_questionnaire = (
                current_state.dish_questionnaire + "\n" + wine_focused
            )
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
    next_question = await handler.question_asker.ask(
        focused_questionnaire,
        conversation_language,
        last_response=last_response,
    )
    normalized_question = next_question.strip()
    if handler.is_categorized_handoff_mode() and not _is_question_like(
        normalized_question
    ):
        fallback_question = _build_deterministic_fallback_question(
            current_state=current_state,
            updated_questionnaire=updated_questionnaire,
            next_question_track=next_question_track,
            conversation_language=conversation_language,
            questionnaire_indices=handler.questionnaire_indices,
        )
        if fallback_question is None:
            raise RuntimeError(
                "Question asker returned an invalid follow-up while intake still has "
                "unanswered fields."
            )
        logger.warning(
            "Question asker returned a non-question follow-up; using deterministic fallback."
        )
        normalized_question = fallback_question
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
    if "?" in normalized:
        return True
    lowered = normalized.lstrip(" .,!;:-").casefold()
    return lowered.startswith(_QUESTION_PREFIXES)


def _build_deterministic_fallback_question(
    *,
    current_state: ChatState,
    updated_questionnaire: str,
    next_question_track: str,
    conversation_language: str,
    questionnaire_indices: list[int],
) -> str | None:
    field_index: int | None
    if next_question_track == "dish":
        field_index = first_blank_field_index(
            current_state.dish_questionnaire,
            DISH_FIELD_INDICES,
        )
    elif next_question_track == "wine":
        field_index = first_blank_field_index(
            current_state.wine_questionnaire,
            WINE_FIELD_INDICES,
        )
    else:
        field_index = first_blank_field_index(
            updated_questionnaire, questionnaire_indices
        )
    if field_index is None:
        return None

    language_code = conversation_language.casefold().split("-")[0]
    localized = _FALLBACK_QUESTIONS.get(language_code) or _FALLBACK_QUESTIONS["en"]
    return localized.get(field_index)


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
