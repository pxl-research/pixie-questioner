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
_FALLBACK_FIELD_QUESTIONS = {
    "nl": {
        0: "Welk gerecht of welke maaltijd wil je met wijn combineren?",
        1: "Gaat het om aperitief, hoofdgerecht, kaas of dessert?",
        2: "Welk hoofdingrediënt zit er in het gerecht, zoals vis, gevogelte, rood vlees of wild?",
        3: "Hoe wordt het gerecht bereid, bijvoorbeeld gegrild/BBQ of als stoofgerecht?",
        4: "Om welk type kaas gaat het: blauw, zacht of hard?",
        5: "Welke kleur wijn heb je in gedachten: wit, rood of rosé?",
        6: "Uit welk land of welke regio mag de wijn komen?",
        7: "Wat is je maximum budget in euro?",
        8: "Welke geur- of smaakstijl zoek je? Bijvoorbeeld citrus of rood fruit in de geur, en droog of fris in de smaak. Je mag ook zeggen wat je niet wil, zoals niet zoet of geen houttoetsen.",
    },
    "en": {
        0: "Which dish or meal would you like to pair with wine?",
        1: "Is it for an aperitif, main course, cheese course, or dessert?",
        2: "What is the main protein in the dish, such as fish, poultry, red meat, or game?",
        3: "How is the dish prepared, for example grilled/BBQ or stewed?",
        4: "What type of cheese is it: blue, soft, or hard?",
        5: "Which wine color do you have in mind: white, red, or rosé?",
        6: "Which country or region should the wine come from?",
        7: "What is your maximum budget in euro?",
        8: "What aroma or taste style are you looking for? For example citrus or red fruit on the nose, and dry or fresh on the palate. You can also say what you do not want, like not sweet or no oak notes.",
    },
    "fr": {
        0: "Quel plat ou quel repas souhaitez-vous accorder avec un vin ?",
        1: "Est-ce pour un aperitif, un plat principal, un plateau de fromages ou un dessert ?",
        2: "Quelle est la proteine principale du plat, par exemple poisson, volaille, viande rouge ou gibier ?",
        3: "Comment le plat est-il prepare, par exemple grille/BBQ ou mijote ?",
        4: "Quel type de fromage est-ce : bleu, doux ou a pate dure ?",
        5: "Quelle couleur de vin preferez-vous : blanc, rouge ou rose ?",
        6: "De quel pays ou de quelle region doit venir le vin ?",
        7: "Quel est votre budget maximum en euro ?",
        8: "Quel style d'aromes ou de saveurs recherchez-vous ? Par exemple des notes d'agrumes ou de fruits rouges au nez, et un style sec ou frais en bouche. Vous pouvez aussi dire ce que vous ne voulez pas, comme pas sucre ou sans notes boisees.",
    },
}
_FALLBACK_GENERIC_QUESTIONS = {
    "nl": "Kun je daar nog iets meer over vertellen?",
    "en": "Could you tell me a bit more about that?",
    "fr": "Pouvez-vous en dire un peu plus ?",
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
    return normalized.endswith(("?", "？"))


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
    field_questions = _FALLBACK_FIELD_QUESTIONS.get(
        language_code,
        _FALLBACK_FIELD_QUESTIONS["en"],
    )
    field_question = field_questions.get(field_index)
    if field_question is not None:
        return field_question
    return _FALLBACK_GENERIC_QUESTIONS.get(
        language_code,
        _FALLBACK_GENERIC_QUESTIONS["en"],
    )


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
