# ruff: noqa: S101,S105,S106
from __future__ import annotations

import asyncio

from pixie_questioner.chat.language import resolve_conversation_language
from pixie_questioner.chat.messages import (
    build_feedback_quick_replies,
    get_localized_options,
    normalize_feedback_response,
)
from pixie_questioner.chat.policy import ChatQuickReply
from pixie_questioner.chat.questionnaire import (
    ensure_initial_question_field,
    mask_later_blank_fields,
    questionnaire_complete,
)
from pixie_questioner.chat.relevance import LLMAnswerRelevanceClassifier
from pixie_questioner.chat.state_codec import (
    merge_partial_questionnaire_update,
    parse_state,
    serialize_state,
)
from pixie_questioner.chat.summary import summary_matches_conversation_language
from pixie_questioner.chat.types import FEEDBACK_NO, FEEDBACK_YES


class FakeLangCandidate:
    def __init__(self, lang: str, prob: float):
        self.lang = lang
        self.prob = prob


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    async def ainvoke(self, messages: list[dict[str, str]]):
        _ = messages
        return self.responses.pop(0)


def test_resolve_conversation_language_keeps_dutch_for_short_history(
    monkeypatch,
) -> None:
    def fake_detect_langs(_: str):
        return [FakeLangCandidate("en", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    conversation_language, recent_user_turns = resolve_conversation_language(
        current_answer="kurzer text",
        conversation_language="",
        recent_user_turns="",
        supported_language_set={"nl", "en", "fr"},
        supported_language_priority={"nl": 0, "en": 1, "fr": 2},
        supported_language_names={"nl": "Dutch", "en": "English", "fr": "French"},
        default_language_code="nl",
    )

    assert conversation_language == "nl"
    assert recent_user_turns == "kurzer text"


def test_resolve_conversation_language_switches_on_explicit_request(
    monkeypatch,
) -> None:
    def fake_detect_langs(_: str):
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    conversation_language, recent_user_turns = resolve_conversation_language(
        current_answer="Kun je in het Engels verdergaan?",
        conversation_language="nl",
        recent_user_turns="Hallo",
        supported_language_set={"nl", "en", "fr"},
        supported_language_priority={"nl": 0, "en": 1, "fr": 2},
        supported_language_names={"nl": "Dutch", "en": "English", "fr": "French"},
        default_language_code="nl",
    )

    assert conversation_language == "en"
    assert recent_user_turns == "Hallo\nKun je in het Engels verdergaan?"


def test_resolve_conversation_language_uses_recent_user_turn_history(
    monkeypatch,
) -> None:
    def fake_detect_langs(text: str):
        if "I am looking for dark blue jeans for work." in text:
            return [FakeLangCandidate("en", 0.99)]
        if text == "Yes":
            return [FakeLangCandidate("en", 0.99)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    conversation_language, recent_user_turns = resolve_conversation_language(
        current_answer="Yes",
        conversation_language="nl",
        recent_user_turns=(
            "I am looking for dark blue jeans for work.\n" "I prefer a straight fit."
        ),
        supported_language_set={"nl", "en", "fr"},
        supported_language_priority={"nl": 0, "en": 1, "fr": 2},
        supported_language_names={"nl": "Dutch", "en": "English", "fr": "French"},
        default_language_code="nl",
    )

    assert conversation_language == "en"
    assert recent_user_turns.endswith("Yes")


def test_messages_fallback_to_english_options() -> None:
    localized_values: dict[str, tuple[str, ...]] = {
        "en": ("English option",),
    }
    quick_replies: dict[str, tuple[ChatQuickReply, ...]] = {
        "en": (
            ChatQuickReply(label="Yes", value="yes"),
            ChatQuickReply(label="No", value="no"),
        )
    }

    assert get_localized_options(localized_values, "de") == ("English option",)
    assert build_feedback_quick_replies(quick_replies, "de") == [
        {"label": "Yes", "value": "yes"},
        {"label": "No", "value": "no"},
    ]


def test_normalize_feedback_response_yes_no_tokens() -> None:
    assert (
        normalize_feedback_response(
            current_answer="Ja zeker",
            feedback_yes_tokens=frozenset({"ja", "yes"}),
            feedback_no_tokens=frozenset({"nee", "no"}),
            feedback_yes_value=FEEDBACK_YES,
            feedback_no_value=FEEDBACK_NO,
        )
        == FEEDBACK_YES
    )
    assert (
        normalize_feedback_response(
            current_answer="Nee",
            feedback_yes_tokens=frozenset({"ja", "yes"}),
            feedback_no_tokens=frozenset({"nee", "no"}),
            feedback_yes_value=FEEDBACK_YES,
            feedback_no_value=FEEDBACK_NO,
        )
        == FEEDBACK_NO
    )


def test_questionnaire_complete_checks_for_remaining_placeholders() -> None:
    assert questionnaire_complete("0. Field: done") is True
    assert questionnaire_complete("0. Field: []") is False


def test_mask_later_blank_fields_keeps_only_first_placeholder() -> None:
    questionnaire = (
        "0. Initial user question: filled\n"
        "1. Occasion: []\n"
        "2. Color: []\n"
        "3. Fit: slim"
    )

    masked = mask_later_blank_fields(questionnaire, [0, 1, 2, 3])

    assert "1. Occasion: []" in masked
    assert "2. Color: [pending later]" in masked
    assert "3. Fit: slim" in masked


def test_ensure_initial_question_field_backfills_first_answer_once() -> None:
    questionnaire = "0. Initial user question: []\n" "1. Occasion, Vibe, Setting: []"

    updated = ensure_initial_question_field(
        questionnaire,
        "Ik zoek een donkerblauwe jeans",
    )
    unchanged = ensure_initial_question_field(updated, "Een ander antwoord")

    assert "0. Initial user question: Ik zoek een donkerblauwe jeans" in updated
    assert unchanged == updated


def test_summary_matches_conversation_language_accepts_short_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "pixie_questioner.chat.summary.detect_language_candidates",
        lambda _text: [("es", 0.99)],
    )

    assert (
        summary_matches_conversation_language(
            summary_text="hola jeans casual",
            conversation_language="nl",
            supported_language_set={"nl", "en", "fr"},
        )
        is True
    )


def test_summary_matches_conversation_language_rejects_long_supported_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "pixie_questioner.chat.summary.detect_language_candidates",
        lambda _text: [("en", 0.99)],
    )

    assert (
        summary_matches_conversation_language(
            summary_text="This customer is looking for dark blue work jeans.",
            conversation_language="nl",
            supported_language_set={"nl", "en", "fr"},
        )
        is False
    )


def test_state_codec_parse_and_serialize_roundtrip() -> None:
    parsed = parse_state(
        state={
            "questionnaire": "0. Initial user question: []",
            "summary": "Summary text",
            "last_question": "Vraag?",
            "phase": "awaiting_feedback",
            "feedback_status": "pending",
            "final_summary": "",
            "post_summary_language": "",
            "boundary_violation_count": "2",
            "handoff_categories_v1": '{"course_type":"hoofdgerecht"}',
            "intake_route": "food_pairing",
            "dish_follow_up_count": "1",
            "wine_follow_up_count": "2",
            "routing_question_asked": "true",
        },
        questionnaire_template="0. Initial user question: []",
        default_language="en",
    )

    serialized = serialize_state(parsed)

    assert serialized["phase"] == "awaiting_feedback"
    assert serialized["post_summary_language"] == "en"
    assert serialized["boundary_violation_count"] == "2"
    assert serialized["handoff_categories_v1"] == '{"course_type":"hoofdgerecht"}'
    assert serialized["intake_route"] == "food_pairing"
    assert serialized["dish_follow_up_count"] == "1"
    assert serialized["wine_follow_up_count"] == "2"
    assert serialized["routing_question_asked"] == "true"


def test_state_codec_defaults_new_cretskens_route_fields_for_legacy_state() -> None:
    parsed = parse_state(
        state={
            "questionnaire": "0. Initial user question: []",
            "summary": "Summary text",
            "last_question": "Vraag?",
        },
        questionnaire_template="0. Initial user question: []",
        default_language="nl",
    )

    serialized = serialize_state(parsed)

    assert serialized["intake_route"] == "unknown"
    assert serialized["dish_follow_up_count"] == "0"
    assert serialized["wine_follow_up_count"] == "0"
    assert serialized["routing_question_asked"] == "false"


def test_state_codec_preserves_conversation_language_and_recent_user_turns() -> None:
    parsed = parse_state(
        state={
            "questionnaire": "0. Initial user question: []",
            "summary": "Summary text",
            "last_question": "Vraag?",
            "conversation_language": "en",
            "recent_user_turns": (
                "Hello there\n" "I am looking for dark blue straight jeans for work."
            ),
        },
        questionnaire_template="0. Initial user question: []",
        default_language="nl",
    )

    serialized = serialize_state(parsed)

    assert serialized.get("conversation_language") == "en"
    assert serialized.get("recent_user_turns") == (
        "Hello there\nI am looking for dark blue straight jeans for work."
    )


def test_relevance_classifier_parses_labels_and_negation() -> None:
    off_topic = LLMAnswerRelevanceClassifier(
        llm=FakeLLM(["OFF_TOPIC"]),
        system_prompt_template="{questionnaire} {current_question} {current_answer}",
    )
    negated = LLMAnswerRelevanceClassifier(
        llm=FakeLLM(["This is NOT OFF_TOPIC."]),
        system_prompt_template="{questionnaire} {current_question} {current_answer}",
    )
    stop = LLMAnswerRelevanceClassifier(
        llm=FakeLLM(["STOP"]),
        system_prompt_template="{questionnaire} {current_question} {current_answer}",
    )

    assert (
        asyncio.run(
            off_topic.classify(
                current_question="Vraag",
                current_answer="Antwoord",
                questionnaire="Q",
            )
        )
        == "off_topic"
    )
    assert (
        asyncio.run(
            negated.classify(
                current_question="Vraag",
                current_answer="Antwoord",
                questionnaire="Q",
            )
        )
        == "on_topic"
    )
    assert (
        asyncio.run(
            stop.classify(
                current_question="Vraag",
                current_answer="Ik wil stoppen",
                questionnaire="Q",
            )
        )
        == "stop"
    )


# --- Field regression protection tests ---

_INDICES = [0, 1, 2]


def test_merge_full_replacement_preserves_filled_fields() -> None:
    """Full replacement (all indices valid) still protects filled fields."""
    current = (
        "0. Initial user question: Ik zoek een jeans\n"
        "1. Occasion: niets speciaals\n"
        "2. Preferences: comfortabel"
    )
    # LLM returns valid questionnaire but blanked field 0
    proposed = (
        "0. Initiële vraag: []\n"
        "1. Gelegenheid: niets speciaals\n"
        "2. Voorkeuren: comfortabel"
    )
    result = merge_partial_questionnaire_update(current, proposed, _INDICES)
    assert result is not None
    # Field 0 must NOT be [] — it was filled before
    assert "Ik zoek een jeans" in result


def test_merge_allows_updating_filled_field_with_new_value() -> None:
    """A filled field CAN be updated to a different non-blank value."""
    current = (
        "0. Initial user question: Ik zoek een jeans\n"
        "1. Occasion: dagelijks\n"
        "2. Preferences: []"
    )
    proposed = (
        "0. Initiële vraag: Ik zoek een jeans\n"
        "1. Gelegenheid: feestje\n"
        "2. Voorkeuren: casual"
    )
    result = merge_partial_questionnaire_update(current, proposed, _INDICES)
    assert result is not None
    assert "feestje" in result


def test_merge_partial_skips_blank_values() -> None:
    """Partial merge must not overwrite a filled field with []."""
    current = (
        "0. Initial user question: Ik zoek een jeans\n"
        "1. Occasion: dagelijks\n"
        "2. Preferences: []"
    )
    # LLM returns only field 2, with field 0 blanked (partial output)
    proposed = "0. Vraag: []\n2. Voorkeuren: casual"
    result = merge_partial_questionnaire_update(current, proposed, _INDICES)
    assert result is not None
    assert "Ik zoek een jeans" in result
    assert "casual" in result


def test_merge_prevents_regression_to_empty_string() -> None:
    """A field reverted to an empty value (no []) must also be protected."""
    current = (
        "0. Initial user question: Ik zoek een jeans\n"
        "1. Occasion: dagelijks\n"
        "2. Preferences: comfortabel"
    )
    proposed = (
        "0. Initiële vraag: \n"
        "1. Gelegenheid: dagelijks\n"
        "2. Voorkeuren: comfortabel"
    )
    result = merge_partial_questionnaire_update(current, proposed, _INDICES)
    assert result is not None
    assert "Ik zoek een jeans" in result


def test_merge_prevents_regression_to_bracket_variants() -> None:
    """Bracket-like blank values ([ ], [ - ]) must also be caught."""
    current = (
        "0. Initial user question: Ik zoek een jeans\n"
        "1. Occasion: dagelijks\n"
        "2. Preferences: comfortabel"
    )
    proposed = (
        "0. Initiële vraag: [ ]\n"
        "1. Gelegenheid: dagelijks\n"
        "2. Voorkeuren: comfortabel"
    )
    result = merge_partial_questionnaire_update(current, proposed, _INDICES)
    assert result is not None
    assert "Ik zoek een jeans" in result
