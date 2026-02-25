from __future__ import annotations

import asyncio

from pixie_questioner.chat.language import detect_preferred_language
from pixie_questioner.chat.messages import (
    build_feedback_quick_replies,
    get_localized_options,
    normalize_feedback_response,
)
from pixie_questioner.chat.policy import ChatQuickReply
from pixie_questioner.chat.relevance import LLMAnswerRelevanceClassifier
from pixie_questioner.chat.state_codec import parse_state, serialize_state
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


def test_detect_preferred_language_prefers_supported_on_small_score_gap(
    monkeypatch,
) -> None:
    def fake_detect_langs(_: str):
        return [FakeLangCandidate("de", 0.56), FakeLangCandidate("nl", 0.49)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    preferred = detect_preferred_language(
        current_answer="kurzer text",
        previous_summary="",
        supported_language_set={"nl", "en", "fr"},
        supported_language_priority={"nl": 0, "en": 1, "fr": 2},
        default_language_code="nl",
    )

    assert preferred == "nl"


def test_detect_preferred_language_allows_other_language_on_strong_signal(
    monkeypatch,
) -> None:
    def fake_detect_langs(_: str):
        return [FakeLangCandidate("de", 0.96), FakeLangCandidate("nl", 0.02)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    preferred = detect_preferred_language(
        current_answer="guten tag ich suche eine jeans",
        previous_summary="",
        supported_language_set={"nl", "en", "fr"},
        supported_language_priority={"nl": 0, "en": 1, "fr": 2},
        default_language_code="nl",
    )

    assert preferred == "de"


def test_messages_fallback_to_english_options() -> None:
    localized_values = {
        "en": ("English option",),
    }
    quick_replies = {
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
        },
        questionnaire_template="0. Initial user question: []",
        detect_language=lambda _text: "en",
    )

    serialized = serialize_state(parsed)

    assert serialized["phase"] == "awaiting_feedback"
    assert serialized["post_summary_language"] == "en"
    assert serialized["boundary_violation_count"] == "2"


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
