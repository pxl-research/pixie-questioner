import asyncio
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from pixie_questioner.chat.policy import ChatDomainPolicy, ChatQuickReply
from pixie_questioner.chat.relevance import (
    ClassificationLabel,
    LLMAnswerRelevanceClassifier,
)
from pixie_questioner.chat_service import ChatService
from pixie_questioner.config import load_domain_profile
from pixie_questioner.notulist import QuestionnaireNotulist
from pixie_questioner.summarizer import ConversationSummarizer


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        await asyncio.sleep(0)
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("No responses left for FakeLLM.")
        return self.responses.pop(0)


class RaisingLLM:
    def __init__(self, exception: Exception):
        self.exception = exception
        self.calls = []

    async def ainvoke(self, messages):
        await asyncio.sleep(0)
        self.calls.append(messages)
        raise self.exception


class FakeLangCandidate:
    def __init__(self, lang: str, prob: float):
        self.lang = lang
        self.prob = prob


class StubRelevanceClassifier:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def classify(
        self, current_question: str, current_answer: str, questionnaire: str
    ) -> ClassificationLabel:
        await asyncio.sleep(0)
        self.calls.append((current_question, current_answer, questionnaire))
        if not self.responses:
            raise AssertionError(
                "No relevance responses left for StubRelevanceClassifier."
            )
        return self.responses.pop(0)


def _make_notulist(llm):
    return QuestionnaireNotulist(
        llm=llm,
        system_prompt_template="Q:{current_question}\nA:{current_answer}\nS:{summary}",
    )


def _make_summarizer(llm):
    return ConversationSummarizer(
        llm=llm,
        system_prompt_template="Q:{current_question}\nA:{current_answer}\nS:{summary}",
    )


QUESTIONNAIRE_TEMPLATE = "0. Initial user question: []\n1. Occasion, Vibe, Setting: []"

_DEFAULT_PROFILE = load_domain_profile(Path("profiles/default/domain_profile.yml"))
_DEFAULT_POLICY = _DEFAULT_PROFILE.chat_domain_policy
_CLASSIFIER_PROMPT = _DEFAULT_PROFILE.relevance_classifier_prompt
HANDOFF_NL = _DEFAULT_POLICY.handoff_lines["nl"]
HANDOFF_EN = _DEFAULT_POLICY.handoff_lines["en"]
HANDOFF_FR = _DEFAULT_POLICY.handoff_lines["fr"]
FEEDBACK_QUESTION_NL = _DEFAULT_POLICY.feedback_questions["nl"]
FEEDBACK_QUESTION_EN = _DEFAULT_POLICY.feedback_questions["en"]
FEEDBACK_QUESTION_FR = _DEFAULT_POLICY.feedback_questions["fr"]
FEEDBACK_THANKS_NL = _DEFAULT_POLICY.feedback_thanks["nl"]
GOODBYE_NL = _DEFAULT_POLICY.goodbye_lines["nl"]
GOODBYE_EN = _DEFAULT_POLICY.goodbye_lines["en"]
GOODBYE_FR = _DEFAULT_POLICY.goodbye_lines["fr"]
OFF_TOPIC_REDIRECT_NL = _DEFAULT_POLICY.off_topic_redirects["nl"]


def _replace_policy(**changes) -> ChatDomainPolicy:
    return replace(_DEFAULT_POLICY, **changes)


def _make_service_for_language_detection() -> ChatService:
    return ChatService(
        llm=FakeLLM([]),
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(FakeLLM([])),
        summarizer=_make_summarizer(FakeLLM([])),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )


def test_ask_updates_state_and_returns_next_question():
    question_llm = FakeLLM(["Next question"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Updated answer\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Updated summary"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_token="TOKEN"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="User answer",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Next question"
    assert (
        result.state["questionnaire"]
        == "0. Initial user question: Updated answer\n1. Occasion, Vibe, Setting: []"
    )
    assert result.state["summary"] == "Updated summary"
    assert result.state["last_question"] == "Next question"
    assert result.state["phase"] == "intake"
    assert result.state["feedback_status"] == "pending"
    assert result.state["final_summary"] == ""
    assert result.state["post_summary_language"] == ""
    assert result.state["boundary_violation_count"] == "0"
    assert result.summary is None
    assert result.quick_replies is None
    assert (
        "0. Initial user question: Updated answer\n1. Occasion, Vibe, Setting: []"
        in question_llm.calls[-1][0]["content"]
    )
    assert "AI question" in notulist_llm.calls[0][0]["content"]
    assert "User answer" in notulist_llm.calls[0][0]["content"]
    assert "Prev []" in notulist_llm.calls[0][0]["content"]
    assert "Language policy:" in question_llm.calls[-1][0]["content"]
    assert "If uncertain, default to Dutch." in question_llm.calls[-1][0]["content"]


def test_ask_prefers_dutch_for_hallo_greeting():
    question_llm = FakeLLM(["Volgende vraag"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Hallo\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Hallo"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Hallo",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Initial question:",
            },
        )
    )

    assert result.response_type == "answer"
    assert "Respond in Dutch." in question_llm.calls[-1][0]["content"]


def test_ask_prefers_german_for_clear_german_sentence():
    question_llm = FakeLLM(["Naechste Frage"])
    notulist_llm = FakeLLM(
        [
            "0. Initial user question: Hallo, ich suche eine neue Jeans.\n1. Occasion, Vibe, Setting: []"
        ]
    )
    summarizer_llm = FakeLLM(["Hallo, ich suche eine neue Jeans."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Hallo, ich suche eine neue Jeans.",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Initial question:",
            },
        )
    )

    assert result.response_type == "answer"
    assert "ISO code 'de'" in question_llm.calls[-1][0]["content"]


def test_detect_preferred_language_prefers_supported_on_small_score_gap(monkeypatch):
    service = _make_service_for_language_detection()

    def fake_detect_langs(_: str):
        return [FakeLangCandidate("de", 0.56), FakeLangCandidate("nl", 0.49)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    preferred_language = service._language.detect_preferred("kurzer text", "")

    assert preferred_language == "nl"


def test_detect_preferred_language_allows_other_language_on_strong_signal(monkeypatch):
    service = _make_service_for_language_detection()

    def fake_detect_langs(_: str):
        return [FakeLangCandidate("de", 0.96), FakeLangCandidate("nl", 0.02)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    preferred_language = service._language.detect_preferred(
        "guten tag ich suche eine jeans", ""
    )

    assert preferred_language == "de"


def test_detect_preferred_language_falls_back_to_dutch_for_short_unsupported_text(
    monkeypatch,
):
    service = _make_service_for_language_detection()

    def fake_detect_langs(_: str):
        return [FakeLangCandidate("fi", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    preferred_language = service._language.detect_preferred("Hallo", "")

    assert preferred_language == "nl"


def test_detect_preferred_language_keeps_dutch_for_short_supported_non_dutch_signal(
    monkeypatch,
):
    service = _make_service_for_language_detection()

    def fake_detect_langs(text: str):
        if text == "Graag":
            return [FakeLangCandidate("en", 0.99)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    preferred_language = service._language.detect_preferred("Graag", "")

    assert preferred_language == "nl"


def test_detect_preferred_language_uses_summary_when_short_answer_is_uncertain(
    monkeypatch,
):
    service = _make_service_for_language_detection()

    def fake_detect_langs(text: str):
        if text == "Yes":
            return [FakeLangCandidate("en", 0.99)]
        if text == "I am looking for dark blue straight jeans for work.":
            return [FakeLangCandidate("en", 0.97)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    preferred_language = service._language.detect_preferred(
        "Yes", "I am looking for dark blue straight jeans for work."
    )

    assert preferred_language == "en"


def test_ask_returns_summary_when_questionnaire_complete():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Filled\n1. Occasion, Vibe, Setting: Filled"]
    )
    summarizer_llm = FakeLLM(["Eind samenvatting"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="User answer",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert any(result.text.startswith(line) for line in HANDOFF_NL)
    assert result.follow_up_question in FEEDBACK_QUESTION_NL
    assert result.summary == "Eind samenvatting"
    assert result.state["phase"] == "awaiting_feedback"
    assert result.state["feedback_status"] == "pending"
    assert result.state["final_summary"] == "Eind samenvatting"
    assert result.state["post_summary_language"] == "nl"
    assert result.state["boundary_violation_count"] == "0"
    assert result.quick_replies == [
        {"label": "Ja", "value": "yes"},
        {"label": "Nee", "value": "no"},
    ]
    assert question_llm.calls == []


def test_ask_enters_feedback_flow_on_stop_classification():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Hulp\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Klant wil stoppen."])
    relevance_classifier = StubRelevanceClassifier(["stop"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
        answer_relevance_classifier=relevance_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik wil stoppen",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text in [*HANDOFF_NL, *HANDOFF_EN, *HANDOFF_FR]
    assert result.follow_up_question is not None
    assert result.summary == "Klant wil stoppen."
    assert result.state["phase"] == "awaiting_feedback"
    assert result.state["feedback_status"] == "pending"
    assert result.state["boundary_violation_count"] == "0"
    assert result.quick_replies is not None
    assert len(question_llm.calls) == 0
    assert len(notulist_llm.calls) == 1
    assert len(summarizer_llm.calls) == 1
    assert len(relevance_classifier.calls) == 1


def test_ask_stop_classification_runs_notulist_and_summarizer():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "0. Initial user question: Slim fit blauw\n"
            "1. Occasion, Vibe, Setting: [casual]"
        ]
    )
    summarizer_llm = FakeLLM(
        ["Klant zoekt een casual slim fit blauwe jeans en wil stoppen."]
    )
    relevance_classifier = StubRelevanceClassifier(["stop"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
        answer_relevance_classifier=relevance_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Dat is alles, gewoon een blauwe slim fit",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welke pasvorm zoek je?",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text in [*HANDOFF_NL, *HANDOFF_EN, *HANDOFF_FR]
    assert result.summary is not None
    assert result.state["phase"] == "awaiting_feedback"
    assert len(notulist_llm.calls) == 1
    assert len(summarizer_llm.calls) == 1


def test_ask_skips_relevance_classifier_for_initial_question_placeholder():
    question_llm = FakeLLM(["Voor welke gelegenheid zoek je een jeans?"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Graag hulp\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["De klant wil hulp bij het kiezen van een jeans."])
    relevance_classifier = StubRelevanceClassifier(["off_topic"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
        answer_relevance_classifier=relevance_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Graag",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Initial question:",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Voor welke gelegenheid zoek je een jeans?"
    assert result.state["last_question"] == "Voor welke gelegenheid zoek je een jeans?"
    assert result.state["boundary_violation_count"] == "0"
    assert len(relevance_classifier.calls) == 0
    assert len(notulist_llm.calls) == 1
    assert len(summarizer_llm.calls) == 1


def test_ask_terminates_after_repeated_boundary_violations():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    relevance_classifier = StubRelevanceClassifier(["off_topic", "off_topic"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
        answer_relevance_classifier=relevance_classifier,
    )
    current_question = "Voor welke gelegenheid zoek je een jeans?"
    expected_redirect = OFF_TOPIC_REDIRECT_NL[0].format(question=current_question)

    first = asyncio.run(
        service.ask(
            current_answer="Ik moet naar het toilet.",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": current_question,
            },
        )
    )
    second = asyncio.run(
        service.ask(
            current_answer="Ik wil geen broek, enkel toilet.", state=first.state
        )
    )

    assert first.response_type == "answer"
    assert first.text == expected_redirect
    assert first.state["boundary_violation_count"] == "1"
    assert first.state["phase"] == "intake"
    assert first.summary is None
    assert first.follow_up_question is None
    assert first.quick_replies is None

    assert second.response_type == "answer"
    assert second.text in [*GOODBYE_NL, *GOODBYE_EN, *GOODBYE_FR]
    assert second.summary is None
    assert second.follow_up_question is None
    assert second.quick_replies is None
    assert second.state["phase"] == "terminated"
    assert second.state["boundary_violation_count"] == "2"

    assert len(question_llm.calls) == 0
    assert len(notulist_llm.calls) == 0
    assert len(summarizer_llm.calls) == 0
    assert len(relevance_classifier.calls) == 2


def test_ask_uses_dutch_off_topic_redirect_for_short_uncertain_answer(monkeypatch):
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    relevance_classifier = StubRelevanceClassifier(["off_topic"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
        answer_relevance_classifier=relevance_classifier,
    )
    current_question = "Voor welke gelegenheid zoek je een jeans?"
    expected_redirect = OFF_TOPIC_REDIRECT_NL[0].format(question=current_question)

    def fake_detect_langs(text: str):
        if text == "Graag":
            return [FakeLangCandidate("en", 0.99)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    result = asyncio.run(
        service.ask(
            current_answer="Graag",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": current_question,
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == expected_redirect
    assert result.state["boundary_violation_count"] == "1"
    assert result.state["phase"] == "intake"
    assert len(relevance_classifier.calls) == 1
    assert len(notulist_llm.calls) == 0
    assert len(summarizer_llm.calls) == 0


def test_llm_answer_relevance_classifier_parses_off_topic_label():
    classifier = LLMAnswerRelevanceClassifier(
        llm=FakeLLM(["OFF_TOPIC"]),
        system_prompt_template=_CLASSIFIER_PROMPT,
    )

    result = asyncio.run(
        classifier.classify(
            current_question="Voor welke gelegenheid zoek je een jeans?",
            current_answer="Ik moet naar het toilet",
            questionnaire=QUESTIONNAIRE_TEMPLATE,
        )
    )

    assert result == "off_topic"


def test_llm_answer_relevance_classifier_parses_stop_label():
    classifier = LLMAnswerRelevanceClassifier(
        llm=FakeLLM(["STOP"]),
        system_prompt_template=_CLASSIFIER_PROMPT,
    )

    result = asyncio.run(
        classifier.classify(
            current_question="Welke pasvorm zoek je?",
            current_answer="Ik wil stoppen",
            questionnaire=QUESTIONNAIRE_TEMPLATE,
        )
    )

    assert result == "stop"


def test_llm_answer_relevance_classifier_defaults_to_on_topic_when_label_missing():
    classifier = LLMAnswerRelevanceClassifier(
        llm=FakeLLM(["uncertain"]),
        system_prompt_template=_CLASSIFIER_PROMPT,
    )

    result = asyncio.run(
        classifier.classify(
            current_question="Welke pasvorm zoek je?",
            current_answer="Misschien slim fit",
            questionnaire=QUESTIONNAIRE_TEMPLATE,
        )
    )

    assert result == "on_topic"


def test_llm_answer_relevance_classifier_defaults_to_on_topic_when_labels_conflict():
    classifier = LLMAnswerRelevanceClassifier(
        llm=FakeLLM(["This is not OFF_TOPIC. ON_TOPIC."]),
        system_prompt_template=_CLASSIFIER_PROMPT,
    )

    result = asyncio.run(
        classifier.classify(
            current_question="Kan ik je helpen bij het vinden van een jeans?",
            current_answer="Ja graag",
            questionnaire=QUESTIONNAIRE_TEMPLATE,
        )
    )

    assert result == "on_topic"


def test_llm_answer_relevance_classifier_defaults_to_on_topic_for_negated_off_topic():
    classifier = LLMAnswerRelevanceClassifier(
        llm=FakeLLM(["The answer is NOT OFF_TOPIC."]),
        system_prompt_template=_CLASSIFIER_PROMPT,
    )

    result = asyncio.run(
        classifier.classify(
            current_question="Kan ik je helpen bij het vinden van een jeans?",
            current_answer="Graag",
            questionnaire=QUESTIONNAIRE_TEMPLATE,
        )
    )

    assert result == "on_topic"


def test_ask_stores_feedback_response_in_state():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ja, zeker",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "Eind samenvatting",
                "last_question": "AI question",
                "phase": "awaiting_feedback",
                "feedback_status": "pending",
                "final_summary": "Eind samenvatting",
                "post_summary_language": "nl",
            },
        )
    )

    assert result.response_type == "summary"
    assert result.text in FEEDBACK_THANKS_NL
    assert result.follow_up_question is None
    assert result.state["phase"] == "completed"
    assert result.state["feedback_status"] == "yes"
    assert result.state["final_summary"] == "Eind samenvatting"
    assert result.state["post_summary_language"] == "nl"
    assert result.summary == "Eind samenvatting"
    assert result.quick_replies is None
    assert question_llm.calls == []
    assert notulist_llm.calls == []
    assert summarizer_llm.calls == []


def test_ask_repeats_feedback_question_for_unclear_feedback_response():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Misschien",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "Eind samenvatting",
                "last_question": "AI question",
                "phase": "awaiting_feedback",
                "feedback_status": "pending",
                "final_summary": "Eind samenvatting",
                "post_summary_language": "nl",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text in FEEDBACK_QUESTION_NL
    assert result.follow_up_question is None
    assert result.state["phase"] == "awaiting_feedback"
    assert result.state["feedback_status"] == "pending"
    assert result.state["final_summary"] == "Eind samenvatting"
    assert result.state["post_summary_language"] == "nl"
    assert result.summary == "Eind samenvatting"
    assert result.quick_replies == [
        {"label": "Ja", "value": "yes"},
        {"label": "Nee", "value": "no"},
    ]
    assert question_llm.calls == []
    assert notulist_llm.calls == []
    assert summarizer_llm.calls == []


def test_ask_keeps_final_summary_immutable_while_feedback_pending():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )
    pending_state = {
        "questionnaire": QUESTIONNAIRE_TEMPLATE,
        "summary": "Deze mag niet gebruikt worden",
        "last_question": "AI question",
        "phase": "awaiting_feedback",
        "feedback_status": "pending",
        "final_summary": "Vastgepinde samenvatting",
        "post_summary_language": "nl",
    }

    first = asyncio.run(service.ask(current_answer="Misschien", state=pending_state))
    second = asyncio.run(service.ask(current_answer="Onduidelijk", state=first.state))

    assert first.response_type == "answer"
    assert first.summary == "Vastgepinde samenvatting"
    assert first.quick_replies == [
        {"label": "Ja", "value": "yes"},
        {"label": "Nee", "value": "no"},
    ]
    assert second.response_type == "answer"
    assert second.summary == "Vastgepinde samenvatting"
    assert second.quick_replies == [
        {"label": "Ja", "value": "yes"},
        {"label": "Nee", "value": "no"},
    ]
    assert question_llm.calls == []
    assert notulist_llm.calls == []
    assert summarizer_llm.calls == []


def test_ask_uses_locked_post_summary_language_for_feedback_retry():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Maybe",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "English summary should not change language lock.",
                "last_question": "AI question",
                "phase": "awaiting_feedback",
                "feedback_status": "pending",
                "final_summary": "English summary should not change language lock.",
                "post_summary_language": "fr",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text in FEEDBACK_QUESTION_FR
    assert result.state["post_summary_language"] == "fr"
    assert result.quick_replies == [
        {"label": "Oui", "value": "yes"},
        {"label": "Non", "value": "no"},
    ]
    assert question_llm.calls == []
    assert notulist_llm.calls == []
    assert summarizer_llm.calls == []


def test_ask_returns_completed_feedback_ack_when_phase_completed():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="extra bericht",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "Samenvatting",
                "last_question": "AI question",
                "phase": "completed",
                "feedback_status": "yes",
                "final_summary": "Samenvatting",
                "post_summary_language": "nl",
            },
        )
    )

    assert result.response_type == "summary"
    assert result.text in FEEDBACK_THANKS_NL
    assert result.summary == "Samenvatting"
    assert result.follow_up_question is None
    assert result.quick_replies is None
    assert result.state["phase"] == "completed"
    assert result.state["feedback_status"] == "yes"
    assert question_llm.calls == []
    assert notulist_llm.calls == []
    assert summarizer_llm.calls == []


def test_ask_returns_english_handoff_for_english_summary():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Filled\n1. Occasion, Vibe, Setting: Filled"]
    )
    summarizer_llm = FakeLLM(["I am looking for blue jeans with a straight fit."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="User answer",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert any(result.text.startswith(line) for line in HANDOFF_EN)
    assert result.follow_up_question in FEEDBACK_QUESTION_EN
    assert result.state["post_summary_language"] == "en"
    assert result.quick_replies == [
        {"label": "Yes", "value": "yes"},
        {"label": "No", "value": "no"},
    ]
    assert question_llm.calls == []


def test_ask_uses_custom_handoff_line_from_domain_policy():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Filled\n1. Occasion, Vibe, Setting: Filled"]
    )
    summarizer_llm = FakeLLM(["I need new jeans."])
    custom_policy = _replace_policy(
        handoff_lines={
            "en": ("Custom handoff line",),
        },
        feedback_questions={
            "en": ("Custom feedback question",),
        },
        quick_replies={
            "en": (
                ChatQuickReply(label="Yes", value="yes"),
                ChatQuickReply(label="No", value="no"),
            )
        },
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=custom_policy,
    )

    result = asyncio.run(
        service.ask(
            current_answer="User answer",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Custom handoff line"
    assert result.follow_up_question == "Custom feedback question"
    assert result.quick_replies == [
        {"label": "Yes", "value": "yes"},
        {"label": "No", "value": "no"},
    ]


def test_ask_respects_custom_off_topic_termination_threshold():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    relevance_classifier = StubRelevanceClassifier(
        ["off_topic", "off_topic", "off_topic"]
    )
    custom_policy = _replace_policy(off_topic_termination_threshold=3)
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        answer_relevance_classifier=relevance_classifier,
        domain_policy=custom_policy,
    )
    current_question = "Voor welke gelegenheid zoek je een jeans?"

    first = asyncio.run(
        service.ask(
            current_answer="Off topic 1",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": current_question,
            },
        )
    )
    second = asyncio.run(service.ask(current_answer="Off topic 2", state=first.state))
    third = asyncio.run(service.ask(current_answer="Off topic 3", state=second.state))

    assert first.state["phase"] == "intake"
    assert first.state["boundary_violation_count"] == "1"
    assert second.state["phase"] == "intake"
    assert second.state["boundary_violation_count"] == "2"
    assert third.state["phase"] == "terminated"
    assert third.state["boundary_violation_count"] == "3"


def test_ask_falls_back_to_english_for_missing_extra_language_copy():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    custom_policy = _replace_policy(
        language_codes=("nl", "en", "fr", "de"),
        language_names={"nl": "Dutch", "en": "English", "fr": "French", "de": "German"},
        feedback_questions={
            "en": ("Would you use this assistant again? (yes/no)",),
        },
        quick_replies={
            "en": (
                ChatQuickReply(label="Yes", value="yes"),
                ChatQuickReply(label="No", value="no"),
            ),
        },
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=custom_policy,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Vielleicht",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "Zusammenfassung",
                "last_question": "AI question",
                "phase": "awaiting_feedback",
                "feedback_status": "pending",
                "final_summary": "Zusammenfassung",
                "post_summary_language": "de",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Would you use this assistant again? (yes/no)"
    assert result.quick_replies == [
        {"label": "Yes", "value": "yes"},
        {"label": "No", "value": "no"},
    ]


def test_ask_returns_french_handoff_for_french_summary():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Filled\n1. Occasion, Vibe, Setting: Filled"]
    )
    summarizer_llm = FakeLLM(["Je cherche un jean bleu avec une coupe droite."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="User answer",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert any(result.text.startswith(line) for line in HANDOFF_FR)
    assert result.follow_up_question in FEEDBACK_QUESTION_FR
    assert result.state["post_summary_language"] == "fr"
    assert result.quick_replies == [
        {"label": "Oui", "value": "yes"},
        {"label": "Non", "value": "no"},
    ]
    assert question_llm.calls == []


def test_ask_falls_back_to_dutch_handoff_when_language_is_undetectable():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Filled\n1. Occasion, Vibe, Setting: Filled"]
    )
    summarizer_llm = FakeLLM(["12345 !!!"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="User answer",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert any(result.text.startswith(line) for line in HANDOFF_NL)
    assert result.follow_up_question in FEEDBACK_QUESTION_NL
    assert result.state["post_summary_language"] == "nl"
    assert result.quick_replies == [
        {"label": "Ja", "value": "yes"},
        {"label": "Nee", "value": "no"},
    ]
    assert question_llm.calls == []


def test_ask_strips_completion_token_from_next_question():
    question_llm = FakeLLM(["Next question TOKEN"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Updated answer\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Updated summary"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_token="TOKEN"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="User answer",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.text == "Next question"


def test_ask_simplifies_dutch_technical_terms_in_next_question():
    question_llm = FakeLLM(["Welke Snit en silhouet zoek je in indigo?"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Updated answer\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Updated summary"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek een jeans",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke Pasvorm en vorm zoek je in donkerblauw?"


def test_ask_seeds_questionnaire_from_template_when_state_empty():
    question_llm = FakeLLM(["Next question"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Updated answer\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Updated summary"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="User answer",
            state={},
        )
    )

    assert result.response_type == "answer"
    assert QUESTIONNAIRE_TEMPLATE in notulist_llm.calls[0][0]["content"]


def test_ask_keeps_previous_questionnaire_on_invalid_notulist_output():
    question_llm = FakeLLM(["Next question"])
    notulist_llm = FakeLLM(["De gebruiker zoekt een nieuwe broek."])
    summarizer_llm = FakeLLM(["Updated summary"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek een nieuwe broek",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.state["questionnaire"] == QUESTIONNAIRE_TEMPLATE


def test_ask_raises_on_invalid_state_types():
    question_llm = FakeLLM(["Next question"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Updated answer\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Updated summary"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    with pytest.raises(RuntimeError, match="summary"):
        asyncio.run(
            service.ask(
                current_answer="User answer",
                state=cast(
                    dict[str, str],
                    {
                        "questionnaire": QUESTIONNAIRE_TEMPLATE,
                        "summary": 123,
                        "last_question": "AI question",
                    },
                ),
            )
        )


def test_ask_wraps_notulist_invocation_error_with_cause():
    question_llm = FakeLLM(["Next question"])
    notulist_llm = RaisingLLM(RuntimeError("upstream notulist failure"))
    summarizer_llm = FakeLLM(["Updated summary"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    with pytest.raises(RuntimeError, match="Failed to update questionnaire.") as exc:
        asyncio.run(
            service.ask(
                current_answer="User answer",
                state={
                    "questionnaire": QUESTIONNAIRE_TEMPLATE,
                    "summary": "",
                    "last_question": "AI question",
                },
            )
        )

    assert exc.value.__cause__ is not None
    assert "Questionnaire notulist invocation failed." in str(exc.value.__cause__)
    assert exc.value.__cause__.__cause__ is not None
    assert "upstream notulist failure" in str(exc.value.__cause__.__cause__)


def test_ask_wraps_summarizer_invocation_error_with_cause():
    question_llm = FakeLLM(["Next question"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Updated answer\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = RaisingLLM(RuntimeError("upstream summarizer failure"))
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    with pytest.raises(RuntimeError, match="Failed to summarize conversation.") as exc:
        asyncio.run(
            service.ask(
                current_answer="User answer",
                state={
                    "questionnaire": QUESTIONNAIRE_TEMPLATE,
                    "summary": "",
                    "last_question": "AI question",
                },
            )
        )

    assert exc.value.__cause__ is not None
    assert "Conversation summarizer invocation failed." in str(exc.value.__cause__)
    assert exc.value.__cause__.__cause__ is not None
    assert "upstream summarizer failure" in str(exc.value.__cause__.__cause__)
