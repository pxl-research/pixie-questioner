import asyncio
from typing import cast

import pytest

from pixie_questioner.chat_service import (
    FEEDBACK_QUESTION_EN,
    FEEDBACK_QUESTION_FR,
    FEEDBACK_QUESTION_NL,
    FEEDBACK_THANKS_NL,
    HANDOFF_EN,
    HANDOFF_FR,
    HANDOFF_NL,
    ChatService,
)
from pixie_questioner.notulist import QuestionnaireNotulist
from pixie_questioner.summarizer import ConversationSummarizer


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("No responses left for FakeLLM.")
        return self.responses.pop(0)


class FakeLangCandidate:
    def __init__(self, lang: str, prob: float):
        self.lang = lang
        self.prob = prob


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


QUESTIONNAIRE_TEMPLATE = (
    "0. Initial user question: []\n" "1. Occasion, Vibe, Setting: []"
)


def _make_service_for_language_detection() -> ChatService:
    return ChatService(
        llm=FakeLLM([]),
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(FakeLLM([])),
        summarizer=_make_summarizer(FakeLLM([])),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
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
        completion_token="TOKEN",
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
    assert result.state["questionnaire"] == (
        "0. Initial user question: Updated answer\n" "1. Occasion, Vibe, Setting: []"
    )
    assert result.state["summary"] == "Updated summary"
    assert result.state["last_question"] == "Next question"
    assert result.state["phase"] == "intake"
    assert result.state["feedback_status"] == "pending"
    assert result.state["final_summary"] == ""
    assert result.state["post_summary_language"] == ""
    assert result.summary is None
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

    monkeypatch.setattr("pixie_questioner.chat_service.detect_langs", fake_detect_langs)

    preferred_language = service._detect_preferred_language("kurzer text", "")

    assert preferred_language == "nl"


def test_detect_preferred_language_allows_other_language_on_strong_signal(monkeypatch):
    service = _make_service_for_language_detection()

    def fake_detect_langs(_: str):
        return [FakeLangCandidate("de", 0.96), FakeLangCandidate("nl", 0.02)]

    monkeypatch.setattr("pixie_questioner.chat_service.detect_langs", fake_detect_langs)

    preferred_language = service._detect_preferred_language(
        "guten tag ich suche eine jeans", ""
    )

    assert preferred_language == "de"


def test_detect_preferred_language_falls_back_to_dutch_for_short_unsupported_text(
    monkeypatch,
):
    service = _make_service_for_language_detection()

    def fake_detect_langs(_: str):
        return [FakeLangCandidate("fi", 0.99)]

    monkeypatch.setattr("pixie_questioner.chat_service.detect_langs", fake_detect_langs)

    preferred_language = service._detect_preferred_language("Hallo", "")

    assert preferred_language == "nl"


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
    assert question_llm.calls == []


def test_ask_returns_summary_on_stop_intent():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: []\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Eind samenvatting"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik wil stoppen",
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
    assert question_llm.calls == []


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
    assert second.response_type == "answer"
    assert second.summary == "Vastgepinde samenvatting"
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
    assert result.state["phase"] == "completed"
    assert result.state["feedback_status"] == "yes"
    assert question_llm.calls == []
    assert notulist_llm.calls == []
    assert summarizer_llm.calls == []


def test_ask_returns_english_handoff_for_english_summary():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: []\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["I am looking for blue jeans with a straight fit."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
    )

    result = asyncio.run(
        service.ask(
            current_answer="done",
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
    assert question_llm.calls == []


def test_ask_returns_french_handoff_for_french_summary():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: []\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Je cherche un jean bleu avec une coupe droite."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
    )

    result = asyncio.run(
        service.ask(
            current_answer="c'est tout",
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
    assert question_llm.calls == []


def test_ask_falls_back_to_dutch_handoff_when_language_is_undetectable():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: []\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["12345 !!!"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
    )

    result = asyncio.run(
        service.ask(
            current_answer="done",
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
        completion_token="TOKEN",
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
