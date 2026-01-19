import asyncio

from pixie_questioner.chat_service import ChatService
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


QUESTIONNAIRE_TEMPLATE = "0. Initial user question: []"


def test_ask_updates_state_and_returns_next_question():
    question_llm = FakeLLM(["Next question"])
    notulist_llm = FakeLLM(["Updated questionnaire []"])
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
    assert result.state["questionnaire"] == "Updated questionnaire []"
    assert result.state["summary"] == "Updated summary"
    assert result.state["last_question"] == "Next question"
    assert "Updated questionnaire []" in question_llm.calls[-1][0]["content"]
    assert "AI question" in notulist_llm.calls[0][0]["content"]
    assert "User answer" in notulist_llm.calls[0][0]["content"]
    assert "Prev []" in notulist_llm.calls[0][0]["content"]


def test_ask_returns_summary_when_questionnaire_complete():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["All fields filled"])
    summarizer_llm = FakeLLM(["Final summary"])
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

    assert result.response_type == "summary"
    assert result.text == "Final summary"
    assert question_llm.calls == []


def test_ask_returns_summary_on_stop_intent():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Still []"])
    summarizer_llm = FakeLLM(["Final summary"])
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

    assert result.response_type == "summary"
    assert result.text == "Final summary"
    assert question_llm.calls == []


def test_ask_strips_completion_token_from_next_question():
    question_llm = FakeLLM(["Next question TOKEN"])
    notulist_llm = FakeLLM(["Updated questionnaire []"])
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
    notulist_llm = FakeLLM(["Updated questionnaire []"])
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
