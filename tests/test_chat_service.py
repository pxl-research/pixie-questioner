# ruff: noqa: S101,S105,S106
import asyncio
import json
import logging
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
from pixie_questioner.llm_client import LLMPromptInvoker


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


class RaisingRelevanceClassifier:
    def __init__(self, exception: Exception) -> None:
        self.exception = exception
        self.calls: list[tuple[str, str, str]] = []

    async def classify(
        self, current_question: str, current_answer: str, questionnaire: str
    ) -> ClassificationLabel:
        await asyncio.sleep(0)
        self.calls.append((current_question, current_answer, questionnaire))
        raise self.exception


class StubIntakeRouteClassifier:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    async def classify(self, current_answer: str) -> str:
        await asyncio.sleep(0)
        self.calls.append(current_answer)
        if not self.responses:
            raise AssertionError(
                "No route responses left for StubIntakeRouteClassifier."
            )
        return self.responses.pop(0)


def _make_notulist(llm):
    return LLMPromptInvoker(
        llm=llm,
        system_prompt_template="Q:{current_question}\nA:{current_answer}\nS:{summary}",
        error_label="Questionnaire notulist",
    )


def _make_summarizer(llm):
    return LLMPromptInvoker(
        llm=llm,
        system_prompt_template="Q:{current_question}\nA:{current_answer}\nS:{summary}",
        error_label="Conversation summarizer",
    )


def _make_handoff_classifier(llm):
    return LLMPromptInvoker(
        llm=llm,
        system_prompt_template="K:{summary}\nQ:{current_question}\nA:{current_answer}",
        error_label="Handoff classifier",
    )


def _make_unsupported_origin_follow_up(llm):
    return LLMPromptInvoker(
        llm=llm,
        system_prompt_template="S:{summary}\nQ:{current_question}\nA:{current_answer}",
        error_label="Unsupported origin follow-up",
    )


QUESTIONNAIRE_TEMPLATE = "0. Initial user question: []\n1. Occasion, Vibe, Setting: []"
CRETSKENS_QUESTIONNAIRE_TEMPLATE = (
    "0. Dish description (free text from customer): []\n"
    "1. Course type (aperitif, main course, cheese course, or dessert): []\n"
    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
    "3. Cooking method (BBQ or grill, stew, or none): []\n"
    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
    "5. Kleur van de wijn (wit, rood, rosé): []\n"
    "6. Herkomst (land en/of regio): []\n"
    "7. Maximum budget in euro: []\n"
    "8. Gewenste geur/smaak (vrije tekst): []"
)

_DEFAULT_PROFILE = load_domain_profile(Path("profiles/brooklyn/domain_profile.yml"))
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
OFF_TOPIC_REDIRECT_EN = _DEFAULT_POLICY.off_topic_redirects["en"]


def _replace_policy(**changes) -> ChatDomainPolicy:
    return replace(_DEFAULT_POLICY, **changes)


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
    assert result.state["conversation_language"] == "nl"
    assert result.state["recent_user_turns"] == "User answer"
    assert result.state["post_summary_language"] == "nl"
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


def test_ask_summary_feedback_does_not_emit_cretskens_stage_fields():
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

    assert "dish_questionnaire" not in result.state
    assert "wine_questionnaire" not in result.state
    assert "intake_stage" not in result.state


def test_ask_merges_partial_notulist_update_instead_of_repeating_same_field():
    questionnaire_template = (
        "0. Initial user question: []\n"
        "1. Occasion, Vibe, Setting: []\n"
        "2. Color & Wash: []"
    )
    question_llm = FakeLLM(["Welke kleur heeft je voorkeur?"])
    notulist_llm = FakeLLM(["1. Gelegenheid en sfeer: dagelijks gebruik"])
    summarizer_llm = FakeLLM(["Klant zoekt een jeans voor dagelijks gebruik."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=questionnaire_template,
        domain_policy=_replace_policy(completion_token="TOKEN"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Gewoon door de dag",
            state={
                "questionnaire": (
                    "0. Initial user question: Ik zoek een nieuwe broek\n"
                    "1. Occasion, Vibe, Setting: []\n"
                    "2. Color & Wash: []"
                ),
                "summary": "",
                "last_question": "Waar wil je de jeans voor dragen?",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke kleur heeft je voorkeur?"
    assert (
        result.state["questionnaire"]
        == "0. Initial user question: Ik zoek een nieuwe broek\n"
        "1. Occasion, Vibe, Setting: dagelijks gebruik\n"
        "2. Color & Wash: []"
    )
    assert result.state["phase"] == "intake"
    assert result.state["last_question"] == "Welke kleur heeft je voorkeur?"


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


def test_ask_keeps_dutch_for_clear_unsupported_sentence():
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
    assert "Respond in Dutch." in question_llm.calls[-1][0]["content"]


def test_ask_switches_question_language_immediately_for_explicit_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question_llm = FakeLLM(["Next question"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Ik zoek een jeans\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Nederlandse samenvatting"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    def fake_detect_langs(_: str):
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    result = asyncio.run(
        service.ask(
            current_answer="Kun je in het Engels verdergaan?",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "Nederlandse samenvatting",
                "last_question": "Welke pasvorm zoek je?",
                "conversation_language": "nl",
                "recent_user_turns": "Hallo",
            },
        )
    )

    assert result.response_type == "answer"
    assert "Respond in English." in question_llm.calls[-1][0]["content"]
    assert result.state.get("conversation_language") == "en"


def test_ask_switches_question_language_from_recent_user_turn_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question_llm = FakeLLM(["Next question"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Ik zoek een jeans\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Nederlandse samenvatting"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    def fake_detect_langs(text: str):
        if "I am looking for dark blue jeans for work." in text:
            return [FakeLangCandidate("en", 0.99)]
        if text == "Yes":
            return [FakeLangCandidate("en", 0.99)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    result = asyncio.run(
        service.ask(
            current_answer="Yes",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "Nederlandse samenvatting",
                "last_question": "Welke pasvorm zoek je?",
                "conversation_language": "nl",
                "recent_user_turns": (
                    "I am looking for dark blue jeans for work.\n"
                    "I prefer a straight fit."
                ),
            },
        )
    )

    assert result.response_type == "answer"
    assert "Respond in English." in question_llm.calls[-1][0]["content"]
    assert result.state.get("conversation_language") == "en"


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


def test_ask_summary_feedback_completion_does_not_emit_cretskens_stage_fields():
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

    assert "dish_questionnaire" not in result.state
    assert "wine_questionnaire" not in result.state
    assert "intake_stage" not in result.state


def test_ask_returns_categorized_handoff_when_questionnaire_complete():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): vol-au-vent\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
            "3. Cooking method (BBQ or grill, stew, or none): stew\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Bordeaux, Frankrijk\n"
            "7. Maximum budget in euro: 20-30 euro\n"
            "8. Gewenste geur/smaak (vrije tekst): citrus en niet zoet"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet vol-au-vent."])
    handoff_llm = FakeLLM(
        [
            (
                '{"foodpair_witvlees_gevogelte":true,'
                '"foodpair_stoofgerecht":true,"kleur":"rood",'
                '"land":"Frankrijk","regio":"Bordeaux","prijs":30}'
            )
        ]
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="We eten vol-au-vent",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welk gerecht eten jullie?",
                "intake_route": "food_pairing",
                "dish_follow_up_count": "1",
            },
        )
    )

    assert result.response_type == "answer"
    assert any(
        result.text.startswith(line) for line in [*HANDOFF_NL, *HANDOFF_EN, *HANDOFF_FR]
    )
    assert '"foodpair_witvlees_gevogelte": true' in result.text
    assert '"foodpair_stoofgerecht": true' in result.text
    assert '"kleur": "rood"' in result.text
    assert '"land": "Frankrijk"' in result.text
    assert '"regio": "Bordeaux"' in result.text
    assert '"prijs": 30' in result.text
    assert result.summary is None
    assert result.follow_up_question is None
    assert result.quick_replies is None
    assert result.state["phase"] == "completed"
    assert result.state["feedback_status"] == "pending"
    assert json.loads(result.state["handoff_categories_v1"]) == {
        "foodpair_witvlees_gevogelte": True,
        "foodpair_stoofgerecht": True,
        "kleur": "rood",
        "land": "Frankrijk",
        "regio": "Bordeaux",
        "prijs": 30,
    }
    assert question_llm.calls == []
    assert len(handoff_llm.calls) == 1


def test_ask_stop_classification_completes_categorized_handoff_with_defaults():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): Ik weet het nog niet\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): []\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
            "3. Cooking method (BBQ or grill, stew, or none): []\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
            "5. Kleur van de wijn (wit, rood, rosé): []\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant wil stoppen."])
    handoff_llm = FakeLLM(["{}"])
    relevance_classifier = StubRelevanceClassifier(["stop"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        answer_relevance_classifier=relevance_classifier,
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik wil stoppen",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welk gerecht eten jullie?",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text.rstrip().endswith("{}")
    assert result.summary is None
    assert result.follow_up_question is None
    assert result.quick_replies is None
    assert result.state["phase"] == "completed"
    assert json.loads(result.state["handoff_categories_v1"]) == {}
    assert question_llm.calls == []
    assert len(relevance_classifier.calls) == 1


def test_ask_categorized_handoff_does_not_force_cheese_type_when_blank():
    question_llm = FakeLLM(["Welke soort kaas is het vooral?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): vol-au-vent\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
            "3. Cooking method (BBQ or grill, stew, or none): stew\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
            "5. Kleur van de wijn (wit, rood, rosé): niet gespecificeerd\n"
            "6. Herkomst (land en/of regio): niet gespecificeerd\n"
            "7. Maximum budget in euro: niet gespecificeerd\n"
            "8. Gewenste geur/smaak (vrije tekst): geen voorkeur"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet vol-au-vent."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )

    state = {
        "questionnaire": (
            "0. Dish description (free text from customer): vol-au-vent\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
            "3. Cooking method (BBQ or grill, stew, or none): stew\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
            "5. Kleur van de wijn (wit, rood, rosé): []\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ),
        "summary": "",
        "last_question": "Welk gerecht eten jullie?",
    }

    result = asyncio.run(
        service.ask(
            current_answer="Met frieten",
            state=state,
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke soort kaas is het vooral?"
    assert result.state["phase"] == "intake"
    assert (
        "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []"
        in result.state["questionnaire"]
    )
    assert result.state["handoff_categories_v1"] == ""
    assert len(question_llm.calls) == 1


def test_ask_cretskens_vague_first_turn_asks_routing_question():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    route_classifier = StubIntakeRouteClassifier(["unknown"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
    )

    result = asyncio.run(service.ask(current_answer="Hallo", state=None))

    assert result.response_type == "answer"
    assert (
        result.text
        == "Zoek je een wijn voor bij een gerecht, of om gewoon zo te drinken?"
    )
    assert result.state["phase"] == "intake"
    assert result.state["intake_route"] == "unknown"
    assert result.state["routing_question_asked"] == "true"
    assert result.state["dish_follow_up_count"] == "0"
    assert result.state["wine_follow_up_count"] == "0"
    assert result.state["last_question"] == result.text
    assert result.state["questionnaire"] == CRETSKENS_QUESTIONNAIRE_TEMPLATE
    assert result.state["summary"] == ""
    assert question_llm.calls == []
    assert notulist_llm.calls == []
    assert summarizer_llm.calls == []
    assert route_classifier.calls == ["Hallo"]


def test_ask_cretskens_first_turn_with_dish_routes_to_food_pairing():
    question_llm = FakeLLM(["Wat zit er vooral in het gerecht?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): gegrilde zalm\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
            "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): []\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde zalm."])
    route_classifier = StubIntakeRouteClassifier(["food_pairing"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="We eten vanavond gegrilde zalm",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Wat zit er vooral in het gerecht?"
    assert result.state["intake_route"] == "food_pairing"
    assert result.state["last_question"] == "Wat zit er vooral in het gerecht?"
    assert result.state["questionnaire"].startswith(
        "0. Dish description (free text from customer): gegrilde zalm"
    )
    assert len(question_llm.calls) == 1
    assert len(notulist_llm.calls) == 1
    assert route_classifier.calls == ["We eten vanavond gegrilde zalm"]


def test_ask_cretskens_unknown_dish_name_prompts_dish_follow_up_first():
    question_llm = FakeLLM(["Wat voor gerecht is dat precies?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): trerdsgseyt\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): []\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
            "3. Cooking method (BBQ or grill, stew, or none): []\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
            "5. Kleur van de wijn (wit, rood, rosé): []\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant vermeldt onbekend gerecht: trerdsgseyt."])
    route_classifier = StubIntakeRouteClassifier(["food_pairing"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="trerdsgseyt",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Wat voor gerecht is dat precies?"
    assert result.state["intake_route"] == "food_pairing"
    assert result.state["dish_follow_up_count"] == "1"
    assert result.state["wine_follow_up_count"] == "0"
    assert result.state["last_question"] == "Wat voor gerecht is dat precies?"
    assert result.state["questionnaire"].startswith(
        "0. Dish description (free text from customer): trerdsgseyt"
    )
    assert len(question_llm.calls) == 1
    assert len(notulist_llm.calls) == 1
    assert route_classifier.calls == ["trerdsgseyt"]


def test_ask_cretskens_first_turn_without_dish_routes_to_general_wine():
    question_llm = FakeLLM(["Welke kleur wijn heb je in gedachten?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): []\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): []\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
            "3. Cooking method (BBQ or grill, stew, or none): []\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
            "5. Kleur van de wijn (wit, rood, rosé): []\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt een wijn om zo te drinken."])
    route_classifier = StubIntakeRouteClassifier(["general_wine"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek gewoon een wijn om zo te drinken",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke kleur wijn heb je in gedachten?"
    assert result.state["intake_route"] == "general_wine"
    assert result.state["intake_stage"] == "wine"
    assert result.state["last_question"] == "Welke kleur wijn heb je in gedachten?"
    assert len(question_llm.calls) == 1
    assert len(notulist_llm.calls) == 1
    assert route_classifier.calls == ["Ik zoek gewoon een wijn om zo te drinken"]


def test_ask_cretskens_general_wine_opening_can_capture_supported_origin():
    question_llm = FakeLLM(["Wat is je maximum budget?"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Frankrijk"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt een wijn om zo te drinken."])
    route_classifier = StubIntakeRouteClassifier(["general_wine"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek een rode wijn uit Frankrijk",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Wat is je maximum budget?"
    assert result.state["intake_route"] == "general_wine"
    assert result.state["intake_stage"] == "wine"
    assert result.state["wine_follow_up_count"] == "1"
    assert (
        result.state["wine_questionnaire"]
        == "5. Kleur van de wijn (wit, rood, rosé): rood\n"
        "6. Herkomst (land en/of regio): Frankrijk\n"
        "7. Maximum budget in euro: []\n"
        "8. Gewenste geur/smaak (vrije tekst): []"
    )
    assert (
        "5. Kleur van de wijn (wit, rood, rosé): rood"
        in question_llm.calls[0][0]["content"]
    )
    assert (
        "6. Herkomst (land en/of regio): Frankrijk"
        in question_llm.calls[0][0]["content"]
    )
    assert (
        "0. Dish description (free text from customer):"
        not in question_llm.calls[0][0]["content"]
    )
    assert len(notulist_llm.calls) == 1


def test_ask_cretskens_general_wine_opening_reports_unsupported_origin():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt een wijn om zo te drinken."])
    unsupported_origin_llm = FakeLLM(
        [
            (
                '{"should_follow_up":true,"question":"Bulgarije hebben we '
                "momenteel niet in ons aanbod. Is een ander land of een "
                "andere regio ook goed, of wil je liever dat ik los van "
                'herkomst iets voorstel?"}'
            )
        ]
    )
    route_classifier = StubIntakeRouteClassifier(["general_wine"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
        unsupported_origin_follow_up=_make_unsupported_origin_follow_up(
            unsupported_origin_llm
        ),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek een rode wijn uit Bulgarije",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert (
        result.text == "Bulgarije hebben we momenteel niet in ons aanbod. "
        "Is een ander land of een andere regio ook goed, of wil je liever "
        "dat ik los van herkomst iets voorstel?"
    )
    assert result.state["intake_route"] == "general_wine"
    assert result.state["intake_stage"] == "wine"
    assert result.state["last_question"] == result.text
    assert result.state["wine_follow_up_count"] == "1"
    assert (
        result.state["wine_questionnaire"]
        == "5. Kleur van de wijn (wit, rood, rosé): rood\n"
        "6. Herkomst (land en/of regio): []\n"
        "7. Maximum budget in euro: []\n"
        "8. Gewenste geur/smaak (vrije tekst): []"
    )
    assert question_llm.calls == []
    assert len(notulist_llm.calls) == 1
    assert len(unsupported_origin_llm.calls) == 1


def test_ask_cretskens_general_wine_opening_without_origin_uses_next_question():
    question_llm = FakeLLM(["Uit welk land of welke regio mag de wijn komen?"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt een wijn om zo te drinken."])
    unsupported_origin_llm = FakeLLM(
        ['{"should_follow_up":false,"no_preference_expressed":false,"question":""}']
    )
    route_classifier = StubIntakeRouteClassifier(["general_wine"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
        unsupported_origin_follow_up=_make_unsupported_origin_follow_up(
            unsupported_origin_llm
        ),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek een rode wijn",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Uit welk land of welke regio mag de wijn komen?"
    assert result.state["intake_route"] == "general_wine"
    assert result.state["intake_stage"] == "wine"
    assert result.state["last_question"] == result.text
    assert result.state["wine_follow_up_count"] == "1"
    # Opening turn: field 6 must remain blank so unsupported-origin check
    # can still fire when the user names a country on the next turn.
    assert "6. Herkomst (land en/of regio): []" in result.state["wine_questionnaire"]
    assert len(question_llm.calls) == 1
    assert len(notulist_llm.calls) == 1
    assert len(unsupported_origin_llm.calls) == 1


def test_ask_cretskens_general_wine_opening_does_not_complete_on_unspecified_values():
    question_llm = FakeLLM(["Welke kleur wijn heeft je voorkeur: wit, rood of rosé?"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): niet gespecificeerd\n"
            "6. Herkomst (land en/of regio): niet gespecificeerd\n"
            "7. Maximum budget in euro: niet gespecificeerd\n"
            "8. Gewenste geur/smaak (vrije tekst): niet gespecificeerd"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt iets goeds om te drinken."])
    route_classifier = StubIntakeRouteClassifier(["general_wine"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek iets goeds om te drinken",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke kleur wijn heeft je voorkeur: wit, rood of rosé?"
    assert result.state["phase"] == "intake"
    assert result.state["intake_route"] == "general_wine"
    assert result.state["wine_follow_up_count"] == "1"
    assert len(question_llm.calls) == 1
    assert len(notulist_llm.calls) == 1


def test_ask_cretskens_general_wine_does_not_complete_when_notulist_fills_all_wine_fields():
    """Regression test: intake must ask at least one wine question for general_wine route.

    When the notulist fills all wine fields with "geen voorkeur" (no preference values
    that are not treated as blank), the intake must still ask a wine question instead of
    completing immediately. The decision is structural (wine_follow_up_count == 0),
    not based on keyword matching.
    """
    question_llm = FakeLLM(["Welke kleur wijn heb je in gedachten?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): []\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): []\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
            "3. Cooking method (BBQ or grill, stew, or none): []\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
            "5. Kleur van de wijn (wit, rood, rosé): geen voorkeur\n"
            "6. Herkomst (land en/of regio): geen voorkeur\n"
            "7. Maximum budget in euro: geen voorkeur\n"
            "8. Gewenste geur/smaak (vrije tekst): geen voorkeur"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt iets om zo te drinken."])
    route_classifier = StubIntakeRouteClassifier(["general_wine"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek gewoon iets om te drinken",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert result.state["phase"] == "intake"
    assert result.state["intake_route"] == "general_wine"
    assert result.state["wine_follow_up_count"] == "1"
    assert len(question_llm.calls) == 1
    assert len(notulist_llm.calls) == 1


def test_ask_cretskens_supported_origin_fills_field_when_notulist_misses_it():
    """Regression: when user answers a supported country (e.g. 'Spanje') but the
    notulist fails to fill field 6, the fallback extraction must fill it directly
    to prevent an infinite re-ask loop.
    """
    question_llm = FakeLLM(["Wat is je maximum budget?"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt rode wijn."])
    unsupported_origin_llm = FakeLLM(["should not be called"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        unsupported_origin_follow_up=_make_unsupported_origin_follow_up(
            unsupported_origin_llm
        ),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Spanje",
            state={
                "questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": "Uit welk land of welke regio mag de wijn komen?",
                "intake_route": "general_wine",
                "intake_stage": "wine",
                "wine_follow_up_count": "1",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Wat is je maximum budget?"
    assert (
        "6. Herkomst (land en/of regio): Spanje" in result.state["wine_questionnaire"]
    )
    assert "6. Herkomst (land en/of regio): Spanje" in result.state["questionnaire"]
    # Unsupported-origin LLM must NOT be called — the fallback fills it directly.
    assert unsupported_origin_llm.calls == []
    assert len(notulist_llm.calls) == 1


def test_ask_cretskens_supported_origin_overrides_unsupported_when_answer_contains_supported_country():
    question_llm = FakeLLM(["Welke geur of smaak zoek je?"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Bulgarije\n"
            "7. Maximum budget in euro: 20\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt rode wijn rond 20 euro."])
    unsupported_origin_llm = FakeLLM(["should not be called"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        unsupported_origin_follow_up=_make_unsupported_origin_follow_up(
            unsupported_origin_llm
        ),
    )

    result = asyncio.run(
        service.ask(
            current_answer="20 euro, liefst Frankrijk",
            state={
                "questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): Bulgarije\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): Bulgarije\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": "Wat is je maximum budget?",
                "intake_route": "general_wine",
                "intake_stage": "wine",
                "wine_follow_up_count": "2",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke geur of smaak zoek je?"
    assert (
        "6. Herkomst (land en/of regio): Frankrijk"
        in result.state["wine_questionnaire"]
    )
    assert "6. Herkomst (land en/of regio): Frankrijk" in result.state["questionnaire"]
    assert unsupported_origin_llm.calls == []
    assert len(notulist_llm.calls) == 1


def test_ask_cretskens_general_wine_unsupported_origin_still_triggers_after_explicit_origin_question():
    """Regression: when origin was explicitly asked and user names an unsupported country,
    the unsupported-origin follow-up must still fire (field 6 must not be pre-marked).
    """
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Bulgarije\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt rode wijn."])
    unsupported_origin_llm = FakeLLM(
        [
            (
                '{"should_follow_up":true,"question":"Bulgarije hebben we '
                "momenteel niet in ons aanbod. Is een ander land of een "
                'andere regio ook goed?"}'
            )
        ]
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        unsupported_origin_follow_up=_make_unsupported_origin_follow_up(
            unsupported_origin_llm
        ),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Bulgarije",
            state={
                "questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": "Heb je een voorkeur voor een land of regio waar de wijn vandaan komt?",
                "intake_route": "general_wine",
                "intake_stage": "wine",
                "wine_follow_up_count": "1",
            },
        )
    )

    assert result.response_type == "answer"
    assert (
        result.text == "Bulgarije hebben we momenteel niet in ons aanbod. "
        "Is een ander land of een andere regio ook goed?"
    )
    assert result.state["last_question"] == result.text
    assert question_llm.calls == []
    assert len(notulist_llm.calls) == 1
    assert len(unsupported_origin_llm.calls) == 1


def test_ask_cretskens_general_wine_unsupported_origin_followup_with_non_blank_field():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): wit\n"
            "6. Herkomst (land en/of regio): Bulgarije\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt witte wijn."])
    unsupported_origin_llm = FakeLLM(
        [
            (
                '{"should_follow_up":true,"question":"Bulgarije hebben we '
                "momenteel niet in ons aanbod. Is een ander land of een "
                'andere regio ook goed?"}'
            )
        ]
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        unsupported_origin_follow_up=_make_unsupported_origin_follow_up(
            unsupported_origin_llm
        ),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Bulgarije",
            state={
                "questionnaire": (
                    "0. Dish description (free text from customer): steak\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): red meat\n"
                    "3. Cooking method (BBQ or grill, stew, or none): stew\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
                    "5. Kleur van de wijn (wit, rood, rosé): wit\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "dish_questionnaire": (
                    "0. Dish description (free text from customer): steak\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): red meat\n"
                    "3. Cooking method (BBQ or grill, stew, or none): stew\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): wit\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": "Vanuit welk land of welke regio zou je die het liefst hebben?",
                "intake_route": "general_wine",
                "intake_stage": "wine",
                "wine_follow_up_count": "2",
            },
        )
    )

    assert result.response_type == "answer"
    assert (
        result.text == "Bulgarije hebben we momenteel niet in ons aanbod. "
        "Is een ander land of een andere regio ook goed?"
    )
    assert result.state["last_question"] == result.text
    assert result.state["wine_follow_up_count"] == "3"
    assert question_llm.calls == []
    assert len(notulist_llm.calls) == 1
    assert len(unsupported_origin_llm.calls) == 1


def test_ask_cretskens_no_preference_origin_marks_field_to_prevent_reask():
    """Regression: when user says 'geen voorkeur' after being explicitly asked for origin,
    field 6 must be marked as 'geen voorkeur' so it is not re-asked in subsequent turns.
    """
    question_llm = FakeLLM(["Wat is je maximale budget?"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant heeft geen voorkeur voor herkomst."])
    unsupported_origin_llm = FakeLLM(
        ['{"should_follow_up":false,"no_preference_expressed":true,"question":""}']
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        unsupported_origin_follow_up=_make_unsupported_origin_follow_up(
            unsupported_origin_llm
        ),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Zonder voorkeur dan",
            state={
                "questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": (
                    "Sorry, we hebben helaas geen wijnen uit Nederland. "
                    "Zou een andere regio ook goed zijn?"
                ),
                "intake_route": "general_wine",
                "intake_stage": "wine",
                "wine_follow_up_count": "2",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Wat is je maximale budget?"
    # Field 6 must be marked so origin is not re-asked.
    assert (
        "6. Herkomst (land en/of regio): geen voorkeur"
        in result.state["wine_questionnaire"]
    )
    assert len(unsupported_origin_llm.calls) == 1


def test_ask_cretskens_unsupported_origin_then_no_preference_moves_to_next_field():
    """Characterization: unsupported origin follow-up can resolve to no-preference.

    Sequence:
    1) Opening general-wine turn names an unsupported country => follow-up question.
    2) User answers "geen voorkeur" => origin is marked and flow continues to field 7.
    """
    question_llm = FakeLLM(["Wat is je maximale budget?"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []",
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []",
        ]
    )
    summarizer_llm = FakeLLM(
        [
            "Klant zoekt rode wijn uit Bulgarije.",
            "Klant heeft geen voorkeur voor herkomst.",
        ]
    )
    unsupported_origin_llm = FakeLLM(
        [
            (
                '{"should_follow_up":true,"question":"Bulgarije hebben we nu niet '
                'in het aanbod. Is een andere herkomst ook goed?"}'
            ),
            '{"should_follow_up":false,"no_preference_expressed":true,"question":""}',
        ]
    )
    route_classifier = StubIntakeRouteClassifier(["general_wine"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
        unsupported_origin_follow_up=_make_unsupported_origin_follow_up(
            unsupported_origin_llm
        ),
    )

    first_turn = asyncio.run(
        service.ask(
            current_answer="Ik zoek een rode wijn uit Bulgarije",
            state=None,
        )
    )

    assert first_turn.response_type == "answer"
    assert (
        first_turn.text
        == "Bulgarije hebben we nu niet in het aanbod. Is een andere herkomst ook goed?"
    )
    assert first_turn.state["wine_follow_up_count"] == "1"

    second_turn = asyncio.run(
        service.ask(
            current_answer="Geen voorkeur dan",
            state=first_turn.state,
        )
    )

    assert second_turn.response_type == "answer"
    assert second_turn.text == "Wat is je maximale budget?"
    assert (
        "6. Herkomst (land en/of regio): geen voorkeur"
        in second_turn.state["wine_questionnaire"]
    )
    assert (
        "6. Herkomst (land en/of regio): geen voorkeur"
        in second_turn.state["questionnaire"]
    )
    assert second_turn.state["wine_follow_up_count"] == "2"
    assert len(unsupported_origin_llm.calls) == 2
    assert len(question_llm.calls) == 1


def test_ask_cretskens_no_preference_origin_marks_field_when_notulist_fills_value():
    """Regression: when user says 'zonder voorkeur dan' after an unsupported-origin
    follow-up and the notulist LLM fills field 6 with a non-supported value like
    'geen voorkeur', the field must still be marked so origin is not re-asked.

    Previously, the stale origin_is_blank=False guard prevented marking field 6
    after the code cleared it, causing an infinite loop.
    """
    question_llm = FakeLLM(["Wat is je maximale budget?"])
    # Notulist fills field 6 with "geen voorkeur" (non-blank, non-supported)
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): geen voorkeur\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant heeft geen voorkeur voor herkomst."])
    unsupported_origin_llm = FakeLLM(
        ['{"should_follow_up":false,"no_preference_expressed":true,"question":""}']
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        unsupported_origin_follow_up=_make_unsupported_origin_follow_up(
            unsupported_origin_llm
        ),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Zonder voorkeur dan",
            state={
                "questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": (
                    "Helaas hebben we op dit moment geen wijnen uit Bulgarije "
                    "in ons assortiment. Zou een wijn uit een ander land of "
                    "regio ook goed zijn, of wilt u liever een suggestie "
                    "zonder herkomstvoorkeur?"
                ),
                "intake_route": "general_wine",
                "intake_stage": "wine",
                "wine_follow_up_count": "2",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Wat is je maximale budget?"
    # Field 6 must be marked even when the notulist filled it with a
    # non-supported value before the unsupported-origin check cleared it.
    assert (
        "6. Herkomst (land en/of regio): geen voorkeur"
        in result.state["wine_questionnaire"]
    )
    assert (
        "6. Herkomst (land en/of regio): geen voorkeur" in result.state["questionnaire"]
    )
    assert len(unsupported_origin_llm.calls) == 1


def test_ask_cretskens_opening_general_wine_no_preference_expressed_marks_origin_field():
    """When user says 'geen voorkeur' on the very first (opening) general-wine turn,
    field 6 must be marked as 'geen voorkeur' to avoid re-asking on the next turn."""
    question_llm = FakeLLM(["Wat is je maximale budget?"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt rode wijn, geen voorkeur voor herkomst."])
    unsupported_origin_llm = FakeLLM(
        ['{"should_follow_up":false,"no_preference_expressed":true,"question":""}']
    )
    route_classifier = StubIntakeRouteClassifier(["general_wine"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
        unsupported_origin_follow_up=_make_unsupported_origin_follow_up(
            unsupported_origin_llm
        ),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek een rode wijn, geen voorkeur voor herkomst",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Wat is je maximale budget?"
    # Even on the opening turn, an explicit no-preference must mark field 6.
    assert (
        "6. Herkomst (land en/of regio): geen voorkeur"
        in result.state["wine_questionnaire"]
    )
    assert len(unsupported_origin_llm.calls) == 1


def test_ask_cretskens_marker_like_opening_does_not_bypass_unknown_route_classifier():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    route_classifier = StubIntakeRouteClassifier(["unknown"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="We eten vanavond gegrilde zalm",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert (
        result.text
        == "Zoek je een wijn voor bij een gerecht, of om gewoon zo te drinken?"
    )
    assert result.state["intake_route"] == "unknown"
    assert route_classifier.calls == ["We eten vanavond gegrilde zalm"]
    assert question_llm.calls == []
    assert notulist_llm.calls == []
    assert summarizer_llm.calls == []


def test_ask_cretskens_food_pairing_stops_dish_followups_after_budget():
    question_llm = FakeLLM(["Welke kleur wijn heb je in gedachten?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): gegrilde zalm\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
            "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): []\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde zalm."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Nog niet duidelijk",
            state={
                "questionnaire": (
                    "0. Dish description (free text from customer): gegrilde zalm\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
                    "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
                    "5. Kleur van de wijn (wit, rood, rosé): []\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": "Wat zit er vooral in het gerecht?",
                "intake_route": "food_pairing",
                "dish_follow_up_count": "3",
                "wine_follow_up_count": "0",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke kleur wijn heb je in gedachten?"
    assert result.state["intake_stage"] == "wine"
    assert result.state["dish_follow_up_count"] == "3"
    assert result.state["wine_follow_up_count"] == "1"
    assert (
        "5. Kleur van de wijn (wit, rood, rosé): []"
        in question_llm.calls[0][0]["content"]
    )
    assert (
        "0. Dish description (free text from customer):"
        in question_llm.calls[0][0]["content"]
    ), "Dish context must be visible to question_asker when in food_pairing mode"
    question_llm = FakeLLM(["Welke kleur wijn heb je in gedachten?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): gegrilde zalm\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): fish or seafood\n"
            "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde zalm."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Het is vis",
            state={
                "questionnaire": (
                    "0. Dish description (free text from customer): gegrilde zalm\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
                    "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
                    "5. Kleur van de wijn (wit, rood, rosé): []\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "dish_questionnaire": (
                    "0. Dish description (free text from customer): gegrilde zalm\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
                    "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): []\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": "Wat zit er vooral in het gerecht?",
                "intake_route": "food_pairing",
                "intake_stage": "dish",
                "dish_follow_up_count": "1",
                "wine_follow_up_count": "0",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke kleur wijn heb je in gedachten?"
    assert result.state["intake_stage"] == "wine"
    assert result.state["dish_follow_up_count"] == "1"
    assert result.state["wine_follow_up_count"] == "1"
    assert (
        "5. Kleur van de wijn (wit, rood, rosé): []"
        in question_llm.calls[0][0]["content"]
    )
    assert (
        "0. Dish description (free text from customer):"
        in question_llm.calls[0][0]["content"]
    ), "Dish context must be visible to question_asker when in food_pairing mode"
    """Regression test: food_pairing route must ask at least one question.

    When the notulist fills all dish+wine fields on the opening turn,
    the intake must still ask a follow-up question instead of completing
    immediately and jumping to recommendation.
    """
    question_llm = FakeLLM(["Hoe is de vis bereid?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): pasta met zalm\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): fish or seafood\n"
            "3. Cooking method (BBQ or grill, stew, or none): none\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): geen voorkeur\n"
            "6. Herkomst (land en/of regio): geen voorkeur\n"
            "7. Maximum budget in euro: geen voorkeur\n"
            "8. Gewenste geur/smaak (vrije tekst): geen voorkeur"
        ]
    )
    summarizer_llm = FakeLLM(["Klant wil wijn bij pasta met zalm."])
    route_classifier = StubIntakeRouteClassifier(["food_pairing"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik eet pasta met zalm",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert result.state["phase"] == "intake"
    assert result.state["intake_route"] == "food_pairing"
    assert (
        int(result.state.get("dish_follow_up_count", "0"))
        + int(result.state.get("wine_follow_up_count", "0"))
        >= 1
    )
    assert len(question_llm.calls) == 1


def test_ask_cretskens_next_question_prompt_includes_updated_field_values():
    """The focused questionnaire sent to the question-asker must reflect the
    current turn's updates so the LLM knows which fields are already filled.

    Scenario: user's answer fills field 5 (wine colour).  The question-asker
    prompt must contain the updated value ("wit"), not the pre-update blank.
    """
    question_llm = FakeLLM(["Uit welk land of regio?"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): wit\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt een witte wijn."])
    route_classifier = StubIntakeRouteClassifier(["general_wine"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        intake_route_classifier=route_classifier,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Een witte wijn graag",
            state=None,
        )
    )

    assert result.response_type == "answer"
    assert len(question_llm.calls) == 1
    prompt_content = question_llm.calls[0][0]["content"]
    assert (
        "5. Kleur van de wijn (wit, rood, rosé): wit" in prompt_content
    ), "Question-asker prompt must include the updated field value from the current turn"
    assert "6. Herkomst (land en/of regio): []" in prompt_content

    question_llm = FakeLLM(["Welke kleur wijn heb je in gedachten?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): []\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): []\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
            "3. Cooking method (BBQ or grill, stew, or none): []\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
            "5. Kleur van de wijn (wit, rood, rosé): []\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt een wijn om zo te drinken."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek gewoon een wijn om zo te drinken",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Zoek je een wijn voor bij een gerecht, of om gewoon zo te drinken?",
                "intake_route": "general_wine",
                "routing_question_asked": "true",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke kleur wijn heb je in gedachten?"
    assert result.state["intake_stage"] == "wine"
    assert result.state["dish_follow_up_count"] == "0"
    assert result.state["wine_follow_up_count"] == "1"
    assert (
        "5. Kleur van de wijn (wit, rood, rosé): []"
        in question_llm.calls[0][0]["content"]
    )
    assert (
        "0. Dish description (free text from customer):"
        not in question_llm.calls[0][0]["content"]
    )


def test_ask_cretskens_no_preference_for_color_still_asks_remaining_wine_fields():
    question_llm = FakeLLM(["Uit welk land of welke regio mag de wijn komen?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): pasta met zalm\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): fish or seafood\n"
            "3. Cooking method (BBQ or grill, stew, or none): none\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): maakt niet uit\n"
            "6. Herkomst (land en/of regio): geen voorkeur\n"
            "7. Maximum budget in euro: geen voorkeur\n"
            "8. Gewenste geur/smaak (vrije tekst): geen voorkeur"
        ]
    )
    summarizer_llm = FakeLLM(["Klant heeft geen voorkeur voor de kleur."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Maakt niet uit",
            state={
                "questionnaire": (
                    "0. Dish description (free text from customer): pasta met zalm\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): fish or seafood\n"
                    "3. Cooking method (BBQ or grill, stew, or none): none\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
                    "5. Kleur van de wijn (wit, rood, rosé): []\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": "Welke kleur wijn heb je in gedachten?",
                "intake_route": "food_pairing",
                "intake_stage": "wine",
                "dish_follow_up_count": "1",
                "wine_follow_up_count": "1",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Uit welk land of welke regio mag de wijn komen?"
    assert result.state["phase"] == "intake"
    assert (
        "5. Kleur van de wijn (wit, rood, rosé): maakt niet uit"
        in result.state["wine_questionnaire"]
    )
    assert "6. Herkomst (land en/of regio): []" in result.state["wine_questionnaire"]
    assert "7. Maximum budget in euro: []" in result.state["wine_questionnaire"]
    assert (
        "8. Gewenste geur/smaak (vrije tekst): []" in result.state["wine_questionnaire"]
    )
    assert result.state["wine_follow_up_count"] == "2"


def test_ask_cretskens_wine_stage_prompt_includes_dish_context_for_pairing() -> None:
    question_llm = FakeLLM(["Welke herkomst heb je in gedachten?"])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): wit\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde zalm en zoekt een frisse wijn."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Wit",
            state={
                "questionnaire": (
                    "0. Dish description (free text from customer): gegrilde zalm\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): fish or seafood\n"
                    "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
                    "5. Kleur van de wijn (wit, rood, rosé): []\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "dish_questionnaire": (
                    "0. Dish description (free text from customer): gegrilde zalm\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): fish or seafood\n"
                    "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): []\n"
                    "6. Herkomst (land en/of regio): []\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": "Welke kleur wijn heb je in gedachten?",
                "intake_route": "food_pairing",
                "intake_stage": "wine",
                "dish_follow_up_count": "2",
                "wine_follow_up_count": "1",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke herkomst heb je in gedachten?"
    assert result.state["intake_stage"] == "wine"
    assert result.state["wine_follow_up_count"] == "2"
    assert (
        "5. Kleur van de wijn (wit, rood, rosé): wit"
        in result.state["wine_questionnaire"]
    )
    assert "6. Herkomst (land en/of regio): []" in question_llm.calls[0][0]["content"]
    assert (
        "0. Dish description (free text from customer):"
        in question_llm.calls[0][0]["content"]
    ), "Dish context must be visible to question_asker when in food_pairing wine stage"


def test_ask_cretskens_stops_wine_followups_after_budget():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): []\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): []\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): []\n"
            "3. Cooking method (BBQ or grill, stew, or none): []\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
            "5. Kleur van de wijn (wit, rood, rosé): []\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt een wijn om zo te drinken."])
    handoff_llm = FakeLLM(["{}"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Nog geen voorkeur",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welke herkomst heb je in gedachten?",
                "intake_route": "general_wine",
                "wine_follow_up_count": "5",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.state["phase"] == "completed"
    assert question_llm.calls == []
    assert json.loads(result.state["handoff_categories_v1"]) == {}


def test_ask_categorized_handoff_asks_cheese_followup_for_cheese_course():
    question_llm = FakeLLM(["Welke soort kaas is het vooral?"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): kaasplank\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): cheese course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): none\n"
            "3. Cooking method (BBQ or grill, stew, or none): none\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): []\n"
            "5. Kleur van de wijn (wit, rood, rosé): []\n"
            "6. Herkomst (land en/of regio): []\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet een kaasplank."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik eet een kaasplank",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welk gerecht eten jullie?",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Welke soort kaas is het vooral?"
    assert result.state["phase"] == "intake"
    assert result.state["last_question"] == "Welke soort kaas is het vooral?"
    assert result.state["handoff_categories_v1"] == ""
    assert len(question_llm.calls) == 1


def test_ask_categorized_handoff_derives_country_from_region_only():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): rundstoofvlees\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): red meat\n"
            "3. Cooking method (BBQ or grill, stew, or none): stew\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Rioja\n"
            "7. Maximum budget in euro: 25\n"
            "8. Gewenste geur/smaak (vrije tekst): fruitig en fris"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet rundstoofvlees."])
    handoff_llm = FakeLLM(
        [
            (
                '{"foodpair_roodvlees":true,"foodpair_stoofgerecht":true,'
                '"regio":"Rioja","land":"Frankrijk","prijs":25}'
            )
        ]
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Rundstoofvlees met rode wijn, liefst uit Rioja",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welk gerecht eten jullie?",
                "intake_route": "food_pairing",
                "dish_follow_up_count": "1",
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff["regio"] == "Rioja"
    assert handoff["land"] == "Spanje"
    assert handoff["prijs"] == 25


def test_ask_categorized_handoff_region_wins_on_country_conflict():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): gegrilde zalm\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): fish or seafood\n"
            "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): wit\n"
            "6. Herkomst (land en/of regio): Spanje, Bordeaux\n"
            "7. Maximum budget in euro: 40\n"
            "8. Gewenste geur/smaak (vrije tekst): kruidig en bruisend"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde zalm."])
    handoff_llm = FakeLLM(
        [
            (
                '{"foodpair_vis_zeevruchten":true,"foodpair_bbq_grill":true,'
                '"regio":"Bordeaux","land":"Spanje","prijs":40}'
            )
        ]
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Zalm, wit, uit Spanje of Bordeaux",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welk gerecht eten jullie?",
                "intake_route": "food_pairing",
                "dish_follow_up_count": "1",
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff["regio"] == "Bordeaux"
    assert handoff["land"] == "Frankrijk"
    assert handoff["prijs"] == 40


def test_ask_categorized_handoff_keeps_wrapped_aroma_text_out_of_handoff():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): gegrilde kip\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
            "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): wit\n"
            "6. Herkomst (land en/of regio): Frankrijk\n"
            "7. Maximum budget in euro: 20\n"
            '8. Gewenste geur/smaak (vrije tekst): Aroma flags: {"smaak_droog":"true","smaak_fris_zuur":true}'
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde kip."])
    handoff_llm = FakeLLM(
        ['{"foodpair_witvlees_gevogelte":true,"foodpair_bbq_grill":true}']
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="droog en fris",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welke geur of smaak zoek je?",
                "intake_route": "food_pairing",
                "wine_follow_up_count": "1",
            },
        )
    )

    assert (
        "8. Gewenste geur/smaak (vrije tekst): "
        'Aroma flags: {"smaak_droog":"true","smaak_fris_zuur":true}'
    ) in result.state["questionnaire"]
    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff == {
        "foodpair_witvlees_gevogelte": True,
        "foodpair_bbq_grill": True,
    }


def test_ask_categorized_handoff_keeps_parseable_aroma_text_out_of_handoff():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): gegrilde kip\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
            "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): wit\n"
            "6. Herkomst (land en/of regio): Frankrijk\n"
            "7. Maximum budget in euro: 20\n"
            "8. Gewenste geur/smaak (vrije tekst): smaak_kruidig: true"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde kip."])
    handoff_llm = FakeLLM(
        ['{"foodpair_witvlees_gevogelte":true,"foodpair_bbq_grill":true}']
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="kruidig",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welke geur of smaak zoek je?",
                "intake_route": "food_pairing",
                "wine_follow_up_count": "1",
            },
        )
    )

    assert (
        "8. Gewenste geur/smaak (vrije tekst): smaak_kruidig: true"
        in result.state["questionnaire"]
    )
    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff == {
        "foodpair_witvlees_gevogelte": True,
        "foodpair_bbq_grill": True,
    }


def test_ask_completed_categorized_handoff_keeps_text_field8_out_of_handoff():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    handoff_llm = FakeLLM(
        ['{"foodpair_witvlees_gevogelte":true,"foodpair_stoofgerecht":true}']
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )
    completed_state = {
        "questionnaire": (
            "0. Dish description (free text from customer): vol-au-vent\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
            "3. Cooking method (BBQ or grill, stew, or none): stew\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Sicilië\n"
            "7. Maximum budget in euro: 30\n"
            "8. Gewenste geur/smaak (vrije tekst): kruidig"
        ),
        "summary": "Klant eet vol-au-vent.",
        "last_question": "Welke geur of smaak zoek je?",
        "phase": "completed",
        "feedback_status": "pending",
        "final_summary": "",
        "post_summary_language": "",
        "boundary_violation_count": "0",
        "handoff_categories_v1": "",
    }

    result = asyncio.run(service.ask(current_answer="Nog iets", state=completed_state))

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff["foodpair_witvlees_gevogelte"] is True
    assert handoff["foodpair_stoofgerecht"] is True
    assert "smaak_kruidig" not in handoff


def test_ask_categorized_handoff_asks_aroma_after_unclear_budget_answer():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): kaasplank\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): cheese course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): none\n"
            "3. Cooking method (BBQ or grill, stew, or none): none\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): hard cheese\n"
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Italië\n"
            "7. Maximum budget in euro: niet gespecificeerd\n"
            "8. Gewenste geur/smaak (vrije tekst): niet gespecificeerd"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet kaas."])
    handoff_llm = FakeLLM(
        ['{"foodpair_harde_kaas":true,"kleur":"rood","land":"Italië"}']
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Geen budgetvoorkeur",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Wat is je maximum budget?",
                "intake_route": "food_pairing",
                "wine_follow_up_count": "1",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.state["phase"] == "intake"
    assert result.state["handoff_categories_v1"] == ""
    assert "Welke geuren of smaken spreken je aan bij wijn?" in result.text
    assert (
        "7. Maximum budget in euro: niet gespecificeerd"
        in result.state["wine_questionnaire"]
    )
    assert (
        "8. Gewenste geur/smaak (vrije tekst): []" in result.state["wine_questionnaire"]
    )


def test_ask_completed_phase_sanitizes_legacy_categorized_handoff_answer():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )
    completed_state = {
        "questionnaire": (
            "0. Dish description (free text from customer): vol-au-vent\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
            "3. Cooking method (BBQ or grill, stew, or none): stew\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Bordeaux, Frankrijk\n"
            "7. Maximum budget in euro: 30\n"
            "8. Gewenste geur/smaak (vrije tekst): citrus en mineraal"
        ),
        "summary": "Klant eet vol-au-vent.",
        "last_question": "Welk gerecht eten jullie?",
        "phase": "completed",
        "feedback_status": "pending",
        "final_summary": "",
        "post_summary_language": "",
        "boundary_violation_count": "0",
        "handoff_categories_v1": (
            '{"foodpair_witvlees_gevogelte":true,"foodpair_stoofgerecht":true,'
            '"kleur":"rood","land":"Frankrijk","regio":"Bordeaux","prijs":30,'
            '"geur_citrus":true,"smaak_mineralig":true}'
        ),
    }

    result = asyncio.run(service.ask(current_answer="Nog iets", state=completed_state))

    assert result.response_type == "answer"
    assert any(
        result.text.startswith(line) for line in [*HANDOFF_NL, *HANDOFF_EN, *HANDOFF_FR]
    )
    assert '"foodpair_witvlees_gevogelte": true' in result.text
    assert '"foodpair_stoofgerecht": true' in result.text
    assert '"kleur": "rood"' in result.text
    assert '"regio": "Bordeaux"' in result.text
    assert '"geur_citrus"' not in result.text
    assert '"smaak_mineralig"' not in result.text
    assert result.summary is None
    assert result.follow_up_question is None
    assert result.quick_replies is None
    assert result.state["phase"] == "completed"
    assert json.loads(result.state["handoff_categories_v1"]) == {
        "foodpair_witvlees_gevogelte": True,
        "foodpair_stoofgerecht": True,
        "kleur": "rood",
        "land": "Frankrijk",
        "regio": "Bordeaux",
        "prijs": 30,
    }
    assert question_llm.calls == []
    assert notulist_llm.calls == []
    assert summarizer_llm.calls == []


def test_ask_completed_phase_sanitizes_legacy_machine_coded_field8_state():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )
    questionnaire_only_state = {
        "questionnaire": (
            "0. Dish description (free text from customer): vol-au-vent\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
            "3. Cooking method (BBQ or grill, stew, or none): stew\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Bordeaux, Frankrijk\n"
            "7. Maximum budget in euro: 30\n"
            '8. Gewenste geur/smaak (vrije tekst): {"smaak_fruitig":true}'
        ),
        "summary": "Klant eet vol-au-vent.",
        "last_question": "Welke geur of smaak zoek je?",
        "phase": "completed",
        "feedback_status": "pending",
        "handoff_categories_v1": '{"foodpair_witvlees_gevogelte":true}',
    }

    questionnaire_only_result = asyncio.run(
        service.ask(current_answer="Nog iets", state=questionnaire_only_state)
    )

    assert (
        "8. Gewenste geur/smaak (vrije tekst): []"
        in questionnaire_only_result.state["questionnaire"]
    )
    assert (
        "8. Gewenste geur/smaak (vrije tekst): []"
        in questionnaire_only_result.state["wine_questionnaire"]
    )
    assert "smaak_" not in questionnaire_only_result.state["questionnaire"]
    assert "smaak_" not in questionnaire_only_result.state["wine_questionnaire"]
    assert "geur_" not in questionnaire_only_result.state["questionnaire"]
    assert "geur_" not in questionnaire_only_result.state["wine_questionnaire"]

    split_section_state = {
        "questionnaire": (
            "0. Dish description (free text from customer): vol-au-vent\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
            "3. Cooking method (BBQ or grill, stew, or none): stew\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Bordeaux, Frankrijk\n"
            "7. Maximum budget in euro: 30\n"
            "8. Gewenste geur/smaak (vrije tekst): citrus en mineraal"
        ),
        "dish_questionnaire": (
            "0. Dish description (free text from customer): vol-au-vent\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
            "3. Cooking method (BBQ or grill, stew, or none): stew\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none"
        ),
        "wine_questionnaire": (
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Bordeaux, Frankrijk\n"
            "7. Maximum budget in euro: 30\n"
            '8. Gewenste geur/smaak (vrije tekst): {"geur_citrus":true}'
        ),
        "summary": "Klant eet vol-au-vent.",
        "last_question": "Welke geur of smaak zoek je?",
        "phase": "completed",
        "feedback_status": "pending",
        "handoff_categories_v1": '{"foodpair_witvlees_gevogelte":true}',
    }

    split_section_result = asyncio.run(
        service.ask(current_answer="Nog iets", state=split_section_state)
    )

    assert (
        "8. Gewenste geur/smaak (vrije tekst): citrus en mineraal"
        in split_section_result.state["questionnaire"]
    )
    assert (
        "8. Gewenste geur/smaak (vrije tekst): citrus en mineraal"
        in split_section_result.state["wine_questionnaire"]
    )
    assert "smaak_" not in split_section_result.state["questionnaire"]
    assert "smaak_" not in split_section_result.state["wine_questionnaire"]
    assert "geur_" not in split_section_result.state["questionnaire"]
    assert "geur_" not in split_section_result.state["wine_questionnaire"]


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


def test_ask_uses_conversation_language_for_off_topic_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    current_question = "Which fit are you looking for?"

    def fake_detect_langs(text: str):
        if "I am looking for dark blue jeans for work." in text:
            return [FakeLangCandidate("en", 0.99)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    result = asyncio.run(
        service.ask(
            current_answer="Prima",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "Nederlandse samenvatting",
                "last_question": current_question,
                "conversation_language": "en",
                "recent_user_turns": "I am looking for dark blue jeans for work.",
            },
        )
    )

    assert result.response_type == "answer"
    assert any(
        result.text == line.format(question=current_question)
        for line in OFF_TOPIC_REDIRECT_EN
    )
    assert result.state.get("conversation_language") == "en"


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


def test_ask_infers_legacy_feedback_language_from_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def fake_detect_langs(text: str):
        if text.startswith("English summary should keep the legacy language."):
            return [FakeLangCandidate("en", 0.99)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    result = asyncio.run(
        service.ask(
            current_answer="Maybe",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "English summary should keep the legacy language.",
                "last_question": "AI question",
                "phase": "awaiting_feedback",
                "feedback_status": "pending",
                "final_summary": "English summary should keep the legacy language.",
                "post_summary_language": "",
                "conversation_language": "",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text in FEEDBACK_QUESTION_EN
    assert result.quick_replies == [
        {"label": "Yes", "value": "yes"},
        {"label": "No", "value": "no"},
    ]
    assert result.state["conversation_language"] == "en"
    assert result.state["post_summary_language"] == "en"


def test_ask_uses_conversation_language_for_feedback_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def fake_detect_langs(text: str):
        if text.startswith("English summary should not decide language"):
            return [FakeLangCandidate("en", 0.99)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    result = asyncio.run(
        service.ask(
            current_answer="Maybe",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "English summary should not decide language.",
                "last_question": "AI question",
                "phase": "awaiting_feedback",
                "feedback_status": "pending",
                "final_summary": "English summary should not decide language.",
                "post_summary_language": "",
                "conversation_language": "nl",
                "recent_user_turns": "Hallo\nIk zoek een jeans",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text in FEEDBACK_QUESTION_NL
    assert result.quick_replies == [
        {"label": "Ja", "value": "yes"},
        {"label": "Nee", "value": "no"},
    ]
    assert result.state.get("conversation_language") == "nl"


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


def test_ask_returns_english_handoff_for_english_conversation(
    monkeypatch: pytest.MonkeyPatch,
):
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

    def fake_detect_langs(text: str):
        if "I am looking for blue jeans with a straight fit." in text:
            return [FakeLangCandidate("en", 0.99)]
        if text == "Yes":
            return [FakeLangCandidate("en", 0.99)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    result = asyncio.run(
        service.ask(
            current_answer="Yes",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
                "conversation_language": "en",
                "recent_user_turns": "I am looking for blue jeans with a straight fit.",
            },
        )
    )

    assert result.response_type == "answer"
    assert any(result.text.startswith(line) for line in HANDOFF_EN)
    assert result.follow_up_question in FEEDBACK_QUESTION_EN
    assert result.state["conversation_language"] == "en"
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


def test_ask_returns_french_handoff_for_french_conversation(
    monkeypatch: pytest.MonkeyPatch,
):
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

    def fake_detect_langs(text: str):
        if "Je cherche un jean bleu avec une coupe droite." in text:
            return [FakeLangCandidate("fr", 0.99)]
        if text == "Oui":
            return [FakeLangCandidate("fr", 0.99)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    result = asyncio.run(
        service.ask(
            current_answer="Oui",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
                "conversation_language": "fr",
                "recent_user_turns": "Je cherche un jean bleu avec une coupe droite.",
            },
        )
    )

    assert result.response_type == "answer"
    assert any(result.text.startswith(line) for line in HANDOFF_FR)
    assert result.follow_up_question in FEEDBACK_QUESTION_FR
    assert result.state["conversation_language"] == "fr"
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


def test_ask_uses_conversation_language_for_terminated_goodbye(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def fake_detect_langs(text: str):
        if "I am looking for dark blue jeans for work." in text:
            return [FakeLangCandidate("en", 0.99)]
        return [FakeLangCandidate("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    result = asyncio.run(
        service.ask(
            current_answer="ok",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "Nederlandse samenvatting",
                "last_question": "AI question",
                "phase": "terminated",
                "conversation_language": "en",
                "recent_user_turns": "I am looking for dark blue jeans for work.",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text in GOODBYE_EN
    assert result.state.get("conversation_language") == "en"


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


def test_ask_simplifies_aanlass_typo_in_next_question():
    question_llm = FakeLLM(
        ["Voor welk aanlass, stemming of setting zoek je deze jeans?"]
    )
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
    assert (
        result.text == "Voor welk aanleiding, stemming of setting zoek je deze jeans?"
    )


def test_ask_rewrites_unsupported_summary_language_to_preferred_language():
    question_llm = FakeLLM(["Klant zoekt een lange mom fit of straight fit jeans."])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Filled\n1. Occasion, Vibe, Setting: Filled"]
    )
    summarizer_llm = FakeLLM(
        [
            "La persona busca unos jeans largos mom o straight fit para un contexto casual."
        ]
    )
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
            current_answer="Ik zoek een lange mom of straight fit jeans",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.summary == "Klant zoekt een lange mom fit of straight fit jeans."
    assert result.state["post_summary_language"] == "nl"
    assert len(question_llm.calls) == 1


def test_ask_rewrites_summary_when_primary_language_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
):
    question_llm = FakeLLM(["Klant zoekt een lange mom fit of straight fit jeans."])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Filled\n1. Occasion, Vibe, Setting: Filled"]
    )
    summarizer_llm = FakeLLM(
        [
            "La persona busca unos jeans largos mom o straight fit para un contexto casual."
        ]
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    def fake_language_candidates(text: str) -> list[tuple[str, float]]:
        if text.startswith("La persona busca"):
            return [("es", 0.61), ("en", 0.39)]
        if text.startswith("Klant zoekt"):
            return [("nl", 0.99)]
        return [("nl", 0.99)]

    monkeypatch.setattr(
        "pixie_questioner.chat.summary.detect_language_candidates",
        fake_language_candidates,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek een lange mom of straight fit jeans",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "AI question",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.summary == "Klant zoekt een lange mom fit of straight fit jeans."
    assert result.state["post_summary_language"] == "nl"
    assert len(question_llm.calls) == 1


def test_ask_logs_warning_when_summary_rewrite_fails(caplog) -> None:
    notulist_llm = FakeLLM(
        ["0. Initial user question: Filled\n1. Occasion, Vibe, Setting: Filled"]
    )
    summarizer_llm = FakeLLM(
        [
            "La persona busca unos jeans largos mom o straight fit para un contexto casual."
        ]
    )
    service = ChatService(
        llm=RaisingLLM(RuntimeError("rewrite failed")),
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
    )

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(
            service.ask(
                current_answer="Ik zoek een lange mom of straight fit jeans",
                state={
                    "questionnaire": "Prev []",
                    "summary": "Bestaande samenvatting",
                    "last_question": "AI question",
                },
            )
        )

    assert result.response_type == "answer"
    assert result.summary == "Bestaande samenvatting"
    assert "rewrite summary" in caplog.text


def test_ask_logs_warning_when_relevance_classifier_fails(caplog) -> None:
    question_llm = FakeLLM(["Volgende vraag"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Hallo\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Klant zoekt hulp."])
    relevance_classifier = RaisingRelevanceClassifier(RuntimeError("classifier failed"))
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_DEFAULT_POLICY,
        answer_relevance_classifier=relevance_classifier,
    )

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(
            service.ask(
                current_answer="Hallo",
                state={
                    "questionnaire": QUESTIONNAIRE_TEMPLATE,
                    "summary": "",
                    "last_question": "Waar wil je de jeans voor dragen?",
                },
            )
        )

    assert result.text == "Volgende vraag"
    assert len(relevance_classifier.calls) == 1
    assert "relevance classifier" in caplog.text


def test_ask_logs_warning_when_handoff_classifier_fails(caplog) -> None:
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(
        [
            "0. Dish description (free text from customer): rundstoofvlees\n"
            "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
            "2. Main protein (fish or seafood, poultry, red meat, wild, or none): red meat\n"
            "3. Cooking method (BBQ or grill, stew, or none): stew\n"
            "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Rioja\n"
            "7. Maximum budget in euro: 25\n"
            "8. Gewenste geur/smaak (vrije tekst): fruitig en fris"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet rundstoofvlees."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        handoff_classifier=_make_handoff_classifier(
            RaisingLLM(RuntimeError("handoff failed"))
        ),
    )

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(
            service.ask(
                current_answer="Rundstoofvlees met rode wijn, liefst uit Rioja",
                state={
                    "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                    "summary": "",
                    "last_question": "Welk gerecht eten jullie?",
                    "intake_route": "food_pairing",
                    "dish_follow_up_count": "1",
                },
            )
        )

    assert result.response_type == "answer"
    assert json.loads(result.state["handoff_categories_v1"]) == {}
    assert "handoff core categories" in caplog.text


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


def test_ask_focuses_next_question_on_first_blank_field():
    questionnaire_template = (
        "0. Initial user question: []\n"
        "1. Occasion, Vibe, Setting: []\n"
        "2. What do you like to wear or what are your preferences: []"
    )
    question_llm = FakeLLM(["Welke gelegenheid heb je in gedachten?"])
    notulist_llm = FakeLLM(
        [
            "0. Initial user question: Ik zoek een jeans\n"
            "1. Occasion, Vibe, Setting: []\n"
            "2. What do you like to wear or what are your preferences: []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt een jeans."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=questionnaire_template,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Ik zoek een nieuwe jeans",
            state={
                "questionnaire": questionnaire_template,
                "summary": "",
                "last_question": "Initial question:",
            },
        )
    )

    assert result.response_type == "answer"
    question_prompt = question_llm.calls[0][0]["content"]
    assert "1. Occasion, Vibe, Setting: []" in question_prompt
    assert (
        "2. What do you like to wear or what are your preferences: [pending later]"
        in question_prompt
    )


def test_ask_backfills_initial_user_question_from_first_answer():
    questionnaire_template = (
        "0. Initial user question: []\n"
        "1. Occasion, Vibe, Setting: []\n"
        "2. What do you like to wear or what are your preferences: []"
    )
    question_llm = FakeLLM(["Voor welke gelegenheid zoek je deze jeans?"])
    notulist_llm = FakeLLM(
        [
            "0. Initial user question: []\n"
            "1. Occasion, Vibe, Setting: []\n"
            "2. What do you like to wear or what are your preferences: Jeans lang mom of straight fit"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt een jeans."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=questionnaire_template,
        domain_policy=_DEFAULT_POLICY,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Jeans lang mom of straight fit",
            state={},
        )
    )

    assert result.response_type == "answer"
    assert (
        "0. Initial user question: Jeans lang mom of straight fit"
        in result.state["questionnaire"]
    )


def test_ask_does_not_reask_initial_user_question_after_backfill():
    questionnaire_template = (
        "0. Initial user question: []\n"
        "1. Occasion, Vibe, Setting: []\n"
        "2. What do you like to wear or what are your preferences: []\n"
        "3. Most Important Aspects: []"
    )
    question_llm = FakeLLM(
        [
            "Voor welke gelegenheid zoek je deze jeans?",
            "Wat is het belangrijkste dat je zoekt in die jeans?",
        ]
    )
    notulist_llm = FakeLLM(
        [
            "0. Initial user question: []\n"
            "1. Occasion, Vibe, Setting: []\n"
            "2. What do you like to wear or what are your preferences: Jeans lang mom of straight fit\n"
            "3. Most Important Aspects: []",
            "0. Initial user question: []\n"
            "1. Occasion, Vibe, Setting: Casual\n"
            "2. What do you like to wear or what are your preferences: Jeans lang mom of straight fit\n"
            "3. Most Important Aspects: []",
        ]
    )
    summarizer_llm = FakeLLM(
        ["Klant zoekt een jeans.", "Klant zoekt een casual jeans."]
    )
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=questionnaire_template,
        domain_policy=_DEFAULT_POLICY,
    )

    first_result = asyncio.run(
        service.ask(
            current_answer="Jeans lang mom of straight fit",
            state={},
        )
    )
    second_result = asyncio.run(
        service.ask(
            current_answer="Casual",
            state=first_result.state,
        )
    )

    second_prompt = question_llm.calls[1][0]["content"]
    assert "0. Initial user question: []" not in second_prompt
    assert (
        "0. Initial user question: Jeans lang mom of straight fit"
        in second_result.state["questionnaire"]
    )


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


def test_question_asker_receives_last_response_on_follow_up_turn():
    """Previous assistant response is injected into the question-asker system prompt."""
    question_llm = FakeLLM(["Next question"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: User answer\n1. Occasion, Vibe, Setting: []"]
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

    asyncio.run(
        service.ask(
            current_answer="User answer",
            state={
                "questionnaire": "Prev []",
                "summary": "Prev summary",
                "last_question": "Eerder gestelde vraag?",
            },
        )
    )

    system_prompt = question_llm.calls[-1][0]["content"]
    assert "Eerder gestelde vraag?" in system_prompt


def test_question_asker_omits_last_response_on_initial_turn():
    """The INITIAL_QUESTION sentinel must not be injected into the question-asker prompt."""
    question_llm = FakeLLM(["Opening question"])
    notulist_llm = FakeLLM(
        ["0. Initial user question: Hallo\n1. Occasion, Vibe, Setting: []"]
    )
    summarizer_llm = FakeLLM(["Summary"])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_token="TOKEN"),
    )

    asyncio.run(
        service.ask(
            current_answer="Hallo",
            state={
                "questionnaire": QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Initial question:",
            },
        )
    )

    system_prompt = question_llm.calls[-1][0]["content"]
    assert "Initial question:" not in system_prompt
    assert "Your previous response" not in system_prompt


def test_aroma_taste_fallback_when_notulist_leaves_field_8_blank():
    """Regression: when the notulist fails to fill field 8 (aroma/taste) after the user
    answered, the fallback must fill it verbatim and complete the intake — instead of
    crashing with 'Question asker returned an empty question'.

    This reproduces the scenario where:
    - All wine fields (5-7) are filled, field 8 is the last blank wine field
    - The notulist returns the questionnaire with field 8 still []
    - Without the fallback, the question asker would refuse to re-ask and return empty
    """
    _fully_filled_except_field_8 = (
        "0. Dish description (free text from customer): gegrilde kip\n"
        "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
        "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
        "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
        "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none\n"
        "5. Kleur van de wijn (wit, rood, rosé): wit\n"
        "6. Herkomst (land en/of regio): Frankrijk\n"
        "7. Maximum budget in euro: 20\n"
        "8. Gewenste geur/smaak (vrije tekst): []"
    )
    # Notulist returns the questionnaire with field 8 still blank (simulates LLM failure
    # to capture the aroma/taste answer).
    notulist_llm = FakeLLM([_fully_filled_except_field_8])
    summarizer_llm = FakeLLM(["Klant wil een frisse wijn bij gegrilde kip."])
    handoff_llm = FakeLLM(['{"foodpair_witvlees_gevogelte":true,"kleur":"wit"}'])
    question_llm = FakeLLM(["Should not be called"])

    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="fris en fruitig",
            state={
                "questionnaire": _fully_filled_except_field_8,
                "dish_questionnaire": (
                    "0. Dish description (free text from customer): gegrilde kip\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): main course\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): poultry\n"
                    "3. Cooking method (BBQ or grill, stew, or none): BBQ or grill\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): none"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): wit\n"
                    "6. Herkomst (land en/of regio): Frankrijk\n"
                    "7. Maximum budget in euro: 20\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "Klant wil wijn bij gegrilde kip.",
                "last_question": "Welke geur of smaak zoek je?",
                "intake_route": "food_pairing",
                "intake_stage": "wine",
                "wine_follow_up_count": "3",
            },
        )
    )

    # Intake must complete (not crash and not re-ask field 8)
    assert result.state["phase"] == "completed"
    # Field 8 must be filled with the verbatim user answer via fallback
    assert "fris en fruitig" in result.state["questionnaire"]
    assert "fris en fruitig" in result.state.get("wine_questionnaire", "")
    # Question asker must NOT have been called
    assert question_llm.calls == []


def test_ask_retries_question_asker_when_first_response_is_not_a_question():
    question_llm = FakeLLM(
        ["Dank u, dat helpt ons goed verder", "Wat is je maximum budget in euro?"]
    )
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Frankrijk\n"
            "7. Maximum budget in euro: []\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt rode wijn uit Frankrijk."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="20 euro",
            state={
                "questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): Frankrijk\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): Frankrijk\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": "Wat is je maximum budget?",
                "intake_route": "general_wine",
                "intake_stage": "wine",
                "wine_follow_up_count": "2",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text == "Wat is je maximum budget in euro?"
    assert result.state["last_question"] == "Wat is je maximum budget in euro?"
    assert result.state["phase"] == "intake"
    assert len(question_llm.calls) == 2
    second_prompt = question_llm.calls[1][0]["content"]
    assert "not a valid customer question" in second_prompt
    assert "Do not use internal field labels." in second_prompt


def test_ask_uses_static_question_for_aroma_taste_field():
    """Field 8 bypasses the question_asker LLM and uses a static question."""
    question_llm = FakeLLM([])
    notulist_llm = FakeLLM(
        [
            "5. Kleur van de wijn (wit, rood, rosé): rood\n"
            "6. Herkomst (land en/of regio): Frankrijk\n"
            "7. Maximum budget in euro: 20\n"
            "8. Gewenste geur/smaak (vrije tekst): []"
        ]
    )
    summarizer_llm = FakeLLM(["Klant zoekt rode wijn uit Frankrijk rond 20 euro."])
    service = ChatService(
        llm=question_llm,
        question_asker_prompt="PROMPT {summary} {completion_token}",
        notulist=_make_notulist(notulist_llm),
        summarizer=_make_summarizer(summarizer_llm),
        questionnaire_template=CRETSKENS_QUESTIONNAIRE_TEMPLATE,
        domain_policy=_replace_policy(completion_mode="categorized_handoff"),
    )

    result = asyncio.run(
        service.ask(
            current_answer="20 euro",
            state={
                "questionnaire": (
                    "0. Dish description (free text from customer): [pending later]\n"
                    "1. Course type (aperitif, main course, cheese course, or dessert): [pending later]\n"
                    "2. Main protein (fish or seafood, poultry, red meat, wild, or none): [pending later]\n"
                    "3. Cooking method (BBQ or grill, stew, or none): [pending later]\n"
                    "4. Cheese type (blue cheese, soft cheese, hard cheese, or none): [pending later]\n"
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): Frankrijk\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "wine_questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): rood\n"
                    "6. Herkomst (land en/of regio): Frankrijk\n"
                    "7. Maximum budget in euro: []\n"
                    "8. Gewenste geur/smaak (vrije tekst): []"
                ),
                "summary": "",
                "last_question": "Wat is je maximum budget?",
                "intake_route": "general_wine",
                "intake_stage": "wine",
                "wine_follow_up_count": "2",
            },
        )
    )

    assert result.response_type == "answer"
    assert result.text.startswith("Welke geuren of smaken spreken je aan bij wijn?")
    assert "citrus" in result.text
    assert "niet" in result.text
    assert "Gewenste geur/smaak" not in result.text
    assert result.state["last_question"] == result.text
    assert result.state["phase"] == "intake"
    assert len(question_llm.calls) == 0
