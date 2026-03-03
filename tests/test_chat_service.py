# ruff: noqa: S101,S105,S106
import asyncio
import json
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


def _make_aroma_taste_classifier(llm):
    return LLMPromptInvoker(
        llm=llm,
        system_prompt_template="K:{summary}\nQ:{current_question}\nA:{current_answer}",
        error_label="Aroma taste classifier",
    )


def _make_handoff_classifier(llm):
    return LLMPromptInvoker(
        llm=llm,
        system_prompt_template="K:{summary}\nQ:{current_question}\nA:{current_answer}",
        error_label="Handoff classifier",
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


def test_detect_preferred_language_prefers_supported_on_small_score_gap(monkeypatch):
    service = _make_service_for_language_detection()

    def fake_detect_langs(_: str):
        return [FakeLangCandidate("de", 0.56), FakeLangCandidate("nl", 0.49)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    preferred_language = service._language.detect_preferred("kurzer text", "")

    assert preferred_language == "nl"


def test_detect_preferred_language_keeps_dutch_on_strong_unsupported_signal(
    monkeypatch,
):
    service = _make_service_for_language_detection()

    def fake_detect_langs(_: str):
        return [FakeLangCandidate("de", 0.96), FakeLangCandidate("nl", 0.02)]

    monkeypatch.setattr(
        "pixie_questioner.chat.language.detect_langs", fake_detect_langs
    )

    preferred_language = service._language.detect_preferred(
        "guten tag ich suche eine jeans", ""
    )

    assert preferred_language == "nl"


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
    aroma_llm = FakeLLM(['{"geur_citrus":true,"smaak_zoet":false}'])
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
        aroma_taste_classifier=_make_aroma_taste_classifier(aroma_llm),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="We eten vol-au-vent",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welk gerecht eten jullie?",
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
    assert '"geur_citrus": true' in result.text
    assert '"smaak_zoet": false' in result.text
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
        "geur_citrus": True,
        "smaak_zoet": False,
    }
    assert question_llm.calls == []
    assert len(aroma_llm.calls) == 1
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
    aroma_llm = FakeLLM(['{"smaak_fruitig":true,"smaak_fris_zuur":true}'])
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
        aroma_taste_classifier=_make_aroma_taste_classifier(aroma_llm),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Rundstoofvlees met rode wijn, liefst uit Rioja",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welk gerecht eten jullie?",
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff["regio"] == "Rioja"
    assert handoff["land"] == "Spanje"
    assert handoff["prijs"] == 25
    assert handoff["smaak_fruitig"] is True
    assert handoff["smaak_fris_zuur"] is True


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
    aroma_llm = FakeLLM(['{"smaak_kruidig":true,"smaak_bruisend":true}'])
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
        aroma_taste_classifier=_make_aroma_taste_classifier(aroma_llm),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="Zalm, wit, uit Spanje of Bordeaux",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welk gerecht eten jullie?",
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff["regio"] == "Bordeaux"
    assert handoff["land"] == "Frankrijk"
    assert handoff["prijs"] == 40
    assert handoff["smaak_kruidig"] is True
    assert handoff["smaak_bruisend"] is True


def test_ask_categorized_handoff_aroma_taste_last_statement_wins():
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
            "8. Gewenste geur/smaak (vrije tekst): fruitig, niet fruitig, maar toch fruitig"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde kip."])
    aroma_llm = FakeLLM(['{"smaak_fruitig":true}'])
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
        aroma_taste_classifier=_make_aroma_taste_classifier(aroma_llm),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="fruitig, niet fruitig, maar toch fruitig",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welke geur of smaak zoek je?",
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff["smaak_fruitig"] is True
    assert len(aroma_llm.calls) == 1


def test_ask_categorized_handoff_ignores_invalid_aroma_classifier_output():
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
            "8. Gewenste geur/smaak (vrije tekst): fruitig en fris"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde kip."])
    aroma_llm = FakeLLM(["geen geldig json antwoord"])
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
        aroma_taste_classifier=_make_aroma_taste_classifier(aroma_llm),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="fruitig en fris",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welke geur of smaak zoek je?",
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert "smaak_fruitig" not in handoff
    assert "smaak_fris_zuur" not in handoff


def test_ask_categorized_handoff_parses_wrapped_aroma_classifier_output():
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
            "8. Gewenste geur/smaak (vrije tekst): droog en fris"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde kip."])
    aroma_llm = FakeLLM(['Aroma flags: {"smaak_droog":"true","smaak_fris_zuur":true}'])
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
        aroma_taste_classifier=_make_aroma_taste_classifier(aroma_llm),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="droog en fris",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welke geur of smaak zoek je?",
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff["smaak_droog"] is True
    assert handoff["smaak_fris_zuur"] is True


def test_ask_categorized_handoff_parses_non_json_aroma_key_values():
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
            "8. Gewenste geur/smaak (vrije tekst): kruidig"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde kip."])
    aroma_llm = FakeLLM(["smaak_kruidig: true"])
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
        aroma_taste_classifier=_make_aroma_taste_classifier(aroma_llm),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="kruidig",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welke geur of smaak zoek je?",
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff["smaak_kruidig"] is True


def test_ask_categorized_handoff_retries_unstructured_aroma_classifier_output():
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
            "8. Gewenste geur/smaak (vrije tekst): kruidig"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde kip."])
    aroma_llm = FakeLLM(["kruidig", '{"smaak_kruidig":true}'])
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
        aroma_taste_classifier=_make_aroma_taste_classifier(aroma_llm),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="kruidig",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welke geur of smaak zoek je?",
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff["smaak_kruidig"] is True
    assert len(aroma_llm.calls) == 2


def test_ask_categorized_handoff_maps_generic_kruidig_key():
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
            "8. Gewenste geur/smaak (vrije tekst): kruidig"
        ]
    )
    summarizer_llm = FakeLLM(["Klant eet gegrilde kip."])
    aroma_llm = FakeLLM(['{"kruidig":true}'])
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
        aroma_taste_classifier=_make_aroma_taste_classifier(aroma_llm),
        handoff_classifier=_make_handoff_classifier(handoff_llm),
    )

    result = asyncio.run(
        service.ask(
            current_answer="kruidig",
            state={
                "questionnaire": CRETSKENS_QUESTIONNAIRE_TEMPLATE,
                "summary": "",
                "last_question": "Welke geur of smaak zoek je?",
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert handoff["geur_kruidig"] is True
    assert handoff["smaak_kruidig"] is True


def test_ask_completed_categorized_handoff_fallback_classifies_text_field8():
    question_llm = FakeLLM(["Should not be used"])
    notulist_llm = FakeLLM(["Should not be used"])
    summarizer_llm = FakeLLM(["Should not be used"])
    aroma_llm = FakeLLM(['{"smaak_kruidig":true}'])
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
        aroma_taste_classifier=_make_aroma_taste_classifier(aroma_llm),
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
    assert handoff["smaak_kruidig"] is True


def test_ask_categorized_handoff_omits_unparseable_price():
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
            },
        )
    )

    handoff = json.loads(result.state["handoff_categories_v1"])
    assert "prijs" not in handoff
    assert handoff["kleur"] == "rood"
    assert handoff["land"] == "Italië"


def test_ask_completed_phase_returns_stable_categorized_handoff_answer():
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
    assert '"geur_citrus": true' in result.text
    assert '"smaak_mineralig": true' in result.text
    assert result.summary is None
    assert result.follow_up_question is None
    assert result.quick_replies is None
    assert result.state["phase"] == "completed"
    assert (
        result.state["handoff_categories_v1"]
        == completed_state["handoff_categories_v1"]
    )
    assert question_llm.calls == []
    assert notulist_llm.calls == []
    assert summarizer_llm.calls == []


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
