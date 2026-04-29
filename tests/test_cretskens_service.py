# ruff: noqa: S101,S105,S106
from __future__ import annotations

import asyncio

from pixie_questioner.chat.types import ChatResult
from pixie_questioner.cretskens_service import (
    CRETSKENS_ADVISOR_STATE_KEY,
    CRETSKENS_FEEDBACK_STATE_KEY,
    CRETSKENS_FEEDBACK_SUMMARY_TEXT_KEY,
    CRETSKENS_FLOW_ADVISOR,
    CRETSKENS_FLOW_INTAKE,
    CRETSKENS_FLOW_KEY,
    CRETSKENS_INTAKE_STATE_KEY,
    CretskensService,
)
from pixie_questioner.wine_advisor.types import WineAdvisorResult


class StubIntakeService:
    def __init__(self, result: ChatResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    async def ask(
        self,
        current_answer: str,
        state: dict[str, object] | None,
    ) -> ChatResult:
        await asyncio.sleep(0)
        self.calls.append((current_answer, state))
        return self.result


class StubRecommendationService:
    def __init__(
        self,
        ask_result: WineAdvisorResult,
        recommend_result: WineAdvisorResult | None = None,
    ) -> None:
        self.ask_result = ask_result
        self.recommend_result = recommend_result or ask_result
        self.ask_calls: list[tuple[str, dict[str, object] | None]] = []
        self.recommend_calls: list[tuple[str, str]] = []

    async def ask(
        self,
        *,
        current_answer: str,
        state: dict[str, object] | None,
    ) -> WineAdvisorResult:
        await asyncio.sleep(0)
        self.ask_calls.append((current_answer, state))
        return self.ask_result

    async def recommend_from_context(
        self,
        *,
        working_context: str,
        meal_mode: str,
    ) -> WineAdvisorResult:
        await asyncio.sleep(0)
        self.recommend_calls.append((working_context, meal_mode))
        return self.recommend_result


def test_ask_wraps_incomplete_intake_state_under_intake_flow_key() -> None:
    intake_service = StubIntakeService(
        ChatResult(
            text="Welke kleur wijn heeft je voorkeur?",
            response_type="answer",
            state={
                "phase": "intake",
                "last_question": "Welke kleur wijn heeft je voorkeur?",
            },
            summary=None,
        )
    )
    recommendation_service = StubRecommendationService(
        WineAdvisorResult(
            text="unused",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        )
    )
    service = CretskensService(
        intake_service=intake_service,
        recommendation_service=recommendation_service,
    )

    result = asyncio.run(service.ask(current_answer="Ik zoek wijn", state=None))

    assert result.response_type == "question"
    assert result.text == "Welke kleur wijn heeft je voorkeur?"
    assert result.state[CRETSKENS_FLOW_KEY] == CRETSKENS_FLOW_INTAKE
    assert result.state[CRETSKENS_INTAKE_STATE_KEY] == {
        "phase": "intake",
        "last_question": "Welke kleur wijn heeft je voorkeur?",
    }
    assert intake_service.calls == [("Ik zoek wijn", None)]
    assert recommendation_service.ask_calls == []
    assert recommendation_service.recommend_calls == []


def test_ask_completed_intake_calls_recommendation_with_built_context_and_meal_mode() -> (
    None
):
    intake_service = StubIntakeService(
        ChatResult(
            text="Dank je, ik geef aanbevelingen.",
            response_type="answer",
            state={
                "phase": "completed",
                "summary": "Klant zoekt wijn bij vis.",
                "questionnaire": (
                    "5. Kleur van de wijn (wit, rood, rosé): wit\n"
                    "6. Herkomst (land en/of regio): Loire (AOC Cheverny)"
                ),
                "handoff_categories_v1": '{"foodpair_vis_zeevruchten":true}',
                "intake_route": "food_pairing",
            },
            summary=None,
        )
    )
    recommendation_service = StubRecommendationService(
        ask_result=WineAdvisorResult(
            text="unused",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        ),
        recommend_result=WineAdvisorResult(
            text="Hier zijn enkele passende opties.",
            response_type="recommendations",
            state={
                "phase": "completed",
                "selected_barcodes": ["w1"],
                "working_context": (
                    "Summary: Klant zoekt wijn bij vis. "
                    "Questionnaire: 5. Kleur van de wijn (wit, rood, rosé): wit "
                    "6. Herkomst (land en/of regio): Loire (AOC Cheverny)"
                ),
            },
            recommendations=[],
        ),
    )
    service = CretskensService(
        intake_service=intake_service,
        recommendation_service=recommendation_service,
    )

    result = asyncio.run(
        service.ask(current_answer="We eten gegrilde zeebaars", state=None)
    )

    assert result.response_type == "recommendations"
    assert result.state[CRETSKENS_FLOW_KEY] == CRETSKENS_FLOW_ADVISOR
    assert result.state[CRETSKENS_ADVISOR_STATE_KEY]["phase"] == "completed"
    assert result.state[CRETSKENS_ADVISOR_STATE_KEY]["selected_barcodes"] == ["w1"]
    assert (
        result.state[CRETSKENS_ADVISOR_STATE_KEY][CRETSKENS_FEEDBACK_SUMMARY_TEXT_KEY]
        == "Hier zijn enkele passende opties."
    )
    assert result.state[CRETSKENS_FEEDBACK_STATE_KEY] == {
        "step": "ease_of_use",
        "prompt_issued": True,
        "ease_of_use": "",
        "reuse": "",
        "suggestions_satisfaction": "",
        "summary_text": "Hier zijn enkele passende opties.",
    }
    assert len(recommendation_service.recommend_calls) == 1
    working_context, meal_mode = recommendation_service.recommend_calls[0]
    assert meal_mode == "pairing"
    assert "Summary:\nKlant zoekt wijn bij vis." in working_context
    assert "Questionnaire:\n5. Kleur van de wijn" in working_context
    assert 'Handoff categories:\n{"foodpair_vis_zeevruchten":true}' in working_context


def test_ask_uses_legacy_advisor_state_shape_as_advisor_flow() -> None:
    intake_service = StubIntakeService(
        ChatResult(
            text="unused",
            response_type="answer",
            state={"phase": "intake"},
            summary=None,
        )
    )
    recommendation_service = StubRecommendationService(
        WineAdvisorResult(
            text="Welke stijl zoek je?",
            response_type="question",
            state={"phase": "intake", "turn_count": 2},
            recommendations=None,
        )
    )
    service = CretskensService(
        intake_service=intake_service,
        recommendation_service=recommendation_service,
    )

    legacy_advisor_state = {
        "phase": "intake",
        "turn_count": 1,
        "meal_mode": "pairing",
        "working_context": "Klant zoekt wijn bij vis.",
        "chat_history": [{"role": "assistant", "text": "Welke kleur?"}],
    }

    result = asyncio.run(service.ask(current_answer="Wit", state=legacy_advisor_state))

    assert intake_service.calls == []
    assert recommendation_service.ask_calls == [("Wit", legacy_advisor_state)]
    assert result.response_type == "question"
    assert result.state[CRETSKENS_FLOW_KEY] == CRETSKENS_FLOW_ADVISOR
    assert result.state[CRETSKENS_ADVISOR_STATE_KEY] == {
        "phase": "intake",
        "turn_count": 2,
    }


def test_ask_prefers_wrapped_intake_state_payload_over_legacy_like_keys() -> None:
    intake_service = StubIntakeService(
        ChatResult(
            text="Welke stijl zoek je?",
            response_type="answer",
            state={"phase": "intake", "last_question": "Welke stijl zoek je?"},
            summary=None,
        )
    )
    recommendation_service = StubRecommendationService(
        WineAdvisorResult(
            text="unused",
            response_type="question",
            state={"phase": "intake", "turn_count": 1},
            recommendations=None,
        )
    )
    service = CretskensService(
        intake_service=intake_service,
        recommendation_service=recommendation_service,
    )

    wrapped_state = {
        CRETSKENS_FLOW_KEY: CRETSKENS_FLOW_INTAKE,
        CRETSKENS_INTAKE_STATE_KEY: {
            "phase": "intake",
            "last_question": "Vorige intakevraag",
            "intake_route": "food_pairing",
        },
        # These keys should not force advisor flow when wrapped intake payload exists.
        "phase": "completed",
        "turn_count": 99,
    }

    result = asyncio.run(service.ask(current_answer="Verder", state=wrapped_state))

    assert intake_service.calls == [
        (
            "Verder",
            {
                "phase": "intake",
                "last_question": "Vorige intakevraag",
                "intake_route": "food_pairing",
            },
        )
    ]
    assert recommendation_service.ask_calls == []
    assert result.state[CRETSKENS_FLOW_KEY] == CRETSKENS_FLOW_INTAKE


def test_ask_collects_three_feedback_questions_after_recommendations() -> None:
    intake_service = StubIntakeService(
        ChatResult(
            text="Dank je, ik geef aanbevelingen.",
            response_type="answer",
            state={
                "phase": "completed",
                "summary": "Klant zoekt wijn.",
                "questionnaire": "5. Kleur van de wijn (wit, rood, rosé): wit",
                "handoff_categories_v1": "{}",
                "intake_route": "food_pairing",
            },
            summary=None,
        )
    )
    recommendation_service = StubRecommendationService(
        ask_result=WineAdvisorResult(
            text="unused",
            response_type="recommendations",
            state={"phase": "completed", "selected_barcodes": ["w1"]},
            recommendations=[],
        ),
        recommend_result=WineAdvisorResult(
            text="Hier zijn enkele passende opties.",
            response_type="recommendations",
            state={
                "phase": "completed",
                "selected_barcodes": ["w1"],
                "working_context": (
                    "Summary:\nKlant zoekt wijn.\n\n"
                    "Questionnaire:\n5. Kleur van de wijn (wit, rood, rosé): wit"
                ),
            },
            recommendations=[],
        ),
    )
    service = CretskensService(
        intake_service=intake_service,
        recommendation_service=recommendation_service,
    )

    first = asyncio.run(
        service.ask(current_answer="We eten gegrilde zeebaars", state=None)
    )
    assert first.response_type == "recommendations"
    assert first.follow_up_question == "Hoe makkelijk vond je de assistent in gebruik?"
    assert first.quick_replies == [
        {"label": "1", "value": "1"},
        {"label": "2", "value": "2"},
        {"label": "3", "value": "3"},
        {"label": "4", "value": "4"},
        {"label": "5", "value": "5"},
    ]
    assert first.state[CRETSKENS_FEEDBACK_STATE_KEY]["prompt_issued"] is True

    second = asyncio.run(service.ask(current_answer="4", state=first.state))
    assert second.response_type == "question"
    assert second.text == "Zou je me nog eens gebruiken?"
    assert second.quick_replies == [
        {"label": "ja", "value": "ja"},
        {"label": "nee", "value": "nee"},
    ]

    third = asyncio.run(service.ask(current_answer="ja", state=second.state))
    assert third.response_type == "question"
    assert third.text == "In welke mate ben je tevreden met de wijnsuggesties?"
    assert third.quick_replies == [
        {"label": "1", "value": "1"},
        {"label": "2", "value": "2"},
        {"label": "3", "value": "3"},
        {"label": "4", "value": "4"},
        {"label": "5", "value": "5"},
    ]

    fourth = asyncio.run(service.ask(current_answer="5", state=third.state))
    assert fourth.response_type == "question"
    assert fourth.text == "Bedankt voor je feedback."
    assert fourth.quick_replies is None
    assert fourth.state[CRETSKENS_FEEDBACK_STATE_KEY]["step"] == "done"


def test_ask_repeats_feedback_question_when_answer_is_invalid() -> None:
    service = CretskensService(
        intake_service=StubIntakeService(
            ChatResult(
                text="unused",
                response_type="answer",
                state={"phase": "intake"},
                summary=None,
            )
        ),
        recommendation_service=StubRecommendationService(
            ask_result=WineAdvisorResult(
                text="unused",
                response_type="recommendations",
                state={"phase": "completed", "selected_barcodes": ["w1"]},
                recommendations=[],
            )
        ),
    )

    state = {
        CRETSKENS_FLOW_KEY: CRETSKENS_FLOW_ADVISOR,
        CRETSKENS_ADVISOR_STATE_KEY: {
            "phase": "completed",
            "selected_barcodes": ["w1"],
            "working_context": (
                "Summary:\nKlant zoekt een soepele rode wijn.\n\nQuestionnaire:\n..."
            ),
        },
        CRETSKENS_FEEDBACK_STATE_KEY: {
            "step": "ease_of_use",
            "prompt_issued": True,
            "ease_of_use": "",
            "reuse": "",
            "suggestions_satisfaction": "",
        },
    }

    result = asyncio.run(service.ask(current_answer="heel goed", state=state))

    assert result.response_type == "question"
    assert result.text == "Hoe makkelijk vond je de assistent in gebruik?"
    assert result.state[CRETSKENS_FEEDBACK_STATE_KEY]["step"] == "ease_of_use"
