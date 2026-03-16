# ruff: noqa: S101,S105,S106
from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pixie_questioner.wine_advisor.catalog_models import WineCatalogRecord
from pixie_questioner.wine_advisor.llm_controller import TurnDecision
from pixie_questioner.wine_advisor.service import WineAdvisorService


class StubController:
    def __init__(self, decisions: list[TurnDecision]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict[str, object]] = []

    async def decide(
        self,
        *,
        current_answer: str,
        working_context: str,
        turn_count: int,
        max_turns: int,
        current_meal_mode: str,
        last_question: str,
        conversation_history: str,
    ) -> TurnDecision:
        await asyncio.sleep(0)
        self.calls.append(
            {
                "current_answer": current_answer,
                "working_context": working_context,
                "turn_count": turn_count,
                "max_turns": max_turns,
                "current_meal_mode": current_meal_mode,
                "last_question": last_question,
                "conversation_history": conversation_history,
            }
        )
        if not self.decisions:
            raise AssertionError("No decision left in StubController")
        return self.decisions.pop(0)


@dataclass(frozen=True)
class _StubRankerSelection:
    barcodes: list[str]
    explanation: str
    intro_text: str = ""


class StubRanker:
    def __init__(
        self, selections: list[list[str]], explanations: list[str] | None = None
    ) -> None:
        self.selections = list(selections)
        self.explanations = list(explanations or [])
        self.calls: list[dict[str, object]] = []

    async def select_barcodes(
        self,
        *,
        working_context: str,
        candidate_records: list[WineCatalogRecord],
        candidate_barcodes: list[str],
        min_count: int,
        max_count: int,
    ) -> list[str]:
        await asyncio.sleep(0)
        self.calls.append(
            {
                "working_context": working_context,
                "candidate_records": list(candidate_records),
                "candidate_barcodes": list(candidate_barcodes),
                "min_count": min_count,
                "max_count": max_count,
            }
        )
        if not self.selections:
            return []
        return self.selections.pop(0)

    async def select_with_explanation(
        self,
        *,
        working_context: str,
        candidate_records: list[WineCatalogRecord],
        candidate_barcodes: list[str],
        min_count: int,
        max_count: int,
    ) -> _StubRankerSelection:
        selected = await self.select_barcodes(
            working_context=working_context,
            candidate_records=candidate_records,
            candidate_barcodes=candidate_barcodes,
            min_count=min_count,
            max_count=max_count,
        )
        explanation = ""
        if self.explanations:
            explanation = self.explanations.pop(0)
        return _StubRankerSelection(barcodes=selected, explanation=explanation)


class EchoRanker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def select_with_explanation(
        self,
        *,
        working_context: str,
        candidate_records: list[WineCatalogRecord],
        candidate_barcodes: list[str],
        min_count: int,
        max_count: int,
    ) -> _StubRankerSelection:
        await asyncio.sleep(0)
        self.calls.append(
            {
                "working_context": working_context,
                "candidate_records": list(candidate_records),
                "candidate_barcodes": list(candidate_barcodes),
                "min_count": min_count,
                "max_count": max_count,
            }
        )
        count = max(min_count, 1)
        return _StubRankerSelection(
            barcodes=list(candidate_barcodes[:count]),
            explanation="echo selection",
        )


async def _observe_coroutine_yield(coro: Awaitable[object]) -> list[str]:
    order: list[str] = []
    loop = asyncio.get_running_loop()
    loop.call_soon(order.append, "loop")
    await coro
    order.append("done")
    await asyncio.sleep(0)
    return order


def _catalog() -> list[WineCatalogRecord]:
    return [
        WineCatalogRecord(
            barcode="w1",
            name="Wijn 1",
            price_eur=9.5,
            country="Frankrijk",
            region="Bordeaux",
            color="rood",
            grape_description="Merlot",
            aroma_description="Rood fruit",
            taste_description="Droog",
            foodpair_description="Rund",
        ),
        WineCatalogRecord(
            barcode="w2",
            name="Wijn 2",
            price_eur=11.0,
            country="Frankrijk",
            region="Rhône",
            color="rood",
            grape_description="Syrah",
            aroma_description="Kruidig",
            taste_description="Vol",
            foodpair_description="Lamsvlees",
        ),
        WineCatalogRecord(
            barcode="w3",
            name="Wijn 3",
            price_eur=8.0,
            country="Italië",
            region="Toscane",
            color="wit",
            grape_description="Trebbiano",
            aroma_description="Citrus",
            taste_description="Fris",
            foodpair_description="Vis",
        ),
        WineCatalogRecord(
            barcode="w4",
            name="Wijn 4",
            price_eur=13.0,
            country="Spanje",
            region="Rioja",
            color="rood",
            grape_description="Tempranillo",
            aroma_description="Donker fruit",
            taste_description="Kruidig",
            foodpair_description="Rund",
        ),
    ]


def test_ask_returns_one_question_in_intake_phase() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="question",
                question_text="Welke wijnkleur heb je in gedachten?",
                meal_mode="pairing",
                query_hints="rund",
            )
        ]
    )
    ranker = StubRanker([])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
    )

    result = asyncio.run(service.ask(current_answer="Ik eet rund", state={}))

    assert result.response_type == "question"
    assert result.text == "Welke wijnkleur heb je in gedachten?"
    assert result.recommendations is None
    assert result.state["phase"] == "intake"
    assert result.state["turn_count"] == 1


def test_service_collaborators_are_not_typed_as_any() -> None:
    hints = get_type_hints(WineAdvisorService)

    assert hints["turn_controller"] is not Any
    assert hints["ranker"] is not Any


def test_stub_controller_decide_yields_to_event_loop() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="question",
                question_text="Welke stijl zoek je?",
                meal_mode="pairing",
                query_hints="",
            )
        ]
    )

    order = asyncio.run(
        _observe_coroutine_yield(
            controller.decide(
                current_answer="Ik eet rund",
                working_context="Klant zoekt wijn",
                turn_count=1,
                max_turns=6,
                current_meal_mode="unknown",
                last_question="Wat eet je?",
                conversation_history="",
            )
        )
    )

    assert order[0] == "loop"


def test_stub_ranker_select_barcodes_yields_to_event_loop() -> None:
    ranker = StubRanker([["w1"]])

    order = asyncio.run(
        _observe_coroutine_yield(
            ranker.select_barcodes(
                working_context="Klant zoekt wijn",
                candidate_records=_catalog(),
                candidate_barcodes=["w1", "w2"],
                min_count=1,
                max_count=2,
            )
        )
    )

    assert order[0] == "loop"


def test_echo_ranker_select_with_explanation_yields_to_event_loop() -> None:
    ranker = EchoRanker()

    order = asyncio.run(
        _observe_coroutine_yield(
            ranker.select_with_explanation(
                working_context="Klant zoekt wijn",
                candidate_records=_catalog(),
                candidate_barcodes=["w1", "w2"],
                min_count=1,
                max_count=2,
            )
        )
    )

    assert order[0] == "loop"


def test_ask_can_complete_before_turn_six() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="pairing",
                query_hints="kruidig rund",
            )
        ]
    )
    ranker = StubRanker([["w4", "w1", "w2"]])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
    )

    result = asyncio.run(service.ask(current_answer="Doe maar suggesties", state={}))

    assert result.response_type == "recommendations"
    assert len(result.recommendations or []) == 3
    assert result.state["phase"] == "completed"
    assert result.state["turn_count"] == 1


def test_ask_forces_completion_at_turn_six() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="question",
                question_text="Nog een vraag",
                meal_mode="pairing",
                query_hints="rund",
            )
        ]
    )
    ranker = StubRanker([["w1", "w2", "w4"]])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Nog extra info",
            state={
                "phase": "intake",
                "turn_count": 5,
                "working_context": "Klant zoekt wijn bij rund",
            },
        )
    )

    assert result.response_type == "recommendations"
    assert result.state["phase"] == "completed"
    assert result.state["turn_count"] == 6


def test_ask_uses_min_count_three_when_enough_candidates() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="pairing",
                query_hints="rund",
            )
        ]
    )
    ranker = StubRanker([["w1", "w2", "w4"]])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
    )

    _ = asyncio.run(service.ask(current_answer="Geef advies", state={}))

    assert ranker.calls
    assert ranker.calls[0]["min_count"] == 3


def test_ask_uses_min_count_one_when_candidates_below_three() -> None:
    limited_catalog = _catalog()[:2]
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="pairing",
                query_hints="rund",
            )
        ]
    )
    ranker = StubRanker([["w1", "w2"]])
    service = WineAdvisorService(
        catalog_records=limited_catalog,
        turn_controller=controller,
        ranker=ranker,
    )

    _ = asyncio.run(service.ask(current_answer="Geef advies", state={}))

    assert ranker.calls
    assert ranker.calls[0]["min_count"] == 1


def test_ask_retries_with_min_count_one_when_ranker_returns_too_few() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="pairing",
                query_hints="rund",
            )
        ]
    )
    ranker = StubRanker([[], ["w1"]])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
    )

    result = asyncio.run(service.ask(current_answer="Geef advies", state={}))

    assert result.response_type == "recommendations"
    assert [item.barcode for item in (result.recommendations or [])] == ["w1"]
    assert len(ranker.calls) == 2
    assert ranker.calls[0]["min_count"] == 3
    assert ranker.calls[1]["min_count"] == 1


def test_ask_completed_phase_is_idempotent() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="question",
                question_text="Deze vraag mag nooit gebruikt worden",
                meal_mode="pairing",
                query_hints="",
            )
        ]
    )
    ranker = StubRanker([["w1", "w2", "w3"]])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Nog iets",
            state={
                "phase": "completed",
                "turn_count": 2,
                "selected_barcodes": ["w2", "w1"],
            },
        )
    )

    assert result.response_type == "recommendations"
    assert [item.barcode for item in (result.recommendations or [])] == ["w2", "w1"]
    assert len(controller.calls) == 0
    assert len(ranker.calls) == 0


def test_ask_recommendations_text_includes_recommended_wines() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="pairing",
                query_hints="kruidig rund",
            )
        ]
    )
    ranker = StubRanker([["w4", "w1", "w2"]])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
    )

    result = asyncio.run(service.ask(current_answer="Doe maar suggesties", state={}))

    assert result.response_type == "recommendations"
    assert result.text.splitlines()[0] == service.recommendation_intro
    assert "Wijn 4" in result.text
    assert "Wijn 1" in result.text
    assert "Wijn 2" in result.text
    assert "EUR" not in result.text
    assert "Geur: Donker fruit" in result.text
    assert "Smaak: Kruidig" in result.text


def test_ask_respects_controller_recommend_action_for_drinking_only() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="drinking_only",
                query_hints="frisse witte wijn",
            )
        ]
    )
    ranker = StubRanker([["w1", "w2", "w3"]])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Gewoon los",
            state={
                "phase": "intake",
                "turn_count": 1,
                "working_context": "Kan je me een wijn aanbevelen om deze avond te drinken?",
            },
        )
    )

    assert result.response_type == "recommendations"
    assert len(ranker.calls) == 1


def test_ask_shortlists_catalog_records_from_query_hints() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="pairing",
                query_hints="rioja",
            )
        ]
    )
    ranker = EchoRanker()
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
        candidate_limit=1,
    )

    result = asyncio.run(service.ask(current_answer="Vol-au-vent", state={}))

    assert result.response_type == "recommendations"
    assert len(ranker.calls) == 1
    ranker_call = ranker.calls[0]
    candidate_records = ranker_call["candidate_records"]
    candidate_barcodes = ranker_call["candidate_barcodes"]
    assert isinstance(candidate_records, list)
    assert isinstance(candidate_barcodes, list)
    assert len(candidate_records) == 1
    assert candidate_barcodes == ["w4"]
    assert [item.barcode for item in (result.recommendations or [])] == ["w4"]


def test_ask_question_without_query_hints_keeps_candidate_list_empty() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="question",
                question_text="Welke kleur wijn heb je in gedachten?",
                meal_mode="pairing",
                query_hints="",
            )
        ]
    )
    ranker = EchoRanker()
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
        candidate_limit=1,
    )

    result = asyncio.run(service.ask(current_answer="Ik zoek wijn", state={}))

    assert result.response_type == "question"
    assert len(ranker.calls) == 0
    assert result.state["candidate_barcodes"] == []


def test_ask_recommend_without_query_hints_uses_context_fallback() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="pairing",
                query_hints="",
            )
        ]
    )
    ranker = StubRanker([["w1"]])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
        candidate_limit=1,
    )

    result = asyncio.run(service.ask(current_answer="Geef suggesties", state={}))

    assert result.response_type == "recommendations"
    assert len(ranker.calls) == 1
    assert ranker.calls[0]["candidate_barcodes"] == ["w1"]


def test_ask_expands_candidates_after_strict_shortlist_ranker_failures() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="pairing",
                query_hints="rose australie",
            )
        ]
    )
    ranker = StubRanker([["w2"], ["w2"], ["w2", "w1", "w3"]])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
        candidate_limit=1,
    )

    result = asyncio.run(service.ask(current_answer="Geef suggesties", state={}))

    assert result.response_type == "recommendations"
    assert [item.barcode for item in (result.recommendations or [])] == [
        "w2",
        "w1",
        "w3",
    ]
    assert len(ranker.calls) == 3
    assert ranker.calls[0]["candidate_barcodes"] == ["w1"]
    expanded_candidate_barcodes = ranker.calls[2]["candidate_barcodes"]
    assert isinstance(expanded_candidate_barcodes, list)
    assert len(expanded_candidate_barcodes) > 1


def test_ask_relaxes_origin_filtered_context_after_empty_ranker_results() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="pairing",
                query_hints="bulgarije rood",
            )
        ]
    )
    ranker = StubRanker([[], [], [], ["w1"]])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
        candidate_limit=1,
    )

    result = asyncio.run(
        service.ask(
            current_answer="Je cherche un vin rouge de Bulgarie",
            state={},
        )
    )

    assert result.response_type == "recommendations"
    assert [item.barcode for item in (result.recommendations or [])] == ["w1"]
    assert len(ranker.calls) == 4
    assert "bulgarie" in str(ranker.calls[0]["working_context"]).casefold()
    assert "bulgarie" not in str(ranker.calls[-1]["working_context"]).casefold()


def test_recommend_from_context_asks_question_when_controller_says_question() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="question",
                question_text="Welke stijl zoek je?",
                meal_mode="drinking_only",
                query_hints="",
            )
        ]
    )
    ranker = EchoRanker()
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
    )

    result = asyncio.run(
        service.recommend_from_context(
            working_context=(
                "Summary:\nOp zoek naar een lekkere wijn zonder gerecht.\n\n"
                "Questionnaire:\n"
                "5. Kleur van de wijn (wit, rood, rosé): rosé\n"
                "6. Herkomst (land en/of regio): australie\n"
                "7. Maximum budget in euro: 20\n"
                "8. Gewenste geur/smaak (vrije tekst): fris"
            ),
            meal_mode="drinking_only",
        )
    )

    assert result.response_type == "question"
    assert result.text == "Welke stijl zoek je?"
    assert result.state["turn_count"] == 1
    assert result.state["meal_mode"] == "drinking_only"


def test_recommend_from_context_injects_candidate_count_for_pairing() -> None:
    """Candidate count is injected into working_context so LLM can decide to recommend early."""
    controller = StubController(
        [
            TurnDecision(
                action="recommend",
                question_text="",
                meal_mode="pairing",
                query_hints="rund",
            )
        ]
    )
    ranker = EchoRanker()
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
    )

    result = asyncio.run(
        service.recommend_from_context(
            working_context="Summary:\nSteak van rund op de BBQ.",
            meal_mode="pairing",
        )
    )

    assert result.response_type == "recommendations"
    assert len(controller.calls) == 1, "Turn controller must always be called"
    context_sent = controller.calls[0]["working_context"]
    assert (
        "Dish candidate wine count:" in context_sent
    ), "Candidate count should be injected into working_context"


def test_ask_truncates_working_context_before_controller_call() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="question",
                question_text="Welke kleur wijn heb je liefst?",
                meal_mode="drinking_only",
                query_hints="",
            )
        ]
    )
    ranker = StubRanker([])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
        max_working_context_chars=40,
    )

    _ = asyncio.run(
        service.ask(
            current_answer="x" * 80,
            state={"working_context": "y" * 80},
        )
    )

    assert controller.calls
    working_context = controller.calls[0]["working_context"]
    assert isinstance(working_context, str)
    assert len(working_context) <= 40


def test_ask_limits_history_items_and_text_lengths_in_prompt() -> None:
    controller = StubController(
        [
            TurnDecision(
                action="question",
                question_text="Nog een vraag",
                meal_mode="pairing",
                query_hints="",
            )
        ]
    )
    ranker = StubRanker([])
    service = WineAdvisorService(
        catalog_records=_catalog(),
        turn_controller=controller,
        ranker=ranker,
        max_history_items=2,
        max_history_item_chars=10,
    )

    _ = asyncio.run(
        service.ask(
            current_answer="akkoord",
            state={
                "chat_history": [
                    {"role": "user", "text": "A" * 40},
                    {"role": "assistant", "text": "B" * 40},
                    {"role": "user", "text": "C" * 40},
                ]
            },
        )
    )

    assert controller.calls
    conversation_history = controller.calls[0]["conversation_history"]
    assert isinstance(conversation_history, str)
    assert "user: CCCCCCCCCC" in conversation_history
    assert "assistant: BBBBBBBBBB" in conversation_history
    assert "AAAAAAAAAA" not in conversation_history
