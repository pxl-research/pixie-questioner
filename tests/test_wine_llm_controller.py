# ruff: noqa: S101,S105,S106
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable

import pytest

from pixie_questioner.wine_advisor.llm_controller import LLMTurnController
from pixie_questioner.wine_advisor.llm_ranker import LLMWineRanker
from pixie_questioner.wine_advisor.catalog_models import WineCatalogRecord


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def ainvoke(self, messages: list[dict[str, str]]) -> str:
        await asyncio.sleep(0)
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("No responses left for FakeLLM")
        return self.responses.pop(0)


class RaisingLLM:
    def __init__(self, exception: Exception):
        self.exception = exception
        self.calls: list[list[dict[str, str]]] = []

    async def ainvoke(self, messages: list[dict[str, str]]) -> str:
        await asyncio.sleep(0)
        self.calls.append(messages)
        raise self.exception


async def _observe_coroutine_yield(coro: Awaitable[object]) -> list[str]:
    order: list[str] = []
    loop = asyncio.get_running_loop()
    loop.call_soon(order.append, "loop")
    try:
        await coro
    except Exception as exc:
        order.append(type(exc).__name__)
    order.append("done")
    await asyncio.sleep(0)
    return order


def _candidate_records() -> list[WineCatalogRecord]:
    return [
        WineCatalogRecord(
            barcode="c1",
            name="Wijn 1",
            price_eur=10.0,
            country="Frankrijk",
            region="Rhône",
            color="rood",
            grape_description="Syrah",
            aroma_description="Kruidig",
            taste_description="Vol",
            foodpair_description="Rund",
        ),
        WineCatalogRecord(
            barcode="c2",
            name="Wijn 2",
            price_eur=11.0,
            country="Italië",
            region="Toscane",
            color="wit",
            grape_description="Trebbiano",
            aroma_description="Citrus",
            taste_description="Fris",
            foodpair_description="Vis",
        ),
        WineCatalogRecord(
            barcode="c3",
            name="Wijn 3",
            price_eur=12.0,
            country="Spanje",
            region="Rioja",
            color="rood",
            grape_description="Tempranillo",
            aroma_description="Donker fruit",
            taste_description="Droog",
            foodpair_description="Lam",
        ),
    ]


def test_turn_controller_returns_structured_decision() -> None:
    controller = LLMTurnController(
        llm=FakeLLM(
            [
                '{"action":"question","question_text":"Welke smaken vind je lekker?","meal_mode":"pairing","query_hints":"kruidig rund"}'
            ]
        ),
        system_prompt_template=(
            "context={working_context};answer={current_answer};turn={turn_count};max={max_turns}"
        ),
    )

    decision = asyncio.run(
        controller.decide(
            current_answer="Ik eet rund",
            working_context="Klant zoekt wijn bij hoofdgerecht",
            turn_count=2,
            max_turns=6,
        )
    )

    assert decision.action == "question"
    assert decision.question_text == "Welke smaken vind je lekker?"
    assert decision.meal_mode == "pairing"


def test_fake_llm_ainvoke_yields_to_event_loop() -> None:
    llm = FakeLLM(['{"action":"question","question_text":"Vraag"}'])

    order = asyncio.run(
        _observe_coroutine_yield(llm.ainvoke([{"role": "system", "content": "test"}]))
    )

    assert order[0] == "loop"


def test_raising_llm_ainvoke_yields_to_event_loop() -> None:
    llm = RaisingLLM(RuntimeError("boom"))

    order = asyncio.run(
        _observe_coroutine_yield(llm.ainvoke([{"role": "system", "content": "test"}]))
    )

    assert order[0] == "loop"


def test_turn_controller_includes_stateful_context_in_prompt() -> None:
    llm = FakeLLM(
        [
            '{"action":"question","question_text":"Wat is je budget?","meal_mode":"drinking_only","query_hints":"budget"}'
        ]
    )
    controller = LLMTurnController(
        llm=llm,
        system_prompt_template=(
            "mode={current_meal_mode};last={last_question};history={conversation_history};"
            "ctx={working_context};answer={current_answer};turn={turn_count};max={max_turns}"
        ),
    )

    _ = asyncio.run(
        controller.decide(
            current_answer="Gewoon los",
            working_context="Ik zoek een wijn",
            turn_count=2,
            max_turns=6,
            current_meal_mode="drinking_only",
            last_question="Wil je los drinken of bij eten?",
            conversation_history=(
                "user: Ik zoek een wijn\n"
                "assistant: Wil je los drinken of bij eten?\n"
                "user: Gewoon los"
            ),
        )
    )

    prompt = llm.calls[0][0]["content"]
    assert "mode=drinking_only" in prompt
    assert "last=Wil je los drinken of bij eten?" in prompt
    assert "history=user: Ik zoek een wijn" in prompt


def test_turn_controller_raises_on_invalid_payload() -> None:
    controller = LLMTurnController(
        llm=FakeLLM(["geen json payload"]),
        system_prompt_template="{current_answer}",
    )

    with pytest.raises(RuntimeError):
        asyncio.run(
            controller.decide(
                current_answer="Geen idee",
                working_context="",
                turn_count=1,
                max_turns=6,
            )
        )


def test_ranker_accepts_valid_unique_candidate_barcodes() -> None:
    ranker = LLMWineRanker(
        llm=FakeLLM(['{"barcodes":["c3","c1","c2"]}']),
        system_prompt_template="{candidate_barcodes}",
    )

    selection = asyncio.run(
        ranker.select_with_explanation(
            working_context="Klant zoekt kruidige rode wijn",
            candidate_records=_candidate_records(),
            candidate_barcodes=["c1", "c2", "c3", "c4"],
            min_count=1,
            max_count=5,
        )
    )

    assert selection.barcodes == ["c3", "c1", "c2"]


def test_ranker_can_return_selection_explanation() -> None:
    ranker = LLMWineRanker(
        llm=FakeLLM(
            [
                '{"barcodes":["c1","c3"],"selection_explanation":"Best fit on food pairing and freshness."}'
            ]
        ),
        system_prompt_template="{candidate_catalog_rows}",
    )

    selection = asyncio.run(
        ranker.select_with_explanation(
            working_context="Klant eet vol-au-vent en wil frisse stijl",
            candidate_records=_candidate_records(),
            candidate_barcodes=["c1", "c2", "c3"],
            min_count=1,
            max_count=3,
        )
    )

    assert selection.barcodes == ["c1", "c3"]
    assert selection.explanation == "Best fit on food pairing and freshness."


def test_ranker_raises_when_two_attempts_are_invalid() -> None:
    llm = FakeLLM(
        [
            '{"barcodes":["x999","c1"]}',
            '{"barcodes":["c1","c1"]}',
        ]
    )
    ranker = LLMWineRanker(
        llm=llm,
        system_prompt_template="context={working_context};candidates={candidate_barcodes}",
    )

    with pytest.raises(RuntimeError):
        asyncio.run(
            ranker.select_with_explanation(
                working_context="Klant zoekt wijn bij rund",
                candidate_records=_candidate_records(),
                candidate_barcodes=["c1", "c2", "c3"],
                min_count=1,
                max_count=3,
            )
        )
    assert len(llm.calls) == 2


def test_ranker_logs_warning_when_attempts_raise(caplog) -> None:
    llm = RaisingLLM(RuntimeError("ranker failed"))
    ranker = LLMWineRanker(
        llm=llm,
        system_prompt_template="context={working_context};candidates={candidate_barcodes}",
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            asyncio.run(
                ranker.select_with_explanation(
                    working_context="Klant zoekt wijn bij rund",
                    candidate_records=_candidate_records(),
                    candidate_barcodes=["c1", "c2", "c3"],
                    min_count=1,
                    max_count=3,
                )
            )

    assert len(llm.calls) == 2
    assert "ranker attempt" in caplog.text
