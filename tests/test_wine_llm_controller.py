# ruff: noqa: S101,S105,S106
from __future__ import annotations

import asyncio

from pixie_questioner.wine_advisor.llm_controller import LLMTurnController
from pixie_questioner.wine_advisor.llm_ranker import LLMWineRanker


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def ainvoke(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("No responses left for FakeLLM")
        return self.responses.pop(0)


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
        fallback_question="Voor welk gerecht zoek je wijn?",
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


def test_turn_controller_uses_fallback_question_on_invalid_payload() -> None:
    controller = LLMTurnController(
        llm=FakeLLM(["geen json payload"]),
        system_prompt_template="{current_answer}",
        fallback_question="Voor welk gerecht zoek je wijn?",
    )

    decision = asyncio.run(
        controller.decide(
            current_answer="Geen idee",
            working_context="",
            turn_count=1,
            max_turns=6,
        )
    )

    assert decision.action == "question"
    assert decision.question_text == "Voor welk gerecht zoek je wijn?"


def test_ranker_accepts_valid_unique_candidate_barcodes() -> None:
    ranker = LLMWineRanker(
        llm=FakeLLM(['{"barcodes":["c3","c1","c2"]}']),
        system_prompt_template="{candidate_barcodes}",
    )

    selected = asyncio.run(
        ranker.select_barcodes(
            working_context="Klant zoekt kruidige rode wijn",
            candidate_barcodes=["c1", "c2", "c3", "c4"],
            min_count=1,
            max_count=5,
        )
    )

    assert selected == ["c3", "c1", "c2"]


def test_ranker_retries_once_then_falls_back_to_deterministic_selection() -> None:
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

    selected = asyncio.run(
        ranker.select_barcodes(
            working_context="Klant zoekt wijn bij rund",
            candidate_barcodes=["c1", "c2", "c3"],
            min_count=1,
            max_count=3,
        )
    )

    assert selected == ["c1", "c2", "c3"]
    assert len(llm.calls) == 2
