"""Core orchestration service for LLM-driven wine advising."""

from __future__ import annotations

import re

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .catalog_models import WineCatalogRecord
from .llm_controller import TurnDecision
from .retrieval import rank_catalog_records
from .types import (
    ConversationTurn,
    MEAL_MODE_DRINKING_ONLY,
    MEAL_MODE_PAIRING,
    MEAL_MODE_UNKNOWN,
    PHASE_COMPLETED,
    PHASE_INTAKE,
    WineAdvisorResult,
    WineAdvisorState,
    WineCatalogRecommendation,
)


class RankerSelectionProtocol(Protocol):
    barcodes: list[str]
    explanation: str


class TurnControllerProtocol(Protocol):
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
    ) -> TurnDecision: ...


class WineRankerProtocol(Protocol):
    def select_with_explanation(
        self,
        *,
        working_context: str,
        candidate_records: list[WineCatalogRecord],
        candidate_barcodes: list[str],
        min_count: int,
        max_count: int,
    ) -> Awaitable[RankerSelectionProtocol]: ...


def _truncate_text(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip()


_ORIGIN_LINE_PATTERN = re.compile(r"(?im)^\s*6\.\s*Herkomst\s*\(land en/of regio\):.*$")
_ORIGIN_PHRASE_PATTERN = re.compile(
    r"(?:\b(?:uit|from|de)\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’ -]{1,40}"
    r"|\bd['’][A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’ -]{1,40})",
    flags=re.IGNORECASE,
)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0)


def _relax_origin_constraints(working_context: str) -> str:
    without_origin_line = _ORIGIN_LINE_PATTERN.sub("", working_context)
    relaxed = _ORIGIN_PHRASE_PATTERN.sub("", without_origin_line)
    relaxed = re.sub(r"\n{3,}", "\n\n", relaxed)
    relaxed = re.sub(r"[ \t]{2,}", " ", relaxed)
    return relaxed.strip()


def _to_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _to_chat_history(
    value: Any, *, max_items: int, max_item_chars: int
) -> list[ConversationTurn]:
    if not isinstance(value, list):
        return []
    history: list[ConversationTurn] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = item.get("text")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(text, str):
            continue
        normalized_text = _truncate_text(text, max_item_chars)
        if not normalized_text:
            continue
        history.append(ConversationTurn(role=role, text=normalized_text))
    if max_items <= 0:
        return []
    return history[-max_items:]


def _append_history(
    history: list[ConversationTurn],
    role: Literal["user", "assistant"],
    text: str,
    *,
    max_chars: int,
    max_items: int,
) -> list[ConversationTurn]:
    normalized = _truncate_text(text, max_chars)
    if not normalized:
        return list(history)
    updated = [*history, ConversationTurn(role=role, text=normalized)]
    if max_items <= 0:
        return []
    return updated[-max_items:]


def _history_to_prompt(
    history: list[ConversationTurn],
    *,
    max_items: int,
    max_item_chars: int,
) -> str:
    if not history:
        return ""
    recent = history[-max_items:]
    return "\n".join(
        f"{turn.role}: {_truncate_text(turn.text, max_item_chars)}" for turn in recent
    )


@dataclass
class WineAdvisorService:
    catalog_records: list[WineCatalogRecord]
    turn_controller: TurnControllerProtocol
    ranker: WineRankerProtocol
    recommendation_intro: str = "Leuk, ik heb een paar mooie opties voor je klaargezet."
    max_turns: int = 6
    candidate_limit: int = 40
    max_input_chars: int = 600
    max_working_context_chars: int = 2_200
    max_history_items: int = 8
    max_history_item_chars: int = 240

    def __post_init__(self) -> None:
        if not self.catalog_records:
            raise RuntimeError("Wine catalog must contain at least one record.")
        self.max_turns = max(self.max_turns, 1)
        self.candidate_limit = max(self.candidate_limit, 0)
        self.max_input_chars = max(self.max_input_chars, 80)
        self.max_working_context_chars = max(self.max_working_context_chars, 40)
        self.max_history_items = max(self.max_history_items, 1)
        self.max_history_item_chars = max(self.max_history_item_chars, 40)
        self._catalog_by_barcode = {
            record.barcode: record for record in self.catalog_records if record.barcode
        }

    def _format_recommendation_text(
        self,
        recommendations: list[WineCatalogRecommendation],
    ) -> str:
        if not recommendations:
            return self.recommendation_intro
        lines = [self.recommendation_intro, ""]
        for index, recommendation in enumerate(recommendations, start=1):
            lines.append(
                f"{index}. {recommendation.name} "
                f"({recommendation.country}, {recommendation.region})"
            )
            lines.append(f"   Geur: {recommendation.aroma_description}")
            lines.append(f"   Smaak: {recommendation.taste_description}")
        return "\n".join(lines)

    def _parse_state(self, state: dict[str, Any] | None) -> WineAdvisorState:
        if not state:
            return WineAdvisorState()
        raw_phase = state.get("phase")
        phase = (
            raw_phase if raw_phase in {PHASE_INTAKE, PHASE_COMPLETED} else PHASE_INTAKE
        )
        raw_meal_mode = state.get("meal_mode")
        meal_mode = (
            raw_meal_mode
            if raw_meal_mode in {"pairing", "drinking_only", MEAL_MODE_UNKNOWN}
            else MEAL_MODE_UNKNOWN
        )
        last_question = state.get("last_question")
        working_context = state.get("working_context")
        return WineAdvisorState(
            phase=phase,
            turn_count=_to_int(state.get("turn_count"), 0),
            last_question=(
                _truncate_text(last_question, self.max_history_item_chars)
                if isinstance(last_question, str)
                else ""
            ),
            meal_mode=meal_mode,
            working_context=(
                _truncate_text(working_context, self.max_working_context_chars)
                if isinstance(working_context, str)
                else ""
            ),
            chat_history=_to_chat_history(
                state.get("chat_history"),
                max_items=self.max_history_items,
                max_item_chars=self.max_history_item_chars,
            ),
            candidate_barcodes=_to_string_list(state.get("candidate_barcodes")),
            selected_barcodes=_to_string_list(state.get("selected_barcodes")),
        )

    def _serialize_state(self, state: WineAdvisorState) -> dict[str, Any]:
        serialized: dict[str, Any] = {
            "phase": state.phase,
            "turn_count": state.turn_count,
            "last_question": state.last_question,
            "meal_mode": state.meal_mode,
            "working_context": state.working_context,
            "chat_history": [
                {"role": turn.role, "text": turn.text} for turn in state.chat_history
            ],
            "candidate_barcodes": list(state.candidate_barcodes),
            "selected_barcodes": list(state.selected_barcodes),
        }
        return serialized

    def _recommendations_for_barcodes(
        self,
        barcodes: list[str],
    ) -> list[WineCatalogRecommendation]:
        recommendations: list[WineCatalogRecommendation] = []
        for barcode in barcodes:
            record = self._catalog_by_barcode.get(barcode)
            if record is None:
                continue
            recommendations.append(
                WineCatalogRecommendation(
                    barcode=record.barcode,
                    name=record.name,
                    country=record.country,
                    region=record.region,
                    aroma_description=record.aroma_description,
                    taste_description=record.taste_description,
                )
            )
        return recommendations

    def _build_working_context(self, previous_context: str, current_answer: str) -> str:
        parts = [
            _truncate_text(part, self.max_working_context_chars)
            for part in [previous_context, current_answer]
            if isinstance(part, str) and part.strip()
        ]
        combined = "\n".join(parts)
        if len(combined) <= self.max_working_context_chars:
            return combined
        return combined[-self.max_working_context_chars :]

    def _normalize_selected(
        self,
        selected: list[str],
        candidates: list[str],
        min_count: int,
        max_count: int,
    ) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        candidate_set = set(candidates)
        for barcode in selected:
            if barcode in seen:
                raise RuntimeError("Ranker returned duplicate barcodes.")
            if barcode not in candidate_set:
                raise RuntimeError("Ranker returned unknown barcodes.")
            seen.add(barcode)
            deduped.append(barcode)
        if len(deduped) < min_count:
            raise RuntimeError("Ranker returned too few barcodes.")
        if len(deduped) > max_count:
            raise RuntimeError("Ranker returned too many barcodes.")
        return deduped

    def _candidate_records_for_decision(
        self,
        parsed_state: WineAdvisorState,
        decision: TurnDecision,
    ) -> tuple[list[WineCatalogRecord], bool]:
        if decision.query_hints.strip():
            return (
                rank_catalog_records(
                    self.catalog_records,
                    query=decision.query_hints,
                    limit=self._resolve_shortlist_limit(),
                ),
                True,
            )

        if parsed_state.candidate_barcodes:
            candidate_records = [
                self._catalog_by_barcode[barcode]
                for barcode in parsed_state.candidate_barcodes
                if barcode in self._catalog_by_barcode
            ]
            if candidate_records:
                return candidate_records, True

        return [], False

    def _resolve_shortlist_limit(self) -> int:
        if self.candidate_limit > 0:
            return self.candidate_limit
        return len(self.catalog_records)

    def _fallback_candidates_from_context(
        self,
        context_query: str,
        *,
        limit: int | None = None,
    ) -> list[WineCatalogRecord]:
        resolved_limit = (
            self._resolve_shortlist_limit() if limit is None else max(limit, 0)
        )
        return rank_catalog_records(
            self.catalog_records,
            query=context_query,
            limit=resolved_limit,
        )

    async def ask(
        self,
        *,
        current_answer: str,
        state: dict[str, Any] | None,
    ) -> WineAdvisorResult:
        normalized_answer = _truncate_text(current_answer, self.max_input_chars)
        if not normalized_answer:
            raise RuntimeError("Current answer cannot be empty.")

        parsed_state = self._parse_state(state)
        if parsed_state.phase == PHASE_COMPLETED and parsed_state.selected_barcodes:
            recommendations = self._recommendations_for_barcodes(
                parsed_state.selected_barcodes
            )
            return WineAdvisorResult(
                text=self._format_recommendation_text(recommendations),
                response_type="recommendations",
                state=self._serialize_state(parsed_state),
                recommendations=recommendations,
            )

        next_turn_count = min(parsed_state.turn_count + 1, self.max_turns)
        working_context = self._build_working_context(
            parsed_state.working_context,
            normalized_answer,
        )
        decision = await self.turn_controller.decide(
            current_answer=normalized_answer,
            working_context=working_context,
            turn_count=next_turn_count,
            max_turns=self.max_turns,
            current_meal_mode=parsed_state.meal_mode,
            last_question=parsed_state.last_question,
            conversation_history=_history_to_prompt(
                parsed_state.chat_history,
                max_items=self.max_history_items,
                max_item_chars=self.max_history_item_chars,
            ),
        )

        candidate_records, persist_candidates = self._candidate_records_for_decision(
            parsed_state,
            decision,
        )
        candidate_barcodes = [record.barcode for record in candidate_records]
        state_candidate_barcodes = candidate_barcodes if persist_candidates else []

        should_recommend = (
            decision.action == "recommend" or next_turn_count >= self.max_turns
        )
        if should_recommend and not candidate_barcodes:
            candidate_records = self._fallback_candidates_from_context(working_context)
            candidate_barcodes = [record.barcode for record in candidate_records]
            state_candidate_barcodes = candidate_barcodes
        if should_recommend and not candidate_barcodes:
            raise RuntimeError(
                "Turn controller must provide query hints or persisted candidates "
                "before recommendations."
            )
        if should_recommend:
            return await self._finalize_recommendation(
                parsed_state,
                normalized_answer,
                next_turn_count,
                working_context,
                decision,
                candidate_records,
                candidate_barcodes,
                state_candidate_barcodes,
            )
        return self._finalize_question(
            parsed_state,
            normalized_answer,
            next_turn_count,
            working_context,
            decision,
            state_candidate_barcodes,
        )

    async def _finalize_recommendation(
        self,
        parsed_state: WineAdvisorState,
        normalized_answer: str,
        turn_count: int,
        working_context: str,
        decision: TurnDecision,
        candidate_records: list[WineCatalogRecord],
        candidate_barcodes: list[str],
        state_candidate_barcodes: list[str],
    ) -> WineAdvisorResult:
        def _min_count_attempts_for(candidates: list[str]) -> list[int]:
            if len(candidates) >= 3:
                return [3, 1]
            return [1]

        normalized: list[str] | None = None
        last_error: RuntimeError | None = None
        for min_count in _min_count_attempts_for(candidate_barcodes):
            try:
                selection = await self.ranker.select_with_explanation(
                    working_context=working_context,
                    candidate_records=candidate_records,
                    candidate_barcodes=candidate_barcodes,
                    min_count=min_count,
                    max_count=min(5, len(candidate_barcodes)),
                )
                normalized = self._normalize_selected(
                    selection.barcodes,
                    candidate_barcodes,
                    min_count,
                    min(5, len(candidate_barcodes)),
                )
                break
            except RuntimeError as exc:
                last_error = exc

        if normalized is None and len(candidate_barcodes) < len(
            self._catalog_by_barcode
        ):
            expanded_records = self._fallback_candidates_from_context(
                working_context,
                limit=len(self._catalog_by_barcode),
            )
            expanded_barcodes = [record.barcode for record in expanded_records]
            if expanded_barcodes and expanded_barcodes != candidate_barcodes:
                for min_count in _min_count_attempts_for(expanded_barcodes):
                    try:
                        selection = await self.ranker.select_with_explanation(
                            working_context=working_context,
                            candidate_records=expanded_records,
                            candidate_barcodes=expanded_barcodes,
                            min_count=min_count,
                            max_count=min(5, len(expanded_barcodes)),
                        )
                        normalized = self._normalize_selected(
                            selection.barcodes,
                            expanded_barcodes,
                            min_count,
                            min(5, len(expanded_barcodes)),
                        )
                        candidate_records = expanded_records
                        candidate_barcodes = expanded_barcodes
                        state_candidate_barcodes = expanded_barcodes
                        break
                    except RuntimeError as exc:
                        last_error = exc

        if normalized is None:
            relaxed_working_context = _relax_origin_constraints(working_context)
            if relaxed_working_context and relaxed_working_context != working_context:
                relaxed_records = self._fallback_candidates_from_context(
                    relaxed_working_context,
                    limit=len(self._catalog_by_barcode),
                )
                relaxed_barcodes = [record.barcode for record in relaxed_records]
                if relaxed_barcodes:
                    try:
                        selection = await self.ranker.select_with_explanation(
                            working_context=relaxed_working_context,
                            candidate_records=relaxed_records,
                            candidate_barcodes=relaxed_barcodes,
                            min_count=1,
                            max_count=min(5, len(relaxed_barcodes)),
                        )
                        normalized = self._normalize_selected(
                            selection.barcodes,
                            relaxed_barcodes,
                            1,
                            min(5, len(relaxed_barcodes)),
                        )
                        candidate_records = relaxed_records
                        candidate_barcodes = relaxed_barcodes
                        state_candidate_barcodes = relaxed_barcodes
                    except RuntimeError as exc:
                        last_error = exc

        if normalized is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Ranker failed to return a valid barcode selection.")

        recommendations = self._recommendations_for_barcodes(normalized)
        if not recommendations:
            raise RuntimeError("Ranker selected barcodes without matching records.")

        completed = WineAdvisorState(
            phase=PHASE_COMPLETED,
            turn_count=turn_count,
            last_question=parsed_state.last_question,
            meal_mode=decision.meal_mode,
            working_context=working_context,
            chat_history=_append_history(
                parsed_state.chat_history,
                "user",
                normalized_answer,
                max_chars=self.max_history_item_chars,
                max_items=self.max_history_items,
            ),
            candidate_barcodes=state_candidate_barcodes,
            selected_barcodes=[item.barcode for item in recommendations],
        )
        return WineAdvisorResult(
            text=self._format_recommendation_text(recommendations),
            response_type="recommendations",
            state=self._serialize_state(completed),
            recommendations=recommendations,
        )

    def _finalize_question(
        self,
        parsed_state: WineAdvisorState,
        normalized_answer: str,
        turn_count: int,
        working_context: str,
        decision: TurnDecision,
        candidate_barcodes: list[str],
    ) -> WineAdvisorResult:
        question_text = decision.question_text.strip()
        if not question_text:
            raise RuntimeError("Turn controller returned an empty question.")

        next_history = _append_history(
            parsed_state.chat_history,
            "user",
            normalized_answer,
            max_chars=self.max_history_item_chars,
            max_items=self.max_history_items,
        )
        next_history = _append_history(
            next_history,
            "assistant",
            question_text,
            max_chars=self.max_history_item_chars,
            max_items=self.max_history_items,
        )
        updated = WineAdvisorState(
            phase=PHASE_INTAKE,
            turn_count=turn_count,
            last_question=question_text,
            meal_mode=decision.meal_mode,
            working_context=working_context,
            chat_history=next_history,
            candidate_barcodes=candidate_barcodes,
            selected_barcodes=[],
        )
        return WineAdvisorResult(
            text=question_text,
            response_type="question",
            state=self._serialize_state(updated),
        )

    async def recommend_from_context(
        self,
        *,
        working_context: str,
        meal_mode: str,
    ) -> WineAdvisorResult:
        normalized_context = _truncate_text(
            working_context,
            self.max_working_context_chars,
        )
        if not normalized_context:
            raise RuntimeError("Working context cannot be empty.")
        normalized_meal_mode: Literal["pairing", "drinking_only", "unknown"] = (
            MEAL_MODE_UNKNOWN
        )
        if meal_mode == MEAL_MODE_PAIRING:
            normalized_meal_mode = MEAL_MODE_PAIRING
        elif meal_mode == MEAL_MODE_DRINKING_ONLY:
            normalized_meal_mode = MEAL_MODE_DRINKING_ONLY

        parsed_state = WineAdvisorState(
            phase=PHASE_INTAKE,
            turn_count=0,
            last_question="",
            meal_mode=normalized_meal_mode,
            working_context=normalized_context,
            chat_history=[],
            candidate_barcodes=[],
            selected_barcodes=[],
        )
        decision = await self.turn_controller.decide(
            current_answer=normalized_context,
            working_context=normalized_context,
            turn_count=1,
            max_turns=self.max_turns,
            current_meal_mode=parsed_state.meal_mode,
            last_question="",
            conversation_history="",
        )

        candidate_records, persist_candidates = self._candidate_records_for_decision(
            parsed_state,
            decision,
        )
        candidate_barcodes = [record.barcode for record in candidate_records]
        state_candidate_barcodes = candidate_barcodes if persist_candidates else []

        if decision.action == "question" and decision.question_text.strip():
            return self._finalize_question(
                parsed_state,
                normalized_context,
                1,
                normalized_context,
                decision,
                state_candidate_barcodes,
            )

        if not candidate_barcodes:
            candidate_records = self._fallback_candidates_from_context(
                normalized_context
            )
            candidate_barcodes = [record.barcode for record in candidate_records]
            state_candidate_barcodes = candidate_barcodes
        if not candidate_barcodes:
            raise RuntimeError(
                "Turn controller must provide query hints or persisted candidates "
                "before recommendations."
            )
        return await self._finalize_recommendation(
            parsed_state,
            normalized_context,
            1,
            normalized_context,
            decision,
            candidate_records,
            candidate_barcodes,
            state_candidate_barcodes,
        )
