"""Core orchestration service for LLM-driven wine advising."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .catalog_models import WineCatalogRecord
from .retrieval import rank_catalog_records
from .types import (
    ConversationTurn,
    MEAL_MODE_UNKNOWN,
    PHASE_COMPLETED,
    PHASE_INTAKE,
    WineAdvisorResult,
    WineAdvisorState,
    WineCatalogRecommendation,
)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0)


def _to_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _to_chat_history(value: Any) -> list[ConversationTurn]:
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
        normalized_text = text.strip()
        if not normalized_text:
            continue
        history.append(ConversationTurn(role=role, text=normalized_text))
    return history


def _append_history(
    history: list[ConversationTurn],
    role: str,
    text: str,
) -> list[ConversationTurn]:
    normalized = text.strip()
    if not normalized:
        return list(history)
    return [*history, ConversationTurn(role=role, text=normalized)]


@dataclass
class WineAdvisorService:
    catalog_records: list[WineCatalogRecord]
    turn_controller: Any
    ranker: Any
    fallback_question: str = "Voor welk gerecht zoek je wijn?"
    recommendation_intro: str = "Dit zijn passende opties uit de catalogus."
    max_turns: int = 6
    candidate_limit: int = 20

    def __post_init__(self) -> None:
        if not self.catalog_records:
            raise RuntimeError("Wine catalog must contain at least one record.")
        self._catalog_by_barcode = {
            record.barcode: record for record in self.catalog_records if record.barcode
        }

    def _parse_state(self, state: dict[str, Any] | None) -> WineAdvisorState:
        if not state:
            return WineAdvisorState()
        raw_phase = state.get("phase")
        phase = raw_phase if raw_phase in {PHASE_INTAKE, PHASE_COMPLETED} else PHASE_INTAKE
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
            last_question=last_question if isinstance(last_question, str) else "",
            meal_mode=meal_mode,
            working_context=working_context if isinstance(working_context, str) else "",
            chat_history=_to_chat_history(state.get("chat_history")),
            candidate_barcodes=_to_string_list(state.get("candidate_barcodes")),
            selected_barcodes=_to_string_list(state.get("selected_barcodes")),
        )

    def _serialize_state(self, state: WineAdvisorState) -> dict[str, Any]:
        return {
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
                    price_eur=record.price_eur,
                    country=record.country,
                    region=record.region,
                )
            )
        return recommendations

    def _build_working_context(self, previous_context: str, current_answer: str) -> str:
        parts = [part.strip() for part in [previous_context, current_answer] if part.strip()]
        return "\n".join(parts)

    def _fallback_candidates(self, count: int) -> list[str]:
        return [record.barcode for record in self.catalog_records[:count]]

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
            if barcode in seen or barcode not in candidate_set:
                continue
            seen.add(barcode)
            deduped.append(barcode)
        if min_count <= len(deduped) <= max_count:
            return deduped
        return candidates[:max_count]

    async def ask(
        self,
        *,
        current_answer: str,
        state: dict[str, Any] | None,
    ) -> WineAdvisorResult:
        parsed_state = self._parse_state(state)
        if parsed_state.phase == PHASE_COMPLETED and parsed_state.selected_barcodes:
            recommendations = self._recommendations_for_barcodes(
                parsed_state.selected_barcodes
            )
            return WineAdvisorResult(
                text=self.recommendation_intro,
                response_type="recommendations",
                state=self._serialize_state(parsed_state),
                recommendations=recommendations,
            )

        next_turn_count = min(parsed_state.turn_count + 1, self.max_turns)
        working_context = self._build_working_context(
            parsed_state.working_context,
            current_answer,
        )
        decision = await self.turn_controller.decide(
            current_answer=current_answer,
            working_context=working_context,
            turn_count=next_turn_count,
            max_turns=self.max_turns,
        )

        query = " ".join([working_context, decision.query_hints]).strip()
        ranked_candidates = rank_catalog_records(
            self.catalog_records,
            query=query,
            limit=self.candidate_limit,
        )
        candidate_barcodes = [record.barcode for record in ranked_candidates if record.barcode]
        if not candidate_barcodes:
            candidate_barcodes = self._fallback_candidates(min(5, len(self.catalog_records)))

        should_recommend = decision.action == "recommend" or next_turn_count >= self.max_turns
        if should_recommend:
            min_count = 3 if len(candidate_barcodes) >= 3 else 1
            max_count = min(5, len(candidate_barcodes))
            selected = await self.ranker.select_barcodes(
                working_context=working_context,
                candidate_barcodes=candidate_barcodes,
                min_count=min_count,
                max_count=max_count,
            )
            normalized = self._normalize_selected(
                selected,
                candidate_barcodes,
                min_count,
                max_count,
            )
            recommendations = self._recommendations_for_barcodes(normalized)
            if not recommendations:
                recommendations = self._recommendations_for_barcodes(candidate_barcodes[:1])
            completed = WineAdvisorState(
                phase=PHASE_COMPLETED,
                turn_count=next_turn_count,
                last_question=parsed_state.last_question,
                meal_mode=decision.meal_mode,
                working_context=working_context,
                chat_history=_append_history(parsed_state.chat_history, "user", current_answer),
                candidate_barcodes=candidate_barcodes,
                selected_barcodes=[item.barcode for item in recommendations],
            )
            return WineAdvisorResult(
                text=self.recommendation_intro,
                response_type="recommendations",
                state=self._serialize_state(completed),
                recommendations=recommendations,
            )

        question_text = decision.question_text.strip() or self.fallback_question
        next_history = _append_history(parsed_state.chat_history, "user", current_answer)
        next_history = _append_history(next_history, "assistant", question_text)
        updated = WineAdvisorState(
            phase=PHASE_INTAKE,
            turn_count=next_turn_count,
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
