"""LLM-backed final barcode selector with strict validation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from ..llm_client import extract_llm_content, resolve_llm_client
from .catalog_models import WineCatalogRecord

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)
logger = logging.getLogger(__name__)


class _RankerPayload(BaseModel):
    barcodes: list[str]
    intro_text: str = ""
    selection_explanation: str = ""


@dataclass(frozen=True)
class RankerSelection:
    barcodes: list[str]
    explanation: str = ""
    intro_text: str = ""


@dataclass
class LLMWineRanker:
    llm: object
    system_prompt_template: str

    def _render_prompt(
        self,
        *,
        working_context: str,
        candidate_catalog_rows: str,
        candidate_barcodes: list[str],
        min_count: int,
        max_count: int,
        retry_reason: str,
    ) -> str:
        prompt = self.system_prompt_template
        prompt = prompt.replace("{working_context}", working_context)
        prompt = prompt.replace("{candidate_catalog_rows}", candidate_catalog_rows)
        prompt = prompt.replace("{candidate_barcodes}", ",".join(candidate_barcodes))
        prompt = prompt.replace("{min_count}", str(min_count))
        prompt = prompt.replace("{max_count}", str(max_count))
        prompt = prompt.replace("{retry_reason}", retry_reason)
        return prompt

    def _serialize_catalog_rows(
        self, candidate_records: list[WineCatalogRecord]
    ) -> str:
        lines: list[str] = []
        seen: set[str] = set()
        for record in candidate_records:
            barcode = record.barcode.strip()
            if not barcode or barcode in seen:
                continue
            seen.add(barcode)
            lines.append(
                json.dumps(
                    {
                        "barcode": record.barcode,
                        "name": record.name,
                        "price_eur": record.price_eur,
                        "country": record.country,
                        "region": record.region,
                        "color": record.color,
                        "grape_description": record.grape_description,
                        "aroma_description": record.aroma_description,
                        "taste_description": record.taste_description,
                        "foodpair_description": record.foodpair_description,
                    }
                )
            )
        return "\n".join(lines)

    def _parse_payload(self, content: str) -> RankerSelection:
        try:
            raw_payload = json.loads(content)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_PATTERN.search(content)
            if match is None:
                raise RuntimeError("Invalid ranker payload")
            raw_payload = json.loads(match.group(0))
        try:
            payload = _RankerPayload.model_validate(raw_payload)
        except ValidationError as exc:
            raise RuntimeError("Invalid ranker payload") from exc
        return RankerSelection(
            barcodes=payload.barcodes,
            explanation=payload.selection_explanation.strip(),
            intro_text=payload.intro_text.strip(),
        )

    def _dedupe_preserving_order(self, barcodes: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for barcode in barcodes:
            if barcode in seen:
                continue
            seen.add(barcode)
            ordered.append(barcode)
        return ordered

    def _validate_selection(
        self,
        selected: list[str],
        candidates: list[str],
        min_count: int,
        max_count: int,
    ) -> tuple[list[str] | None, str]:
        candidate_set = set(candidates)
        deduped = self._dedupe_preserving_order(selected)
        if not deduped:
            return None, "No barcodes selected"
        if len(deduped) > max_count:
            return None, "Too many barcodes selected"
        if len(deduped) < min_count:
            return None, "Too few barcodes selected"
        invalid = [barcode for barcode in deduped if barcode not in candidate_set]
        if invalid:
            return None, f"Unknown barcodes selected: {', '.join(invalid)}"
        if len(deduped) != len(selected):
            return None, "Duplicate barcodes selected"
        return deduped, ""

    async def _invoke_ranker(
        self,
        *,
        working_context: str,
        candidate_catalog_rows: str,
        candidate_barcodes: list[str],
        min_count: int,
        max_count: int,
        retry_reason: str,
    ) -> RankerSelection:
        prompt = self._render_prompt(
            working_context=working_context,
            candidate_catalog_rows=candidate_catalog_rows,
            candidate_barcodes=candidate_barcodes,
            min_count=min_count,
            max_count=max_count,
            retry_reason=retry_reason,
        )
        response = await resolve_llm_client(self.llm).ainvoke(
            [{"role": "system", "content": prompt}]
        )
        return self._parse_payload(extract_llm_content(response))

    async def select_with_explanation(
        self,
        *,
        working_context: str,
        candidate_records: list[WineCatalogRecord],
        candidate_barcodes: list[str],
        min_count: int,
        max_count: int,
    ) -> RankerSelection:
        candidates = self._dedupe_preserving_order(candidate_barcodes)
        if not candidates:
            return RankerSelection(barcodes=[], explanation="")
        candidate_catalog_rows = self._serialize_catalog_rows(candidate_records)
        if not candidate_catalog_rows:
            raise RuntimeError("No candidate catalog rows available.")

        try:
            first_attempt = await self._invoke_ranker(
                working_context=working_context,
                candidate_catalog_rows=candidate_catalog_rows,
                candidate_barcodes=candidates,
                min_count=min_count,
                max_count=max_count,
                retry_reason="",
            )
        except Exception:
            logger.warning("Wine ranker attempt 1 failed.", exc_info=True)
            first_attempt = RankerSelection(barcodes=[], explanation="")

        first_valid, reason = self._validate_selection(
            first_attempt.barcodes, candidates, min_count, max_count
        )
        if first_valid is not None:
            return RankerSelection(
                barcodes=first_valid,
                explanation=first_attempt.explanation,
            )

        try:
            second_attempt = await self._invoke_ranker(
                working_context=working_context,
                candidate_catalog_rows=candidate_catalog_rows,
                candidate_barcodes=candidates,
                min_count=min_count,
                max_count=max_count,
                retry_reason=reason,
            )
        except Exception:
            logger.warning("Wine ranker attempt 2 failed.", exc_info=True)
            second_attempt = RankerSelection(barcodes=[], explanation="")

        second_valid, _ = self._validate_selection(
            second_attempt.barcodes, candidates, min_count, max_count
        )
        if second_valid is not None:
            return RankerSelection(
                barcodes=second_valid,
                explanation=second_attempt.explanation,
            )

        raise RuntimeError("Ranker failed to return a valid barcode selection.")
