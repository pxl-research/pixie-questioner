"""Helpers for categorized handoff payloads."""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass

from ..llm_client import LLMPromptInvoker

logger = logging.getLogger(__name__)

FOODPAIR_KEY_ORDER: tuple[str, ...] = (
    "foodpair_aperitief",
    "foodpair_vis_zeevruchten",
    "foodpair_witvlees_gevogelte",
    "foodpair_roodvlees",
    "foodpair_wild",
    "foodpair_bbq_grill",
    "foodpair_stoofgerecht",
    "foodpair_blauwschimmel_kaas",
    "foodpair_zachte_kaas",
    "foodpair_harde_kaas",
    "foodpair_dessert",
)

WINE_COLOR_VALUES = frozenset(("wit", "rood", "rosé"))
COUNTRY_AUSTRALIA = "Australië"
COUNTRY_ITALY = "Italië"
WINE_COUNTRY_VALUES = frozenset(
    (
        COUNTRY_AUSTRALIA,
        "Chili",
        "Frankrijk",
        COUNTRY_ITALY,
        "Luxemburg",
        "Oostenrijk",
        "Portugal",
        "Spanje",
        "Verenigde Staten",
        "Zuid-Afrika",
    )
)

WINE_REGION_TO_COUNTRY: dict[str, str] = {
    "Alentejo": "Portugal",
    "Alsace": "Frankrijk",
    "Bergerac": "Frankrijk",
    "Bordeaux": "Frankrijk",
    "Bourgogne": "Frankrijk",
    "Bourgogne (Chablis)": "Frankrijk",
    "Californië": "Verenigde Staten",
    "Central Valley": "Chili",
    "Champagne": "Frankrijk",
    "Corsica": "Frankrijk",
    "Côte du Rhône": "Frankrijk",
    "Côtes du Rhône": "Frankrijk",
    "Côtes du Rhône, Plan de Dieu": "Frankrijk",
    "Côtes du Rhône, Séguret": "Frankrijk",
    "Côtes du Rhône, Vacqueyras": "Frankrijk",
    "Elzas": "Frankrijk",
    "Friuli-Venezia Giulia": COUNTRY_ITALY,
    "Haut-Médoc, Bordeaux": "Frankrijk",
    "IGP Méditerranée": "Frankrijk",
    "Languedoc": "Frankrijk",
    "Languedoc-Roussillon": "Frankrijk",
    "Loire (AOC Cheverny)": "Frankrijk",
    "Moezel": "Luxemburg",
    "Navarra": "Spanje",
    "Neder-Oostenrijk": "Oostenrijk",
    "Pays d’Oc": "Frankrijk",
    "Penedès": "Spanje",
    "Provence": "Frankrijk",
    "Puglia": COUNTRY_ITALY,
    "Rioja": "Spanje",
    "Setúbal": "Portugal",
    "Sicilië": COUNTRY_ITALY,
    "Tejo / Bairrada": "Portugal",
    "Toscane": COUNTRY_ITALY,
    "Valencia (Utiel-Requena)": "Spanje",
    "Valle Central": "Chili",
    "Veneto": COUNTRY_ITALY,
    "Victoria": COUNTRY_AUSTRALIA,
    "Vin de France": "Frankrijk",
    "Vin de Pays d’Oc": "Frankrijk",
    "Zuidoost-Australië": COUNTRY_AUSTRALIA,
}

WINE_REGION_VALUES = frozenset(WINE_REGION_TO_COUNTRY)


def extract_questionnaire_field_value(questionnaire: str, index: int) -> str:
    match = re.search(rf"(?m)^\s*{index}\.\s*[^:\n]*:[^\S\n]*(.*?)$", questionnaire)
    if match:
        return match.group(1).strip()
    return ""


def replace_questionnaire_field_value(
    questionnaire: str,
    index: int,
    new_value: str,
) -> str:
    pattern = re.compile(rf"(?m)^(\s*{index}\.\s*[^:\n]*:[^\S\n]*).*$")
    updated, count = pattern.subn(
        lambda match: f"{match.group(1)}{new_value}",
        questionnaire,
        count=1,
    )
    if count == 0:
        return questionnaire
    return updated


def is_blank_questionnaire_value(value: str) -> bool:
    stripped = value.strip().strip("[]").strip()
    return not stripped


def parse_json_object(value: str) -> dict[str, object]:
    stripped = value.strip()
    if not stripped:
        return {}
    candidates = [stripped]
    if "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}") + 1
        candidates.append(stripped[start:end])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def sanitize_handoff_core_categories(
    raw_categories: dict[str, object],
) -> dict[str, bool | str | int]:
    categories: dict[str, bool | str | int] = {}
    categories.update(_extract_foodpair_categories(raw_categories))
    kleur = _extract_valid_color(raw_categories)
    if kleur is not None:
        categories["kleur"] = kleur
    categories.update(_extract_location_categories(raw_categories))
    prijs = _extract_valid_price(raw_categories)
    if prijs is not None:
        categories["prijs"] = prijs
    return categories


def sanitize_handoff_categories_v1(handoff_categories_v1: str) -> str:
    parsed = parse_json_object(handoff_categories_v1)
    categories = sanitize_handoff_core_categories(parsed)
    return json.dumps(categories, ensure_ascii=False, separators=(",", ":"))


def _extract_foodpair_categories(
    raw_categories: dict[str, object],
) -> dict[str, bool | str | int]:
    return {key: True for key in FOODPAIR_KEY_ORDER if raw_categories.get(key) is True}


def _extract_valid_color(raw_categories: dict[str, object]) -> str | None:
    kleur = raw_categories.get("kleur")
    if isinstance(kleur, str) and kleur in WINE_COLOR_VALUES:
        return kleur
    return None


def _extract_location_categories(
    raw_categories: dict[str, object],
) -> dict[str, str]:
    regio = raw_categories.get("regio")
    if isinstance(regio, str) and regio in WINE_REGION_VALUES:
        return {"regio": regio, "land": WINE_REGION_TO_COUNTRY[regio]}

    land = raw_categories.get("land")
    if isinstance(land, str) and land in WINE_COUNTRY_VALUES:
        return {"land": land}
    return {}


def _extract_valid_price(raw_categories: dict[str, object]) -> int | None:
    prijs = raw_categories.get("prijs")
    if isinstance(prijs, int) and not isinstance(prijs, bool) and prijs >= 0:
        return prijs
    return None


def format_handoff_categories_for_display(handoff_categories_v1: str) -> str:
    try:
        parsed = json.loads(handoff_categories_v1)
    except json.JSONDecodeError:
        return handoff_categories_v1
    if not isinstance(parsed, dict):
        return handoff_categories_v1
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _build_core_classifier_rules() -> str:
    allowed_foodpair = "\n".join(f"- {key}" for key in FOODPAIR_KEY_ORDER)
    allowed_countries = "\n".join(
        f"- {country}" for country in sorted(WINE_COUNTRY_VALUES)
    )
    allowed_regions = "\n".join(f"- {region}" for region in sorted(WINE_REGION_VALUES))
    return (
        "Allowed foodpair keys (boolean true only when applicable):\n"
        f"{allowed_foodpair}\n"
        "Allowed color values: wit, rood, rosé\n"
        "Allowed country values:\n"
        f"{allowed_countries}\n"
        "Allowed region values:\n"
        f"{allowed_regions}\n"
        "Price must be a non-negative integer."
    )


@dataclass
class CategorizedHandoffBuilder:
    handoff_classifier: LLMPromptInvoker | None

    async def build_handoff_categories_v1(self, questionnaire: str) -> str:
        categories = await self._classify_handoff_core_categories(questionnaire)
        return json.dumps(categories, ensure_ascii=False, separators=(",", ":"))

    def build_handoff_text(self, handoff_line: str, handoff_categories_v1: str) -> str:
        formatted = format_handoff_categories_for_display(handoff_categories_v1)
        return f"{handoff_line.rstrip()}\n{formatted}"

    async def _classify_handoff_core_categories(
        self, questionnaire: str
    ) -> dict[str, bool | str | int]:
        if self.handoff_classifier is None:
            return {}
        try:
            classifier_output = await self.handoff_classifier.generate(
                current_question="Build categorized handoff JSON.",
                current_answer=questionnaire,
                summary=_build_core_classifier_rules(),
            )
        except Exception:
            logger.warning(
                "Failed to classify categorized handoff core categories.",
                exc_info=True,
            )
            return {}
        parsed = parse_json_object(classifier_output)
        return sanitize_handoff_core_categories(parsed)
