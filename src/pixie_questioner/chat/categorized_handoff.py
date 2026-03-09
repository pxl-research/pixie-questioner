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
WINE_COUNTRY_VALUES = frozenset(
    (
        "Australië",
        "Chili",
        "Frankrijk",
        "Italië",
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
    "Friuli-Venezia Giulia": "Italië",
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
    "Puglia": "Italië",
    "Rioja": "Spanje",
    "Setúbal": "Portugal",
    "Sicilië": "Italië",
    "Tejo / Bairrada": "Portugal",
    "Toscane": "Italië",
    "Valencia (Utiel-Requena)": "Spanje",
    "Valle Central": "Chili",
    "Veneto": "Italië",
    "Victoria": "Australië",
    "Vin de France": "Frankrijk",
    "Vin de Pays d’Oc": "Frankrijk",
    "Zuidoost-Australië": "Australië",
}

WINE_REGION_VALUES = frozenset(WINE_REGION_TO_COUNTRY)

AROMA_TASTE_KEY_ORDER: tuple[str, ...] = (
    "geur_citrus",
    "geur_groen_vegetaal",
    "geur_wit_fruit",
    "geur_tropisch_fruit",
    "geur_steenfruit",
    "geur_rood_fruit",
    "geur_donker_fruit",
    "geur_floraal",
    "geur_kruidig",
    "geur_mineraal_aards",
    "geur_zoet_rijping",
    "geur_hout_geroosterd",
    "smaak_zoet",
    "smaak_droog",
    "smaak_fris_zuur",
    "smaak_fruitig",
    "smaak_kruidig",
    "smaak_bruisend",
    "smaak_houtgerijpt",
    "smaak_mineralig",
    "smaak_romig",
)

_NORMALIZED_AROMA_KEYS: dict[str, tuple[str, ...]] = {}
for _key in AROMA_TASTE_KEY_ORDER:
    _normalized = re.sub(r"[^a-z0-9]+", "_", _key.casefold()).strip("_")
    _NORMALIZED_AROMA_KEYS[_normalized] = (_key,)

_AROMA_SUFFIX_MAP: dict[str, tuple[str, ...]] = {}
for _key in AROMA_TASTE_KEY_ORDER:
    _suffix = _key.split("_", maxsplit=1)[1]
    _AROMA_SUFFIX_MAP.setdefault(_suffix, tuple())
    _AROMA_SUFFIX_MAP[_suffix] = (*_AROMA_SUFFIX_MAP[_suffix], _key)

for _ambiguous_key in ("kruidig",):
    if _ambiguous_key in _AROMA_SUFFIX_MAP:
        _AROMA_SUFFIX_MAP[_ambiguous_key] = tuple(
            sorted(set(_AROMA_SUFFIX_MAP[_ambiguous_key]))
        )


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


def parse_aroma_taste_flags(value: str) -> dict[str, bool]:
    if is_blank_questionnaire_value(value):
        return {}
    parsed = parse_json_object(value)
    flags = _extract_flags_from_mapping(parsed)
    if flags:
        return flags
    return _extract_flags_from_text(value)


def _extract_bool_value(raw_value: object) -> bool | None:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().casefold().strip('"').strip("'")
        if normalized in {"true", "false"}:
            return normalized == "true"
    return None


def _candidate_aroma_keys(raw_key: str) -> tuple[str, ...]:
    normalized_key = re.sub(r"[^a-z0-9]+", "_", raw_key.casefold()).strip("_")
    direct = _NORMALIZED_AROMA_KEYS.get(normalized_key)
    if direct is not None:
        return direct
    suffix = _AROMA_SUFFIX_MAP.get(normalized_key)
    if suffix is not None:
        return suffix
    return ()


def _extract_flags_from_mapping(parsed: dict[str, object]) -> dict[str, bool]:
    if not parsed:
        return {}

    mappings: list[dict[str, object]] = [parsed]
    for value in parsed.values():
        if isinstance(value, dict):
            normalized_mapping = {
                key: nested_value
                for key, nested_value in value.items()
                if isinstance(key, str)
            }
            mappings.append(normalized_mapping)

    flags: dict[str, bool] = {}
    for mapping in mappings:
        for raw_key, raw_value in mapping.items():
            if not isinstance(raw_key, str):
                continue
            bool_value = _extract_bool_value(raw_value)
            if bool_value is not None:
                for candidate_key in _candidate_aroma_keys(raw_key):
                    flags[candidate_key] = bool_value
    return flags


def _extract_flags_from_text(value: str) -> dict[str, bool]:
    pattern = re.compile(
        r"(?P<key>[a-z0-9_\-/]+)\s*[:=]\s*(?P<bool>true|false)",
        flags=re.IGNORECASE,
    )
    flags: dict[str, bool] = {}
    for match in pattern.finditer(value):
        bool_value = match.group("bool").strip().casefold() == "true"
        for candidate_key in _candidate_aroma_keys(match.group("key")):
            flags[candidate_key] = bool_value
    return flags


def serialize_aroma_taste_flags(flags: dict[str, bool]) -> str:
    ordered = {key: flags[key] for key in AROMA_TASTE_KEY_ORDER if key in flags}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def sanitize_handoff_core_categories(
    raw_categories: dict[str, object],
) -> dict[str, bool | str | int]:
    categories: dict[str, bool | str | int] = {}
    for key in FOODPAIR_KEY_ORDER:
        if raw_categories.get(key) is True:
            categories[key] = True

    kleur = raw_categories.get("kleur")
    if isinstance(kleur, str) and kleur in WINE_COLOR_VALUES:
        categories["kleur"] = kleur

    regio = raw_categories.get("regio")
    if isinstance(regio, str) and regio in WINE_REGION_VALUES:
        categories["regio"] = regio
        categories["land"] = WINE_REGION_TO_COUNTRY[regio]
    else:
        land = raw_categories.get("land")
        if isinstance(land, str) and land in WINE_COUNTRY_VALUES:
            categories["land"] = land

    prijs = raw_categories.get("prijs")
    if isinstance(prijs, int) and not isinstance(prijs, bool) and prijs >= 0:
        categories["prijs"] = prijs
    return categories


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
    aroma_taste_classifier: LLMPromptInvoker | None

    async def normalize_questionnaire(self, questionnaire: str) -> str:
        field_value = extract_questionnaire_field_value(questionnaire, 8)
        if is_blank_questionnaire_value(field_value):
            return questionnaire

        parsed_flags = parse_aroma_taste_flags(field_value)
        if parsed_flags:
            return replace_questionnaire_field_value(
                questionnaire=questionnaire,
                index=8,
                new_value=serialize_aroma_taste_flags(parsed_flags),
            )

        classified_flags = await self._classify_aroma_taste_flags(field_value)
        if not classified_flags:
            return questionnaire
        return replace_questionnaire_field_value(
            questionnaire=questionnaire,
            index=8,
            new_value=serialize_aroma_taste_flags(classified_flags),
        )

    async def build_handoff_categories_v1(self, questionnaire: str) -> str:
        categories = await self._classify_handoff_core_categories(questionnaire)
        field_value = extract_questionnaire_field_value(questionnaire, 8)
        aroma_taste_flags = parse_aroma_taste_flags(field_value)
        if not aroma_taste_flags and not is_blank_questionnaire_value(field_value):
            aroma_taste_flags = await self._classify_aroma_taste_flags(field_value)
        for key in AROMA_TASTE_KEY_ORDER:
            if key in aroma_taste_flags:
                categories[key] = aroma_taste_flags[key]
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

    async def _classify_aroma_taste_flags(self, value: str) -> dict[str, bool]:
        if self.aroma_taste_classifier is None:
            return {}
        allowed_keys = "\n".join(f"- {key}" for key in AROMA_TASTE_KEY_ORDER)
        try:
            classified = await self.aroma_taste_classifier.generate(
                current_question="Classify aroma and taste preferences.",
                current_answer=value,
                summary=allowed_keys,
            )
        except Exception:
            logger.warning(
                "Failed to classify aroma and taste flags.",
                exc_info=True,
            )
            return {}
        parsed = parse_aroma_taste_flags(classified)
        if parsed:
            return parsed
        strict_summary = (
            f"{allowed_keys}\n"
            "Return strict JSON only with exact allowed keys and boolean values."
        )
        try:
            retried = await self.aroma_taste_classifier.generate(
                current_question="Return strict JSON aroma and taste flags.",
                current_answer=value,
                summary=strict_summary,
            )
        except Exception:
            logger.warning(
                "Failed to reclassify aroma and taste flags with strict JSON retry.",
                exc_info=True,
            )
            return {}
        return parse_aroma_taste_flags(retried)
