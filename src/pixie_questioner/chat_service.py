"""Chat orchestration service."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from .chat.flow import (
    ensure_feedback_context,
    mark_feedback_completed,
    to_categorized_handoff_completed_state,
    to_feedback_state,
    to_intake_state_with_next_question,
    to_off_topic_redirect_state,
    to_terminated_state,
)
from .chat.language import create_language_support
from .chat.messages import PolicyMessages
from .chat.policy import ChatDomainPolicy
from .chat.question_asker import QuestionAsker
from .chat.relevance import AnswerRelevanceClassifierProtocol, ClassificationLabel
from .chat.state_codec import (
    extract_questionnaire_indices,
    merge_partial_questionnaire_update,
    parse_state,
    serialize_state,
)
from .chat.types import (
    INITIAL_QUESTION,
    PHASE_AWAITING_FEEDBACK,
    PHASE_COMPLETED,
    PHASE_TERMINATED,
    ChatResult,
    ChatState,
)
from .llm_client import LLMPromptInvoker

_WINE_COUNTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "Australië": ("australia",),
    "Chili": ("chile",),
    "Frankrijk": ("france",),
    "Italië": ("italie", "italy", "italia"),
    "Luxemburg": ("luxembourg",),
    "Oostenrijk": ("austria",),
    "Portugal": (),
    "Spanje": ("spain", "espana", "espania"),
    "Verenigde Staten": ("united states", "usa", "us", "vs", "u s"),
    "Zuid-Afrika": ("south africa", "south-africa"),
}

_WINE_REGION_TO_COUNTRY: dict[str, str] = {
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

_WINE_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "Bourgogne (Chablis)": ("bourgogne chablis", "chablis"),
    "Californië": ("californie", "california"),
    "Côte du Rhône": ("cote du rhone",),
    "Côtes du Rhône": ("cotes du rhone",),
    "Côtes du Rhône, Plan de Dieu": ("cotes du rhone plan de dieu",),
    "Côtes du Rhône, Séguret": ("cotes du rhone seguret",),
    "Côtes du Rhône, Vacqueyras": ("cotes du rhone vacqueyras",),
    "Friuli-Venezia Giulia": ("friuli venezia giulia",),
    "Haut-Médoc, Bordeaux": ("haut medoc bordeaux",),
    "IGP Méditerranée": ("igp mediterranee",),
    "Languedoc-Roussillon": ("languedoc roussillon",),
    "Loire (AOC Cheverny)": ("loire aoc cheverny", "cheverny"),
    "Neder-Oostenrijk": ("neder oostenrijk", "lower austria"),
    "Pays d’Oc": ("pays doc", "pays d oc"),
    "Penedès": ("penedes",),
    "Setúbal": ("setubal",),
    "Sicilië": ("sicilie", "sicily"),
    "Tejo / Bairrada": ("tejo bairrada",),
    "Valencia (Utiel-Requena)": ("valencia utiel requena", "utiel requena"),
    "Vin de Pays d’Oc": ("vin de pays doc", "vin de pays d oc"),
    "Zuidoost-Australië": ("zuidoost australie", "southeast australia"),
}


@dataclass
class ChatService:
    llm: object
    question_asker_prompt: str
    notulist: LLMPromptInvoker
    summarizer: LLMPromptInvoker
    questionnaire_template: str
    domain_policy: ChatDomainPolicy
    answer_relevance_classifier: AnswerRelevanceClassifierProtocol | None = None

    def __post_init__(self) -> None:
        if not self.questionnaire_template.strip():
            raise RuntimeError("Questionnaire template is required.")
        self._questionnaire_indices = extract_questionnaire_indices(
            self.questionnaire_template
        )
        if not self._questionnaire_indices:
            raise RuntimeError("Questionnaire template must contain numbered fields.")
        if self.domain_policy.off_topic_termination_threshold < 1:
            raise RuntimeError(
                "off_topic_termination_threshold must be greater than or equal to 1."
            )

        self._language = create_language_support(
            language_codes=self.domain_policy.language_codes,
            language_names=self.domain_policy.language_names,
        )
        self._messages = PolicyMessages(policy=self.domain_policy)
        self._question_asker = QuestionAsker(
            llm=self.llm,
            prompt_template=self.question_asker_prompt,
            completion_token=self.domain_policy.completion_token,
            language=self._language,
            messages=self._messages,
        )

    # -- Helpers ---------------------------------------------------------------

    def _questionnaire_complete(self, questionnaire: str) -> bool:
        return "[]" not in questionnaire

    def _parse_state(self, state: dict[str, str] | None) -> ChatState:
        return parse_state(
            state,
            questionnaire_template=self.questionnaire_template,
            detect_language=self._language.detect,
        )

    def _answer_result(
        self,
        text: str,
        state: ChatState,
        summary: str | None = None,
        follow_up_question: str | None = None,
        quick_replies: list[dict[str, str]] | None = None,
    ) -> ChatResult:
        return ChatResult(
            text=text,
            response_type="answer",
            state=serialize_state(state),
            summary=summary,
            follow_up_question=follow_up_question,
            quick_replies=quick_replies,
        )

    def _summary_result(self, text: str, state: ChatState, summary: str) -> ChatResult:
        return ChatResult(
            text=text,
            response_type="summary",
            state=serialize_state(state),
            summary=summary,
        )

    def _is_categorized_handoff_mode(self) -> bool:
        return self.domain_policy.completion_mode == "categorized_handoff"

    def _extract_questionnaire_field_value(self, questionnaire: str, index: int) -> str:
        match = re.search(rf"(?m)^\s*{index}\.\s*[^:\n]*:[^\S\n]*(.*?)$", questionnaire)
        if match:
            return match.group(1).strip()
        return ""

    def _replace_questionnaire_field_value(
        self, questionnaire: str, index: int, new_value: str
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

    def _is_blank_questionnaire_value(self, value: str) -> bool:
        stripped = value.strip().strip("[]").strip()
        return not stripped

    def _normalize_text(self, value: str) -> str:
        lowered = value.casefold()
        deaccented = "".join(
            char
            for char in unicodedata.normalize("NFKD", lowered)
            if not unicodedata.combining(char)
        )
        return re.sub(r"[^a-z0-9]+", " ", deaccented).strip()

    def _contains_any(self, normalized_text: str, candidates: tuple[str, ...]) -> bool:
        haystack = f" {normalized_text} "
        return any(f" {candidate} " in haystack for candidate in candidates)

    def _is_unspecified(self, normalized_text: str) -> bool:
        return self._contains_any(
            normalized_text,
            (
                "niet gespecificeerd",
                "not specified",
                "non specifie",
                "unspecified",
            ),
        )

    def _is_none_value(self, normalized_text: str) -> bool:
        return self._contains_any(
            normalized_text,
            ("none", "geen", "aucun", "aucune", "pas de", "zonder"),
        )

    def _is_no_preference(self, normalized_text: str) -> bool:
        return self._contains_any(
            normalized_text,
            (
                "geen voorkeur",
                "maakt niet uit",
                "maakt me niet uit",
                "eender",
                "whatever",
                "any",
                "no preference",
                "doesnt matter",
                "does not matter",
                "niet van belang",
                "ik weet het niet",
                "geen idee",
                "onbekend",
            ),
        )

    def _match_alias(
        self, normalized_text: str, aliases: dict[str, tuple[str, ...]]
    ) -> str | None:
        haystack = f" {normalized_text} "
        best_match: str | None = None
        best_length = 0
        for canonical, candidate_aliases in aliases.items():
            for alias in (canonical, *candidate_aliases):
                normalized_alias = self._normalize_text(alias)
                if not normalized_alias:
                    continue
                if f" {normalized_alias} " not in haystack:
                    continue
                alias_length = len(normalized_alias)
                if alias_length > best_length:
                    best_match = canonical
                    best_length = alias_length
        return best_match

    def _match_country(self, normalized_text: str) -> str | None:
        return self._match_alias(normalized_text, _WINE_COUNTRY_ALIASES)

    def _match_region(self, normalized_text: str) -> str | None:
        region_aliases: dict[str, tuple[str, ...]] = {}
        for region in _WINE_REGION_TO_COUNTRY:
            region_aliases[region] = _WINE_REGION_ALIASES.get(region, ())
        return self._match_alias(normalized_text, region_aliases)

    def _canonical_course_type(self, value: str) -> str:
        if self._is_blank_questionnaire_value(value):
            return "onbekend"
        normalized = self._normalize_text(value)
        if self._is_unspecified(normalized):
            return "onbekend"
        if self._contains_any(normalized, ("aperitief", "aperitif", "starter")):
            return "aperitief"
        if self._contains_any(
            normalized,
            ("hoofdgerecht", "main course", "plat principal"),
        ):
            return "hoofdgerecht"
        if self._contains_any(normalized, ("kaasgerecht", "cheese", "fromage")):
            return "kaasgerecht"
        if self._contains_any(normalized, ("dessert",)):
            return "dessert"
        return "onbekend"

    def _canonical_main_protein(self, value: str) -> str:
        if self._is_blank_questionnaire_value(value):
            return "onbekend"
        normalized = self._normalize_text(value)
        if self._is_unspecified(normalized):
            return "onbekend"
        if self._contains_any(
            normalized,
            (
                "fish",
                "seafood",
                "sea food",
                "vis",
                "zeevruchten",
                "poisson",
                "fruit de mer",
                "fruits de mer",
            ),
        ):
            return "vis_zeevruchten"
        if self._contains_any(
            normalized,
            (
                "poultry",
                "white meat",
                "witvlees",
                "chicken",
                "turkey",
                "duck",
                "gevogelte",
                "kip",
                "kalkoen",
                "eend",
                "volaille",
                "poulet",
            ),
        ):
            return "witvlees_gevogelte"
        if self._contains_any(
            normalized,
            (
                "wild",
                "game",
                "venison",
                "gibier",
            ),
        ):
            return "wild"
        if self._contains_any(
            normalized,
            (
                "red meat",
                "rood vlees",
                "beef",
                "steak",
                "lamb",
                "pork",
                "varken",
                "porc",
                "rund",
                "boeuf",
                "agneau",
                "veau",
            ),
        ):
            return "roodvlees"
        if self._is_none_value(normalized):
            return "geen"
        return "onbekend"

    def _canonical_cooking_method(self, value: str) -> str:
        if self._is_blank_questionnaire_value(value):
            return "onbekend"
        normalized = self._normalize_text(value)
        if self._is_unspecified(normalized):
            return "onbekend"
        if self._contains_any(normalized, ("bbq", "barbecue", "barbeque")):
            return "bbq_grill"
        if self._contains_any(normalized, ("grill", "grilled", "gegrild", "grille")):
            return "bbq_grill"
        if self._contains_any(
            normalized,
            ("stoofpot", "stoofpotje", "stoof", "stew", "ragout", "braise"),
        ):
            return "stoofgerecht"
        if self._is_none_value(normalized):
            return "geen"
        return "onbekend"

    def _canonical_cheese_type(self, value: str) -> str:
        if self._is_blank_questionnaire_value(value):
            return "onbekend"
        normalized = self._normalize_text(value)
        if self._is_unspecified(normalized):
            return "onbekend"
        if self._contains_any(
            normalized,
            (
                "blauwschimmel",
                "blue cheese",
                "bleu",
                "roquefort",
                "gorgonzola",
                "stilton",
            ),
        ):
            return "blauwschimmel"
        if self._contains_any(
            normalized,
            (
                "zachte kaas",
                "soft cheese",
                "brie",
                "camembert",
                "chevre",
                "goat cheese",
            ),
        ):
            return "zachte"
        if self._contains_any(
            normalized,
            (
                "harde kaas",
                "hard cheese",
                "parmezaan",
                "parmesan",
                "pecorino",
                "gruyere",
                "comte",
                "manchego",
            ),
        ):
            return "harde"
        if self._is_none_value(normalized):
            return "geen"
        return "onbekend"

    def _canonical_wine_color(self, value: str) -> str | None:
        if self._is_blank_questionnaire_value(value):
            return None
        normalized = self._normalize_text(value)
        if self._is_unspecified(normalized) or self._is_no_preference(normalized):
            return None
        if self._contains_any(normalized, ("rose", "rose wijn", "roze", "rosé")):
            return "rosé"
        if self._contains_any(normalized, ("wit", "white", "blanc", "bianco")):
            return "wit"
        if self._contains_any(normalized, ("rood", "red", "rouge", "rosso")):
            return "rood"
        return None

    def _extract_country_region(
        self, questionnaire_value: str
    ) -> tuple[str | None, str | None]:
        if self._is_blank_questionnaire_value(questionnaire_value):
            return None, None
        normalized = self._normalize_text(questionnaire_value)
        if self._is_unspecified(normalized) or self._is_no_preference(normalized):
            return None, None
        matched_region = self._match_region(normalized)
        matched_country = self._match_country(normalized)
        if matched_region is not None:
            return _WINE_REGION_TO_COUNTRY[matched_region], matched_region
        if matched_country is not None:
            return matched_country, None
        return None, None

    def _parse_price_upper_bound(self, value: str) -> int | None:
        if self._is_blank_questionnaire_value(value):
            return None
        normalized = self._normalize_text(value)
        if self._is_unspecified(normalized) or self._is_no_preference(normalized):
            return None
        numeric_matches = re.findall(r"\d+(?:[.,]\d+)?", value)
        if not numeric_matches:
            return None
        max_value = max(float(match.replace(",", ".")) for match in numeric_matches)
        if max_value < 0:
            return None
        return int(max_value)

    def _normalize_cheese_type_for_categorized_flow(self, questionnaire: str) -> str:
        course_type = self._canonical_course_type(
            self._extract_questionnaire_field_value(questionnaire, 1)
        )
        if course_type not in {"aperitief", "hoofdgerecht", "dessert"}:
            return questionnaire
        cheese_type_value = self._extract_questionnaire_field_value(questionnaire, 4)
        if not self._is_blank_questionnaire_value(cheese_type_value):
            return questionnaire
        return self._replace_questionnaire_field_value(
            questionnaire=questionnaire,
            index=4,
            new_value="none",
        )

    def _build_handoff_categories_v1(self, questionnaire: str) -> str:
        course_type = self._canonical_course_type(
            self._extract_questionnaire_field_value(questionnaire, 1)
        )
        main_protein = self._canonical_main_protein(
            self._extract_questionnaire_field_value(questionnaire, 2)
        )
        cooking_method = self._canonical_cooking_method(
            self._extract_questionnaire_field_value(questionnaire, 3)
        )
        cheese_type = self._canonical_cheese_type(
            self._extract_questionnaire_field_value(questionnaire, 4)
        )
        category_flags = {
            "foodpair_aperitief": course_type == "aperitief",
            "foodpair_vis_zeevruchten": main_protein == "vis_zeevruchten",
            "foodpair_witvlees_gevogelte": main_protein == "witvlees_gevogelte",
            "foodpair_roodvlees": main_protein == "roodvlees",
            "foodpair_wild": main_protein == "wild",
            "foodpair_bbq_grill": cooking_method == "bbq_grill",
            "foodpair_stoofgerecht": cooking_method == "stoofgerecht",
            "foodpair_blauwschimmel_kaas": cheese_type == "blauwschimmel",
            "foodpair_zachte_kaas": cheese_type == "zachte",
            "foodpair_harde_kaas": cheese_type == "harde",
            "foodpair_dessert": course_type == "dessert",
        }
        categories: dict[str, bool | str | int] = {
            key: True for key, is_true in category_flags.items() if is_true
        }
        wine_color = self._canonical_wine_color(
            self._extract_questionnaire_field_value(questionnaire, 5)
        )
        if wine_color is not None:
            categories["kleur"] = wine_color
        land, regio = self._extract_country_region(
            self._extract_questionnaire_field_value(questionnaire, 6)
        )
        if land is not None:
            categories["land"] = land
        if regio is not None:
            categories["regio"] = regio
        prijs = self._parse_price_upper_bound(
            self._extract_questionnaire_field_value(questionnaire, 7)
        )
        if prijs is not None:
            categories["prijs"] = prijs
        return json.dumps(categories, ensure_ascii=False, separators=(",", ":"))

    def _format_handoff_categories_for_display(self, handoff_categories_v1: str) -> str:
        try:
            parsed = json.loads(handoff_categories_v1)
        except json.JSONDecodeError:
            return handoff_categories_v1
        if not isinstance(parsed, dict):
            return handoff_categories_v1
        return json.dumps(parsed, ensure_ascii=False, indent=2)

    def _build_categorized_handoff_text(
        self, language: str, handoff_categories_v1: str
    ) -> str:
        handoff_line = self._messages.handoff(language).rstrip()
        formatted = self._format_handoff_categories_for_display(handoff_categories_v1)
        return f"{handoff_line}\n{formatted}"

    def _build_categorized_completion_result(
        self,
        current_state: ChatState,
        current_answer: str,
        updated_questionnaire: str,
        updated_summary: str,
    ) -> ChatResult:
        language = self._language.detect_preferred(
            current_answer=current_answer,
            previous_summary=updated_summary,
        )
        handoff_categories_v1 = self._build_handoff_categories_v1(updated_questionnaire)
        completed_state = to_categorized_handoff_completed_state(
            state=current_state,
            updated_questionnaire=updated_questionnaire,
            updated_summary=updated_summary,
            handoff_categories_v1=handoff_categories_v1,
        )
        return self._answer_result(
            text=self._build_categorized_handoff_text(language, handoff_categories_v1),
            state=completed_state,
        )

    # -- Phase handlers --------------------------------------------------------

    def _resolve_goodbye_language(
        self, current_state: ChatState, current_answer: str
    ) -> str:
        if current_state.post_summary_language:
            return current_state.post_summary_language
        return self._language.detect_preferred(
            current_answer=current_answer,
            previous_summary=current_state.summary,
        )

    def _handle_terminated_phase(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        language = self._resolve_goodbye_language(current_state, current_answer)
        current_state.post_summary_language = language
        return self._answer_result(
            text=self._messages.goodbye(language),
            state=current_state,
        )

    def _handle_awaiting_feedback_phase(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        feedback_state, summary_text, language = ensure_feedback_context(
            current_state,
            detect_language=self._language.detect,
        )
        feedback_response = self._messages.normalize_feedback(current_answer)
        if feedback_response is None:
            return self._answer_result(
                text=self._messages.feedback_question(language),
                state=feedback_state,
                summary=summary_text,
                quick_replies=self._messages.feedback_quick_replies(language),
            )
        completed_state = mark_feedback_completed(feedback_state, feedback_response)
        return self._summary_result(
            text=self._messages.feedback_thanks(language),
            state=completed_state,
            summary=summary_text,
        )

    def _handle_completed_phase(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        if self._is_categorized_handoff_mode():
            handoff_categories_v1 = (
                current_state.handoff_categories_v1
                if current_state.handoff_categories_v1.strip()
                else self._build_handoff_categories_v1(current_state.questionnaire)
            )
            language = self._language.detect_preferred(
                current_answer=current_answer,
                previous_summary=current_state.summary,
            )
            completed_state = to_categorized_handoff_completed_state(
                state=current_state,
                updated_questionnaire=current_state.questionnaire,
                updated_summary=current_state.summary,
                handoff_categories_v1=handoff_categories_v1,
            )
            return self._answer_result(
                text=self._build_categorized_handoff_text(
                    language, handoff_categories_v1
                ),
                state=completed_state,
            )

        completed_state, summary_text, language = ensure_feedback_context(
            current_state,
            detect_language=self._language.detect,
        )
        return self._summary_result(
            text=self._messages.feedback_thanks(language),
            state=completed_state,
            summary=summary_text,
        )

    # -- Intake phase ----------------------------------------------------------

    async def _classify_intake_answer(
        self, current_state: ChatState, current_answer: str
    ) -> ClassificationLabel:
        if current_state.last_question.strip() == INITIAL_QUESTION:
            return "on_topic"
        if self.answer_relevance_classifier is None:
            return "on_topic"
        try:
            return await self.answer_relevance_classifier.classify(
                current_question=current_state.last_question,
                current_answer=current_answer,
                questionnaire=current_state.questionnaire,
            )
        except Exception:
            return "on_topic"

    def _handle_off_topic_intake(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        current_state.boundary_violation_count += 1
        language = self._language.detect_preferred(
            current_answer=current_answer,
            previous_summary=current_state.summary,
        )
        if (
            current_state.boundary_violation_count
            >= self.domain_policy.off_topic_termination_threshold
        ):
            terminated_state = to_terminated_state(current_state, language)
            return self._answer_result(
                text=self._messages.goodbye(language),
                state=terminated_state,
            )
        redirected_state = to_off_topic_redirect_state(current_state)
        return self._answer_result(
            text=self._messages.off_topic_redirect(
                language=language,
                current_question=current_state.last_question,
            ),
            state=redirected_state,
        )

    async def _update_questionnaire(
        self, current_state: ChatState, current_answer: str
    ) -> str:
        try:
            updated = await self.notulist.generate(
                current_question=current_state.last_question,
                current_answer=current_answer,
                summary=current_state.questionnaire,
            )
        except Exception as exc:
            raise RuntimeError("Failed to update questionnaire.") from exc
        merged = merge_partial_questionnaire_update(
            current_questionnaire=current_state.questionnaire,
            proposed_questionnaire=updated,
            questionnaire_indices=self._questionnaire_indices,
        )
        if merged is None:
            return current_state.questionnaire
        return merged

    async def _update_summary(
        self, current_state: ChatState, current_answer: str
    ) -> str:
        try:
            updated = await self.summarizer.generate(
                current_question=current_state.last_question,
                current_answer=current_answer,
                summary=current_state.summary,
            )
        except Exception as exc:
            raise RuntimeError("Failed to summarize conversation.") from exc
        return updated

    async def _handle_on_topic_intake(
        self,
        current_state: ChatState,
        current_answer: str,
        *,
        force_complete: bool = False,
    ) -> ChatResult:
        current_state.boundary_violation_count = 0
        updated_questionnaire = await self._update_questionnaire(
            current_state, current_answer
        )
        if self._is_categorized_handoff_mode():
            updated_questionnaire = self._normalize_cheese_type_for_categorized_flow(
                updated_questionnaire
            )
        updated_summary = await self._update_summary(current_state, current_answer)

        if self._questionnaire_complete(updated_questionnaire) or force_complete:
            if self._is_categorized_handoff_mode():
                return self._build_categorized_completion_result(
                    current_state=current_state,
                    current_answer=current_answer,
                    updated_questionnaire=updated_questionnaire,
                    updated_summary=updated_summary,
                )
            language = self._language.detect(updated_summary)
            feedback_state = to_feedback_state(
                state=current_state,
                updated_questionnaire=updated_questionnaire,
                updated_summary=updated_summary,
                language=language,
            )
            return self._answer_result(
                text=self._messages.handoff(language),
                state=feedback_state,
                summary=updated_summary,
                follow_up_question=self._messages.feedback_question(language),
                quick_replies=self._messages.feedback_quick_replies(language),
            )

        preferred_language = self._language.detect_preferred(
            current_answer=current_answer,
            previous_summary=current_state.summary,
        )
        next_question = await self._question_asker.ask(
            updated_questionnaire, preferred_language
        )
        next_state = to_intake_state_with_next_question(
            state=current_state,
            updated_questionnaire=updated_questionnaire,
            updated_summary=updated_summary,
            next_question=next_question,
        )
        return self._answer_result(text=next_question, state=next_state)

    async def _handle_intake_phase(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        classification = await self._classify_intake_answer(
            current_state, current_answer
        )
        if classification == "off_topic":
            return self._handle_off_topic_intake(current_state, current_answer)
        if classification == "stop":
            return await self._handle_on_topic_intake(
                current_state, current_answer, force_complete=True
            )
        return await self._handle_on_topic_intake(current_state, current_answer)

    # -- Public entry point ----------------------------------------------------

    async def ask(
        self,
        current_answer: str,
        state: dict[str, str] | None,
    ) -> ChatResult:
        current_state = self._parse_state(state)

        # Turn routing is phase-first. This keeps the orchestration flow explicit:
        # terminated -> feedback -> completed -> intake (incl. stop detection).
        if current_state.phase == PHASE_TERMINATED:
            return self._handle_terminated_phase(current_state, current_answer)

        if current_state.phase == PHASE_AWAITING_FEEDBACK:
            return self._handle_awaiting_feedback_phase(current_state, current_answer)

        if current_state.phase == PHASE_COMPLETED:
            return self._handle_completed_phase(current_state, current_answer)

        return await self._handle_intake_phase(current_state, current_answer)
