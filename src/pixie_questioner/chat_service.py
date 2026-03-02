"""Chat orchestration service."""

from __future__ import annotations

import json
import re
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

_FOODPAIR_KEY_ORDER: tuple[str, ...] = (
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

_WINE_COLOR_VALUES = frozenset(("wit", "rood", "rosé"))
_WINE_COUNTRY_VALUES = frozenset(
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

_WINE_REGION_VALUES = frozenset(_WINE_REGION_TO_COUNTRY)

_AROMA_TASTE_KEY_ORDER: tuple[str, ...] = (
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


@dataclass
class ChatService:
    llm: object
    question_asker_prompt: str
    notulist: LLMPromptInvoker
    summarizer: LLMPromptInvoker
    questionnaire_template: str
    domain_policy: ChatDomainPolicy
    answer_relevance_classifier: AnswerRelevanceClassifierProtocol | None = None
    aroma_taste_classifier: LLMPromptInvoker | None = None
    handoff_classifier: LLMPromptInvoker | None = None

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

    def _parse_json_object(self, value: str) -> dict[str, object]:
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
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _sanitize_handoff_core_categories(
        self, raw_categories: dict[str, object]
    ) -> dict[str, bool | str | int]:
        categories: dict[str, bool | str | int] = {}
        for key in _FOODPAIR_KEY_ORDER:
            if raw_categories.get(key) is True:
                categories[key] = True

        kleur = raw_categories.get("kleur")
        if isinstance(kleur, str) and kleur in _WINE_COLOR_VALUES:
            categories["kleur"] = kleur

        regio = raw_categories.get("regio")
        if isinstance(regio, str) and regio in _WINE_REGION_VALUES:
            categories["regio"] = regio
            categories["land"] = _WINE_REGION_TO_COUNTRY[regio]
        else:
            land = raw_categories.get("land")
            if isinstance(land, str) and land in _WINE_COUNTRY_VALUES:
                categories["land"] = land

        prijs = raw_categories.get("prijs")
        if isinstance(prijs, int) and not isinstance(prijs, bool) and prijs >= 0:
            categories["prijs"] = prijs
        return categories

    async def _classify_handoff_core_categories(
        self, questionnaire: str
    ) -> dict[str, bool | str | int]:
        if self.handoff_classifier is None:
            return {}
        allowed_foodpair = "\n".join(f"- {key}" for key in _FOODPAIR_KEY_ORDER)
        allowed_countries = "\n".join(
            f"- {country}" for country in sorted(_WINE_COUNTRY_VALUES)
        )
        allowed_regions = "\n".join(
            f"- {region}" for region in sorted(_WINE_REGION_VALUES)
        )
        rules = (
            "Allowed foodpair keys (boolean true only when applicable):\n"
            f"{allowed_foodpair}\n"
            "Allowed color values: wit, rood, rosé\n"
            "Allowed country values:\n"
            f"{allowed_countries}\n"
            "Allowed region values:\n"
            f"{allowed_regions}\n"
            "Price must be a non-negative integer."
        )
        try:
            classifier_output = await self.handoff_classifier.generate(
                current_question="Build categorized handoff JSON.",
                current_answer=questionnaire,
                summary=rules,
            )
        except Exception:
            return {}
        parsed = self._parse_json_object(classifier_output)
        return self._sanitize_handoff_core_categories(parsed)

    def _parse_aroma_taste_flags(self, value: str) -> dict[str, bool]:
        if self._is_blank_questionnaire_value(value):
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        flags: dict[str, bool] = {}
        for key in _AROMA_TASTE_KEY_ORDER:
            raw_value = parsed.get(key)
            if isinstance(raw_value, bool):
                flags[key] = raw_value
        return flags

    def _serialize_aroma_taste_flags(self, flags: dict[str, bool]) -> str:
        ordered = {key: flags[key] for key in _AROMA_TASTE_KEY_ORDER if key in flags}
        return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))

    async def _classify_aroma_taste_flags(self, value: str) -> dict[str, bool]:
        if self.aroma_taste_classifier is None:
            return {}
        allowed_keys = "\n".join(f"- {key}" for key in _AROMA_TASTE_KEY_ORDER)
        try:
            classified = await self.aroma_taste_classifier.generate(
                current_question="Classify aroma and taste preferences.",
                current_answer=value,
                summary=allowed_keys,
            )
        except Exception:
            return {}
        return self._parse_aroma_taste_flags(classified)

    async def _normalize_aroma_taste_for_categorized_flow(
        self, questionnaire: str
    ) -> str:
        field_value = self._extract_questionnaire_field_value(questionnaire, 8)
        if self._is_blank_questionnaire_value(field_value):
            return questionnaire

        parsed_flags = self._parse_aroma_taste_flags(field_value)
        if parsed_flags:
            return self._replace_questionnaire_field_value(
                questionnaire=questionnaire,
                index=8,
                new_value=self._serialize_aroma_taste_flags(parsed_flags),
            )

        classified_flags = await self._classify_aroma_taste_flags(field_value)
        if not classified_flags:
            return questionnaire
        return self._replace_questionnaire_field_value(
            questionnaire=questionnaire,
            index=8,
            new_value=self._serialize_aroma_taste_flags(classified_flags),
        )

    def _extract_aroma_taste_flags(self, value: str) -> dict[str, bool]:
        return self._parse_aroma_taste_flags(value)

    async def _build_handoff_categories_v1(self, questionnaire: str) -> str:
        categories = await self._classify_handoff_core_categories(questionnaire)
        aroma_taste_flags = self._extract_aroma_taste_flags(
            self._extract_questionnaire_field_value(questionnaire, 8)
        )
        for key in _AROMA_TASTE_KEY_ORDER:
            if key in aroma_taste_flags:
                categories[key] = aroma_taste_flags[key]
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

    async def _build_categorized_completion_result(
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
        handoff_categories_v1 = await self._build_handoff_categories_v1(
            updated_questionnaire
        )
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

    async def _handle_completed_phase(
        self, current_state: ChatState, current_answer: str
    ) -> ChatResult:
        if self._is_categorized_handoff_mode():
            handoff_categories_v1 = (
                current_state.handoff_categories_v1
                if current_state.handoff_categories_v1.strip()
                else await self._build_handoff_categories_v1(
                    current_state.questionnaire
                )
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
            updated_questionnaire = (
                await self._normalize_aroma_taste_for_categorized_flow(
                    updated_questionnaire
                )
            )
        updated_summary = await self._update_summary(current_state, current_answer)

        if self._questionnaire_complete(updated_questionnaire) or force_complete:
            if self._is_categorized_handoff_mode():
                return await self._build_categorized_completion_result(
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
            return await self._handle_completed_phase(current_state, current_answer)

        return await self._handle_intake_phase(current_state, current_answer)
