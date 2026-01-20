"""Chat orchestration service."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Protocol

from .notulist import QuestionnaireNotulist
from .summarizer import ConversationSummarizer

DEFAULT_COMPLETION_TOKEN = "PIXIE_SUMMARY_READY"
_COMPLETION_STRIP_CHARS = " \t\n\r\"'`.,:;!?()[]{}<>"
_INITIAL_QUESTION = "Initial question:"
HANDOFF_NL = [
    "Helder. Tijd om over te schakelen naar een collega met menselijke ogen. Je mag onderstaande samenvatting gerust aan hen laten zien.",
    "Duidelijk. Dit is het moment om een menselijke collega te betrekken. Laat gerust de samenvatting hieronder zien.",
    "Prima. We schakelen nu over naar een menselijke collega. Laat gerust de samenvatting hieronder zien.",
    "Top, ik geef het over aan een collega. Toon hen gerust de samenvatting hieronder.",
    "Oké. Tijd om een menselijke collega in te schakelen. Je kan de samenvatting hieronder aan hen tonen.",
]
HANDOFF_EN = [
    "Got it. Time to hand over to a human colleague. Feel free to show them the summary below.",
    "Understood. Switching to a human colleague now. You can show them the summary below.",
    "All set. Handing this off to a human colleague. Please share the summary below.",
    "Thanks. I will pass this to a human colleague. Feel free to show them the summary below.",
]
HANDOFF_FR = [
    "D'accord. Il est temps de passer a un collegue humain. Vous pouvez leur montrer le resume ci-dessous.",
    "Compris. Nous passons a un collegue humain. Vous pouvez leur montrer le resume ci-dessous.",
    "Tres bien. Je passe le relais a un collegue humain. Montrez-leur le resume ci-dessous.",
    "Parfait. Passage a un collegue humain. Vous pouvez leur montrer le resume ci-dessous.",
]
_LANGUAGE_HINTS = {
    "nl": {
        "de",
        "het",
        "een",
        "ik",
        "je",
        "jij",
        "mijn",
        "broek",
        "spijkerbroek",
        "samenvatting",
        "kleur",
        "wassing",
        "maat",
        "pasvorm",
        "stof",
        "lengte",
        "zoeken",
        "liefst",
    },
    "en": {
        "the",
        "and",
        "with",
        "fit",
        "size",
        "color",
        "wash",
        "fabric",
        "length",
        "looking",
        "want",
        "prefer",
        "summary",
    },
    "fr": {
        "le",
        "la",
        "les",
        "un",
        "une",
        "taille",
        "couleur",
        "lavage",
        "tissu",
        "longueur",
        "resume",
        "veux",
        "cherche",
    },
}
_STOP_PHRASES = (
    "stop",
    "stoppen",
    "ik wil stoppen",
    "ik ben klaar",
    "klaar",
    "genoeg",
    "dat was het",
    "dat is alles",
    "dit is alles",
    "niets meer",
    "done",
    "i am done",
    "finished",
    "that's all",
    "that is all",
    "that's it",
    "that is it",
    "no more",
    "i want to stop",
    "c'est tout",
    "cest tout",
    "j'ai fini",
    "jai fini",
    "fini",
    "arrete",
    "arreter",
)


class LLMClientProtocol(Protocol):
    async def ainvoke(self, messages: list[dict[str, str]]):
        ...


@dataclass(frozen=True)
class ChatResult:
    text: str
    response_type: str
    state: dict[str, str]
    summary: str | None


@dataclass
class ChatState:
    questionnaire: str = ""
    summary: str = ""
    last_question: str = _INITIAL_QUESTION


@dataclass
class ChatService:
    llm: object
    question_asker_prompt: str
    notulist: QuestionnaireNotulist
    summarizer: ConversationSummarizer
    questionnaire_template: str
    completion_token: str = DEFAULT_COMPLETION_TOKEN

    def __post_init__(self) -> None:
        if not self.questionnaire_template.strip():
            raise RuntimeError("Questionnaire template is required.")
        self._questionnaire_indices = self._extract_questionnaire_indices(
            self.questionnaire_template
        )
        if not self._questionnaire_indices:
            raise RuntimeError("Questionnaire template must contain numbered fields.")

    def _get_client(self) -> LLMClientProtocol:
        if hasattr(self.llm, "ainvoke"):
            return self.llm  # type: ignore[return-value]
        model = getattr(self.llm, "model", None)
        if model is None:
            raise RuntimeError("LLM client is not configured.")
        return model

    def _render_question_asker_prompt(self, questionnaire: str) -> str:
        prompt = self.question_asker_prompt
        if "{summary}" in prompt:
            prompt = prompt.replace("{summary}", questionnaire)
        if "{completion_token}" in prompt:
            prompt = prompt.replace("{completion_token}", self.completion_token)
        return prompt

    def _extract_content(self, response: object) -> str:
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        return content

    def _strip_completion_token(self, content: str) -> str:
        stripped = re.sub(re.escape(self.completion_token), "", content, flags=re.IGNORECASE)
        return stripped.strip(_COMPLETION_STRIP_CHARS)

    def _detect_language(self, text: str) -> str:
        words = re.findall(r"[a-zA-Z']+", text.casefold())
        if not words:
            return "nl"
        scores = {lang: 0 for lang in _LANGUAGE_HINTS}
        for word in words:
            for lang, hints in _LANGUAGE_HINTS.items():
                if word in hints:
                    scores[lang] += 1
        max_score = max(scores.values())
        if max_score == 0:
            return "nl"
        for lang in ("nl", "en", "fr"):
            if scores[lang] == max_score:
                return lang
        return "nl"

    def _build_handoff_line(self, language: str) -> str:
        if language.startswith("nl"):
            return random.choice(HANDOFF_NL)
        if language.startswith("fr"):
            return random.choice(HANDOFF_FR)
        return random.choice(HANDOFF_EN)

    def _extract_questionnaire_indices(self, questionnaire: str) -> list[int]:
        indices = set()
        for match in re.finditer(r"(?m)^\s*(\d+)\.", questionnaire):
            indices.add(int(match.group(1)))
        return sorted(indices)

    def _is_valid_questionnaire(self, questionnaire: str) -> bool:
        for index in self._questionnaire_indices:
            if not re.search(rf"(?m)^\s*{index}\.", questionnaire):
                return False
        return True

    def _parse_state(self, state: dict[str, str] | None) -> ChatState:
        if not state:
            return ChatState(questionnaire=self.questionnaire_template)
        questionnaire = state.get("questionnaire", "")
        summary = state.get("summary", "")
        last_question = state.get("last_question", _INITIAL_QUESTION)
        if not isinstance(questionnaire, str):
            raise RuntimeError("State field 'questionnaire' must be a string.")
        if not isinstance(summary, str):
            raise RuntimeError("State field 'summary' must be a string.")
        if not isinstance(last_question, str):
            raise RuntimeError("State field 'last_question' must be a string.")
        if not questionnaire.strip():
            questionnaire = self.questionnaire_template
        return ChatState(
            questionnaire=questionnaire,
            summary=summary,
            last_question=last_question,
        )

    def _serialize_state(self, state: ChatState) -> dict[str, str]:
        return {
            "questionnaire": state.questionnaire,
            "summary": state.summary,
            "last_question": state.last_question,
        }

    def _user_requested_stop(self, current_answer: str) -> bool:
        normalized = " ".join(current_answer.casefold().split())
        return any(phrase in normalized for phrase in _STOP_PHRASES)

    def _questionnaire_complete(self, questionnaire: str) -> bool:
        return "[]" not in questionnaire

    async def _ask_next_question(self, questionnaire: str) -> str:
        system_prompt = self._render_question_asker_prompt(questionnaire)
        response = await self._get_client().ainvoke(
            [{"role": "system", "content": system_prompt}]
        )
        content = self._extract_content(response)
        if self.completion_token and self.completion_token.casefold() in content.casefold():
            content = self._strip_completion_token(content)
        return content

    async def ask(
        self,
        current_answer: str,
        state: dict[str, str] | None,
    ) -> ChatResult:
        current_state = self._parse_state(state)
        current_question = current_state.last_question
        updated_questionnaire = await self.notulist.generate(
            current_question=current_question,
            current_answer=current_answer,
            summary=current_state.questionnaire,
        )
        if updated_questionnaire is None:
            raise RuntimeError("Failed to update questionnaire.")
        if not self._is_valid_questionnaire(updated_questionnaire):
            updated_questionnaire = current_state.questionnaire

        updated_summary = await self.summarizer.generate(
            current_question=current_question,
            current_answer=current_answer,
            summary=current_state.summary,
        )
        if updated_summary is None:
            raise RuntimeError("Failed to summarize conversation.")

        if self._questionnaire_complete(updated_questionnaire) or self._user_requested_stop(
            current_answer
        ):
            language = self._detect_language(updated_summary)
            handoff_line = self._build_handoff_line(language)
            final_state = ChatState(
                questionnaire=updated_questionnaire,
                summary=updated_summary,
                last_question=current_question,
            )
            return ChatResult(
                text=handoff_line,
                response_type="summary",
                state=self._serialize_state(final_state),
                summary=updated_summary,
            )

        next_question = await self._ask_next_question(updated_questionnaire)
        next_state = ChatState(
            questionnaire=updated_questionnaire,
            summary=updated_summary,
            last_question=next_question,
        )
        return ChatResult(
            text=next_question,
            response_type="answer",
            state=self._serialize_state(next_state),
            summary=None,
        )
