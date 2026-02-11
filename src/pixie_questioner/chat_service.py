"""Chat orchestration service."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Protocol

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

from .notulist import QuestionnaireNotulist
from .summarizer import ConversationSummarizer

DEFAULT_COMPLETION_TOKEN = "PIXIE_SUMMARY_READY"
_COMPLETION_STRIP_CHARS = " \t\n\r\"'`.,:;!?()[]{}<>"
_INITIAL_QUESTION = "Initial question:"
_SUPPORTED_LANGUAGE_CODES = {"nl", "en", "fr"}
_SUPPORTED_LANGUAGE_PRIORITY = {"nl": 0, "en": 1, "fr": 2}
_SUPPORTED_LANGUAGE_NAMES = {
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
}
_SUPPORTED_LANGUAGE_CLOSE_MARGIN = 0.15
_OTHER_LANGUAGE_STRONG_PROBABILITY = 0.85
_OTHER_LANGUAGE_MIN_WORDS = 4
_PHASE_INTAKE = "intake"
_PHASE_AWAITING_FEEDBACK = "awaiting_feedback"
_PHASE_COMPLETED = "completed"
_PHASE_TERMINATED = "terminated"
_VALID_PHASES = {
    _PHASE_INTAKE,
    _PHASE_AWAITING_FEEDBACK,
    _PHASE_COMPLETED,
    _PHASE_TERMINATED,
}
_FEEDBACK_PENDING = "pending"
_FEEDBACK_YES = "yes"
_FEEDBACK_NO = "no"
_VALID_FEEDBACK_STATUSES = {_FEEDBACK_PENDING, _FEEDBACK_YES, _FEEDBACK_NO}
DetectorFactory.seed = 0
HANDOFF_NL = [
    "Dank je wel, dit geeft al een duidelijk beeld naar wat je op zoek bent:",
]
HANDOFF_EN = [
    "Thanks. This gives a clear picture of what you're looking for: ",
]
HANDOFF_FR = [
    "Merci. Cela donne une idée claire de ce que vous recherchez: ",
]
FEEDBACK_QUESTION_NL = [
    "Zou je deze assistent bij een volgend winkelbezoek opnieuw gebruiken? (ja/nee)",
]
FEEDBACK_QUESTION_EN = [
    "Would you use this assistant again on a future store visit? (yes/no)",
]
FEEDBACK_QUESTION_FR = [
    "Utiliseriez-vous a nouveau cet assistant lors d'une prochaine visite en magasin ? (oui/non)",
]
FEEDBACK_THANKS_NL = [
    "Bedankt voor je feedback.",
]
FEEDBACK_THANKS_EN = [
    "Thanks for your feedback.",
]
FEEDBACK_THANKS_FR = [
    "Merci pour votre retour.",
]
GOODBYE_NL = [
    "Dank je wel. Ik rond het gesprek hier af. Veel succes met je keuze.",
]
GOODBYE_EN = [
    "Thanks. I'll close the conversation here. Good luck with your choice.",
]
GOODBYE_FR = [
    "Merci. Je vais cloturer la conversation ici. Bonne chance pour votre choix.",
]
OFF_TOPIC_REDIRECT_NL = [
    "Ik kan je alleen helpen met jeansbroeken. Kun je deze vraag beantwoorden: {question}",
]
OFF_TOPIC_REDIRECT_EN = [
    "I can only help with jeans. Could you answer this question: {question}",
]
OFF_TOPIC_REDIRECT_FR = [
    "Je peux seulement vous aider avec des jeans. Pouvez-vous repondre a cette question: {question}",
]
_FEEDBACK_YES_TOKENS = {"ja", "yes", "oui", "zeker", "absoluut", "y", "si", "jaa"}
_FEEDBACK_NO_TOKENS = {"nee", "no", "non", "n", "nein"}
_OFF_TOPIC_TERMINATION_THRESHOLD = 2
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
DEFAULT_RELEVANCE_CLASSIFIER_PROMPT = (
    "You are a strict relevance classifier for a jeans-store assistant.\n"
    "Decide if the customer's latest answer is a reasonable attempt to answer the assistant's latest question.\n"
    "Use this context:\n"
    "- The store only helps customers choose jeans.\n"
    "- The assistant asks targeted intake questions to narrow down the right jeans.\n"
    "- If the customer answer is clearly about unrelated topics, mark OFF_TOPIC.\n"
    "- If the answer is a reasonable attempt, even if short or imperfect, mark ON_TOPIC.\n"
    "- If unsure, mark ON_TOPIC.\n\n"
    "Questionnaire context:\n"
    "{questionnaire}\n\n"
    "Assistant question:\n"
    "{current_question}\n\n"
    "Customer answer:\n"
    "{current_answer}\n\n"
    "Return exactly one label: ON_TOPIC or OFF_TOPIC."
)


class LLMClientProtocol(Protocol):
    async def ainvoke(self, messages: list[dict[str, str]]): ...


class AnswerRelevanceClassifierProtocol(Protocol):
    async def is_on_topic(
        self,
        current_question: str,
        current_answer: str,
        questionnaire: str,
    ) -> bool: ...


@dataclass
class LLMAnswerRelevanceClassifier:
    llm: object
    system_prompt_template: str = DEFAULT_RELEVANCE_CLASSIFIER_PROMPT

    def _get_client(self) -> LLMClientProtocol:
        if hasattr(self.llm, "ainvoke"):
            return self.llm  # type: ignore[return-value]
        model = getattr(self.llm, "model", None)
        if model is None:
            raise RuntimeError("LLM client is not configured.")
        return model

    def _extract_content(self, response: object) -> str:
        content = getattr(response, "content", None)
        if content is None:
            content = str(response)
        return content

    async def is_on_topic(
        self,
        current_question: str,
        current_answer: str,
        questionnaire: str,
    ) -> bool:
        system_prompt = self.system_prompt_template.format(
            questionnaire=questionnaire,
            current_question=current_question,
            current_answer=current_answer,
        )
        response = await self._get_client().ainvoke(
            [{"role": "system", "content": system_prompt}]
        )
        content = self._extract_content(response)
        match = re.search(r"\b(ON_TOPIC|OFF_TOPIC)\b", content.upper())
        if match is None:
            return True
        return match.group(1) == "ON_TOPIC"


@dataclass(frozen=True)
class ChatResult:
    text: str
    response_type: str
    state: dict[str, str]
    summary: str | None
    follow_up_question: str | None = None
    quick_replies: list[dict[str, str]] | None = None


@dataclass
class ChatState:
    questionnaire: str = ""
    summary: str = ""
    last_question: str = _INITIAL_QUESTION
    phase: str = _PHASE_INTAKE
    feedback_status: str = _FEEDBACK_PENDING
    final_summary: str = ""
    post_summary_language: str = ""
    boundary_violation_count: int = 0


@dataclass
class ChatService:
    llm: object
    question_asker_prompt: str
    notulist: QuestionnaireNotulist
    summarizer: ConversationSummarizer
    questionnaire_template: str
    completion_token: str = DEFAULT_COMPLETION_TOKEN
    answer_relevance_classifier: AnswerRelevanceClassifierProtocol | None = None

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
        stripped = re.sub(
            re.escape(self.completion_token), "", content, flags=re.IGNORECASE
        )
        return stripped.strip(_COMPLETION_STRIP_CHARS)

    def _detect_language(self, text: str) -> str:
        candidates = self._detect_language_candidates(text)
        if not candidates:
            return "nl"

        for language_code, _ in candidates:
            if language_code in _SUPPORTED_LANGUAGE_CODES:
                return language_code
        return "nl"

    def _detect_language_candidates(self, text: str) -> list[tuple[str, float]]:
        if not text.strip():
            return []
        try:
            detected = detect_langs(text)
        except LangDetectException:
            return []
        return [(candidate.lang, candidate.prob) for candidate in detected]

    def _select_best_supported_language(
        self, candidates: list[tuple[str, float]]
    ) -> tuple[str, float] | None:
        supported_candidates = [
            (language_code, probability)
            for language_code, probability in candidates
            if language_code in _SUPPORTED_LANGUAGE_CODES
        ]
        if not supported_candidates:
            return None
        return min(
            supported_candidates,
            key=lambda item: (
                -item[1],
                _SUPPORTED_LANGUAGE_PRIORITY[item[0]],
            ),
        )

    def _select_preferred_language_from_text(self, text: str) -> tuple[str, bool]:
        candidates = self._detect_language_candidates(text)
        if not candidates:
            return "nl", True

        top_language, top_probability = candidates[0]
        best_supported = self._select_best_supported_language(candidates)
        word_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text))

        if top_language in _SUPPORTED_LANGUAGE_CODES:
            return top_language, False

        if best_supported is not None:
            best_supported_language, best_supported_probability = best_supported
            score_gap = top_probability - best_supported_probability
            if score_gap <= _SUPPORTED_LANGUAGE_CLOSE_MARGIN:
                return best_supported_language, True
            if (
                top_probability >= _OTHER_LANGUAGE_STRONG_PROBABILITY
                and word_count >= _OTHER_LANGUAGE_MIN_WORDS
            ):
                return top_language, False
            return best_supported_language, True

        if (
            top_probability >= _OTHER_LANGUAGE_STRONG_PROBABILITY
            and word_count >= _OTHER_LANGUAGE_MIN_WORDS
        ):
            return top_language, False
        return "nl", True

    def _detect_preferred_language(
        self, current_answer: str, previous_summary: str
    ) -> str:
        answer_language, answer_is_uncertain = (
            self._select_preferred_language_from_text(current_answer)
        )
        if not answer_is_uncertain:
            return answer_language

        summary_language, summary_is_uncertain = (
            self._select_preferred_language_from_text(previous_summary)
        )
        if not summary_is_uncertain:
            return summary_language

        if answer_language in _SUPPORTED_LANGUAGE_CODES:
            return answer_language
        if summary_language in _SUPPORTED_LANGUAGE_CODES:
            return summary_language
        return "nl"

    def _build_language_instruction(self, language_code: str) -> str:
        if language_code in _SUPPORTED_LANGUAGE_NAMES:
            language_target = _SUPPORTED_LANGUAGE_NAMES[language_code]
            default_hint = "- If uncertain, default to Dutch."
        else:
            language_target = f"the language with ISO code '{language_code}'"
            default_hint = "- If uncertain, default to Dutch, English, or French."
        return (
            "\n\nLanguage policy:\n"
            f"- Respond in {language_target}.\n"
            f"{default_hint}\n"
            "- Prefer Dutch when there is no strong language signal."
        )

    def _build_handoff_line(self, language: str) -> str:
        if language.startswith("nl"):
            return random.choice(HANDOFF_NL)
        if language.startswith("fr"):
            return random.choice(HANDOFF_FR)
        return random.choice(HANDOFF_EN)

    def _build_feedback_question(self, language: str) -> str:
        if language.startswith("nl"):
            return random.choice(FEEDBACK_QUESTION_NL)
        if language.startswith("fr"):
            return random.choice(FEEDBACK_QUESTION_FR)
        return random.choice(FEEDBACK_QUESTION_EN)

    def _build_feedback_quick_replies(self, language: str) -> list[dict[str, str]]:
        if language.startswith("nl"):
            return [
                {"label": "Ja", "value": "yes"},
                {"label": "Nee", "value": "no"},
            ]
        if language.startswith("fr"):
            return [
                {"label": "Oui", "value": "yes"},
                {"label": "Non", "value": "no"},
            ]
        return [
            {"label": "Yes", "value": "yes"},
            {"label": "No", "value": "no"},
        ]

    def _build_feedback_thanks(self, language: str) -> str:
        if language.startswith("nl"):
            return random.choice(FEEDBACK_THANKS_NL)
        if language.startswith("fr"):
            return random.choice(FEEDBACK_THANKS_FR)
        return random.choice(FEEDBACK_THANKS_EN)

    def _build_goodbye_line(self, language: str) -> str:
        if language.startswith("nl"):
            return random.choice(GOODBYE_NL)
        if language.startswith("fr"):
            return random.choice(GOODBYE_FR)
        return random.choice(GOODBYE_EN)

    def _build_off_topic_redirect(self, language: str, current_question: str) -> str:
        if language.startswith("nl"):
            return random.choice(OFF_TOPIC_REDIRECT_NL).format(
                question=current_question
            )
        if language.startswith("fr"):
            return random.choice(OFF_TOPIC_REDIRECT_FR).format(
                question=current_question
            )
        return random.choice(OFF_TOPIC_REDIRECT_EN).format(question=current_question)

    def _normalize_feedback_response(self, current_answer: str) -> str | None:
        tokens = set(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", current_answer.casefold()))
        has_yes = bool(tokens & _FEEDBACK_YES_TOKENS)
        has_no = bool(tokens & _FEEDBACK_NO_TOKENS)
        if has_yes and not has_no:
            return _FEEDBACK_YES
        if has_no and not has_yes:
            return _FEEDBACK_NO
        return None

    async def _is_on_topic_answer(
        self,
        current_question: str,
        current_answer: str,
        questionnaire: str,
    ) -> bool:
        if self.answer_relevance_classifier is None:
            return True
        try:
            return await self.answer_relevance_classifier.is_on_topic(
                current_question=current_question,
                current_answer=current_answer,
                questionnaire=questionnaire,
            )
        except Exception:
            return True

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
        phase = state.get("phase", _PHASE_INTAKE)
        feedback_status = state.get("feedback_status", _FEEDBACK_PENDING)
        final_summary = state.get("final_summary", "")
        post_summary_language = state.get("post_summary_language", "")
        boundary_violation_count_raw = state.get("boundary_violation_count", "0")
        if not isinstance(questionnaire, str):
            raise RuntimeError("State field 'questionnaire' must be a string.")
        if not isinstance(summary, str):
            raise RuntimeError("State field 'summary' must be a string.")
        if not isinstance(last_question, str):
            raise RuntimeError("State field 'last_question' must be a string.")
        if not isinstance(phase, str):
            raise RuntimeError("State field 'phase' must be a string.")
        if not isinstance(feedback_status, str):
            raise RuntimeError("State field 'feedback_status' must be a string.")
        if not isinstance(final_summary, str):
            raise RuntimeError("State field 'final_summary' must be a string.")
        if not isinstance(post_summary_language, str):
            raise RuntimeError("State field 'post_summary_language' must be a string.")
        if not isinstance(boundary_violation_count_raw, str):
            raise RuntimeError(
                "State field 'boundary_violation_count' must be a string."
            )
        try:
            boundary_violation_count = int(boundary_violation_count_raw)
        except ValueError:
            boundary_violation_count = 0
        boundary_violation_count = max(boundary_violation_count, 0)
        if not questionnaire.strip():
            questionnaire = self.questionnaire_template
        if phase not in _VALID_PHASES:
            phase = _PHASE_INTAKE
        if feedback_status not in _VALID_FEEDBACK_STATUSES:
            feedback_status = _FEEDBACK_PENDING
        if phase == _PHASE_INTAKE:
            feedback_status = _FEEDBACK_PENDING
            final_summary = ""
            post_summary_language = ""
        elif phase == _PHASE_TERMINATED:
            feedback_status = _FEEDBACK_PENDING
            final_summary = ""
        else:
            if not final_summary.strip():
                final_summary = summary
            if not post_summary_language.strip():
                post_summary_language = self._detect_language(final_summary)
            if (
                phase == _PHASE_AWAITING_FEEDBACK
                and feedback_status != _FEEDBACK_PENDING
            ):
                phase = _PHASE_COMPLETED
        return ChatState(
            questionnaire=questionnaire,
            summary=summary,
            last_question=last_question,
            phase=phase,
            feedback_status=feedback_status,
            final_summary=final_summary,
            post_summary_language=post_summary_language,
            boundary_violation_count=boundary_violation_count,
        )

    def _serialize_state(self, state: ChatState) -> dict[str, str]:
        return {
            "questionnaire": state.questionnaire,
            "summary": state.summary,
            "last_question": state.last_question,
            "phase": state.phase,
            "feedback_status": state.feedback_status,
            "final_summary": state.final_summary,
            "post_summary_language": state.post_summary_language,
            "boundary_violation_count": str(state.boundary_violation_count),
        }

    def _user_requested_stop(self, current_answer: str) -> bool:
        normalized = " ".join(current_answer.casefold().split())
        return any(phrase in normalized for phrase in _STOP_PHRASES)

    def _questionnaire_complete(self, questionnaire: str) -> bool:
        return "[]" not in questionnaire

    async def _ask_next_question(
        self, questionnaire: str, preferred_language: str
    ) -> str:
        system_prompt = self._render_question_asker_prompt(questionnaire)
        system_prompt = (
            f"{system_prompt}{self._build_language_instruction(preferred_language)}"
        )
        response = await self._get_client().ainvoke(
            [{"role": "system", "content": system_prompt}]
        )
        content = self._extract_content(response)
        if (
            self.completion_token
            and self.completion_token.casefold() in content.casefold()
        ):
            content = self._strip_completion_token(content)
        return content

    async def ask(
        self,
        current_answer: str,
        state: dict[str, str] | None,
    ) -> ChatResult:
        current_state = self._parse_state(state)
        if current_state.phase == _PHASE_TERMINATED:
            language = current_state.post_summary_language
            if not language:
                language = self._detect_preferred_language(
                    current_answer=current_answer,
                    previous_summary=current_state.summary,
                )
                current_state.post_summary_language = language
            return ChatResult(
                text=self._build_goodbye_line(language),
                response_type="answer",
                state=self._serialize_state(current_state),
                summary=None,
            )
        if self._user_requested_stop(current_answer):
            language = self._detect_preferred_language(
                current_answer=current_answer,
                previous_summary=current_state.summary,
            )
            terminated_state = ChatState(
                questionnaire=current_state.questionnaire,
                summary=current_state.summary,
                last_question=current_state.last_question,
                phase=_PHASE_TERMINATED,
                feedback_status=_FEEDBACK_PENDING,
                final_summary="",
                post_summary_language=language,
                boundary_violation_count=current_state.boundary_violation_count,
            )
            return ChatResult(
                text=self._build_goodbye_line(language),
                response_type="answer",
                state=self._serialize_state(terminated_state),
                summary=None,
            )
        if current_state.phase == _PHASE_AWAITING_FEEDBACK:
            summary_text = current_state.final_summary
            if not summary_text.strip():
                summary_text = current_state.summary
                current_state.final_summary = summary_text
            language = current_state.post_summary_language
            if not language:
                language = self._detect_language(summary_text)
                current_state.post_summary_language = language
            feedback_response = self._normalize_feedback_response(current_answer)
            if feedback_response is not None:
                current_state.phase = _PHASE_COMPLETED
                current_state.feedback_status = feedback_response
                return ChatResult(
                    text=self._build_feedback_thanks(language),
                    response_type="summary",
                    state=self._serialize_state(current_state),
                    summary=summary_text,
                )
            current_state.feedback_status = _FEEDBACK_PENDING
            return ChatResult(
                text=self._build_feedback_question(language),
                response_type="answer",
                state=self._serialize_state(current_state),
                summary=summary_text,
                quick_replies=self._build_feedback_quick_replies(language),
            )
        if current_state.phase == _PHASE_COMPLETED:
            summary_text = current_state.final_summary
            if not summary_text.strip():
                summary_text = current_state.summary
                current_state.final_summary = summary_text
            language = current_state.post_summary_language
            if not language:
                language = self._detect_language(summary_text)
                current_state.post_summary_language = language
            return ChatResult(
                text=self._build_feedback_thanks(language),
                response_type="summary",
                state=self._serialize_state(current_state),
                summary=summary_text,
            )
        is_on_topic = await self._is_on_topic_answer(
            current_question=current_state.last_question,
            current_answer=current_answer,
            questionnaire=current_state.questionnaire,
        )
        if not is_on_topic:
            current_state.boundary_violation_count += 1
            language = self._detect_preferred_language(
                current_answer=current_answer,
                previous_summary=current_state.summary,
            )
            if (
                current_state.boundary_violation_count
                >= _OFF_TOPIC_TERMINATION_THRESHOLD
            ):
                language = self._detect_preferred_language(
                    current_answer=current_answer,
                    previous_summary=current_state.summary,
                )
                terminated_state = ChatState(
                    questionnaire=current_state.questionnaire,
                    summary=current_state.summary,
                    last_question=current_state.last_question,
                    phase=_PHASE_TERMINATED,
                    feedback_status=_FEEDBACK_PENDING,
                    final_summary="",
                    post_summary_language=language,
                    boundary_violation_count=current_state.boundary_violation_count,
                )
                return ChatResult(
                    text=self._build_goodbye_line(language),
                    response_type="answer",
                    state=self._serialize_state(terminated_state),
                    summary=None,
                )
            redirected_state = ChatState(
                questionnaire=current_state.questionnaire,
                summary=current_state.summary,
                last_question=current_state.last_question,
                phase=_PHASE_INTAKE,
                feedback_status=_FEEDBACK_PENDING,
                final_summary="",
                post_summary_language="",
                boundary_violation_count=current_state.boundary_violation_count,
            )
            return ChatResult(
                text=self._build_off_topic_redirect(
                    language=language,
                    current_question=current_state.last_question,
                ),
                response_type="answer",
                state=self._serialize_state(redirected_state),
                summary=None,
            )
        else:
            current_state.boundary_violation_count = 0
        current_question = current_state.last_question
        try:
            updated_questionnaire = await self.notulist.generate(
                current_question=current_question,
                current_answer=current_answer,
                summary=current_state.questionnaire,
            )
        except Exception as exc:
            raise RuntimeError("Failed to update questionnaire.") from exc
        if updated_questionnaire is None:
            raise RuntimeError("Failed to update questionnaire.")
        if not self._is_valid_questionnaire(updated_questionnaire):
            updated_questionnaire = current_state.questionnaire

        try:
            updated_summary = await self.summarizer.generate(
                current_question=current_question,
                current_answer=current_answer,
                summary=current_state.summary,
            )
        except Exception as exc:
            raise RuntimeError("Failed to summarize conversation.") from exc
        if updated_summary is None:
            raise RuntimeError("Failed to summarize conversation.")

        if self._questionnaire_complete(
            updated_questionnaire
        ) or self._user_requested_stop(current_answer):
            language = self._detect_language(updated_summary)
            handoff_line = self._build_handoff_line(language)
            feedback_question = self._build_feedback_question(language)
            final_state = ChatState(
                questionnaire=updated_questionnaire,
                summary=updated_summary,
                last_question=current_question,
                phase=_PHASE_AWAITING_FEEDBACK,
                feedback_status=_FEEDBACK_PENDING,
                final_summary=updated_summary,
                post_summary_language=language,
                boundary_violation_count=current_state.boundary_violation_count,
            )
            return ChatResult(
                text=handoff_line,
                response_type="answer",
                state=self._serialize_state(final_state),
                summary=updated_summary,
                follow_up_question=feedback_question,
                quick_replies=self._build_feedback_quick_replies(language),
            )

        preferred_language = self._detect_preferred_language(
            current_answer=current_answer,
            previous_summary=current_state.summary,
        )
        next_question = await self._ask_next_question(
            updated_questionnaire, preferred_language
        )
        next_state = ChatState(
            questionnaire=updated_questionnaire,
            summary=updated_summary,
            last_question=next_question,
            phase=_PHASE_INTAKE,
            feedback_status=_FEEDBACK_PENDING,
            boundary_violation_count=current_state.boundary_violation_count,
        )
        return ChatResult(
            text=next_question,
            response_type="answer",
            state=self._serialize_state(next_state),
            summary=None,
        )
