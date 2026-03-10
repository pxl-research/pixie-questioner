"""LLM-backed next-question generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..llm_client import extract_llm_content, resolve_llm_client
from .language import LanguageSupport
from .messages import PolicyMessages

_COMPLETION_STRIP_CHARS = " \t\n\r\"'`.,:;!?()[]{}<>"


@dataclass
class QuestionAsker:
    """Generates the next question via an LLM.

    Owns prompt rendering, completion-token stripping, language
    instruction injection and term simplification.
    """

    llm: object
    prompt_template: str
    completion_token: str
    language: LanguageSupport
    messages: PolicyMessages

    def _render_prompt(self, questionnaire: str) -> str:
        prompt = self.prompt_template
        if "{summary}" in prompt:
            prompt = prompt.replace("{summary}", questionnaire)
        if "{completion_token}" in prompt:
            prompt = prompt.replace("{completion_token}", self.completion_token)
        return prompt

    def _strip_completion_token(self, content: str) -> str:
        stripped = re.sub(
            re.escape(self.completion_token), "", content, flags=re.IGNORECASE
        )
        return stripped.strip(_COMPLETION_STRIP_CHARS)

    async def ask(self, questionnaire: str, conversation_language: str) -> str:
        """Ask the LLM for the next question and return the cleaned text."""
        system_prompt = self._render_prompt(questionnaire)
        system_prompt = (
            f"{system_prompt}{self.language.build_instruction(conversation_language)}"
        )
        response = await resolve_llm_client(self.llm).ainvoke(
            [{"role": "system", "content": system_prompt}]
        )
        content = extract_llm_content(response)
        if (
            self.completion_token
            and self.completion_token.casefold() in content.casefold()
        ):
            content = self._strip_completion_token(content)
        return self.messages.simplify_terms(content, conversation_language)
