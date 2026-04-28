"""Shared LLM client utilities and prompt invocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast


class LLMClientProtocol(Protocol):
    async def ainvoke(self, messages: list[dict[str, str]]): ...


def resolve_llm_client(llm: object) -> LLMClientProtocol:
    """Resolve an LLM wrapper to a client that supports ``ainvoke``."""
    if hasattr(llm, "ainvoke"):
        return cast(LLMClientProtocol, llm)
    raise RuntimeError("LLM client must support ainvoke().")


def extract_llm_content(response: object) -> str:
    """Extract text content from an LLM response object."""
    content = getattr(response, "content", None)
    if content is None:
        content = str(response)
    return content


CONVERSATION_SUMMARIZER_PROMPT = (
    "Given:\n"
    "* The current question of an AI\n"
    "* The answer of a user\n"
    "* The summary of previous information.\n"
    "Summarize everything what the user wants in 1 paragraph without losing any information.\n"
    "Formulate in a gender-neutral way.\n"
    "Internally analyze in which language all this information is. Use this language in the summary.\n"
    "\n"
    "AI current question:\n"
    "{current_question}\n"
    "User current answer:\n"
    "{current_answer}\n"
    "Summary of previous information:\n"
    "{summary}\n"
    "\n"
    "New summary of what user wants (translated to the language of all this information):"
)


@dataclass
class LLMPromptInvoker:
    """Invokes an LLM with a formatted system prompt template.

    Used for both questionnaire notulist and conversation summarizer roles.
    The ``error_label`` customizes the error message on failure.
    """

    llm: object
    system_prompt_template: str
    error_label: str = "LLM prompt"

    async def generate(
        self, current_question: str, current_answer: str, summary: str
    ) -> str:
        system_prompt = self.system_prompt_template.format(
            summary=summary,
            current_question=current_question,
            current_answer=current_answer,
        )
        client = resolve_llm_client(self.llm)
        try:
            response = await client.ainvoke(
                [{"role": "system", "content": system_prompt}]
            )
        except Exception as exc:
            raise RuntimeError(f"{self.error_label} invocation failed.") from exc
        return extract_llm_content(response)
