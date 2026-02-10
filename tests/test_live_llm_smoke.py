from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from pixie_questioner.azure_openai_llm import AzureOpenAILLM
from pixie_questioner.chat_service import ChatService
from pixie_questioner.notulist import QuestionnaireNotulist
from pixie_questioner.summarizer import ConversationSummarizer


_REQUIRED_ENV_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "ADVANCED_AZURE_OPENAI_API_VERSION",
    "ADVANCED_AZURE_OPENAI_MODEL",
    "SYSTEM_PROMPT_PATH",
    "QUESTIONNAIRE_NOTULIST_PROMPT_PATH",
    "QUESTIONNAIRE_TEMPLATE_PATH",
)


def _read_prompt(path_env_var: str) -> str:
    prompt_path = Path(os.environ[path_env_var])
    return prompt_path.read_text(encoding="utf-8")


def _build_live_service() -> ChatService:
    llm = AzureOpenAILLM(
        api_version=os.environ["ADVANCED_AZURE_OPENAI_API_VERSION"],
        api_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        model_version=os.environ["ADVANCED_AZURE_OPENAI_MODEL"],
        temperature=float(os.environ.get("ADVANCED_AZURE_OPENAI_TEMPERATURE", "0.2")),
    )
    notulist = QuestionnaireNotulist(
        llm=llm,
        system_prompt_template=_read_prompt("QUESTIONNAIRE_NOTULIST_PROMPT_PATH"),
    )

    summarizer_prompt_path = os.environ.get("SUMMARIZER_PROMPT_PATH")
    if summarizer_prompt_path:
        summarizer = ConversationSummarizer(
            llm=llm,
            system_prompt_template=Path(summarizer_prompt_path).read_text(
                encoding="utf-8"
            ),
        )
    else:
        summarizer = ConversationSummarizer(llm=llm)

    return ChatService(
        llm=llm,
        question_asker_prompt=_read_prompt("SYSTEM_PROMPT_PATH"),
        notulist=notulist,
        summarizer=summarizer,
        questionnaire_template=_read_prompt("QUESTIONNAIRE_TEMPLATE_PATH"),
        completion_token=os.environ.get("COMPLETION_TOKEN", ""),
    )


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TESTS", "1") != "1",
    reason="Set RUN_LIVE_LLM_TESTS=0 to skip live Azure OpenAI smoke tests.",
)
def test_live_llm_post_summary_feedback_flow() -> None:
    missing = [key for key in _REQUIRED_ENV_VARS if not os.environ.get(key)]
    if missing:
        pytest.skip(
            f"Missing required env vars for live LLM test: {', '.join(missing)}"
        )

    service = _build_live_service()

    first = asyncio.run(service.ask(current_answer="stop", state={}))
    assert first.response_type == "answer"
    assert first.summary is not None
    assert first.follow_up_question is not None
    assert first.state.get("phase") == "awaiting_feedback"
    assert first.state.get("feedback_status") == "pending"
    assert first.state.get("final_summary") == first.summary
    assert first.state.get("post_summary_language")

    second = asyncio.run(service.ask(current_answer="ja", state=first.state))
    assert second.response_type == "summary"
    assert second.summary is not None
    assert second.state.get("phase") == "completed"
    assert second.state.get("feedback_status") == "yes"
    assert second.state.get("final_summary") == second.summary
