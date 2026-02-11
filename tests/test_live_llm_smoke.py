from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Mapping

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


def _collect_live_env_issues(env: Mapping[str, str]) -> list[str]:
    issues: list[str] = []
    for key in _REQUIRED_ENV_VARS:
        value = env.get(key, "").strip()
        if not value:
            issues.append(f"{key} is missing or empty.")

    endpoint = env.get("AZURE_OPENAI_ENDPOINT", "").strip().casefold()
    if endpoint and (
        "your-resource.openai.azure.com" in endpoint or "replace-me" in endpoint
    ):
        issues.append("AZURE_OPENAI_ENDPOINT looks like a placeholder value.")

    key = env.get("AZURE_OPENAI_KEY", "").strip().casefold()
    if key and (key == "replace-me" or key == "your-key" or "replace-me" in key):
        issues.append("AZURE_OPENAI_KEY looks like a placeholder value.")

    model = env.get("ADVANCED_AZURE_OPENAI_MODEL", "").strip().casefold()
    if model and (
        model == "replace-me" or model == "your-model" or "replace-me" in model
    ):
        issues.append("ADVANCED_AZURE_OPENAI_MODEL looks like a placeholder value.")

    return issues


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
def test_live_llm_stop_terminates_without_summary() -> None:
    issues = _collect_live_env_issues(os.environ)
    if issues:
        pytest.fail("Live LLM configuration issues:\n- " + "\n- ".join(issues))

    service = _build_live_service()

    result = asyncio.run(service.ask(current_answer="stop", state={}))
    assert result.response_type == "answer"
    assert result.summary is None
    assert result.follow_up_question is None
    assert result.state.get("phase") == "terminated"


def test_collect_live_env_issues_reports_missing_and_placeholder_values() -> None:
    issues = _collect_live_env_issues(
        {
            "AZURE_OPENAI_ENDPOINT": "https://your-resource.openai.azure.com/",
            "AZURE_OPENAI_KEY": "replace-me",
            "ADVANCED_AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
            "ADVANCED_AZURE_OPENAI_MODEL": "replace-me",
            "SYSTEM_PROMPT_PATH": "prompts/question_asker.txt",
            "QUESTIONNAIRE_NOTULIST_PROMPT_PATH": "prompts/questionnaire_notulist.txt",
        }
    )

    assert "QUESTIONNAIRE_TEMPLATE_PATH is missing or empty." in issues
    assert "AZURE_OPENAI_ENDPOINT looks like a placeholder value." in issues
    assert "AZURE_OPENAI_KEY looks like a placeholder value." in issues
    assert "ADVANCED_AZURE_OPENAI_MODEL looks like a placeholder value." in issues


def test_collect_live_env_issues_accepts_valid_configuration() -> None:
    issues = _collect_live_env_issues(
        {
            "AZURE_OPENAI_ENDPOINT": "https://tai-for-gpt-4.openai.azure.com/",
            "AZURE_OPENAI_KEY": "live-secret",
            "ADVANCED_AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
            "ADVANCED_AZURE_OPENAI_MODEL": "gpt-5-chat",
            "SYSTEM_PROMPT_PATH": "prompts/question_asker.txt",
            "QUESTIONNAIRE_NOTULIST_PROMPT_PATH": "prompts/questionnaire_notulist.txt",
            "QUESTIONNAIRE_TEMPLATE_PATH": "prompts/questionnaire_template.txt",
        }
    )

    assert issues == []
