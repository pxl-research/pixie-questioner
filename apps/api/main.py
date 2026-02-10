"""FastAPI app for Pixie Questioner."""

from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI

from pixie_questioner.audit_client import AuditClient
from pixie_questioner.azure_openai_llm import AzureOpenAILLM
from pixie_questioner.chat_service import ChatService
from pixie_questioner.config import get_settings
from pixie_questioner.notulist import QuestionnaireNotulist
from pixie_questioner.schemas import (
    ChatAnswerResponse,
    ChatRequest,
    ChatResponse,
    ChatSummaryResponse,
)
from pixie_questioner.summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)

settings = get_settings()
llm = AzureOpenAILLM(
    api_version=settings.advanced_azure_openai_api_version,
    api_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_key,
    model_version=settings.advanced_azure_openai_model,
    temperature=settings.advanced_azure_openai_temperature,
)
summarizer_prompt = settings.summarizer_prompt
if summarizer_prompt:
    summarizer = ConversationSummarizer(
        llm=llm,
        system_prompt_template=summarizer_prompt,
    )
else:
    summarizer = ConversationSummarizer(llm=llm)
notulist_prompt = settings.questionnaire_notulist_prompt
if notulist_prompt is None:
    raise RuntimeError("QUESTIONNAIRE_NOTULIST_PROMPT_PATH must be set.")
notulist = QuestionnaireNotulist(
    llm=llm,
    system_prompt_template=notulist_prompt,
)
chat_service = ChatService(
    llm=llm,
    question_asker_prompt=settings.system_prompt,
    notulist=notulist,
    summarizer=summarizer,
    questionnaire_template=settings.questionnaire_template,
    completion_token=settings.completion_token,
)
audit_client: AuditClient | None = None
if settings.audit_api_url:
    if not settings.audit_profile:
        raise RuntimeError(
            "AUDIT_PROFILE must be set when AUDIT_API_URL is configured."
        )
    audit_client = AuditClient(
        base_url=settings.audit_api_url,
        profile=settings.audit_profile,
        api_key=settings.audit_api_key,
        timeout_seconds=settings.audit_timeout_seconds,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if audit_client is not None:
        await audit_client.aclose()


app = FastAPI(title="Pixie Questioner", lifespan=lifespan)


def _build_audit_messages(
    user_text: str,
    assistant_text: str,
    summary_text: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if user_text.strip():
        messages.append({"role": "user", "content": user_text})
    if assistant_text.strip():
        messages.append({"role": "assistant", "content": assistant_text})
    if summary_text is not None and summary_text.strip():
        messages.append({"role": "summary", "content": summary_text})
    return messages


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):
    result = await chat_service.ask(
        current_answer=request.current_answer,
        state=request.state,
    )
    if audit_client is not None:
        payload: dict[str, object] = {
            "conversation_id": request.session_id,
            "profile": audit_client.profile,
            "messages": _build_audit_messages(
                request.current_answer,
                result.text,
                result.summary,
            ),
        }
        if result.response_type == "summary" and result.summary is not None:
            payload["summary"] = result.summary
        try:
            await audit_client.post_event(payload)
        except Exception as exc:
            logger.warning("Failed to post audit event: %s", exc)
    if result.response_type == "summary":
        if result.summary is None:
            raise RuntimeError("Summary response requires a summary payload.")
        return ChatSummaryResponse(
            type="summary",
            text=result.text,
            summary=result.summary,
            follow_up_question=result.follow_up_question,
            conversation_id=request.session_id,
            state=result.state,
        )
    return ChatAnswerResponse(
        type="answer",
        text=result.text,
        conversation_id=request.session_id,
        state=result.state,
    )
