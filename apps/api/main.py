"""FastAPI app for Pixie Questioner."""

from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from pydantic import ValidationError

from pixie_questioner.audit_client import AuditClient
from pixie_questioner.azure_openai_llm import AzureOpenAILLM
from pixie_questioner.chat.relevance import LLMAnswerRelevanceClassifier
from pixie_questioner.chat_service import ChatService
from pixie_questioner.config import get_settings
from pixie_questioner.llm_client import CONVERSATION_SUMMARIZER_PROMPT, LLMPromptInvoker
from pixie_questioner.schemas import (
    AUDIT_EVENT_CONTRACT_VERSION,
    CHAT_CONTRACT_VERSION,
    AuditEventMessage,
    AuditEventMetadata,
    AuditEventPayload,
    ChatAnswerResponse,
    ChatRequest,
    ChatResponse,
    ChatSummaryResponse,
    QuickReplyOption,
)

logger = logging.getLogger(__name__)

settings = get_settings()
domain_profile = settings.domain_profile
llm = AzureOpenAILLM(
    api_version=settings.advanced_azure_openai_api_version,
    api_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_key,
    model_version=settings.advanced_azure_openai_model,
    temperature=settings.advanced_azure_openai_temperature,
)
notulist = LLMPromptInvoker(
    llm=llm,
    system_prompt_template=domain_profile.questionnaire_notulist_prompt,
    error_label="Questionnaire notulist",
)
summarizer = LLMPromptInvoker(
    llm=llm,
    system_prompt_template=CONVERSATION_SUMMARIZER_PROMPT,
    error_label="Conversation summarizer",
)
chat_service = ChatService(
    llm=llm,
    question_asker_prompt=domain_profile.question_asker_prompt,
    notulist=notulist,
    summarizer=summarizer,
    questionnaire_template=domain_profile.questionnaire_template,
    domain_policy=domain_profile.chat_domain_policy,
    answer_relevance_classifier=LLMAnswerRelevanceClassifier(
        llm=llm,
        system_prompt_template=domain_profile.relevance_classifier_prompt,
    ),
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
) -> list[AuditEventMessage]:
    messages: list[AuditEventMessage] = []
    normalized_user = user_text.strip()
    normalized_assistant = assistant_text.strip()
    normalized_summary = summary_text.strip() if summary_text is not None else None
    if normalized_user:
        messages.append(AuditEventMessage(role="user", content=normalized_user))
    if normalized_assistant:
        messages.append(
            AuditEventMessage(role="assistant", content=normalized_assistant)
        )
    if normalized_summary:
        messages.append(AuditEventMessage(role="summary", content=normalized_summary))
    return messages


def _extract_audit_location(state: dict[str, str]) -> str | None:
    location = state.get("location")
    if location is None:
        return None
    normalized = location.strip()
    if not normalized:
        return None
    return normalized


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
async def chat(request: ChatRequest, response: Response):
    result = await chat_service.ask(
        current_answer=request.current_answer,
        state=request.state,
    )
    response.headers["X-Pixie-Chat-Contract"] = CHAT_CONTRACT_VERSION
    if audit_client is not None:
        try:
            metadata: AuditEventMetadata | None = None
            audit_location = _extract_audit_location(request.state)
            if audit_location is not None:
                metadata = AuditEventMetadata(location=audit_location)

            payload = AuditEventPayload(
                conversation_id=request.session_id,
                profile=audit_client.profile,
                messages=_build_audit_messages(
                    request.current_answer,
                    result.text,
                    result.summary,
                ),
                metadata=metadata,
            )
            await audit_client.post_event(payload.model_dump(exclude_none=True))
        except ValidationError as exc:
            logger.warning(
                "Skipped audit event due to invalid payload (%s): %s",
                AUDIT_EVENT_CONTRACT_VERSION,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to post audit event (%s): %s",
                AUDIT_EVENT_CONTRACT_VERSION,
                exc,
            )
    if result.response_type == "summary":
        if result.summary is None:
            raise RuntimeError("Summary response requires a summary payload.")
        summary_response = ChatSummaryResponse(
            type="summary",
            text=result.text,
            summary=result.summary,
            follow_up_question=result.follow_up_question,
            conversation_id=request.session_id,
            state=result.state,
        )
        return summary_response
    answer_response = ChatAnswerResponse(
        type="answer",
        text=result.text,
        summary=result.summary,
        follow_up_question=result.follow_up_question,
        quick_replies=(
            [QuickReplyOption(**item) for item in result.quick_replies]
            if result.quick_replies is not None
            else None
        ),
        conversation_id=request.session_id,
        state=result.state,
    )
    return answer_response
