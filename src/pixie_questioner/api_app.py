"""FastAPI app construction and request handling."""

from __future__ import annotations

import logging

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import ValidationError

from .app_runtime import AppRuntime, CASE_BROOKLYN, build_runtime
from .chat.types import ChatResult
from .cretskens_service import CretskensServiceUnavailableError
from .schemas import (
    AUDIT_EVENT_CONTRACT_VERSION,
    CHAT_CONTRACT_VERSION,
    AuditEventMessage,
    AuditEventMetadata,
    AuditEventPayload,
    CaseChatResponse,
    ChatAnswerResponse,
    ChatRecommendation,
    ChatRecommendationsResponse,
    ChatQuestionResponse,
    ChatRequest,
    ChatSummaryResponse,
    QuickReplyOption,
)
from .wine_advisor.types import WineAdvisorResult

logger = logging.getLogger(__name__)

runtime = build_runtime()
settings = runtime.settings

DEFAULT_RECOMMENDATION_TEXT = "Not specified"
CRETSKENS_TEMPORARILY_UNAVAILABLE = "Tijdelijk niet beschikbaar."


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


def _sanitize_recommendation_text(value: str) -> str:
    normalized = value.strip()
    if normalized:
        return normalized
    return DEFAULT_RECOMMENDATION_TEXT


def _extract_audit_location(state: dict[str, object]) -> str | None:
    location = state.get("location")
    if not isinstance(location, str):
        return None
    normalized = location.strip()
    if not normalized:
        return None
    return normalized


async def _post_audit_event(
    app_runtime: AppRuntime,
    request: ChatRequest,
    *,
    assistant_text: str,
    summary_text: str | None = None,
) -> None:
    if app_runtime.audit_client is None:
        return
    try:
        metadata: AuditEventMetadata | None = None
        audit_location = _extract_audit_location(request.state)
        if audit_location is not None:
            metadata = AuditEventMetadata(location=audit_location)

        payload = AuditEventPayload(
            conversation_id=request.session_id,
            profile=app_runtime.audit_client.profile,
            messages=_build_audit_messages(
                request.current_answer,
                assistant_text,
                summary_text,
            ),
            metadata=metadata,
        )
        await app_runtime.audit_client.post_event(payload.model_dump(exclude_none=True))
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


def _build_cretskens_chat_response(
    request: ChatRequest,
    result: WineAdvisorResult,
    *,
    signed_state: dict[str, Any],
) -> ChatQuestionResponse | ChatRecommendationsResponse:
    if result.response_type == "recommendations":
        recommendations = result.recommendations or []
        return ChatRecommendationsResponse(
            type="recommendations",
            text=result.text,
            recommendations=[
                ChatRecommendation(
                    name=item.name,
                    country=item.country,
                    region=item.region,
                    aroma_description=_sanitize_recommendation_text(
                        item.aroma_description
                    ),
                    taste_description=_sanitize_recommendation_text(
                        item.taste_description
                    ),
                )
                for item in recommendations
            ],
            conversation_id=request.session_id,
            state=signed_state,
        )

    return ChatQuestionResponse(
        type="question",
        text=result.text,
        conversation_id=request.session_id,
        state=signed_state,
    )


def _build_cretskens_unavailable_response(
    request: ChatRequest,
    *,
    signed_state: dict[str, Any],
) -> ChatQuestionResponse:
    return ChatQuestionResponse(
        type="question",
        text=CRETSKENS_TEMPORARILY_UNAVAILABLE,
        conversation_id=request.session_id,
        state=signed_state,
    )


def _build_brooklyn_chat_response(
    request: ChatRequest,
    result: ChatResult,
    *,
    signed_state: dict[str, Any],
) -> ChatAnswerResponse | ChatSummaryResponse:
    if result.response_type == "summary":
        if result.summary is None:
            raise RuntimeError("Summary response requires a summary payload.")
        return ChatSummaryResponse(
            type="summary",
            text=result.text,
            summary=result.summary,
            follow_up_question=result.follow_up_question,
            conversation_id=request.session_id,
            state=signed_state,
        )

    return ChatAnswerResponse(
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
        state=signed_state,
    )


def _get_client_ip(raw_request: Request, *, trusted_proxy_ips: frozenset[str]) -> str:
    immediate_client_ip = raw_request.client.host if raw_request.client else "unknown"
    if immediate_client_ip.casefold() not in trusted_proxy_ips:
        return immediate_client_ip

    forwarded_for = raw_request.headers.get("x-forwarded-for", "")
    # Walk right-to-left through the XFF list and return the first entry that
    # is not itself a trusted proxy.  Appending proxies write the real client
    # IP as the rightmost entry, but in multi-proxy setups (e.g. CDN + ingress)
    # several rightmost entries may be trusted.  Skipping them avoids collapsing
    # many users into one rate-limit bucket.  Entries to the left of the first
    # untrusted hop are client-supplied and cannot be trusted.
    for candidate in reversed(
        [c.strip() for c in forwarded_for.split(",") if c.strip()]
    ):
        if candidate.casefold() not in trusted_proxy_ips:
            return candidate
    return immediate_client_ip


async def _handle_chat_request(
    *,
    app_runtime: AppRuntime,
    case_id: str,
    request: ChatRequest,
    response: Response,
    raw_request: Request,
) -> CaseChatResponse:
    normalized_case_id = app_runtime.normalize_case_id(case_id)
    client_ip = _get_client_ip(
        raw_request,
        trusted_proxy_ips=app_runtime.settings.trusted_proxy_ips,
    )
    if not app_runtime.ip_rate_limiter.is_allowed(client_ip):
        logger.warning("Rate limit exceeded for IP %s", client_ip)
        raise HTTPException(status_code=429, detail="Too many requests.")
    if not app_runtime.session_rate_limiter.is_allowed(request.session_id):
        logger.warning("Rate limit exceeded for session %s", request.session_id)
        raise HTTPException(status_code=429, detail="Session request limit exceeded.")

    signer = app_runtime.state_signer_for(normalized_case_id)
    try:
        verified_state = signer.verify_and_strip(request.state)
    except ValueError as exc:
        logger.warning(
            "State verification failed for case %s, session %s: %s",
            normalized_case_id,
            request.session_id,
            exc,
        )
        raise HTTPException(
            status_code=400, detail="Invalid or expired state."
        ) from exc

    response.headers["X-Pixie-Chat-Contract"] = CHAT_CONTRACT_VERSION
    response.headers["X-Pixie-Case"] = normalized_case_id
    response.headers["X-Pixie-Case-Contract"] = app_runtime.case_contract_version_for(
        normalized_case_id
    )

    if normalized_case_id == CASE_BROOKLYN:
        result = await app_runtime.brooklyn_chat_service.ask(
            current_answer=request.current_answer,
            state=verified_state,
        )
        signed_state = signer.sign(result.state)
        await _post_audit_event(
            app_runtime,
            request,
            assistant_text=result.text,
            summary_text=result.summary,
        )
        return _build_brooklyn_chat_response(
            request,
            result,
            signed_state=signed_state,
        )

    try:
        result = await app_runtime.cretskens_service.ask(
            current_answer=request.current_answer,
            state=verified_state,
        )
    except CretskensServiceUnavailableError as exc:
        logger.warning(
            "Cretskens service unavailable for session %s: %s",
            request.session_id,
            exc,
            exc_info=True,
        )
        return _build_cretskens_unavailable_response(
            request,
            signed_state=signer.sign(verified_state),
        )
    signed_state = signer.sign(result.state)
    await _post_audit_event(
        app_runtime,
        request,
        assistant_text=result.text,
    )
    return _build_cretskens_chat_response(
        request,
        result,
        signed_state=signed_state,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if runtime.audit_client is not None:
        await runtime.audit_client.aclose()


app = FastAPI(title="Pixie Questioner", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/cases/{case_id}/chat",
    response_model=CaseChatResponse,
    response_model_exclude_none=True,
)
async def chat_for_case(
    case_id: str,
    request: ChatRequest,
    response: Response,
    raw_request: Request,
):
    return await _handle_chat_request(
        app_runtime=runtime,
        case_id=case_id,
        request=request,
        response=response,
        raw_request=raw_request,
    )
