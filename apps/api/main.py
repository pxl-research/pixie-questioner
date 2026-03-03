"""FastAPI app for Pixie Questioner."""

from __future__ import annotations

import logging
import secrets as _secrets

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import ValidationError

from pixie_questioner.audit_client import AuditClient
from pixie_questioner.azure_openai_llm import AzureOpenAILLM
from pixie_questioner.chat.relevance import LLMAnswerRelevanceClassifier
from pixie_questioner.chat.types import ChatResult
from pixie_questioner.chat_service import ChatService
from pixie_questioner.config import ResolvedDomainProfile, get_settings
from pixie_questioner.llm_client import CONVERSATION_SUMMARIZER_PROMPT, LLMPromptInvoker
from pixie_questioner.rate_limiter import SlidingWindowRateLimiter
from pixie_questioner.schemas import (
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
from pixie_questioner.state_signer import StateSigner
from pixie_questioner.wine_advisor.catalog_loader import load_wine_catalog
from pixie_questioner.wine_advisor.llm_controller import LLMTurnController
from pixie_questioner.wine_advisor.llm_ranker import LLMWineRanker
from pixie_questioner.wine_advisor.service import WineAdvisorService
from pixie_questioner.wine_advisor.types import WineAdvisorResult

logger = logging.getLogger(__name__)

CASE_BROOKLYN = "brooklyn"
CASE_CASE2 = "case2"
CaseId = Literal["brooklyn", "case2"]

TURN_CONTROLLER_PROMPT = """
You are a wine-advice intake controller.
Decide if you need ONE next question or if you can recommend now.

Rules:
- Ask at most one question in this turn.
- If this is the first turn, always choose action "question".
- Question style for question_text:
  - Sound like a friendly in-store wine advisor: warm, upbeat, and human.
  - Use upbeat openers only occasionally; they are optional, not required.
  - Vary how questions start across turns and avoid repeating the same opener.
  - Keep it concise and natural, not robotic.
  - Mirror the customer's language (NL/EN/FR) used in recent messages.
  - Ask for one preference at a time in plain everyday wording.
- Do not recommend until wine preference information is available:
  color OR budget OR preferred style/aroma/taste OR origin (country/region),
  unless the customer explicitly states no preference.
- Preference loop before recommendation:
  1) Proactively ask whether the customer has preferences for
     budget, country/region of origin, grape variety, and style.
  2) If the customer gives one preference, continue with the next
     highest-value missing preference dimension.
  3) Keep asking while the customer still has additional preferences.
  4) Recommend only after the customer says there are no additional
     preferences, or clearly asks to proceed with current constraints.
- If meal_mode is "drinking_only", ask only wine preference questions
  (never dish/taste-of-dish questions).
- If meal_mode is "pairing", first clarify dish context only when still unclear,
  then ask the highest-value missing wine preference question.
- Recommend only when the context is specific enough for a confident shortlist.
- If unsure between question vs recommend, choose action "question".
- Return only JSON with keys:
  action: "question" or "recommend"
  question_text: string (required when action is question)
  meal_mode: "pairing" | "drinking_only" | "unknown"
  query_hints: concise search hints
  decision_explanation: short factual reason for this action (no hidden reasoning)

Current meal mode guess:
{current_meal_mode}

Last assistant question:
{last_question}

Recent conversation history:
{conversation_history}

Working context:
{working_context}

Latest user answer:
{current_answer}

Turn:
{turn_count} / {max_turns}
""".strip()

RANKER_PROMPT = """
Select the best candidate barcodes for this customer.
Return only JSON with keys:
- barcodes: ordered array
- selection_explanation: short factual reason for this shortlist
Use only candidate barcodes from the catalog rows below.

Working context:
{working_context}

Candidate catalog rows (one JSON object per line):
{candidate_catalog_rows}

Candidate barcodes:
{candidate_barcodes}

Target count between {min_count} and {max_count}.

Retry reason (if any):
{retry_reason}
""".strip()

settings = get_settings()
llm = AzureOpenAILLM(
    api_version=settings.advanced_azure_openai_api_version,
    api_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_key,
    model_version=settings.advanced_azure_openai_model,
    temperature=settings.advanced_azure_openai_temperature,
    request_timeout_seconds=settings.advanced_azure_openai_timeout_seconds,
    max_tokens=settings.advanced_azure_openai_max_tokens,
)


def _build_optional_prompt_invoker(
    *,
    llm_client: object,
    prompt_template: str,
    error_label: str,
) -> LLMPromptInvoker | None:
    if not prompt_template.strip():
        return None
    return LLMPromptInvoker(
        llm=llm_client,
        system_prompt_template=prompt_template,
        error_label=error_label,
    )


def _build_brooklyn_chat_service(
    *,
    llm_client: object,
    profile: ResolvedDomainProfile,
) -> ChatService:
    notulist = _build_optional_prompt_invoker(
        llm_client=llm_client,
        prompt_template=profile.questionnaire_notulist_prompt,
        error_label="Questionnaire notulist",
    )
    if notulist is None:
        raise RuntimeError(
            "Brooklyn profile must define questionnaire_notulist prompt."
        )

    summarizer = LLMPromptInvoker(
        llm=llm_client,
        system_prompt_template=CONVERSATION_SUMMARIZER_PROMPT,
        error_label="Conversation summarizer",
    )

    relevance_classifier: LLMAnswerRelevanceClassifier | None = None
    if profile.relevance_classifier_prompt.strip():
        relevance_classifier = LLMAnswerRelevanceClassifier(
            llm=llm_client,
            system_prompt_template=profile.relevance_classifier_prompt,
        )

    return ChatService(
        llm=llm_client,
        question_asker_prompt=profile.question_asker_prompt,
        notulist=notulist,
        summarizer=summarizer,
        questionnaire_template=profile.questionnaire_template,
        domain_policy=profile.chat_domain_policy,
        answer_relevance_classifier=relevance_classifier,
        aroma_taste_classifier=_build_optional_prompt_invoker(
            llm_client=llm_client,
            prompt_template=profile.aroma_taste_classifier_prompt,
            error_label="Aroma taste classifier",
        ),
        handoff_classifier=_build_optional_prompt_invoker(
            llm_client=llm_client,
            prompt_template=profile.handoff_classifier_prompt,
            error_label="Handoff classifier",
        ),
    )


brooklyn_chat_service = _build_brooklyn_chat_service(
    llm_client=llm,
    profile=settings.brooklyn_domain_profile,
)

catalog_records = load_wine_catalog(settings.case2_wine_catalog_path)
turn_controller = LLMTurnController(
    llm=llm,
    system_prompt_template=TURN_CONTROLLER_PROMPT,
)
ranker = LLMWineRanker(
    llm=llm,
    system_prompt_template=RANKER_PROMPT,
)
case2_wine_advisor_service = WineAdvisorService(
    catalog_records=catalog_records,
    turn_controller=turn_controller,
    ranker=ranker,
    candidate_limit=settings.wine_advisor_candidate_limit,
    max_input_chars=settings.wine_advisor_max_input_chars,
    max_working_context_chars=settings.wine_advisor_max_working_context_chars,
    max_history_items=settings.wine_advisor_max_history_items,
    max_history_item_chars=settings.wine_advisor_max_history_item_chars,
    debug_mode_allowed=settings.wine_advisor_debug_enabled,
)

# Backward-compatible alias for existing tests and local scripts.
wine_advisor_service = case2_wine_advisor_service

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

_signing_key = settings.state_signing_key
if not _signing_key:
    _signing_key = _secrets.token_hex(32)
    logger.warning(
        "STATE_SIGNING_KEY not set — using ephemeral key. "
        "Client sessions will not survive server restarts."
    )
brooklyn_state_signer = StateSigner(
    secret_key=_signing_key,
    ttl_seconds=settings.state_session_ttl_seconds,
    scope=CASE_BROOKLYN,
)
case2_state_signer = StateSigner(
    secret_key=_signing_key,
    ttl_seconds=settings.state_session_ttl_seconds,
    scope=CASE_CASE2,
)
ip_rate_limiter = SlidingWindowRateLimiter(max_requests=30, window_seconds=60)
session_rate_limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=3600)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    if audit_client is not None:
        await audit_client.aclose()


app = FastAPI(title="Pixie Questioner", lifespan=lifespan)


def _normalize_case_id(case_id: str) -> CaseId:
    normalized = case_id.strip().casefold()
    if normalized == CASE_BROOKLYN:
        return CASE_BROOKLYN
    if normalized == CASE_CASE2:
        return CASE_CASE2
    raise HTTPException(status_code=404, detail="Unknown case.")


def _get_state_signer(case_id: CaseId) -> StateSigner:
    if case_id == CASE_BROOKLYN:
        return brooklyn_state_signer
    return case2_state_signer


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


def _extract_audit_location(state: dict[str, object]) -> str | None:
    location = state.get("location")
    if not isinstance(location, str):
        return None
    normalized = location.strip()
    if not normalized:
        return None
    return normalized


async def _post_audit_event(
    request: ChatRequest,
    *,
    assistant_text: str,
    summary_text: str | None = None,
) -> None:
    if audit_client is None:
        return
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
                assistant_text,
                summary_text,
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


def _build_case2_chat_response(
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
                    price_eur=item.price_eur,
                    country=item.country,
                    region=item.region,
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


def _get_client_ip(raw_request: Request) -> str:
    forwarded = raw_request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if raw_request.client:
        return raw_request.client.host
    return "unknown"


async def _handle_chat_request(
    *,
    case_id: CaseId,
    request: ChatRequest,
    response: Response,
    raw_request: Request,
) -> CaseChatResponse:
    client_ip = _get_client_ip(raw_request)
    if not ip_rate_limiter.is_allowed(client_ip):
        logger.warning("Rate limit exceeded for IP %s", client_ip)
        raise HTTPException(status_code=429, detail="Too many requests.")
    if not session_rate_limiter.is_allowed(request.session_id):
        logger.warning("Rate limit exceeded for session %s", request.session_id)
        raise HTTPException(status_code=429, detail="Session request limit exceeded.")

    signer = _get_state_signer(case_id)
    try:
        verified_state = signer.verify_and_strip(request.state)
    except ValueError as exc:
        logger.warning(
            "State verification failed for case %s, session %s: %s",
            case_id,
            request.session_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response.headers["X-Pixie-Chat-Contract"] = CHAT_CONTRACT_VERSION
    response.headers["X-Pixie-Case"] = case_id

    if case_id == CASE_BROOKLYN:
        result = await brooklyn_chat_service.ask(
            current_answer=request.current_answer,
            state=verified_state,
        )
        signed_state = signer.sign(result.state)
        await _post_audit_event(
            request,
            assistant_text=result.text,
            summary_text=result.summary,
        )
        return _build_brooklyn_chat_response(
            request,
            result,
            signed_state=signed_state,
        )

    result = await case2_wine_advisor_service.ask(
        current_answer=request.current_answer,
        state=verified_state,
    )
    signed_state = signer.sign(result.state)
    await _post_audit_event(
        request,
        assistant_text=result.text,
    )
    return _build_case2_chat_response(
        request,
        result,
        signed_state=signed_state,
    )


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
    normalized_case_id = _normalize_case_id(case_id)
    return await _handle_chat_request(
        case_id=normalized_case_id,
        request=request,
        response=response,
        raw_request=raw_request,
    )


@app.post(
    "/chat",
    response_model=CaseChatResponse,
    response_model_exclude_none=True,
)
async def chat(request: ChatRequest, response: Response, raw_request: Request):
    default_case = _normalize_case_id(settings.default_case)
    return await _handle_chat_request(
        case_id=default_case,
        request=request,
        response=response,
        raw_request=raw_request,
    )
