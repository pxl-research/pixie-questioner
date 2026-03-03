"""FastAPI app for Pixie Questioner."""

from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from pydantic import ValidationError

from pixie_questioner.audit_client import AuditClient
from pixie_questioner.azure_openai_llm import AzureOpenAILLM
from pixie_questioner.config import get_settings
from pixie_questioner.schemas import (
    AUDIT_EVENT_CONTRACT_VERSION,
    CHAT_CONTRACT_VERSION,
    AuditEventMessage,
    AuditEventMetadata,
    AuditEventPayload,
    ChatRecommendation,
    ChatRecommendationsResponse,
    ChatQuestionResponse,
    ChatRequest,
    ChatResponse,
)
from pixie_questioner.wine_advisor.catalog_loader import load_wine_catalog
from pixie_questioner.wine_advisor.llm_controller import LLMTurnController
from pixie_questioner.wine_advisor.llm_ranker import LLMWineRanker
from pixie_questioner.wine_advisor.service import WineAdvisorService
from pixie_questioner.wine_advisor.types import WineAdvisorResult

logger = logging.getLogger(__name__)

TURN_CONTROLLER_PROMPT = """
You are a wine-advice intake controller.
Decide if you need ONE next question or if you can recommend now.

Rules:
- Ask at most one question in this turn.
- If this is the first turn, always choose action "question".
- Question style for question_text:
  - Warm, approachable, and conversational.
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
)
catalog_records = load_wine_catalog(settings.wine_catalog_path)
turn_controller = LLMTurnController(
    llm=llm,
    system_prompt_template=TURN_CONTROLLER_PROMPT,
)
ranker = LLMWineRanker(
    llm=llm,
    system_prompt_template=RANKER_PROMPT,
)
wine_advisor_service = WineAdvisorService(
    catalog_records=catalog_records,
    turn_controller=turn_controller,
    ranker=ranker,
    debug_mode_allowed=settings.wine_advisor_debug_enabled,
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
) -> list[AuditEventMessage]:
    messages: list[AuditEventMessage] = []
    normalized_user = user_text.strip()
    normalized_assistant = assistant_text.strip()
    if normalized_user:
        messages.append(AuditEventMessage(role="user", content=normalized_user))
    if normalized_assistant:
        messages.append(
            AuditEventMessage(role="assistant", content=normalized_assistant)
        )
    return messages


def _extract_audit_location(state: dict[str, object]) -> str | None:
    location = state.get("location")
    if not isinstance(location, str):
        return None
    normalized = location.strip()
    if not normalized:
        return None
    return normalized


async def _post_audit_event(request: ChatRequest, result: WineAdvisorResult) -> None:
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
                result.text,
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


def _build_chat_response(
    request: ChatRequest, result: WineAdvisorResult
) -> ChatResponse:
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
            state=result.state,
        )

    return ChatQuestionResponse(
        type="question",
        text=result.text,
        conversation_id=request.session_id,
        state=result.state,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
async def chat(request: ChatRequest, response: Response):
    result = await wine_advisor_service.ask(
        current_answer=request.current_answer,
        state=request.state,
    )
    response.headers["X-Pixie-Chat-Contract"] = CHAT_CONTRACT_VERSION
    await _post_audit_event(request, result)
    return _build_chat_response(request, result)
