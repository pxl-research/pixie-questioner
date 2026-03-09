"""Runtime construction for the Pixie Questioner API."""

from __future__ import annotations

import logging
import secrets as _secrets

from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException

from .audit_client import AuditClient
from .azure_openai_llm import AzureOpenAILLM
from .chat.relevance import LLMAnswerRelevanceClassifier
from .chat_service import ChatService
from .config import ResolvedDomainProfile, Settings, get_settings
from .llm_client import CONVERSATION_SUMMARIZER_PROMPT, LLMPromptInvoker
from .rate_limiter import SlidingWindowRateLimiter
from .schemas import BROOKLYN_CHAT_CONTRACT_VERSION, CRETSKENS_CHAT_CONTRACT_VERSION
from .state_signer import StateSigner
from .wine_advisor.catalog_loader import load_wine_catalog
from .wine_advisor.llm_controller import LLMTurnController
from .wine_advisor.llm_ranker import LLMWineRanker
from .wine_advisor.service import WineAdvisorService

logger = logging.getLogger(__name__)

CASE_BROOKLYN = "brooklyn"
CASE_CRETSKENS = "cretskens"
CASE_CASE2 = "case2"
CaseId = Literal["brooklyn", "cretskens"]

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


@dataclass
class AppRuntime:
    settings: Settings
    brooklyn_chat_service: ChatService
    case2_wine_advisor_service: WineAdvisorService
    brooklyn_state_signer: StateSigner
    case2_state_signer: StateSigner
    audit_client: AuditClient | None
    ip_rate_limiter: SlidingWindowRateLimiter
    session_rate_limiter: SlidingWindowRateLimiter

    @property
    def wine_advisor_service(self) -> WineAdvisorService:
        return self.case2_wine_advisor_service

    def normalize_case_id(self, case_id: str) -> CaseId:
        normalized = case_id.strip().casefold()
        if normalized == CASE_BROOKLYN:
            return CASE_BROOKLYN
        if normalized in {CASE_CRETSKENS, CASE_CASE2}:
            return CASE_CRETSKENS
        raise HTTPException(status_code=404, detail="Unknown case.")

    def state_signer_for(self, case_id: CaseId) -> StateSigner:
        if case_id == CASE_BROOKLYN:
            return self.brooklyn_state_signer
        return self.case2_state_signer

    def case_contract_version_for(self, case_id: CaseId) -> str:
        if case_id == CASE_BROOKLYN:
            return BROOKLYN_CHAT_CONTRACT_VERSION
        return CRETSKENS_CHAT_CONTRACT_VERSION


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


def _build_case2_wine_advisor_service(
    *, llm_client: object, settings: Settings
) -> WineAdvisorService:
    catalog_records = load_wine_catalog(settings.case2_wine_catalog_path)
    turn_controller = LLMTurnController(
        llm=llm_client,
        system_prompt_template=TURN_CONTROLLER_PROMPT,
    )
    ranker = LLMWineRanker(
        llm=llm_client,
        system_prompt_template=RANKER_PROMPT,
    )
    return WineAdvisorService(
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


def _build_audit_client(settings: Settings) -> AuditClient | None:
    if not settings.audit_api_url:
        return None
    if not settings.audit_profile:
        raise RuntimeError(
            "AUDIT_PROFILE must be set when AUDIT_API_URL is configured."
        )
    return AuditClient(
        base_url=settings.audit_api_url,
        profile=settings.audit_profile,
        api_key=settings.audit_api_key,
        timeout_seconds=settings.audit_timeout_seconds,
    )


def _resolve_signing_key(settings: Settings) -> str:
    signing_key = settings.state_signing_key
    if signing_key:
        return signing_key
    ephemeral_key = _secrets.token_hex(32)
    logger.warning(
        "STATE_SIGNING_KEY not set — using ephemeral key. "
        "Client sessions will not survive server restarts."
    )
    return ephemeral_key


def build_runtime(settings: Settings | None = None) -> AppRuntime:
    resolved_settings = settings or get_settings()
    llm = AzureOpenAILLM(
        api_version=resolved_settings.advanced_azure_openai_api_version,
        api_endpoint=resolved_settings.azure_openai_endpoint,
        api_key=resolved_settings.azure_openai_key,
        model_version=resolved_settings.advanced_azure_openai_model,
        temperature=resolved_settings.advanced_azure_openai_temperature,
        request_timeout_seconds=resolved_settings.advanced_azure_openai_timeout_seconds,
        max_tokens=resolved_settings.advanced_azure_openai_max_tokens,
    )
    signing_key = _resolve_signing_key(resolved_settings)
    return AppRuntime(
        settings=resolved_settings,
        brooklyn_chat_service=_build_brooklyn_chat_service(
            llm_client=llm,
            profile=resolved_settings.brooklyn_domain_profile,
        ),
        case2_wine_advisor_service=_build_case2_wine_advisor_service(
            llm_client=llm,
            settings=resolved_settings,
        ),
        brooklyn_state_signer=StateSigner(
            secret_key=signing_key,
            ttl_seconds=resolved_settings.state_session_ttl_seconds,
            scope=CASE_BROOKLYN,
        ),
        case2_state_signer=StateSigner(
            secret_key=signing_key,
            ttl_seconds=resolved_settings.state_session_ttl_seconds,
            # Keep the legacy scope so existing signed state remains valid.
            scope=CASE_CASE2,
        ),
        audit_client=_build_audit_client(resolved_settings),
        ip_rate_limiter=SlidingWindowRateLimiter(max_requests=30, window_seconds=60),
        session_rate_limiter=SlidingWindowRateLimiter(
            max_requests=100, window_seconds=3600
        ),
    )
