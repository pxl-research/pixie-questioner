"""Runtime construction for the Pixie Questioner API."""

from __future__ import annotations

import logging
import secrets as _secrets

from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException

from .audit_client import AuditClient
from .azure_openai_llm import AzureOpenAILLM
from .chat.intake_route_classifier import LLMIntakeRouteClassifier
from .chat.relevance import LLMAnswerRelevanceClassifier
from .chat_service import ChatService
from .config import ResolvedDomainProfile, Settings, get_settings
from .cretskens_service import CretskensService
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
CaseId = Literal["brooklyn", "cretskens"]


@dataclass
class AppRuntime:
    settings: Settings
    brooklyn_chat_service: ChatService
    cretskens_chat_service: ChatService
    cretskens_service: CretskensService
    cretskens_wine_advisor_service: WineAdvisorService
    brooklyn_state_signer: StateSigner
    cretskens_state_signer: StateSigner
    audit_client: AuditClient | None
    ip_rate_limiter: SlidingWindowRateLimiter
    session_rate_limiter: SlidingWindowRateLimiter

    def normalize_case_id(self, case_id: str) -> CaseId:
        normalized = case_id.strip().casefold()
        if normalized == CASE_BROOKLYN:
            return CASE_BROOKLYN
        if normalized == CASE_CRETSKENS:
            return CASE_CRETSKENS
        raise HTTPException(status_code=404, detail="Unknown case.")

    def state_signer_for(self, case_id: CaseId) -> StateSigner:
        if case_id == CASE_BROOKLYN:
            return self.brooklyn_state_signer
        return self.cretskens_state_signer

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


def _build_chat_service(
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
        raise RuntimeError("Domain profile must define questionnaire_notulist prompt.")

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
    intake_route_classifier: LLMIntakeRouteClassifier | None = None
    if profile.intake_route_classifier_prompt.strip():
        intake_route_classifier = LLMIntakeRouteClassifier(
            llm=llm_client,
            system_prompt_template=profile.intake_route_classifier_prompt,
        )

    return ChatService(
        llm=llm_client,
        question_asker_prompt=profile.question_asker_prompt,
        notulist=notulist,
        summarizer=summarizer,
        questionnaire_template=profile.questionnaire_template,
        domain_policy=profile.chat_domain_policy,
        answer_relevance_classifier=relevance_classifier,
        intake_route_classifier=intake_route_classifier,
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
        unsupported_origin_follow_up=_build_optional_prompt_invoker(
            llm_client=llm_client,
            prompt_template=profile.unsupported_origin_follow_up_prompt,
            error_label="Unsupported origin follow-up",
        ),
    )


def _build_cretskens_wine_advisor_service(
    *, llm_client: object, settings: Settings
) -> WineAdvisorService:
    profile = settings.cretskens_profile
    catalog_records = load_wine_catalog(settings.cretskens_wine_catalog_path)
    turn_controller = LLMTurnController(
        llm=llm_client,
        system_prompt_template=profile.turn_controller_prompt,
    )
    ranker = LLMWineRanker(
        llm=llm_client,
        system_prompt_template=profile.ranker_prompt,
    )
    return WineAdvisorService(
        catalog_records=catalog_records,
        turn_controller=turn_controller,
        ranker=ranker,
        recommendation_intro=profile.recommendation_intro,
        candidate_limit=settings.wine_advisor_candidate_limit,
        max_input_chars=settings.wine_advisor_max_input_chars,
        max_working_context_chars=settings.wine_advisor_max_working_context_chars,
        max_history_items=settings.wine_advisor_max_history_items,
        max_history_item_chars=settings.wine_advisor_max_history_item_chars,
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
    brooklyn_chat_service = _build_chat_service(
        llm_client=llm,
        profile=resolved_settings.brooklyn_domain_profile,
    )
    cretskens_chat_service = _build_chat_service(
        llm_client=llm,
        profile=resolved_settings.cretskens_domain_profile,
    )
    cretskens_wine_advisor_service = _build_cretskens_wine_advisor_service(
        llm_client=llm,
        settings=resolved_settings,
    )
    return AppRuntime(
        settings=resolved_settings,
        brooklyn_chat_service=brooklyn_chat_service,
        cretskens_chat_service=cretskens_chat_service,
        cretskens_service=CretskensService(
            intake_service=cretskens_chat_service,
            recommendation_service=cretskens_wine_advisor_service,
        ),
        cretskens_wine_advisor_service=cretskens_wine_advisor_service,
        brooklyn_state_signer=StateSigner(
            secret_key=signing_key,
            ttl_seconds=resolved_settings.state_session_ttl_seconds,
            scope=CASE_BROOKLYN,
        ),
        cretskens_state_signer=StateSigner(
            secret_key=signing_key,
            ttl_seconds=resolved_settings.state_session_ttl_seconds,
            scope=CASE_CRETSKENS,
        ),
        audit_client=_build_audit_client(resolved_settings),
        ip_rate_limiter=SlidingWindowRateLimiter(max_requests=30, window_seconds=60),
        session_rate_limiter=SlidingWindowRateLimiter(
            max_requests=100, window_seconds=3600
        ),
    )
