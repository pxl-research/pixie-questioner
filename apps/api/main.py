"""FastAPI entrypoint for Pixie Questioner."""

from __future__ import annotations

from pixie_questioner.api_app import app as _app
from pixie_questioner.api_app import runtime
from pixie_questioner.api_app import settings as _settings
from pixie_questioner.app_runtime import (
    CASE_BROOKLYN as _CASE_BROOKLYN,
    CASE_CRETSKENS as _CASE_CRETSKENS,
    CASE_CASE2 as _CASE_CASE2,
    RANKER_PROMPT as _RANKER_PROMPT,
    TURN_CONTROLLER_PROMPT as _TURN_CONTROLLER_PROMPT,
)

app = _app
settings = _settings
CASE_BROOKLYN = _CASE_BROOKLYN
CASE_CRETSKENS = _CASE_CRETSKENS
CASE_CASE2 = _CASE_CASE2
TURN_CONTROLLER_PROMPT = _TURN_CONTROLLER_PROMPT
RANKER_PROMPT = _RANKER_PROMPT
brooklyn_chat_service = runtime.brooklyn_chat_service
case2_wine_advisor_service = runtime.case2_wine_advisor_service
wine_advisor_service = runtime.wine_advisor_service
audit_client = runtime.audit_client
brooklyn_state_signer = runtime.brooklyn_state_signer
case2_state_signer = runtime.case2_state_signer

__all__ = [
    "app",
    "runtime",
    "settings",
    "CASE_BROOKLYN",
    "CASE_CRETSKENS",
    "CASE_CASE2",
    "TURN_CONTROLLER_PROMPT",
    "RANKER_PROMPT",
    "brooklyn_chat_service",
    "case2_wine_advisor_service",
    "wine_advisor_service",
    "audit_client",
    "brooklyn_state_signer",
    "case2_state_signer",
]
