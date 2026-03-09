"""FastAPI entrypoint for Pixie Questioner."""

from __future__ import annotations

from pixie_questioner.api_app import app as _app
from pixie_questioner.api_app import runtime
from pixie_questioner.api_app import settings as _settings
from pixie_questioner.app_runtime import (
    CASE_BROOKLYN as _CASE_BROOKLYN,
    CASE_CRETSKENS as _CASE_CRETSKENS,
)

app = _app
settings = _settings
CASE_BROOKLYN = _CASE_BROOKLYN
CASE_CRETSKENS = _CASE_CRETSKENS
brooklyn_chat_service = runtime.brooklyn_chat_service
audit_client = runtime.audit_client
brooklyn_state_signer = runtime.brooklyn_state_signer
cretskens_state_signer = runtime.cretskens_state_signer

__all__ = [
    "app",
    "runtime",
    "settings",
    "CASE_BROOKLYN",
    "CASE_CRETSKENS",
    "brooklyn_chat_service",
    "audit_client",
    "brooklyn_state_signer",
    "cretskens_state_signer",
]
