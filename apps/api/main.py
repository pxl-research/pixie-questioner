"""FastAPI entrypoint for Pixie Questioner."""

from __future__ import annotations

from pixie_questioner.api_app import app as _app
from pixie_questioner.api_app import runtime
from pixie_questioner.api_app import settings as _settings

app = _app
settings = _settings

__all__ = [
    "app",
    "runtime",
    "settings",
]
