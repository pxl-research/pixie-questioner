"""Audit API client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class AuditClient:
    base_url: str
    profile: str
    api_key: str | None = None
    timeout_seconds: float = 5.0
    _client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client

    async def post_event(self, payload: dict[str, Any]) -> None:
        url = f"{self.base_url.rstrip('/')}/events"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        response = await self._get_client().post(url, json=payload, headers=headers)
        response.raise_for_status()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
