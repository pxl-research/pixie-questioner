"""HMAC-based state signing with TTL enforcement."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

_UNPROTECTED_STATE_KEYS: frozenset[str] = frozenset({"location", "_sig", "_ts"})


@dataclass
class StateSigner:
    """Sign and verify opaque state dicts using HMAC-SHA256."""

    secret_key: str
    ttl_seconds: int = 1800

    def __post_init__(self) -> None:
        if not self.secret_key:
            raise ValueError("secret_key must not be empty.")
        self._secret = self.secret_key.encode("utf-8")

    def _compute_mac(self, data: dict[str, Any]) -> str:
        canonical = json.dumps(
            data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hmac.new(
            self._secret, canonical.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def sign(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of *state* with ``_ts`` and ``_sig`` appended."""
        payload = {k: v for k, v in state.items() if k not in ("_sig", "_ts")}
        payload["_ts"] = int(time.time())
        payload["_sig"] = self._compute_mac(payload)
        return payload

    def verify_and_strip(self, state: dict[str, Any]) -> dict[str, Any]:
        """Verify signature and TTL, returning state without ``_sig``/``_ts``.

        Empty or unprotected-only states are passed through (initial request).
        Raises ``ValueError`` on tampered, expired, or missing signatures.
        """
        if not state or all(k in _UNPROTECTED_STATE_KEYS for k in state):
            return {k: v for k, v in state.items() if k not in ("_sig", "_ts")}

        sig = state.get("_sig")
        if not isinstance(sig, str):
            raise ValueError("State signature missing.")

        ts = state.get("_ts")
        if not isinstance(ts, (int, float)):
            raise ValueError("State timestamp missing.")

        data = {k: v for k, v in state.items() if k != "_sig"}
        expected = self._compute_mac(data)
        if not hmac.compare_digest(sig, expected):
            raise ValueError("State signature invalid.")

        age = time.time() - float(ts)
        if age > self.ttl_seconds:
            raise ValueError("Session expired.")
        if age < -60:
            raise ValueError("State timestamp is in the future.")

        return {k: v for k, v in state.items() if k not in ("_sig", "_ts")}
