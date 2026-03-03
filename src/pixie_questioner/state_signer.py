"""HMAC-based state signing with TTL enforcement."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

# Keys the client is allowed to add or modify without invalidating the signature.
# These are excluded from MAC computation on both sign and verify.
_CLIENT_ADDABLE_KEYS: frozenset[str] = frozenset({"location"})

# Keys used only as control fields.
_CONTROL_KEYS: frozenset[str] = frozenset({"_sig", "_ts", "_scope"})

# A state containing only these keys requires no signature (initial request).
_UNPROTECTED_STATE_KEYS: frozenset[str] = _CLIENT_ADDABLE_KEYS | _CONTROL_KEYS

# Keys excluded from MAC computation on both sides.
_MAC_EXCLUDED_KEYS: frozenset[str] = _CLIENT_ADDABLE_KEYS | _CONTROL_KEYS


@dataclass
class StateSigner:
    """Sign and verify opaque state dicts using HMAC-SHA256."""

    secret_key: str
    ttl_seconds: int = 1800
    scope: str = ""

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
        """Return a copy of *state* with ``_ts`` and ``_sig`` appended.

        ``location`` and other client-addable keys are preserved in the
        returned dict but excluded from MAC computation so clients can
        freely attach them on subsequent requests without breaking the signature.
        """
        protected = {k: v for k, v in state.items() if k not in _MAC_EXCLUDED_KEYS}
        ts = int(time.time())
        mac_payload = {**protected, "_ts": ts}
        if self.scope:
            mac_payload["_scope"] = self.scope
        sig = self._compute_mac(mac_payload)
        result = {k: v for k, v in state.items() if k not in _CONTROL_KEYS}
        result["_ts"] = ts
        if self.scope:
            result["_scope"] = self.scope
        result["_sig"] = sig
        return result

    def verify_and_strip(self, state: dict[str, Any]) -> dict[str, Any]:
        """Verify signature and TTL, returning state without ``_sig``/``_ts``.

        Empty or unprotected-only states are passed through (initial request).
        Raises ``ValueError`` on tampered, expired, or missing signatures.
        """
        if not state or all(k in _UNPROTECTED_STATE_KEYS for k in state):
            self._verify_scope(state)
            return {k: v for k, v in state.items() if k not in _CONTROL_KEYS}

        sig = state.get("_sig")
        if not isinstance(sig, str):
            raise ValueError("State signature missing.")

        ts = state.get("_ts")
        if not isinstance(ts, (int, float)):
            raise ValueError("State timestamp missing.")

        self._verify_scope(state)
        data = {k: v for k, v in state.items() if k not in _MAC_EXCLUDED_KEYS}
        data["_ts"] = ts
        if self.scope:
            data["_scope"] = self.scope
        expected = self._compute_mac(data)
        if not hmac.compare_digest(sig, expected):
            raise ValueError("State signature invalid.")

        age = time.time() - float(ts)
        if age > self.ttl_seconds:
            raise ValueError("Session expired.")
        if age < -60:
            raise ValueError("State timestamp is in the future.")

        return {k: v for k, v in state.items() if k not in _CONTROL_KEYS}

    def _verify_scope(self, state: dict[str, Any]) -> None:
        if not self.scope:
            return
        scope = state.get("_scope")
        if scope is None:
            return
        if not isinstance(scope, str) or scope != self.scope:
            raise ValueError("State scope invalid.")
