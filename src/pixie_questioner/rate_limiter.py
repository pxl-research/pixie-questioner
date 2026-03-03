"""In-memory sliding-window rate limiter."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class SlidingWindowRateLimiter:
    """Reject requests once a key exceeds *max_requests* in *window_seconds*."""

    max_requests: int
    window_seconds: int

    def __post_init__(self) -> None:
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._call_count: int = 0

    def is_allowed(self, key: str) -> bool:
        """Return True if the request is within the rate limit for *key*."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            self._call_count += 1
            timestamps = self._buckets[key]
            pruned = [t for t in timestamps if t > cutoff]
            if len(pruned) >= self.max_requests:
                self._buckets[key] = pruned
                return False
            pruned.append(now)
            self._buckets[key] = pruned
            if self._call_count % 500 == 0:
                self._cleanup(cutoff)
            return True

    def _cleanup(self, cutoff: float) -> None:
        """Remove keys whose timestamps are all expired."""
        stale = [
            key
            for key, timestamps in self._buckets.items()
            if not timestamps or all(t <= cutoff for t in timestamps)
        ]
        for key in stale:
            del self._buckets[key]
