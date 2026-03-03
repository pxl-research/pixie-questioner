# ruff: noqa: S101
"""Tests for in-memory sliding-window rate limiter."""

from __future__ import annotations

from unittest.mock import patch

from pixie_questioner.rate_limiter import SlidingWindowRateLimiter


class TestSlidingWindowRateLimiter:
    def test_allows_within_limit(self) -> None:
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        assert limiter.is_allowed("k") is True
        assert limiter.is_allowed("k") is True
        assert limiter.is_allowed("k") is True

    def test_rejects_over_limit(self) -> None:
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("k") is True
        assert limiter.is_allowed("k") is True
        assert limiter.is_allowed("k") is False

    def test_keys_are_independent(self) -> None:
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("a") is True
        assert limiter.is_allowed("b") is True
        assert limiter.is_allowed("a") is False
        assert limiter.is_allowed("b") is False

    def test_window_expires_old_requests(self) -> None:
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10)
        with patch("pixie_questioner.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0
            assert limiter.is_allowed("k") is True
            assert limiter.is_allowed("k") is False

            mock_time.monotonic.return_value = 1011.0
            assert limiter.is_allowed("k") is True

    def test_boundary_at_exactly_window(self) -> None:
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10)
        with patch("pixie_questioner.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = 1000.0
            assert limiter.is_allowed("k") is True

            mock_time.monotonic.return_value = 1010.0
            assert limiter.is_allowed("k") is True
