# ruff: noqa: S101,S106
"""Tests for HMAC state signing with TTL."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pixie_questioner.state_signer import StateSigner


class TestStateSigner:
    def test_sign_verify_round_trip(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        original = {"phase": "intake", "turn_count": 1}
        signed = signer.sign(original)
        result = signer.verify_and_strip(signed)
        assert result == original

    def test_signed_state_contains_sig_and_ts(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        signed = signer.sign({"phase": "intake"})
        assert "_sig" in signed
        assert "_ts" in signed
        assert isinstance(signed["_sig"], str)
        assert isinstance(signed["_ts"], int)

    def test_tampered_value_rejected(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        signed = signer.sign({"phase": "intake", "turn_count": 1})
        signed["turn_count"] = 999
        with pytest.raises(ValueError, match="signature invalid"):
            signer.verify_and_strip(signed)

    def test_different_key_rejected(self) -> None:
        signer_a = StateSigner(secret_key="key-a", ttl_seconds=1800)
        signer_b = StateSigner(secret_key="key-b", ttl_seconds=1800)
        signed = signer_a.sign({"phase": "intake"})
        with pytest.raises(ValueError, match="signature invalid"):
            signer_b.verify_and_strip(signed)

    def test_expired_state_rejected(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=60)
        with patch("pixie_questioner.state_signer.time") as mock_time:
            mock_time.time.return_value = 1000.0
            signed = signer.sign({"phase": "intake"})
        with patch("pixie_questioner.state_signer.time") as mock_time:
            mock_time.time.return_value = 1061.0
            with pytest.raises(ValueError, match="expired"):
                signer.verify_and_strip(signed)

    def test_future_timestamp_rejected(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        with patch("pixie_questioner.state_signer.time") as mock_time:
            mock_time.time.return_value = 2000.0
            signed = signer.sign({"phase": "intake"})
        with patch("pixie_questioner.state_signer.time") as mock_time:
            mock_time.time.return_value = 1000.0
            with pytest.raises(ValueError, match="future"):
                signer.verify_and_strip(signed)

    def test_empty_state_allowed_unsigned(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        assert signer.verify_and_strip({}) == {}

    def test_unprotected_keys_allowed_unsigned(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        state = {"location": "Antwerpen"}
        assert signer.verify_and_strip(state) == {"location": "Antwerpen"}

    def test_protected_keys_without_signature_rejected(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        with pytest.raises(ValueError, match="signature missing"):
            signer.verify_and_strip({"phase": "intake", "turn_count": 0})

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            StateSigner(secret_key="", ttl_seconds=1800)

    def test_signature_stripped_from_result(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        signed = signer.sign({"phase": "intake", "working_context": "red wine"})
        result = signer.verify_and_strip(signed)
        assert "_sig" not in result
        assert "_ts" not in result
        assert result == {"phase": "intake", "working_context": "red wine"}
