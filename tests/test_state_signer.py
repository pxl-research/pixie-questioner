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

    def test_location_added_by_client_does_not_break_signature(self) -> None:
        """Regression: client adds location to a signed state (turn 2+).

        The server signs {phase, turn_count, ...} without location.
        The client echoes the state back and adds location for audit purposes.
        The signature must still verify correctly.
        """
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        server_state = {"phase": "intake", "turn_count": 1, "working_context": "rood"}
        signed = signer.sign(server_state)
        # Client attaches location before sending back
        signed["location"] = "Antwerpen"
        result = signer.verify_and_strip(signed)
        assert result == server_state | {"location": "Antwerpen"}

    def test_location_change_by_client_does_not_break_signature(self) -> None:
        """Client may change or remove location across turns without tampering."""
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        server_state = {"phase": "intake", "turn_count": 2}
        signed = signer.sign(server_state)
        signed["location"] = "Gent"
        result = signer.verify_and_strip(signed)
        assert result == {"phase": "intake", "turn_count": 2, "location": "Gent"}

    def test_server_key_tamper_still_rejected_even_with_location(self) -> None:
        """Adding location does not help bypass MAC verification on server fields."""
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        signed = signer.sign({"phase": "intake", "turn_count": 1})
        signed["location"] = "Antwerpen"
        signed["turn_count"] = 999  # tamper a server-controlled field
        with pytest.raises(ValueError, match="signature invalid"):
            signer.verify_and_strip(signed)

    def test_signature_stripped_from_result(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800)
        signed = signer.sign({"phase": "intake", "working_context": "red wine"})
        result = signer.verify_and_strip(signed)
        assert "_sig" not in result
        assert "_ts" not in result
        assert result == {"phase": "intake", "working_context": "red wine"}

    def test_scope_mismatch_rejected(self) -> None:
        brooklyn_signer = StateSigner(
            secret_key="test-secret", ttl_seconds=1800, scope="brooklyn"
        )
        case2_signer = StateSigner(
            secret_key="test-secret", ttl_seconds=1800, scope="case2"
        )
        signed = case2_signer.sign({"phase": "intake", "turn_count": 1})

        with pytest.raises(ValueError, match="scope invalid"):
            brooklyn_signer.verify_and_strip(signed)

    def test_unsigned_scope_mismatch_rejected(self) -> None:
        signer = StateSigner(secret_key="test-secret", ttl_seconds=1800, scope="case2")

        with pytest.raises(ValueError, match="scope invalid"):
            signer.verify_and_strip({"location": "Antwerpen", "_scope": "brooklyn"})
