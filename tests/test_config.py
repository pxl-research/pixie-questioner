from __future__ import annotations

from pathlib import Path

import pytest

from pixie_questioner import config


def _set_required_env(
    monkeypatch: pytest.MonkeyPatch,
    system_path: Path,
    template_path: Path,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_API_VERSION", "2024-01-01")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("SYSTEM_PROMPT_PATH", str(system_path))
    monkeypatch.setenv("QUESTIONNAIRE_TEMPLATE_PATH", str(template_path))


def test_load_prompt_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"
    with pytest.raises(RuntimeError, match="Prompt file not found"):
        config.load_prompt(missing_path)


def test_get_settings_loads_prompts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    system_path = tmp_path / "system_prompt.txt"
    template_path = tmp_path / "template.txt"
    system_path.write_text("system prompt", encoding="utf-8")
    template_path.write_text("0. Initial user question: []", encoding="utf-8")

    _set_required_env(monkeypatch, system_path, template_path)
    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.system_prompt == "system prompt"
    assert settings.questionnaire_template == "0. Initial user question: []"
    config.get_settings.cache_clear()
