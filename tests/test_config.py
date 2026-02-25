from __future__ import annotations

from pathlib import Path

import pytest

from pixie_questioner import config


def _set_required_env(
    monkeypatch: pytest.MonkeyPatch,
    profile_path: Path,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_API_VERSION", "2024-01-01")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("DOMAIN_PROFILE_PATH", str(profile_path))


def _write_default_profile_assets(tmp_path: Path) -> Path:
    (tmp_path / "question_asker.txt").write_text(
        "asker {summary} {completion_token}", encoding="utf-8"
    )
    (tmp_path / "questionnaire_notulist.txt").write_text(
        "notulist {summary} {current_question} {current_answer}", encoding="utf-8"
    )
    (tmp_path / "questionnaire_template.txt").write_text(
        "0. Initial user question: []\n1. Occasion, Vibe, Setting: []", encoding="utf-8"
    )
    (tmp_path / "relevance_classifier.txt").write_text(
        "Return ON_TOPIC or OFF_TOPIC for {current_question} and {current_answer}",
        encoding="utf-8",
    )

    profile_path = tmp_path / "domain_profile.yml"
    profile_path.write_text(
        """
version: 1
languages:
  base: [nl, en, fr]
  extra: [de]
  names:
    nl: Dutch
    en: English
    fr: French
    de: German
prompts:
  question_asker_path: question_asker.txt
  questionnaire_notulist_path: questionnaire_notulist.txt
  questionnaire_template_path: questionnaire_template.txt
  relevance_classifier_path: relevance_classifier.txt
messages:
  handoff:
    nl: ["handoff nl"]
    en: ["handoff en"]
    fr: ["handoff fr"]
  feedback_question:
    nl: ["feedback q nl"]
    en: ["feedback q en"]
    fr: ["feedback q fr"]
  feedback_thanks:
    nl: ["feedback thanks nl"]
    en: ["feedback thanks en"]
    fr: ["feedback thanks fr"]
  goodbye:
    nl: ["goodbye nl"]
    en: ["goodbye en"]
    fr: ["goodbye fr"]
  off_topic_redirect:
    nl: ["redirect nl: {question}"]
    en: ["redirect en: {question}"]
    fr: ["redirect fr: {question}"]
  off_topic_initial_question:
    nl: ["initial nl"]
    en: ["initial en"]
    fr: ["initial fr"]
quick_replies:
  nl:
    - { label: "Ja", value: "yes" }
    - { label: "Nee", value: "no" }
  en:
    - { label: "Yes", value: "yes" }
    - { label: "No", value: "no" }
  fr:
    - { label: "Oui", value: "yes" }
    - { label: "Non", value: "no" }
tokens:
  feedback_yes: ["ja", "yes", "oui"]
  feedback_no: ["nee", "no", "non"]
  stop_phrases: ["stop", "that's all"]
  stop_context_tokens: ["please", "thanks", "ok"]
behavior:
  completion_token: "READY"
  off_topic_termination_threshold: 3
term_replacements:
  nl:
    snit: pasvorm
""".strip(),
        encoding="utf-8",
    )
    return profile_path


def test_load_prompt_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"
    with pytest.raises(RuntimeError, match="Prompt file not found"):
        config.load_prompt(missing_path)


def test_get_settings_loads_domain_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    profile_path = _write_default_profile_assets(tmp_path)

    _set_required_env(monkeypatch, profile_path)
    config.get_settings.cache_clear()
    settings = config.get_settings()
    profile = settings.domain_profile

    assert profile.question_asker_prompt == "asker {summary} {completion_token}"
    assert (
        profile.questionnaire_template
        == "0. Initial user question: []\n1. Occasion, Vibe, Setting: []"
    )
    assert profile.relevance_classifier_prompt.startswith("Return ON_TOPIC")
    assert profile.off_topic_termination_threshold == 3
    assert profile.completion_token == "READY"
    assert profile.language_codes == ("nl", "en", "fr", "de")
    config.get_settings.cache_clear()


def test_domain_profile_missing_prompt_file_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    profile_path = _write_default_profile_assets(tmp_path)
    (tmp_path / "question_asker.txt").unlink()

    _set_required_env(monkeypatch, profile_path)
    config.get_settings.cache_clear()
    settings = config.get_settings()

    with pytest.raises(RuntimeError, match="Prompt file not found"):
        _ = settings.domain_profile
    config.get_settings.cache_clear()


def test_domain_profile_invalid_yaml_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    profile_path = tmp_path / "domain_profile.yml"
    profile_path.write_text("version: [", encoding="utf-8")
    _set_required_env(monkeypatch, profile_path)

    config.get_settings.cache_clear()
    settings = config.get_settings()
    with pytest.raises(RuntimeError, match="Invalid domain profile YAML"):
        _ = settings.domain_profile
    config.get_settings.cache_clear()


def test_domain_profile_path_is_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_KEY", "test-key")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_API_VERSION", "2024-01-01")
    monkeypatch.setenv("ADVANCED_AZURE_OPENAI_MODEL", "gpt-test")
    monkeypatch.delenv("DOMAIN_PROFILE_PATH", raising=False)

    config.get_settings.cache_clear()
    with pytest.raises(Exception):
        config.get_settings()
    config.get_settings.cache_clear()
