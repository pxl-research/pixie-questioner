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
    assert profile.completion_mode == "summary_feedback"
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


# --- Profile layering (extends + variables) tests ---


def _write_base_profile(base_dir: Path) -> Path:
    """Create a base profile YAML with shared prompts and messages."""
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "relevance_classifier.txt").write_text(
        "Classifier for {topic}: {current_question} {current_answer}",
        encoding="utf-8",
    )
    (base_dir / "questionnaire_notulist.txt").write_text(
        "notulist for {topic}:\n{questionnaire_template}\n{summary} {current_question} {current_answer}",
        encoding="utf-8",
    )
    profile_path = base_dir / "domain_profile.yml"
    profile_path.write_text(
        """
version: 1
languages:
  base: [nl, en, fr]
  extra: []
  names:
    nl: Dutch
    en: English
    fr: French
prompts:
  questionnaire_notulist_path: questionnaire_notulist.txt
  relevance_classifier_path: relevance_classifier.txt
messages:
  handoff:
    nl: ["handoff nl"]
    en: ["handoff en"]
    fr: ["handoff fr"]
  feedback_question:
    nl: ["feedback {visit_type} nl"]
    en: ["feedback {visit_type} en"]
    fr: ["feedback {visit_type} fr"]
  feedback_thanks:
    nl: ["thanks nl"]
    en: ["thanks en"]
    fr: ["thanks fr"]
  goodbye:
    nl: ["bye nl"]
    en: ["bye en"]
    fr: ["bye fr"]
  off_topic_redirect:
    nl: ["alleen {topic}: {question}"]
    en: ["only {topic}: {question}"]
    fr: ["seulement {topic}: {question}"]
  off_topic_initial_question:
    nl: ["help {topic} nl"]
    en: ["help {topic} en"]
    fr: ["help {topic} fr"]
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
behavior:
  completion_token: "BASE_TOKEN"
  off_topic_termination_threshold: 2
""".strip(),
        encoding="utf-8",
    )
    return profile_path


def _write_domain_profile_with_extends(domain_dir: Path, base_relative: str) -> Path:
    """Create a domain profile that extends a base and adds domain-specific parts."""
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / "question_asker.txt").write_text("asker {summary}", encoding="utf-8")
    (domain_dir / "questionnaire_template.txt").write_text(
        "0. Initial user question: []", encoding="utf-8"
    )

    profile_path = domain_dir / "domain_profile.yml"
    profile_path.write_text(
        f"""
version: 1
extends: {base_relative}
variables:
  topic:
    nl: jeansbroeken
    en: jeans
    fr: jeans
  visit_type:
    nl: winkelbezoek
    en: store visit
    fr: visite en magasin
prompts:
  question_asker_path: question_asker.txt
  questionnaire_template_path: questionnaire_template.txt
term_replacements:
  nl:
    snit: pasvorm
""".strip(),
        encoding="utf-8",
    )
    return profile_path


def test_profile_extends_merges_base_and_domain(tmp_path: Path) -> None:
    base_dir = tmp_path / "_base"
    domain_dir = tmp_path / "brooklyn"
    _write_base_profile(base_dir)
    _write_domain_profile_with_extends(domain_dir, "../_base/domain_profile.yml")

    profile = config.load_domain_profile(domain_dir / "domain_profile.yml")

    # Inherited from base
    assert profile.language_codes == ("nl", "en", "fr")
    assert profile.completion_token == "BASE_TOKEN"
    assert profile.off_topic_termination_threshold == 2
    assert profile.feedback_yes_tokens == frozenset({"ja", "yes", "oui"})

    # Domain-specific
    assert profile.question_asker_prompt == "asker {summary}"
    assert profile.questionnaire_template == "0. Initial user question: []"
    assert profile.term_replacements == {"nl": (("snit", "pasvorm"),)}


def test_profile_variables_resolve_in_messages(tmp_path: Path) -> None:
    base_dir = tmp_path / "_base"
    domain_dir = tmp_path / "brooklyn"
    _write_base_profile(base_dir)
    _write_domain_profile_with_extends(domain_dir, "../_base/domain_profile.yml")

    profile = config.load_domain_profile(domain_dir / "domain_profile.yml")
    policy = profile.chat_domain_policy

    # {topic} resolved, {question} kept as runtime placeholder
    assert policy.off_topic_redirects["nl"] == ("alleen jeansbroeken: {question}",)
    assert policy.off_topic_redirects["en"] == ("only jeans: {question}",)
    assert policy.off_topic_redirects["fr"] == ("seulement jeans: {question}",)

    # {visit_type} resolved
    assert policy.feedback_questions["nl"] == ("feedback winkelbezoek nl",)
    assert policy.feedback_questions["en"] == ("feedback store visit en",)
    assert policy.feedback_questions["fr"] == ("feedback visite en magasin fr",)

    # {topic} resolved in initial question
    assert policy.off_topic_initial_questions["nl"] == ("help jeansbroeken nl",)
    assert policy.off_topic_initial_questions["en"] == ("help jeans en",)


def test_profile_without_extends_still_works(tmp_path: Path) -> None:
    """A standalone profile without extends must keep working."""
    profile_path = _write_default_profile_assets(tmp_path)
    profile = config.load_domain_profile(profile_path)

    assert profile.language_codes == ("nl", "en", "fr", "de")
    assert profile.completion_token == "READY"
    assert profile.completion_mode == "summary_feedback"


def test_profile_classifier_prompt_resolves_topic_from_base(tmp_path: Path) -> None:
    """The classifier prompt from _base should have {topic} substituted."""
    base_dir = tmp_path / "_base"
    domain_dir = tmp_path / "brooklyn"
    _write_base_profile(base_dir)
    _write_domain_profile_with_extends(domain_dir, "../_base/domain_profile.yml")

    profile = config.load_domain_profile(domain_dir / "domain_profile.yml")

    # {topic} resolved with EN value, runtime placeholders kept
    assert "jeans" in profile.relevance_classifier_prompt
    assert "{topic}" not in profile.relevance_classifier_prompt
    assert "{current_question}" in profile.relevance_classifier_prompt
    assert "{current_answer}" in profile.relevance_classifier_prompt


def test_profile_notulist_prompt_resolves_topic_from_base(tmp_path: Path) -> None:
    """The notulist prompt from _base should have {topic} substituted."""
    base_dir = tmp_path / "_base"
    domain_dir = tmp_path / "brooklyn"
    _write_base_profile(base_dir)
    _write_domain_profile_with_extends(domain_dir, "../_base/domain_profile.yml")

    profile = config.load_domain_profile(domain_dir / "domain_profile.yml")

    # {topic} resolved with EN value, runtime placeholders kept
    assert "jeans" in profile.questionnaire_notulist_prompt
    assert "{topic}" not in profile.questionnaire_notulist_prompt
    assert "{summary}" in profile.questionnaire_notulist_prompt
    assert "{current_question}" in profile.questionnaire_notulist_prompt

    # {questionnaire_template} replaced with field names only (no ': []')
    assert "{questionnaire_template}" not in profile.questionnaire_notulist_prompt
    assert "0. Initial user question" in profile.questionnaire_notulist_prompt
    assert ": []" not in profile.questionnaire_notulist_prompt


def test_profile_domain_overrides_base_values(tmp_path: Path) -> None:
    base_dir = tmp_path / "_base"
    domain_dir = tmp_path / "shop"
    _write_base_profile(base_dir)
    _write_domain_profile_with_extends(domain_dir, "../_base/domain_profile.yml")

    # Override behavior in domain profile
    domain_profile_path = domain_dir / "domain_profile.yml"
    content = domain_profile_path.read_text(encoding="utf-8")
    content += (
        "\nbehavior:\n"
        "  completion_token: SHOP_DONE\n"
        "  off_topic_termination_threshold: 5\n"
        "  completion_mode: categorized_handoff\n"
    )
    domain_profile_path.write_text(content, encoding="utf-8")

    profile = config.load_domain_profile(domain_profile_path)

    assert profile.completion_token == "SHOP_DONE"
    assert profile.off_topic_termination_threshold == 5
    assert profile.completion_mode == "categorized_handoff"


def test_cretskens_prompts_reduce_irrelevant_followup_questions() -> None:
    profile = config.load_domain_profile(Path("profiles/cretskens/domain_profile.yml"))

    assert "Avoid unnatural follow-up questions." in profile.question_asker_prompt
    assert (
        "If a dish has multiple common protein variants"
        in profile.question_asker_prompt
    )
    assert "If course type is cheese course" in profile.question_asker_prompt
    assert "do not ask it" in profile.question_asker_prompt
    assert (
        "origin (country and/or region in one question)"
        in profile.question_asker_prompt
    )
    assert 'If the customer says "geen voorkeur"' in profile.question_asker_prompt
    assert (
        "ask at most one extra clarification question" in profile.question_asker_prompt
    )
    assert (
        "Do NOT expose internal categorization language or field labels"
        in profile.question_asker_prompt
    )
    assert "Avoid checklist wording" in profile.question_asker_prompt
    assert (
        "For main protein, use a stricter threshold"
        in profile.questionnaire_notulist_prompt
    )
    assert (
        "If a dish has multiple common protein variants, keep main protein as []"
        in (profile.questionnaire_notulist_prompt)
    )
    assert (
        "If the dish is clearly not a cheese dish"
        in profile.questionnaire_notulist_prompt
    )
    assert "cheese type = none" in profile.questionnaire_notulist_prompt
    assert "Region priority:" in profile.questionnaire_notulist_prompt
    assert "If both country and region are present but conflicting, region wins." in (
        profile.questionnaire_notulist_prompt
    )
    assert "If the user gives a range, use the upper bound." in (
        profile.questionnaire_notulist_prompt
    )
