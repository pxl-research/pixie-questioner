# ruff: noqa: S101,S105,S106
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
    monkeypatch.setenv("BROOKLYN_DOMAIN_PROFILE_PATH", str(profile_path))


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


def _write_cretskens_profile_assets(tmp_path: Path) -> Path:
    (tmp_path / "turn_controller_prompt.txt").write_text(
        "friendly in-store wine advisor\nPreference loop before recommendation",
        encoding="utf-8",
    )
    (tmp_path / "ranker_prompt.txt").write_text(
        "Select the best candidate barcodes\nselection_explanation",
        encoding="utf-8",
    )

    profile_path = tmp_path / "advisor_profile.yml"
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "prompts:",
                "  turn_controller_path: turn_controller_prompt.txt",
                "  ranker_path: ranker_prompt.txt",
                'recommendation_intro: "Leuk, ik heb een paar mooie opties voor je klaargezet."',
            ]
        ),
        encoding="utf-8",
    )
    return profile_path


def test_load_prompt_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"
    with pytest.raises(RuntimeError, match="Prompt file not found"):
        config.load_prompt(missing_path)


def test_get_settings_loads_brooklyn_domain_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    profile_path = _write_default_profile_assets(tmp_path)

    _set_required_env(monkeypatch, profile_path)
    config.get_settings.cache_clear()
    settings = config.get_settings()
    profile = settings.brooklyn_domain_profile

    assert profile.question_asker_prompt == "asker {summary} {completion_token}"
    assert (
        profile.questionnaire_template
        == "0. Initial user question: []\n1. Occasion, Vibe, Setting: []"
    )
    assert profile.relevance_classifier_prompt.startswith("Return ON_TOPIC")
    assert profile.off_topic_termination_threshold == 3
    assert profile.completion_token == "READY"
    assert profile.completion_mode == "summary_feedback"
    assert profile.aroma_taste_classifier_prompt == ""
    assert profile.handoff_classifier_prompt == ""
    assert profile.intake_route_classifier_prompt == ""
    assert profile.language_codes == ("nl", "en", "fr", "de")
    config.get_settings.cache_clear()


def test_get_settings_prefers_cretskens_profile_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    brooklyn_profile_path = _write_default_profile_assets(tmp_path)
    cretskens_profile_path = _write_cretskens_profile_assets(tmp_path)

    _set_required_env(monkeypatch, brooklyn_profile_path)
    monkeypatch.setenv("CRETSKENS_PROFILE_PATH", str(cretskens_profile_path))
    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.cretskens_profile_path == cretskens_profile_path
    config.get_settings.cache_clear()


def test_get_settings_prefers_cretskens_wine_catalog_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    profile_path = _write_default_profile_assets(tmp_path)
    catalog_path = tmp_path / "catalog.xlsx"

    _set_required_env(monkeypatch, profile_path)
    monkeypatch.setenv("CRETSKENS_WINE_CATALOG_PATH", str(catalog_path))
    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.cretskens_wine_catalog_path == catalog_path
    config.get_settings.cache_clear()


def test_get_settings_uses_default_cretskens_wine_catalog_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    profile_path = _write_default_profile_assets(tmp_path)

    _set_required_env(monkeypatch, profile_path)
    monkeypatch.delenv("CRETSKENS_WINE_CATALOG_PATH", raising=False)
    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.cretskens_wine_catalog_path == config.DEFAULT_WINE_CATALOG_PATH
    config.get_settings.cache_clear()


def test_get_settings_ignores_blank_path_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    profile_path = _write_default_profile_assets(tmp_path)

    _set_required_env(monkeypatch, profile_path)
    monkeypatch.setenv("BROOKLYN_DOMAIN_PROFILE_PATH", "   ")
    monkeypatch.setenv("CRETSKENS_PROFILE_PATH", "")
    monkeypatch.setenv("CRETSKENS_WINE_CATALOG_PATH", " ")
    config.get_settings.cache_clear()
    settings = config.get_settings()

    assert settings.brooklyn_domain_profile_path == config.DEFAULT_BROOKLYN_PROFILE_PATH
    assert settings.cretskens_profile_path == config.DEFAULT_CRETSKENS_PROFILE_PATH
    assert settings.cretskens_wine_catalog_path == config.DEFAULT_WINE_CATALOG_PATH
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
        _ = settings.brooklyn_domain_profile
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
        _ = settings.brooklyn_domain_profile
    config.get_settings.cache_clear()


def test_load_cretskens_profile_reads_prompts_and_intro(tmp_path: Path) -> None:
    profile_path = _write_cretskens_profile_assets(tmp_path)

    from pixie_questioner.wine_advisor.profile_loader import load_cretskens_profile

    profile = load_cretskens_profile(profile_path)

    assert "friendly in-store wine advisor" in profile.turn_controller_prompt
    assert "Select the best candidate barcodes" in profile.ranker_prompt
    assert profile.recommendation_intro == (
        "Leuk, ik heb een paar mooie opties voor je klaargezet."
    )


def test_load_cretskens_profile_supports_tilde_prompt_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home_dir = tmp_path / "home"
    prompts_dir = home_dir / "prompts"
    prompts_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home_dir))
    (prompts_dir / "turn_controller_prompt.txt").write_text(
        "tilde controller prompt",
        encoding="utf-8",
    )
    (prompts_dir / "ranker_prompt.txt").write_text(
        "tilde ranker prompt",
        encoding="utf-8",
    )
    profile_path = tmp_path / "advisor_profile.yml"
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "prompts:",
                "  turn_controller_path: ~/prompts/turn_controller_prompt.txt",
                "  ranker_path: ~/prompts/ranker_prompt.txt",
                'recommendation_intro: "Tilde intro"',
            ]
        ),
        encoding="utf-8",
    )

    from pixie_questioner.wine_advisor.profile_loader import load_cretskens_profile

    profile = load_cretskens_profile(profile_path)

    assert profile.turn_controller_prompt == "tilde controller prompt"
    assert profile.ranker_prompt == "tilde ranker prompt"
    assert profile.recommendation_intro == "Tilde intro"


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


def test_cretskens_profile_loads_runtime_prompt_text() -> None:
    from pixie_questioner.wine_advisor.profile_loader import load_cretskens_profile

    profile = load_cretskens_profile(Path("profiles/cretskens/advisor_profile.yml"))

    assert "Preference loop before recommendation" in profile.turn_controller_prompt
    assert "friendly in-store wine advisor" in profile.turn_controller_prompt
    assert "Mirror the customer's language (NL/EN/FR)" in (
        profile.turn_controller_prompt
    )
    assert "Select the best candidate barcodes" in profile.ranker_prompt
    assert "selection_explanation" in profile.ranker_prompt
    assert profile.recommendation_intro


def test_cretskens_domain_profile_loads_intake_route_classifier_prompt() -> None:
    profile = config.load_domain_profile(Path("profiles/cretskens/domain_profile.yml"))

    assert "Return exactly one label" in profile.intake_route_classifier_prompt
    assert "FOOD_PAIRING" in profile.intake_route_classifier_prompt
    assert "GENERAL_WINE" in profile.intake_route_classifier_prompt


def test_cretskens_domain_profile_loads_unsupported_origin_follow_up_prompt() -> None:
    profile = config.load_domain_profile(Path("profiles/cretskens/domain_profile.yml"))

    assert '"should_follow_up"' in profile.unsupported_origin_follow_up_prompt
    assert "customer-facing follow-up question" in (
        profile.unsupported_origin_follow_up_prompt
    )


def test_cretskens_notulist_prompt_encourages_main_course_inference() -> None:
    profile = config.load_domain_profile(Path("profiles/cretskens/domain_profile.yml"))

    assert '"pad thai" → course type = main course (clearly implied) ✓' in (
        profile.questionnaire_notulist_prompt
    )


def test_cretskens_notulist_prompt_has_generic_main_course_rule() -> None:
    profile = config.load_domain_profile(Path("profiles/cretskens/domain_profile.yml"))

    assert (
        "For clearly savory standalone meal dishes, set course type = main course"
        in profile.questionnaire_notulist_prompt
    )
    assert (
        "unless the customer explicitly indicates otherwise."
        in profile.questionnaire_notulist_prompt
    )


def test_brooklyn_prompt_encourages_warm_non_repetitive_tone() -> None:
    profile = config.load_domain_profile(Path("profiles/brooklyn/domain_profile.yml"))

    assert (
        "Use a warm, genuinely interested in-store tone."
        in profile.question_asker_prompt
    )
    assert "Vary your acknowledgements naturally across turns." in (
        profile.question_asker_prompt
    )
    assert 'Avoid repetitive openers such as "Je geeft aan dat".' in (
        profile.question_asker_prompt
    )
