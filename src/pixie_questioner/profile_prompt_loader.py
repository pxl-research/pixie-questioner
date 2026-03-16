"""Prompt loading helpers for domain profiles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable


def _resolve_prompt_path(base_dir: Path, prompt_path: str) -> Path:
    candidate = Path(prompt_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def _resolve_prompt_with_fallback(
    profile_dir: Path,
    base_dir: Path | None,
    configured_path: str,
    *,
    default_name: str | None = None,
) -> Path:
    if configured_path:
        candidate = _resolve_prompt_path(profile_dir, configured_path)
        if not candidate.exists() and base_dir is not None:
            return _resolve_prompt_path(base_dir, configured_path)
        return candidate
    if default_name is None:
        raise RuntimeError("Missing required prompt path.")
    return _resolve_prompt_path(profile_dir, default_name)


def _load_optional_prompt(
    profile_dir: Path,
    base_dir: Path | None,
    configured_path: str,
    load_prompt: Callable[[Path], str],
) -> str:
    if not configured_path:
        return ""
    return load_prompt(
        _resolve_prompt_with_fallback(profile_dir, base_dir, configured_path)
    )


def load_profile_prompts(
    profile: Any,
    profile_dir: Path,
    base_dir: Path | None,
    load_prompt: Callable[[Path], str],
) -> tuple[str, str, str, str, str, str, str, str]:
    prompt_paths = profile.prompts
    question_asker_prompt = load_prompt(
        _resolve_prompt_path(profile_dir, prompt_paths.question_asker_path)
    )
    questionnaire_notulist_prompt = load_prompt(
        _resolve_prompt_with_fallback(
            profile_dir,
            base_dir,
            prompt_paths.questionnaire_notulist_path,
            default_name="questionnaire_notulist.txt",
        )
    )
    unsupported_origin_follow_up_prompt = _load_optional_prompt(
        profile_dir,
        base_dir,
        prompt_paths.unsupported_origin_follow_up_path,
        load_prompt,
    )
    questionnaire_template = load_prompt(
        _resolve_prompt_path(profile_dir, prompt_paths.questionnaire_template_path)
    )
    relevance_classifier_prompt = load_prompt(
        _resolve_prompt_with_fallback(
            profile_dir,
            base_dir,
            prompt_paths.relevance_classifier_path,
            default_name="relevance_classifier.txt",
        )
    )
    intake_route_prompt = _load_optional_prompt(
        profile_dir,
        base_dir,
        prompt_paths.intake_route_classifier_path,
        load_prompt,
    )
    aroma_prompt = _load_optional_prompt(
        profile_dir,
        base_dir,
        prompt_paths.aroma_taste_classifier_path,
        load_prompt,
    )
    handoff_prompt = _load_optional_prompt(
        profile_dir,
        base_dir,
        prompt_paths.handoff_classifier_path,
        load_prompt,
    )
    return (
        question_asker_prompt,
        questionnaire_notulist_prompt,
        unsupported_origin_follow_up_prompt,
        questionnaire_template,
        relevance_classifier_prompt,
        intake_route_prompt,
        aroma_prompt,
        handoff_prompt,
    )


def inject_questionnaire_template(
    questionnaire_notulist_prompt: str,
    questionnaire_template: str,
) -> str:
    if "{questionnaire_template}" not in questionnaire_notulist_prompt:
        return questionnaire_notulist_prompt
    field_names_only = re.sub(r":\s*\[\]", "", questionnaire_template.strip())
    return questionnaire_notulist_prompt.replace(
        "{questionnaire_template}", field_names_only
    )


def substitute_prompt_variables(
    variables: dict[str, dict[str, str]],
    relevance_classifier_prompt: str,
    intake_route_classifier_prompt: str,
    questionnaire_notulist_prompt: str,
    unsupported_origin_follow_up_prompt: str,
    aroma_taste_classifier_prompt: str,
    handoff_classifier_prompt: str,
    substitute_variables: Callable[[str, dict[str, str]], str],
) -> tuple[str, str, str, str, str, str]:
    if not variables:
        return (
            relevance_classifier_prompt,
            intake_route_classifier_prompt,
            questionnaire_notulist_prompt,
            unsupported_origin_follow_up_prompt,
            aroma_taste_classifier_prompt,
            handoff_classifier_prompt,
        )
    en_vars = {
        var_name: lang_values.get("en", next(iter(lang_values.values()), ""))
        for var_name, lang_values in variables.items()
    }
    return (
        substitute_variables(relevance_classifier_prompt, en_vars),
        substitute_variables(intake_route_classifier_prompt, en_vars),
        substitute_variables(questionnaire_notulist_prompt, en_vars),
        substitute_variables(unsupported_origin_follow_up_prompt, en_vars),
        substitute_variables(aroma_taste_classifier_prompt, en_vars),
        substitute_variables(handoff_classifier_prompt, en_vars),
    )
