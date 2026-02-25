"""Configuration and settings loading."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_prompt(path: Path) -> str:
    try:
        resolved_path = path.expanduser()
        return resolved_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Prompt file not found: {path}") from exc


def _normalize_language_code(value: str) -> str:
    return value.strip().casefold()


def _normalize_localized_lines(
    values: dict[str, list[str]],
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_code, raw_lines in values.items():
        code = _normalize_language_code(raw_code)
        lines = tuple(line.strip() for line in raw_lines if line.strip())
        if not lines:
            raise RuntimeError(
                f"Localized message list for language '{raw_code}' cannot be empty."
            )
        normalized[code] = lines
    return normalized


def _resolve_prompt_path(base_dir: Path, prompt_path: str) -> Path:
    candidate = Path(prompt_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


@dataclass(frozen=True)
class DomainQuickReply:
    label: str
    value: str


@dataclass(frozen=True)
class ResolvedDomainProfile:
    question_asker_prompt: str
    questionnaire_notulist_prompt: str
    questionnaire_template: str
    relevance_classifier_prompt: str
    language_codes: tuple[str, ...]
    language_names: dict[str, str]
    handoff_lines: dict[str, tuple[str, ...]]
    feedback_questions: dict[str, tuple[str, ...]]
    feedback_thanks: dict[str, tuple[str, ...]]
    goodbye_lines: dict[str, tuple[str, ...]]
    off_topic_redirects: dict[str, tuple[str, ...]]
    off_topic_initial_questions: dict[str, tuple[str, ...]]
    quick_replies: dict[str, tuple[DomainQuickReply, ...]]
    feedback_yes_tokens: frozenset[str]
    feedback_no_tokens: frozenset[str]
    stop_phrases: tuple[str, ...]
    stop_context_tokens: frozenset[str]
    completion_token: str
    off_topic_termination_threshold: int
    term_replacements: dict[str, tuple[tuple[str, str], ...]]

    @property
    def chat_domain_policy(self):
        from .chat.policy import ChatDomainPolicy, ChatQuickReply

        return ChatDomainPolicy(
            language_codes=self.language_codes,
            language_names=self.language_names,
            handoff_lines=self.handoff_lines,
            feedback_questions=self.feedback_questions,
            feedback_thanks=self.feedback_thanks,
            goodbye_lines=self.goodbye_lines,
            off_topic_redirects=self.off_topic_redirects,
            off_topic_initial_questions=self.off_topic_initial_questions,
            quick_replies={
                language: tuple(
                    ChatQuickReply(label=item.label, value=item.value)
                    for item in quick_reply_options
                )
                for language, quick_reply_options in self.quick_replies.items()
            },
            feedback_yes_tokens=self.feedback_yes_tokens,
            feedback_no_tokens=self.feedback_no_tokens,
            stop_phrases=self.stop_phrases,
            stop_context_tokens=self.stop_context_tokens,
            completion_token=self.completion_token,
            off_topic_termination_threshold=self.off_topic_termination_threshold,
            term_replacements=self.term_replacements,
        )


class DomainProfileLanguages(BaseModel):
    base: list[str]
    extra: list[str] = Field(default_factory=list)
    names: dict[str, str]

    @model_validator(mode="after")
    def validate_required_base_languages(self) -> DomainProfileLanguages:
        normalized_base = {_normalize_language_code(code) for code in self.base}
        required = {"nl", "en", "fr"}
        missing = sorted(required - normalized_base)
        if missing:
            raise ValueError(
                f"languages.base must include nl, en, fr. Missing: {', '.join(missing)}."
            )
        return self


class DomainProfilePrompts(BaseModel):
    question_asker_path: str
    questionnaire_notulist_path: str
    questionnaire_template_path: str
    relevance_classifier_path: str


class DomainProfileMessages(BaseModel):
    handoff: dict[str, list[str]]
    feedback_question: dict[str, list[str]]
    feedback_thanks: dict[str, list[str]]
    goodbye: dict[str, list[str]]
    off_topic_redirect: dict[str, list[str]]
    off_topic_initial_question: dict[str, list[str]]


class DomainProfileQuickReply(BaseModel):
    label: str
    value: Literal["yes", "no"]


class DomainProfileTokens(BaseModel):
    feedback_yes: list[str]
    feedback_no: list[str]
    stop_phrases: list[str]
    stop_context_tokens: list[str]


class DomainProfileBehavior(BaseModel):
    completion_token: str = Field(default="")
    off_topic_termination_threshold: int = Field(default=2, ge=1)


class DomainProfileFile(BaseModel):
    version: Literal[1]
    languages: DomainProfileLanguages
    prompts: DomainProfilePrompts
    messages: DomainProfileMessages
    quick_replies: dict[str, list[DomainProfileQuickReply]]
    tokens: DomainProfileTokens
    behavior: DomainProfileBehavior
    term_replacements: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_profile(self) -> DomainProfileFile:
        normalized_base = {
            _normalize_language_code(code) for code in self.languages.base
        }
        missing_in_names = normalized_base - {
            _normalize_language_code(code) for code in self.languages.names
        }
        if missing_in_names:
            missing = ", ".join(sorted(missing_in_names))
            raise ValueError(f"languages.names is missing base language(s): {missing}.")

        quick_reply_languages = {
            _normalize_language_code(code) for code in self.quick_replies
        }
        missing_quick_replies = normalized_base - quick_reply_languages
        if missing_quick_replies:
            missing = ", ".join(sorted(missing_quick_replies))
            raise ValueError(f"quick_replies is missing base language(s): {missing}.")

        localized_groups: tuple[tuple[str, dict[str, list[str]]], ...] = (
            ("messages.handoff", self.messages.handoff),
            ("messages.feedback_question", self.messages.feedback_question),
            ("messages.feedback_thanks", self.messages.feedback_thanks),
            ("messages.goodbye", self.messages.goodbye),
            ("messages.off_topic_redirect", self.messages.off_topic_redirect),
            (
                "messages.off_topic_initial_question",
                self.messages.off_topic_initial_question,
            ),
        )
        for group_name, localized_values in localized_groups:
            available = {_normalize_language_code(code) for code in localized_values}
            missing = normalized_base - available
            if missing:
                missing_languages = ", ".join(sorted(missing))
                raise ValueError(
                    f"{group_name} is missing base language(s): {missing_languages}."
                )

        for language, lines in self.messages.off_topic_redirect.items():
            for line in lines:
                if "{question}" not in line:
                    raise ValueError(
                        "messages.off_topic_redirect entries must contain '{question}' "
                        f"(language '{language}')."
                    )
        return self


def load_domain_profile(path: Path) -> ResolvedDomainProfile:
    resolved_path = path.expanduser()
    try:
        raw_yaml = resolved_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Domain profile file not found: {path}") from exc

    try:
        loaded = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid domain profile YAML: {path}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Invalid domain profile YAML: {path}")

    try:
        profile = DomainProfileFile.model_validate(loaded)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid domain profile configuration: {exc}") from exc

    profile_dir = resolved_path.resolve().parent
    prompt_paths = profile.prompts

    question_asker_prompt = load_prompt(
        _resolve_prompt_path(profile_dir, prompt_paths.question_asker_path)
    )
    questionnaire_notulist_prompt = load_prompt(
        _resolve_prompt_path(profile_dir, prompt_paths.questionnaire_notulist_path)
    )
    questionnaire_template = load_prompt(
        _resolve_prompt_path(profile_dir, prompt_paths.questionnaire_template_path)
    )
    relevance_classifier_prompt = load_prompt(
        _resolve_prompt_path(profile_dir, prompt_paths.relevance_classifier_path)
    )

    normalized_base = tuple(
        dict.fromkeys(_normalize_language_code(code) for code in profile.languages.base)
    )
    normalized_extra = tuple(
        code
        for code in dict.fromkeys(
            _normalize_language_code(code) for code in profile.languages.extra
        )
        if code not in normalized_base
    )
    language_codes = (*normalized_base, *normalized_extra)
    language_names = {
        _normalize_language_code(code): name
        for code, name in profile.languages.names.items()
    }
    language_names.setdefault("en", "English")
    for language_code in language_codes:
        language_names.setdefault(language_code, language_code.upper())

    quick_replies = {
        _normalize_language_code(language): tuple(
            DomainQuickReply(label=item.label, value=item.value)
            for item in localized_quick_replies
        )
        for language, localized_quick_replies in profile.quick_replies.items()
    }
    term_replacements = {
        _normalize_language_code(language): tuple(values.items())
        for language, values in profile.term_replacements.items()
    }

    return ResolvedDomainProfile(
        question_asker_prompt=question_asker_prompt,
        questionnaire_notulist_prompt=questionnaire_notulist_prompt,
        questionnaire_template=questionnaire_template,
        relevance_classifier_prompt=relevance_classifier_prompt,
        language_codes=language_codes,
        language_names=language_names,
        handoff_lines=_normalize_localized_lines(profile.messages.handoff),
        feedback_questions=_normalize_localized_lines(
            profile.messages.feedback_question
        ),
        feedback_thanks=_normalize_localized_lines(profile.messages.feedback_thanks),
        goodbye_lines=_normalize_localized_lines(profile.messages.goodbye),
        off_topic_redirects=_normalize_localized_lines(
            profile.messages.off_topic_redirect
        ),
        off_topic_initial_questions=_normalize_localized_lines(
            profile.messages.off_topic_initial_question
        ),
        quick_replies=quick_replies,
        feedback_yes_tokens=frozenset(
            token.casefold().strip()
            for token in profile.tokens.feedback_yes
            if token.strip()
        ),
        feedback_no_tokens=frozenset(
            token.casefold().strip()
            for token in profile.tokens.feedback_no
            if token.strip()
        ),
        stop_phrases=tuple(
            phrase.strip() for phrase in profile.tokens.stop_phrases if phrase.strip()
        ),
        stop_context_tokens=frozenset(
            token.casefold().strip()
            for token in profile.tokens.stop_context_tokens
            if token.strip()
        ),
        completion_token=profile.behavior.completion_token,
        off_topic_termination_threshold=profile.behavior.off_topic_termination_threshold,
        term_replacements=term_replacements,
    )


class Settings(BaseSettings):
    azure_openai_endpoint: str = Field(alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_key: str = Field(alias="AZURE_OPENAI_KEY")
    advanced_azure_openai_api_version: str = Field(
        alias="ADVANCED_AZURE_OPENAI_API_VERSION"
    )
    advanced_azure_openai_model: str = Field(alias="ADVANCED_AZURE_OPENAI_MODEL")
    advanced_azure_openai_temperature: float = Field(
        default=0.2, alias="ADVANCED_AZURE_OPENAI_TEMPERATURE"
    )
    domain_profile_path: Path = Field(alias="DOMAIN_PROFILE_PATH")
    audit_api_url: str | None = Field(default=None, alias="AUDIT_API_URL")
    audit_api_key: str | None = Field(default=None, alias="AUDIT_API_KEY")
    audit_profile: str | None = Field(default=None, alias="AUDIT_PROFILE")
    audit_timeout_seconds: float = Field(default=5.0, alias="AUDIT_TIMEOUT_SECONDS")

    @property
    def domain_profile(self) -> ResolvedDomainProfile:
        return load_domain_profile(self.domain_profile_path)

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.dev"),
        env_file_encoding="utf-8",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
