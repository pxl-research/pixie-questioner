"""Configuration and settings loading."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .chat.policy import ChatDomainPolicy, ChatQuickReply
from .profile_prompt_loader import (
    inject_questionnaire_template,
    load_profile_prompts,
    substitute_prompt_variables,
)
from .wine_advisor.profile_loader import (
    ResolvedCretskensProfile,
    load_cretskens_profile,
)

DEFAULT_WINE_CATALOG_PATH = Path("profiles/cretskens/cretskens_wijnen_reduced.xlsx")
DEFAULT_BROOKLYN_PROFILE_PATH = Path("profiles/brooklyn/domain_profile.yml")
DEFAULT_CRETSKENS_DOMAIN_PROFILE_PATH = Path("profiles/cretskens/domain_profile.yml")
DEFAULT_CRETSKENS_PROFILE_PATH = Path("profiles/cretskens/advisor_profile.yml")


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


@dataclass(frozen=True)
class ResolvedDomainProfile:
    question_asker_prompt: str
    questionnaire_notulist_prompt: str
    unsupported_origin_follow_up_prompt: str
    questionnaire_template: str
    relevance_classifier_prompt: str
    intake_route_classifier_prompt: str
    handoff_classifier_prompt: str
    off_topic_redirect_prompt: str
    language_codes: tuple[str, ...]
    language_names: dict[str, str]
    handoff_lines: dict[str, tuple[str, ...]]
    feedback_questions: dict[str, tuple[str, ...]]
    feedback_thanks: dict[str, tuple[str, ...]]
    goodbye_lines: dict[str, tuple[str, ...]]
    off_topic_redirects: dict[str, tuple[str, ...]]
    off_topic_initial_questions: dict[str, tuple[str, ...]]
    quick_replies: dict[str, tuple[ChatQuickReply, ...]]
    feedback_yes_tokens: frozenset[str]
    feedback_no_tokens: frozenset[str]
    completion_token: str
    completion_mode: Literal["summary_feedback", "categorized_handoff"]
    off_topic_termination_threshold: int
    term_replacements: dict[str, tuple[tuple[str, str], ...]]

    @property
    def chat_domain_policy(self) -> ChatDomainPolicy:
        return ChatDomainPolicy(
            language_codes=self.language_codes,
            language_names=self.language_names,
            handoff_lines=self.handoff_lines,
            feedback_questions=self.feedback_questions,
            feedback_thanks=self.feedback_thanks,
            goodbye_lines=self.goodbye_lines,
            off_topic_redirects=self.off_topic_redirects,
            off_topic_initial_questions=self.off_topic_initial_questions,
            quick_replies=self.quick_replies,
            feedback_yes_tokens=self.feedback_yes_tokens,
            feedback_no_tokens=self.feedback_no_tokens,
            completion_token=self.completion_token,
            completion_mode=self.completion_mode,
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
    questionnaire_notulist_path: str = Field(default="")
    unsupported_origin_follow_up_path: str = Field(default="")
    questionnaire_template_path: str
    relevance_classifier_path: str = Field(default="")
    intake_route_classifier_path: str = Field(default="")
    handoff_classifier_path: str = Field(default="")
    off_topic_redirect_path: str = Field(default="")


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


class DomainProfileBehavior(BaseModel):
    completion_token: str = Field(default="")
    completion_mode: Literal["summary_feedback", "categorized_handoff"] = Field(
        default="summary_feedback"
    )
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


_VARIABLE_PATTERN = re.compile(r"\{(\w+)\}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* into *base*. Override values take precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _substitute_variables(text: str, variables: dict[str, str]) -> str:
    """Replace ``{var}`` placeholders whose key exists in *variables*.

    Unknown placeholders (e.g. runtime ``{question}``) are left intact.
    """

    def _replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, match.group(0))

    return _VARIABLE_PATTERN.sub(_replacer, text)


def _resolve_message_variables(
    messages: dict[str, Any],
    variables: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Resolve ``{var}`` placeholders in message strings using localized variables."""
    resolved: dict[str, Any] = {}
    for msg_key, localized in messages.items():
        if not isinstance(localized, dict):
            resolved[msg_key] = localized
            continue
        resolved[msg_key] = {}
        for lang, lines in localized.items():
            lang_vars = {
                var_name: lang_values.get(lang, lang_values.get("en", ""))
                for var_name, lang_values in variables.items()
            }
            if isinstance(lines, list):
                resolved[msg_key][lang] = [
                    _substitute_variables(line, lang_vars) for line in lines
                ]
            else:
                resolved[msg_key][lang] = lines
    return resolved


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file, returning the raw dict."""
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
    return loaded


def _prepare_profile_payload(
    path: Path,
) -> tuple[DomainProfileFile, dict[str, dict[str, str]], Path, Path | None]:
    resolved_path = path.expanduser()
    loaded = _load_yaml_file(resolved_path)
    extends = loaded.pop("extends", None)
    variables: dict[str, dict[str, str]] = loaded.pop("variables", {})

    base_dir: Path | None = None
    if extends:
        base_path = (resolved_path.resolve().parent / extends).resolve()
        base_dir = base_path.parent
        base_loaded = _load_yaml_file(base_path)
        base_loaded.pop("extends", None)
        base_loaded.pop("variables", None)
        loaded = _deep_merge(base_loaded, loaded)

    if variables and "messages" in loaded:
        loaded["messages"] = _resolve_message_variables(loaded["messages"], variables)

    try:
        profile = DomainProfileFile.model_validate(loaded)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid domain profile configuration: {exc}") from exc
    return profile, variables, resolved_path.resolve().parent, base_dir


def _build_language_codes_and_names(
    profile: DomainProfileFile,
) -> tuple[tuple[str, ...], dict[str, str]]:
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
    return language_codes, language_names


def _build_quick_replies(
    profile: DomainProfileFile,
) -> dict[str, tuple[ChatQuickReply, ...]]:
    return {
        _normalize_language_code(language): tuple(
            ChatQuickReply(label=item.label, value=item.value)
            for item in localized_quick_replies
        )
        for language, localized_quick_replies in profile.quick_replies.items()
    }


def _build_term_replacements(
    profile: DomainProfileFile,
) -> dict[str, tuple[tuple[str, str], ...]]:
    return {
        _normalize_language_code(language): tuple(values.items())
        for language, values in profile.term_replacements.items()
    }


def _to_resolved_domain_profile(
    profile: DomainProfileFile,
    question_asker_prompt: str,
    questionnaire_notulist_prompt: str,
    unsupported_origin_follow_up_prompt: str,
    questionnaire_template: str,
    relevance_classifier_prompt: str,
    intake_route_classifier_prompt: str,
    handoff_classifier_prompt: str,
    off_topic_redirect_prompt: str,
    language_codes: tuple[str, ...],
    language_names: dict[str, str],
    quick_replies: dict[str, tuple[ChatQuickReply, ...]],
    term_replacements: dict[str, tuple[tuple[str, str], ...]],
) -> ResolvedDomainProfile:
    return ResolvedDomainProfile(
        question_asker_prompt=question_asker_prompt,
        questionnaire_notulist_prompt=questionnaire_notulist_prompt,
        unsupported_origin_follow_up_prompt=unsupported_origin_follow_up_prompt,
        questionnaire_template=questionnaire_template,
        relevance_classifier_prompt=relevance_classifier_prompt,
        intake_route_classifier_prompt=intake_route_classifier_prompt,
        handoff_classifier_prompt=handoff_classifier_prompt,
        off_topic_redirect_prompt=off_topic_redirect_prompt,
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
        completion_token=profile.behavior.completion_token,
        completion_mode=profile.behavior.completion_mode,
        off_topic_termination_threshold=profile.behavior.off_topic_termination_threshold,
        term_replacements=term_replacements,
    )


def load_domain_profile(path: Path) -> ResolvedDomainProfile:
    profile, variables, profile_dir, base_dir = _prepare_profile_payload(path)
    (
        question_asker_prompt,
        questionnaire_notulist_prompt,
        unsupported_origin_follow_up_prompt,
        questionnaire_template,
        relevance_classifier_prompt,
        intake_route_classifier_prompt,
        handoff_classifier_prompt,
        off_topic_redirect_prompt,
    ) = load_profile_prompts(
        profile=profile,
        profile_dir=profile_dir,
        base_dir=base_dir,
        load_prompt=load_prompt,
    )
    questionnaire_notulist_prompt = inject_questionnaire_template(
        questionnaire_notulist_prompt,
        questionnaire_template,
    )
    (
        relevance_classifier_prompt,
        intake_route_classifier_prompt,
        questionnaire_notulist_prompt,
        unsupported_origin_follow_up_prompt,
        handoff_classifier_prompt,
    ) = substitute_prompt_variables(
        variables=variables,
        relevance_classifier_prompt=relevance_classifier_prompt,
        intake_route_classifier_prompt=intake_route_classifier_prompt,
        questionnaire_notulist_prompt=questionnaire_notulist_prompt,
        unsupported_origin_follow_up_prompt=unsupported_origin_follow_up_prompt,
        handoff_classifier_prompt=handoff_classifier_prompt,
        substitute_variables=_substitute_variables,
    )
    language_codes, language_names = _build_language_codes_and_names(profile)
    quick_replies = _build_quick_replies(profile)
    term_replacements = _build_term_replacements(profile)
    return _to_resolved_domain_profile(
        profile=profile,
        question_asker_prompt=question_asker_prompt,
        questionnaire_notulist_prompt=questionnaire_notulist_prompt,
        unsupported_origin_follow_up_prompt=unsupported_origin_follow_up_prompt,
        questionnaire_template=questionnaire_template,
        relevance_classifier_prompt=relevance_classifier_prompt,
        intake_route_classifier_prompt=intake_route_classifier_prompt,
        handoff_classifier_prompt=handoff_classifier_prompt,
        off_topic_redirect_prompt=off_topic_redirect_prompt,
        language_codes=language_codes,
        language_names=language_names,
        quick_replies=quick_replies,
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
    advanced_azure_openai_timeout_seconds: float = Field(
        default=15.0, alias="ADVANCED_AZURE_OPENAI_TIMEOUT_SECONDS"
    )
    advanced_azure_openai_max_tokens: int = Field(
        default=450, alias="ADVANCED_AZURE_OPENAI_MAX_TOKENS"
    )
    brooklyn_domain_profile_path: Path = Field(
        default=DEFAULT_BROOKLYN_PROFILE_PATH,
        alias="BROOKLYN_DOMAIN_PROFILE_PATH",
    )
    cretskens_domain_profile_path: Path = Field(
        default=DEFAULT_CRETSKENS_DOMAIN_PROFILE_PATH,
        alias="CRETSKENS_DOMAIN_PROFILE_PATH",
    )
    cretskens_profile_path: Path = Field(
        default=DEFAULT_CRETSKENS_PROFILE_PATH,
        alias="CRETSKENS_PROFILE_PATH",
    )
    cretskens_wine_catalog_path: Path = Field(
        default=DEFAULT_WINE_CATALOG_PATH,
        alias="CRETSKENS_WINE_CATALOG_PATH",
    )
    wine_advisor_candidate_limit: int = Field(
        default=40, alias="WINE_ADVISOR_CANDIDATE_LIMIT"
    )
    wine_advisor_max_input_chars: int = Field(
        default=600, alias="WINE_ADVISOR_MAX_INPUT_CHARS"
    )
    wine_advisor_max_state_bytes: int = Field(
        default=12_000, alias="WINE_ADVISOR_MAX_STATE_BYTES"
    )
    wine_advisor_max_working_context_chars: int = Field(
        default=2_200, alias="WINE_ADVISOR_MAX_WORKING_CONTEXT_CHARS"
    )
    wine_advisor_max_history_items: int = Field(
        default=8, alias="WINE_ADVISOR_MAX_HISTORY_ITEMS"
    )
    wine_advisor_max_history_item_chars: int = Field(
        default=240, alias="WINE_ADVISOR_MAX_HISTORY_ITEM_CHARS"
    )
    audit_api_url: str | None = Field(default=None, alias="AUDIT_API_URL")
    audit_api_key: str | None = Field(default=None, alias="AUDIT_API_KEY")
    audit_profile: str | None = Field(default=None, alias="AUDIT_PROFILE")
    audit_timeout_seconds: float = Field(default=5.0, alias="AUDIT_TIMEOUT_SECONDS")
    state_signing_key: str = Field(default="", alias="STATE_SIGNING_KEY")
    state_session_ttl_seconds: int = Field(
        default=1800, alias="STATE_SESSION_TTL_SECONDS"
    )
    trusted_proxy_ips_raw: str = Field(default="", alias="TRUSTED_PROXY_IPS")

    @field_validator("brooklyn_domain_profile_path", mode="before")
    @classmethod
    def _normalize_blank_brooklyn_domain_profile_path(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return DEFAULT_BROOKLYN_PROFILE_PATH
        return value

    @field_validator("cretskens_domain_profile_path", mode="before")
    @classmethod
    def _normalize_blank_cretskens_domain_profile_path(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return DEFAULT_CRETSKENS_DOMAIN_PROFILE_PATH
        return value

    @field_validator("cretskens_profile_path", mode="before")
    @classmethod
    def _normalize_blank_cretskens_profile_path(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return DEFAULT_CRETSKENS_PROFILE_PATH
        return value

    @field_validator("cretskens_wine_catalog_path", mode="before")
    @classmethod
    def _normalize_blank_cretskens_wine_catalog_path(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return DEFAULT_WINE_CATALOG_PATH
        return value

    @property
    def brooklyn_domain_profile(self) -> ResolvedDomainProfile:
        return load_domain_profile(self.brooklyn_domain_profile_path)

    @property
    def cretskens_domain_profile(self) -> ResolvedDomainProfile:
        return load_domain_profile(self.cretskens_domain_profile_path)

    @property
    def cretskens_profile(self) -> ResolvedCretskensProfile:
        return load_cretskens_profile(self.cretskens_profile_path)

    @property
    def trusted_proxy_ips(self) -> frozenset[str]:
        return frozenset(
            item.strip().casefold()
            for item in self.trusted_proxy_ips_raw.split(",")
            if item.strip()
        )

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.dev"),
        env_file_encoding="utf-8",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
