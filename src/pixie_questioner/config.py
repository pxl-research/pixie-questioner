"""Configuration and settings loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_prompt(path: Path) -> str:
    try:
        resolved_path = path.expanduser()
        return resolved_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Prompt file not found: {path}") from exc


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
    system_prompt_path: Path = Field(alias="SYSTEM_PROMPT_PATH")
    summarizer_prompt_path: Path | None = Field(
        default=None, alias="SUMMARIZER_PROMPT_PATH"
    )
    questionnaire_notulist_prompt_path: Path | None = Field(
        default=None, alias="QUESTIONNAIRE_NOTULIST_PROMPT_PATH"
    )
    questionnaire_template_path: Path = Field(alias="QUESTIONNAIRE_TEMPLATE_PATH")
    completion_token: str = Field(default="", alias="COMPLETION_TOKEN")
    audit_api_url: str | None = Field(default=None, alias="AUDIT_API_URL")
    audit_api_key: str | None = Field(default=None, alias="AUDIT_API_KEY")
    audit_profile: str | None = Field(default=None, alias="AUDIT_PROFILE")
    audit_timeout_seconds: float = Field(
        default=5.0, alias="AUDIT_TIMEOUT_SECONDS"
    )

    @property
    def system_prompt(self) -> str:
        return load_prompt(self.system_prompt_path)

    @property
    def summarizer_prompt(self) -> str | None:
        if self.summarizer_prompt_path is None:
            return None
        return load_prompt(self.summarizer_prompt_path)

    @property
    def questionnaire_notulist_prompt(self) -> str | None:
        if self.questionnaire_notulist_prompt_path is None:
            return None
        return load_prompt(self.questionnaire_notulist_prompt_path)

    @property
    def questionnaire_template(self) -> str:
        return load_prompt(self.questionnaire_template_path)

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.dev"),
        env_file_encoding="utf-8",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
