"""Configuration and settings loading."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SYSTEM_PROMPT = (
    "You are Pixie Questioner, a concise assistant that answers clearly and "
    "directly."
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
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.dev"),
        env_file_encoding="utf-8",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
