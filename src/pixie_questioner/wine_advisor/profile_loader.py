"""Load Cretskens advisor profile assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


@dataclass(frozen=True)
class ResolvedCretskensProfile:
    turn_controller_prompt: str
    ranker_prompt: str
    recommendation_intro: str


class _CretskensPromptPaths(BaseModel):
    turn_controller_path: str = Field(..., min_length=1)
    ranker_path: str = Field(..., min_length=1)


class _CretskensProfileFile(BaseModel):
    version: Literal[1]
    prompts: _CretskensPromptPaths
    recommendation_intro: str = Field(..., min_length=1)


def _load_yaml_file(path: Path) -> dict[str, Any]:
    resolved_path = path.expanduser()
    try:
        raw_yaml = resolved_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Cretskens profile file not found: {path}") from exc

    try:
        loaded = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid Cretskens profile YAML: {path}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Invalid Cretskens profile YAML: {path}")
    return loaded


def _read_prompt_file(profile_dir: Path, relative_path: str) -> str:
    prompt_path = (profile_dir / relative_path).resolve()
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Cretskens prompt file not found: {prompt_path}") from exc


def load_cretskens_profile(path: Path) -> ResolvedCretskensProfile:
    resolved_path = path.expanduser().resolve()
    loaded = _load_yaml_file(resolved_path)
    try:
        profile = _CretskensProfileFile.model_validate(loaded)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid Cretskens profile configuration: {exc}") from exc

    profile_dir = resolved_path.parent
    return ResolvedCretskensProfile(
        turn_controller_prompt=_read_prompt_file(
            profile_dir,
            profile.prompts.turn_controller_path,
        ),
        ranker_prompt=_read_prompt_file(
            profile_dir,
            profile.prompts.ranker_path,
        ),
        recommendation_intro=profile.recommendation_intro,
    )
