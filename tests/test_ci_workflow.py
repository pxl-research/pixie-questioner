# ruff: noqa: S101
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOW_PATH = Path(".github/workflows/ci.yml")


def _load_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, Any], loaded)


def _workflow_on() -> dict[str, Any]:
    workflow = cast(dict[Any, Any], _load_workflow())
    on_config = workflow["on"] if "on" in workflow else workflow[True]
    assert isinstance(on_config, dict)
    return cast(dict[str, Any], on_config)


def test_ci_workflow_runs_on_pull_requests_and_main_pushes_only() -> None:
    on_config = _workflow_on()

    assert "pull_request" in on_config
    assert on_config["push"]["branches"] == ["main"]
