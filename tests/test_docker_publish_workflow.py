# ruff: noqa: S101
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOW_PATH = Path(".github/workflows/docker-publish.yml")


def _load_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, Any], loaded)


def _workflow_on() -> dict[str, Any]:
    workflow = cast(dict[Any, Any], _load_workflow())
    on_config = workflow["on"] if "on" in workflow else workflow[True]
    assert isinstance(on_config, dict)
    return cast(dict[str, Any], on_config)


def test_docker_publish_runs_on_main_and_manual_dispatch() -> None:
    on_config = _workflow_on()

    assert on_config["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in on_config
    assert "ref" in on_config["workflow_dispatch"]["inputs"]
