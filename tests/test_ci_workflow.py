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


def _workflow_steps() -> list[dict[str, Any]]:
    workflow = cast(dict[Any, Any], _load_workflow())
    jobs = cast(dict[str, Any], workflow["jobs"])
    validate_job = cast(dict[str, Any], jobs["validate"])
    steps = validate_job["steps"]
    assert isinstance(steps, list)
    return cast(list[dict[str, Any]], steps)


def _step_by_name(step_name: str) -> dict[str, Any]:
    for step in _workflow_steps():
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Step not found: {step_name}")


def test_ci_workflow_runs_on_pull_requests_and_main_pushes_only() -> None:
    on_config = _workflow_on()

    assert "pull_request" in on_config
    assert on_config["push"]["branches"] == ["main"]


def test_ci_workflow_performs_api_health_check_lifecycle() -> None:
    step_names = [step.get("name", "") for step in _workflow_steps()]

    assert "Start API for health check" in step_names
    assert "Wait for /health endpoint" in step_names
    assert "Stop API health-check container" in step_names
    assert _step_by_name("Stop API health-check container")["if"] == "always()"


def test_ci_workflow_runs_live_enabled_tests_for_pr_and_main() -> None:
    step = _step_by_name("Run live-enabled test suite")

    assert (
        step["if"]
        == "github.event_name == 'pull_request' || github.ref == 'refs/heads/main'"
    )
    assert step["env"]["RUN_LIVE_LLM_TESTS"] == "1"
