# ruff: noqa: S101
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOW_PATH = Path(".github/workflows/issue-to-agent.yml")


def _load_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, Any], loaded)


def _workflow_script() -> str:
    workflow = _load_workflow()
    jobs = cast(dict[str, Any], workflow["jobs"])
    bootstrap = cast(dict[str, Any], jobs["bootstrap-pr"])
    steps = cast(list[dict[str, Any]], bootstrap["steps"])
    return cast(str, steps[0]["with"]["script"])


def test_issue_to_agent_workflow_triggers_on_codex_ready_label() -> None:
    workflow = cast(dict[Any, Any], _load_workflow())
    on_config = workflow["on"] if "on" in workflow else workflow[True]
    assert isinstance(on_config, dict)

    assert on_config["issues"]["types"] == ["labeled"]


def test_issue_to_agent_workflow_contains_codex_handoff_flow() -> None:
    script = _workflow_script()

    assert "label.name !== 'codex:ready'" in script
    assert "git commit --allow-empty" in script
    assert "github.rest.pulls.create" in script
    assert "@codex" in script
    assert "codex:in-progress" in script


def test_issue_to_agent_workflow_uses_project_pat() -> None:
    workflow = _load_workflow()
    jobs = cast(dict[str, Any], workflow["jobs"])
    bootstrap = cast(dict[str, Any], jobs["bootstrap-pr"])
    steps = cast(list[dict[str, Any]], bootstrap["steps"])

    assert steps[0]["with"]["github-token"] == "${{ secrets.PROJECT_AUTOMATION_TOKEN }}"
