# ruff: noqa: S101
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOW_PATH = Path(".github/workflows/issue-intake.yml")


def _load_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, Any], loaded)


def _workflow_on() -> dict[str, Any]:
    workflow = cast(dict[Any, Any], _load_workflow())
    on_config = workflow["on"] if "on" in workflow else workflow[True]
    assert isinstance(on_config, dict)
    return cast(dict[str, Any], on_config)


def _workflow_script() -> str:
    workflow = _load_workflow()
    jobs = cast(dict[str, Any], workflow["jobs"])
    handle = cast(dict[str, Any], jobs["handle-intake"])
    steps = cast(list[dict[str, Any]], handle["steps"])
    return cast(str, steps[0]["with"]["script"])


def test_issue_intake_workflow_triggers_on_issue_open_and_comment() -> None:
    on_config = _workflow_on()

    issues_types = on_config["issues"]["types"]
    comment_types = on_config["issue_comment"]["types"]

    assert "opened" in issues_types
    assert "reopened" in issues_types
    assert "created" in comment_types


def test_issue_intake_workflow_posts_questions_and_creates_ticket() -> None:
    script = _workflow_script()

    assert "<!-- intake-questions -->" in script
    assert "/create-ticket" in script
    assert "github.rest.issues.create" in script
    assert "codex:ready" in script


def test_issue_intake_workflow_uses_project_pat() -> None:
    workflow = _load_workflow()
    jobs = cast(dict[str, Any], workflow["jobs"])
    handle = cast(dict[str, Any], jobs["handle-intake"])
    steps = cast(list[dict[str, Any]], handle["steps"])

    assert steps[0]["with"]["github-token"] == "${{ secrets.PROJECT_AUTOMATION_TOKEN }}"
