# ruff: noqa: S101
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOW_PATH = Path(".github/workflows/copilot-handle-pr-reviews.yml")


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
    trigger = cast(dict[str, Any], jobs["trigger-codex"])
    steps = cast(list[dict[str, Any]], trigger["steps"])
    return cast(str, steps[0]["with"]["script"])


def test_review_workflow_listens_for_supported_review_events() -> None:
    on_config = _workflow_on()

    assert on_config["pull_request_review"]["types"] == ["submitted"]
    assert on_config["pull_request_review_comment"]["types"] == [
        "created",
        "edited",
        "deleted",
    ]
    assert on_config["issue_comment"]["types"] == ["created"]


def test_review_workflow_skips_closed_or_merged_prs() -> None:
    script = _workflow_script()

    assert 'if (!pr || pr.state !== "open")' in script
    assert (
        "github.event.issue.pull_request != null"
        in _load_workflow()["jobs"]["trigger-codex"]["if"]
    )


def test_review_workflow_only_enables_auto_merge_when_threads_are_resolved() -> None:
    script = _workflow_script()

    assert "unresolvedThreads.length === 0" in script
    assert "enablePullRequestAutoMerge" in script
    assert "chatgpt-codex-connector[bot]" in script
    assert "<!-- codex-review-trigger -->" in script
    assert "@codex A reviewer" in script
