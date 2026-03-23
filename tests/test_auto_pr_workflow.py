# ruff: noqa: S101
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOW_PATH = Path(".github/workflows/auto-pr.yml")


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
    jobs = workflow["jobs"]
    ensure_pr = jobs["ensure-pr"]
    steps = ensure_pr["steps"]
    return steps[0]["with"]["script"]


def _workflow_step() -> dict[str, Any]:
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    ensure_pr = jobs["ensure-pr"]
    return ensure_pr["steps"][0]


def _workflow_job_env() -> dict[str, Any]:
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    ensure_pr = jobs["ensure-pr"]
    env = ensure_pr["env"]
    assert isinstance(env, dict)
    return cast(dict[str, Any], env)


def _branch_pattern(script: str) -> re.Pattern[str]:
    match = re.search(r"branch\.match\(/(.+?)/\);", script)
    assert match is not None
    return re.compile(match.group(1))


def test_auto_pr_workflow_accepts_current_branch_prefixes() -> None:
    branches = _workflow_on()["push"]["branches"]
    assert "feat/**" in branches

    pattern = _branch_pattern(_workflow_script())

    assert pattern.fullmatch("feat/llm-driven-wine-advisor")
    assert pattern.fullmatch("chore/add-auto-pr-workflow")
    assert pattern.fullmatch("feature/123-runtime-separation")


def test_auto_pr_workflow_only_recovers_duplicate_project_adds() -> None:
    script = _workflow_script()

    assert "already exists" in script
    assert "items(first:" in script
    assert (
        "Skipping field updates because item already exists or add failed."
        not in script
    )


def test_auto_pr_workflow_uses_pat_for_project_automation() -> None:
    workflow = _load_workflow()
    permissions = workflow["permissions"]
    step = _workflow_step()

    assert "projects" not in permissions
    assert step["with"]["github-token"] == "${{ secrets.PROJECT_AUTOMATION_TOKEN }}"


def test_auto_pr_workflow_does_not_enable_auto_merge_eagerly() -> None:
    script = _workflow_script()

    assert "enablePullRequestAutoMerge" not in script


def test_auto_pr_workflow_uses_env_driven_project_configuration() -> None:
    env = _workflow_job_env()
    script = _workflow_script()

    assert "PROJECT_V2_ID" in env
    assert "PROJECT_FIELD_STATUS_ID" in env
    assert "PROJECT_FIELD_WORK_TYPE_ID" in env
    assert "PROJECT_FIELD_SHIPPED_ID" in env
    assert "requiredEnv('PROJECT_V2_ID')" in script
    assert "requiredEnv('PROJECT_STATUS_OPTION_REVIEW_ID')" in script
