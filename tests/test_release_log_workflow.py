# ruff: noqa: S101
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOW_PATH = Path(".github/workflows/release-log.yml")


def _load_workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, Any], loaded)


def _workflow_step() -> dict[str, Any]:
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    release_job = jobs["release"]
    return release_job["steps"][0]


def _workflow_script() -> str:
    step = _workflow_step()
    return cast(str, step["with"]["script"])


def test_release_log_workflow_uses_pat_for_project_updates() -> None:
    step = _workflow_step()

    assert step["with"]["github-token"] == "${{ secrets.PROJECT_AUTOMATION_TOKEN }}"


def test_release_log_workflow_stops_after_release_creation_failure() -> None:
    script = _workflow_script()
    catch_block = re.search(r"\} catch \(e\) \{(?P<body>.*?)\n\s+\}", script, re.DOTALL)

    assert catch_block is not None

    catch_body = catch_block.group("body")
    assert "core.setFailed(" in catch_body
    assert "return;" in catch_body
    assert "core.warning(" not in catch_body


def test_release_log_workflow_seeds_versions_consistently() -> None:
    script = _workflow_script()

    assert "let highest = { major: 0, minor: 0, patch: 0, tag: 'v0.0.0' };" in script
