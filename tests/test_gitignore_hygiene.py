# ruff: noqa: S101
from __future__ import annotations

from pathlib import Path

GITIGNORE_PATH = Path(".gitignore")


def _gitignore_lines() -> set[str]:
    return {
        line.strip()
        for line in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_gitignore_covers_local_workflow_artifacts() -> None:
    lines = _gitignore_lines()

    assert ".dual-graph/" in lines
    assert ".dual-graph-context/" in lines
    assert "docs/reports/*" in lines
    assert "!docs/reports/.gitkeep" in lines
