# ruff: noqa: S101
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

TEMPLATE_PATH = Path(".github/ISSUE_TEMPLATE/raw-intake.yml")


def _load_template() -> dict[str, Any]:
    loaded = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, Any], loaded)


def test_raw_intake_template_labels_issue_for_automation() -> None:
    template = _load_template()
    labels = cast(list[str], template["labels"])

    assert "intake" in labels


def test_raw_intake_template_contains_required_problem_field() -> None:
    template = _load_template()
    body = cast(list[dict[str, Any]], template["body"])

    problem_field = next(item for item in body if item.get("id") == "problem_raw")
    assert cast(dict[str, Any], problem_field["validations"])["required"] is True
