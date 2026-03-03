"""Domain policy models for chat behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ChatQuickReply:
    label: str
    value: str


@dataclass(frozen=True)
class ChatDomainPolicy:
    language_codes: tuple[str, ...]
    language_names: dict[str, str]
    handoff_lines: dict[str, tuple[str, ...]]
    feedback_questions: dict[str, tuple[str, ...]]
    feedback_thanks: dict[str, tuple[str, ...]]
    goodbye_lines: dict[str, tuple[str, ...]]
    off_topic_redirects: dict[str, tuple[str, ...]]
    off_topic_initial_questions: dict[str, tuple[str, ...]]
    quick_replies: dict[str, tuple[ChatQuickReply, ...]]
    feedback_yes_tokens: frozenset[str]
    feedback_no_tokens: frozenset[str]
    completion_token: str
    completion_mode: Literal["summary_feedback", "categorized_handoff"]
    off_topic_termination_threshold: int
    term_replacements: dict[str, tuple[tuple[str, str], ...]]
