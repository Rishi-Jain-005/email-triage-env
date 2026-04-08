"""
models.py — Typed contracts for the Email Triage Environment.

Action      : The agent's decision for an email.
Observation : What the agent sees after each step.
State       : Episode-level metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Base stubs (mirrors openenv-core ABC surface so the file works standalone)
# ---------------------------------------------------------------------------
class Action:
    pass


class Observation:
    pass


class State:
    pass


# ---------------------------------------------------------------------------
# Email Triage domain types
# ---------------------------------------------------------------------------

VALID_LABELS = {"urgent", "follow_up", "archive", "spam", "delegate"}

VALID_PRIORITIES = {"high", "medium", "low"}


@dataclass
class EmailTriageAction(Action):
    """
    The agent's decision for the current email.

    Fields
    ------
    label       : One of 'urgent' | 'follow_up' | 'archive' | 'spam' | 'delegate'
    priority    : One of 'high' | 'medium' | 'low'
    reply_draft : Optional short reply the agent wants to draft (≤ 500 chars).
                  Leave None / empty if no reply is needed.
    reasoning   : Optional free-text explanation (not scored, helps debugging).
    """

    label: str = "archive"
    priority: str = "low"
    reply_draft: Optional[str] = None
    reasoning: Optional[str] = None

    def __post_init__(self) -> None:
        self.label = self.label.lower().strip()
        self.priority = self.priority.lower().strip()


@dataclass
class EmailTriageObservation(Observation):
    """
    What the agent receives after reset() or step().

    Fields
    ------
    done            : Whether the episode has ended.
    reward          : Reward for the last action (None after reset).
    email_id        : Unique identifier for the current email.
    subject         : Email subject line.
    sender          : Sender address / name.
    body            : Email body text (may be truncated).
    thread_length   : Number of prior messages in the thread.
    has_attachment  : Whether the email has attachments.
    emails_remaining: How many emails are left in this episode.
    feedback        : Human-readable feedback on the last action.
    task_description: Description of what the agent must accomplish.
    """

    done: bool = False
    reward: Optional[float] = None

    email_id: str = ""
    subject: str = ""
    sender: str = ""
    body: str = ""
    thread_length: int = 0
    has_attachment: bool = False
    emails_remaining: int = 0
    feedback: str = ""
    task_description: str = (
        "Triage each email: assign a label (urgent/follow_up/archive/spam/delegate) "
        "and a priority (high/medium/low). Optionally draft a reply."
    )


@dataclass
class EmailTriageState(State):
    """
    Episode-level metadata.

    Fields
    ------
    episode_id      : Unique episode UUID.
    step_count      : Number of steps taken so far.
    task_name       : Name of the active task (easy / medium / hard).
    total_emails    : Total emails in this episode.
    cumulative_score: Running sum of normalised rewards [0–1] per step.
    max_steps       : Maximum allowed steps.
    """

    episode_id: Optional[str] = None
    step_count: int = 0
    task_name: str = ""
    total_emails: int = 0
    cumulative_score: float = 0.0
    max_steps: int = 20