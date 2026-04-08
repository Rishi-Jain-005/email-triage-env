"""
environment.py — EmailTriage server-side environment logic.

Implements the OpenEnv ABC:
    reset()  → EmailTriageObservation
    step()   → EmailTriageObservation
    state    → EmailTriageState  (property)
"""

from __future__ import annotations

import uuid
from typing import Optional

from envs.email_triage_env.models import (
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    VALID_LABELS,
    VALID_PRIORITIES,
)
from envs.email_triage_env.tasks import TASKS, Task, Email


class EmailTriageEnvironment:
    """
    Server-side environment for the Email Triage task.

    One episode = one full inbox (list of emails for the chosen task).
    The agent calls step() once per email.  The episode ends when all
    emails have been triaged OR max_steps is reached.
    """

    def __init__(self, task_name: str = "easy") -> None:
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}"
            )
        self._task: Task = TASKS[task_name]
        self._state: EmailTriageState = EmailTriageState()
        self._email_queue: list[Email] = []
        self._current_email: Optional[Email] = None
        self._email_index: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> EmailTriageObservation:
        """Start a fresh episode — return the first email."""
        self._email_queue = list(self._task.emails)  # fresh copy
        self._email_index = 0
        self._current_email = self._email_queue[0]

        self._state = EmailTriageState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_name=self._task.name,
            total_emails=len(self._email_queue),
            cumulative_score=0.0,
            max_steps=len(self._email_queue) + 2,  # slight buffer
        )

        return self._build_observation(
            done=False,
            reward=None,
            feedback=f"New episode started. Task: {self._task.description}",
        )

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        """Process the agent's decision for the current email."""
        if self._current_email is None:
            return self._build_observation(
                done=True,
                reward=0.0,
                feedback="Episode has already ended. Call reset() to start a new one.",
            )

        # Validate action fields — graceful fallback instead of crash
        action = self._sanitize_action(action)

        # Grade the action
        reward = self._task.grader(self._current_email, action)
        self._state.cumulative_score += reward
        self._state.step_count += 1

        # Build feedback string
        feedback = self._build_feedback(action, reward)

        # Advance to next email
        self._email_index += 1
        done = self._email_index >= len(self._email_queue)

        if not done:
            self._current_email = self._email_queue[self._email_index]
        else:
            self._current_email = None

        return self._build_observation(done=done, reward=reward, feedback=feedback)

    @property
    def state(self) -> EmailTriageState:
        return self._state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sanitize_action(self, action: EmailTriageAction) -> EmailTriageAction:
        """Ensure action fields are valid; apply defaults if not."""
        label = action.label if action.label in VALID_LABELS else "archive"
        priority = action.priority if action.priority in VALID_PRIORITIES else "low"
        reply_draft = action.reply_draft
        if reply_draft and len(reply_draft) > 500:
            reply_draft = reply_draft[:500]
        return EmailTriageAction(
            label=label,
            priority=priority,
            reply_draft=reply_draft,
            reasoning=action.reasoning,
        )

    def _build_observation(
        self,
        done: bool,
        reward: Optional[float],
        feedback: str,
    ) -> EmailTriageObservation:
        if self._current_email is not None:
            email = self._current_email
            emails_remaining = len(self._email_queue) - self._email_index
        else:
            # Episode over — use empty sentinel
            return EmailTriageObservation(
                done=True,
                reward=reward,
                email_id="",
                subject="",
                sender="",
                body="",
                thread_length=0,
                has_attachment=False,
                emails_remaining=0,
                feedback=feedback,
                task_description=self._task.description,
            )

        return EmailTriageObservation(
            done=done,
            reward=reward,
            email_id=email.email_id,
            subject=email.subject,
            sender=email.sender,
            body=email.body,
            thread_length=email.thread_length,
            has_attachment=email.has_attachment,
            emails_remaining=emails_remaining,
            feedback=feedback,
            task_description=self._task.description,
        )

    def _build_feedback(self, action: EmailTriageAction, reward: float) -> str:
        email = self._current_email
        if email is None:
            return ""
        parts = [f"Reward: {reward:.2f}/1.00"]
        if action.label != email.correct_label:
            parts.append(
                f"Label '{action.label}' incorrect (expected '{email.correct_label}')"
            )
        else:
            parts.append(f"Label '{action.label}' correct ✓")
        if action.priority != email.correct_priority:
            parts.append(
                f"Priority '{action.priority}' incorrect (expected '{email.correct_priority}')"
            )
        else:
            parts.append(f"Priority '{action.priority}' correct ✓")
        if email.requires_reply and (
            not action.reply_draft or len(action.reply_draft.strip()) < 10
        ):
            parts.append("Reply was required but not provided ✗")
        return " | ".join(parts)