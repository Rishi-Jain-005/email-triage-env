"""
client.py — EmailTriageEnv client.

Usage
-----
    from envs.email_triage_env import EmailTriageEnv, EmailTriageAction

    with EmailTriageEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        obs = result.observation

        action = EmailTriageAction(label="urgent", priority="high")
        result = env.step(action)
        print(result.reward)

The client translates between typed Python models and the JSON wire format.
It supports both a remote server and a local (in-process) mode for testing.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict
from typing import Generator, Optional

import requests

from envs.email_triage_env.models import (
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
)


# ---------------------------------------------------------------------------
# StepResult wrapper
# ---------------------------------------------------------------------------


class StepResult:
    """Mirrors openenv-core StepResult so training code is consistent."""

    def __init__(
        self,
        observation: EmailTriageObservation,
        reward: float,
        done: bool,
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done

    def __repr__(self) -> str:
        return (
            f"StepResult(reward={self.reward:.3f}, done={self.done}, "
            f"email_id='{self.observation.email_id}')"
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class EmailTriageEnv:
    """
    Client for the Email Triage OpenEnv environment.

    Parameters
    ----------
    base_url : str
        URL of the running environment server,
        e.g. "http://localhost:8000" or "https://your-space.hf.space".
    task_name : str
        One of 'easy' | 'medium' | 'hard'.  Sent as a query param to the server.
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        task_name: str = "easy",
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._task_name = task_name
        self._timeout = timeout
        self._session: Optional[requests.Session] = None

    # ------------------------------------------------------------------
    # Context-manager helpers (mirrors openenv-core .sync() pattern)
    # ------------------------------------------------------------------

    def sync(self) -> "EmailTriageEnv":
        """Return self so you can use `with env.sync() as env:`."""
        return self

    def __enter__(self) -> "EmailTriageEnv":
        self._session = requests.Session()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if self._session:
            self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> StepResult:
        payload = {"task_name": self._task_name}
        data = self._post("/reset", payload)
        obs = self._parse_observation(data)
        return StepResult(observation=obs, reward=0.0, done=obs.done)

    def step(self, action: EmailTriageAction) -> StepResult:
        payload = {
            "label": action.label,
            "priority": action.priority,
            "reply_draft": action.reply_draft or "",
            "reasoning": action.reasoning or "",
        }
        data = self._post("/step", payload)
        obs = self._parse_observation(data)
        reward = data.get("reward") or 0.0
        return StepResult(observation=obs, reward=float(reward), done=obs.done)

    def state(self) -> EmailTriageState:
        data = self._get("/state")
        return EmailTriageState(
            episode_id=data.get("episode_id"),
            step_count=data.get("step_count", 0),
            task_name=data.get("task_name", ""),
            total_emails=data.get("total_emails", 0),
            cumulative_score=data.get("cumulative_score", 0.0),
            max_steps=data.get("max_steps", 20),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        session = self._session or requests.Session()
        resp = session.post(
            f"{self._base_url}{path}",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict:
        session = self._session or requests.Session()
        resp = session.get(
            f"{self._base_url}{path}",
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_observation(data: dict) -> EmailTriageObservation:
        return EmailTriageObservation(
            done=data.get("done", False),
            reward=data.get("reward"),
            email_id=data.get("email_id", ""),
            subject=data.get("subject", ""),
            sender=data.get("sender", ""),
            body=data.get("body", ""),
            thread_length=data.get("thread_length", 0),
            has_attachment=data.get("has_attachment", False),
            emails_remaining=data.get("emails_remaining", 0),
            feedback=data.get("feedback", ""),
            task_description=data.get("task_description", ""),
        )