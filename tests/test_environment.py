"""
tests/test_environment.py — Unit tests for the Email Triage environment.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from envs.email_triage_env.environment import EmailTriageEnvironment
from envs.email_triage_env.models import EmailTriageAction, EmailTriageObservation, EmailTriageState
from envs.email_triage_env.tasks import TASKS, grade_easy, grade_medium, grade_hard


# ---------------------------------------------------------------------------
# Environment lifecycle tests
# ---------------------------------------------------------------------------


class TestEnvironmentLifecycle:
    def test_reset_returns_observation(self):
        env = EmailTriageEnvironment(task_name="easy")
        obs = env.reset()
        assert isinstance(obs, EmailTriageObservation)
        assert obs.done is False
        assert obs.reward is None
        assert obs.email_id != ""

    def test_state_after_reset(self):
        env = EmailTriageEnvironment(task_name="easy")
        env.reset()
        state = env.state
        assert isinstance(state, EmailTriageState)
        assert state.step_count == 0
        assert state.task_name == "easy"
        assert state.total_emails == 5

    def test_step_increments_count(self):
        env = EmailTriageEnvironment(task_name="easy")
        env.reset()
        action = EmailTriageAction(label="urgent", priority="high")
        obs = env.step(action)
        assert env.state.step_count == 1

    def test_episode_ends_after_all_emails(self):
        env = EmailTriageEnvironment(task_name="easy")
        env.reset()
        action = EmailTriageAction(label="archive", priority="low")
        obs = None
        for _ in range(5):
            obs = env.step(action)
        assert obs.done is True

    def test_reward_in_range(self):
        env = EmailTriageEnvironment(task_name="easy")
        env.reset()
        action = EmailTriageAction(label="urgent", priority="high", reply_draft="Acknowledged, working on it now.")
        obs = env.step(action)
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0

    def test_invalid_label_falls_back(self):
        env = EmailTriageEnvironment(task_name="easy")
        env.reset()
        action = EmailTriageAction(label="NONSENSE_LABEL", priority="low")
        obs = env.step(action)  # should not raise
        assert obs is not None

    def test_all_tasks_load(self):
        for task_name in ["easy", "medium", "hard"]:
            env = EmailTriageEnvironment(task_name=task_name)
            obs = env.reset()
            assert obs.email_id != ""


# ---------------------------------------------------------------------------
# Grader tests
# ---------------------------------------------------------------------------


class TestGraders:
    def _email(self, task_name="easy", idx=0):
        return TASKS[task_name].emails[idx]

    def test_perfect_action_scores_high(self):
        email = self._email("easy", 0)  # urgent, high, requires_reply
        action = EmailTriageAction(
            label=email.correct_label,
            priority=email.correct_priority,
            reply_draft="I'm on it immediately, escalating now.",
        )
        score = grade_easy(email, action)
        assert score >= 0.9

    def test_wrong_label_loses_points(self):
        email = self._email("easy", 0)  # correct: urgent
        action_wrong = EmailTriageAction(label="spam", priority=email.correct_priority)
        action_right = EmailTriageAction(label=email.correct_label, priority=email.correct_priority)
        assert grade_easy(email, action_wrong) < grade_easy(email, action_right)

    def test_spam_email_no_reply_needed(self):
        email = self._email("easy", 1)  # spam, no reply
        action = EmailTriageAction(label="spam", priority="low", reply_draft=None)
        score = grade_easy(email, action)
        assert score == 1.0

    def test_score_range(self):
        for task_name, task in TASKS.items():
            for email in task.emails:
                action = EmailTriageAction(label="archive", priority="low")
                score = task.grader(email, action)
                assert 0.0 <= score <= 1.0, f"Out of range for {task_name}/{email.email_id}"

    def test_hard_grader_penalises_wrong_label_more(self):
        email = TASKS["hard"].emails[0]  # urgent
        action = EmailTriageAction(label="archive", priority=email.correct_priority)
        hard_score = grade_hard(email, action)
        easy_email = TASKS["easy"].emails[0]
        easy_action = EmailTriageAction(label="archive", priority=easy_email.correct_priority)
        easy_score = grade_easy(easy_email, easy_action)
        # Hard penalises wrong label more → hard score should be <= easy score
        # (both have wrong label but hard applies extra -0.15)
        assert hard_score <= easy_score


# ---------------------------------------------------------------------------
# Task registry tests
# ---------------------------------------------------------------------------


class TestTaskRegistry:
    def test_three_tasks_registered(self):
        assert set(TASKS.keys()) == {"easy", "medium", "hard"}

    def test_each_task_has_5_emails(self):
        for task_name, task in TASKS.items():
            assert len(task.emails) == 5, f"{task_name} should have 5 emails"

    def test_email_ids_unique_within_task(self):
        for task_name, task in TASKS.items():
            ids = [e.email_id for e in task.emails]
            assert len(ids) == len(set(ids)), f"Duplicate email IDs in {task_name}"

    def test_correct_labels_valid(self):
        from envs.email_triage_env.models import VALID_LABELS, VALID_PRIORITIES
        for task_name, task in TASKS.items():
            for email in task.emails:
                assert email.correct_label in VALID_LABELS
                assert email.correct_priority in VALID_PRIORITIES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])