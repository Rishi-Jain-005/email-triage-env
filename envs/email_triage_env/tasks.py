"""
tasks.py — Task definitions, email datasets, and graders.

Each task defines:
    - A list of Email objects (the inbox).
    - A grader function:  grade(action) -> float in [0.0, 1.0]
    - A difficulty label: 'easy' | 'medium' | 'hard'

Graders are deterministic and programmatic (no LLM calls).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .models import EmailTriageAction


# ---------------------------------------------------------------------------
# Raw email data structure
# ---------------------------------------------------------------------------


@dataclass
class Email:
    email_id: str
    subject: str
    sender: str
    body: str
    thread_length: int = 1
    has_attachment: bool = False

    # Ground-truth labels used by graders
    correct_label: str = "archive"
    correct_priority: str = "low"
    requires_reply: bool = False  # True → agent should provide reply_draft


# ---------------------------------------------------------------------------
# Helper: partial-credit scorer
# ---------------------------------------------------------------------------


def _score_action(email: Email, action: EmailTriageAction) -> float:
    """
    Returns a float in [0.0, 1.0]:
      0.5 for correct label
      0.3 for correct priority
      0.2 for reply draft when required (or no penalty when not required)
    """
    score = 0.0

    # Label (50 % weight)
    if action.label == email.correct_label:
        score += 0.5

    # Priority (30 % weight)
    if action.priority == email.correct_priority:
        score += 0.3

    # Reply (20 % weight)
    if email.requires_reply:
        if action.reply_draft and len(action.reply_draft.strip()) >= 10:
            score += 0.2
        # else: no credit
    else:
        # Penalise drafting a reply when none is needed (−0.1)
        if action.reply_draft and len(action.reply_draft.strip()) >= 10:
            score = max(0.0, score - 0.1)
        else:
            score += 0.2  # full credit for correctly not replying

    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# TASK 1 — EASY
# Clearly labelled, unambiguous emails. Single-step per email.
# ---------------------------------------------------------------------------

EASY_EMAILS: List[Email] = [
    Email(
        email_id="easy_001",
        subject="URGENT: Server down in production",
        sender="ops-team@company.com",
        body=(
            "Hi,\n\nOur production server (prod-us-east-1) has been unreachable "
            "since 14:32 UTC. Revenue impact is approximately $5k/minute. "
            "Please escalate immediately.\n\nOps Team"
        ),
        thread_length=1,
        has_attachment=False,
        correct_label="urgent",
        correct_priority="high",
        requires_reply=True,
    ),
    Email(
        email_id="easy_002",
        subject="Newsletter: Top 10 productivity hacks",
        sender="noreply@marketingblast.com",
        body=(
            "Hi there!\n\nYou're receiving this because you signed up for our "
            "weekly productivity tips. This week: top 10 hacks to boost output. "
            "Click here to read more.\n\nUnsubscribe | View in browser"
        ),
        thread_length=1,
        has_attachment=False,
        correct_label="spam",
        correct_priority="low",
        requires_reply=False,
    ),
    Email(
        email_id="easy_003",
        subject="Q3 Report — please review by Friday",
        sender="manager@company.com",
        body=(
            "Hi,\n\nAttached is the Q3 financial report. Please review and send "
            "your comments by end of day Friday. Let me know if you have questions.\n\n"
            "Thanks,\nManager"
        ),
        thread_length=1,
        has_attachment=True,
        correct_label="follow_up",
        correct_priority="medium",
        requires_reply=True,
    ),
    Email(
        email_id="easy_004",
        subject="Meeting notes from yesterday",
        sender="colleague@company.com",
        body=(
            "Hey,\n\nJust sharing the notes from yesterday's standup. "
            "Nothing requires action from you — FYI only.\n\nCheers"
        ),
        thread_length=2,
        has_attachment=False,
        correct_label="archive",
        correct_priority="low",
        requires_reply=False,
    ),
    Email(
        email_id="easy_005",
        subject="Congratulations! You've won a prize",
        sender="winner@lucky-draw-2025.net",
        body=(
            "Dear Winner,\n\nYou have been selected to receive a $1,000 gift card. "
            "Click the link below to claim your reward within 24 hours. "
            "Provide your credit card details to cover shipping.\n\nClaim now!"
        ),
        thread_length=1,
        has_attachment=False,
        correct_label="spam",
        correct_priority="low",
        requires_reply=False,
    ),
]


def grade_easy(email: Email, action: EmailTriageAction) -> float:
    return _score_action(email, action)


# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM
# Ambiguous subjects, multi-thread context, mixed signals.
# ---------------------------------------------------------------------------

MEDIUM_EMAILS: List[Email] = [
    Email(
        email_id="med_001",
        subject="Re: Re: Re: Budget approval",
        sender="cfo@company.com",
        body=(
            "Following up — we need a decision on the $50k budget line today "
            "or the vendor will cancel. Can you approve or escalate?\n\n"
            "--- Previous ---\nAwaiting sign-off from your team per last week's call."
        ),
        thread_length=4,
        has_attachment=True,
        correct_label="urgent",
        correct_priority="high",
        requires_reply=True,
    ),
    Email(
        email_id="med_002",
        subject="Checking in",
        sender="client@bigcorp.com",
        body=(
            "Hi,\n\nJust wanted to see how the project is coming along. "
            "No rush, but let me know if there's anything you need from our side.\n\nBest"
        ),
        thread_length=3,
        has_attachment=False,
        correct_label="follow_up",
        correct_priority="medium",
        requires_reply=True,
    ),
    Email(
        email_id="med_003",
        subject="Offsite planning — venue options",
        sender="hr@company.com",
        body=(
            "Team,\n\nWe're planning the Q4 offsite. Please vote on the three "
            "venue options in the attached form by next Wednesday. "
            "This is not urgent but we need responses.\n\nHR"
        ),
        thread_length=1,
        has_attachment=True,
        correct_label="follow_up",
        correct_priority="low",
        requires_reply=False,
    ),
    Email(
        email_id="med_004",
        subject="Invoice #INV-2094 overdue",
        sender="billing@supplier.com",
        body=(
            "Dear Customer,\n\nInvoice #INV-2094 for $3,200 is now 30 days overdue. "
            "Please arrange payment or contact us to discuss. "
            "Late fees may apply after 45 days.\n\nAccounts Team"
        ),
        thread_length=2,
        has_attachment=True,
        correct_label="delegate",
        correct_priority="medium",
        requires_reply=True,
    ),
    Email(
        email_id="med_005",
        subject="Re: Team lunch tomorrow",
        sender="teammate@company.com",
        body=(
            "Hey! Are you coming to the team lunch tomorrow at noon? "
            "We're going to the Italian place on 5th. Let me know so I can book.\n\nThanks!"
        ),
        thread_length=2,
        has_attachment=False,
        correct_label="follow_up",
        correct_priority="low",
        requires_reply=True,
    ),
]


def grade_medium(email: Email, action: EmailTriageAction) -> float:
    base = _score_action(email, action)
    # Medium task: slight penalty for wrong label (label is trickier)
    if action.label != email.correct_label:
        base = max(0.0, base - 0.1)
    return round(base, 4)


# ---------------------------------------------------------------------------
# TASK 3 — HARD
# Complex multi-signal, requires reasoning about tone/urgency/delegation.
# ---------------------------------------------------------------------------

HARD_EMAILS: List[Email] = [
    Email(
        email_id="hard_001",
        subject="Re: Data breach — legal hold",
        sender="legal@company.com",
        body=(
            "This email and all related correspondence are subject to a legal hold "
            "order effective immediately. Do not delete any data related to the "
            "breach incident reported on 2025-11-01. Forward this to your IT lead "
            "and respond to confirm receipt.\n\nLegal Counsel"
        ),
        thread_length=1,
        has_attachment=False,
        correct_label="urgent",
        correct_priority="high",
        requires_reply=True,
    ),
    Email(
        email_id="hard_002",
        subject="FW: Vendor contract — redlines inside",
        sender="procurement@company.com",
        body=(
            "Please review the attached contract from VendorCo. "
            "Legal has flagged sections 4.2 and 7.1 for renegotiation. "
            "We need your sign-off before Thursday's call. "
            "This is time-sensitive — VendorCo has a hard deadline.\n\nProcurement"
        ),
        thread_length=5,
        has_attachment=True,
        correct_label="delegate",
        correct_priority="high",
        requires_reply=True,
    ),
    Email(
        email_id="hard_003",
        subject="Your account has been compromised",
        sender="security-alert@company-secure.net",
        body=(
            "We have detected unusual login attempts on your account. "
            "Click here immediately to verify your identity and secure your account. "
            "Failure to act within 2 hours will result in account suspension."
        ),
        thread_length=1,
        has_attachment=False,
        correct_label="spam",
        correct_priority="low",
        requires_reply=False,
    ),
    Email(
        email_id="hard_004",
        subject="Re: Re: API integration — still broken",
        sender="enterprise-client@bigbank.com",
        body=(
            "This is the third time we're raising this. Our payments integration "
            "has been broken for 6 days now, costing us an estimated $200k in failed "
            "transactions. We expect a war-room call TODAY at 16:00 UTC. "
            "If we don't hear back in 2 hours, we are escalating to your CEO.\n\n"
            "CTO, BigBank"
        ),
        thread_length=6,
        has_attachment=False,
        correct_label="urgent",
        correct_priority="high",
        requires_reply=True,
    ),
    Email(
        email_id="hard_005",
        subject="Re: Re: Proposal feedback",
        sender="prospective-client@startup.io",
        body=(
            "Thanks for the proposal. We loved sections 2 and 4. "
            "A few questions: (1) Can you reduce the timeline by 2 weeks? "
            "(2) Is there flexibility on pricing for a 2-year commitment? "
            "(3) Who would be our dedicated account manager?\n\nLooking forward to your reply."
        ),
        thread_length=3,
        has_attachment=False,
        correct_label="follow_up",
        correct_priority="high",
        requires_reply=True,
    ),
]


def grade_hard(email: Email, action: EmailTriageAction) -> float:
    base = _score_action(email, action)
    # Hard task: extra penalty for label errors (higher stakes)
    if action.label != email.correct_label:
        base = max(0.0, base - 0.15)
    # Extra credit for reply quality on hard emails
    if email.requires_reply and action.reply_draft:
        reply = action.reply_draft.lower()
        # Very basic quality signal: reward longer, substantive replies
        if len(reply) >= 80:
            base = min(1.0, base + 0.05)
    return round(base, 4)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

@dataclass
class Task:
    name: str
    difficulty: str
    emails: List[Email]
    grader: Callable[[Email, EmailTriageAction], float]
    description: str


TASKS: Dict[str, Task] = {
    "easy": Task(
        name="easy",
        difficulty="easy",
        emails=EASY_EMAILS,
        grader=grade_easy,
        description=(
            "Triage 5 clearly labelled emails. Labels and priorities are unambiguous. "
            "Expected score: 0.8–1.0"
        ),
    ),
    "medium": Task(
        name="medium",
        difficulty="medium",
        emails=MEDIUM_EMAILS,
        grader=grade_medium,
        description=(
            "Triage 5 emails with ambiguous subjects and multi-turn thread context. "
            "Expected score: 0.5–0.8"
        ),
    ),
    "hard": Task(
        name="hard",
        difficulty="hard",
        emails=HARD_EMAILS,
        grader=grade_hard,
        description=(
            "Triage 5 high-stakes emails requiring nuanced reasoning about urgency, "
            "delegation, and phishing detection. Expected score: 0.3–0.6"
        ),
    ),
}