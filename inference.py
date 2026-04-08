"""
inference.py — Email Triage OpenEnv Baseline Inference Script
=============================================================

MANDATORY VARIABLES (set in your environment before running):
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face / API key

Usage
-----
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="hf_..."
    export ENV_BASE_URL="http://localhost:8000"   # your running Space URL

    python inference.py

The script runs the model against all 3 tasks and prints a score summary.
Total runtime must be < 20 min on vcpu=2, memory=8GB.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Load local environment variables from .env if present.
load_dotenv(dotenv_path=Path(".env"))

from envs.email_triage_env import EmailTriageEnv, EmailTriageAction

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TEMPERATURE: float = 0.1
MAX_TOKENS: int = 400

TASKS_TO_RUN: List[str] = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email assistant. Your job is to triage emails with accuracy.

    For each email you receive, respond with a JSON object containing:
    {
      "label":       one of "urgent" | "follow_up" | "archive" | "spam" | "delegate",
      "priority":    one of "high" | "medium" | "low",
      "reply_draft": a short reply if needed, or null if no reply is required,
      "reasoning":   brief explanation of your decision
    }

    ═══════════════════════════════════════════════════════════════════════════
    DECISION TREE (use in order):
    ═══════════════════════════════════════════════════════════════════════════

    1. IS IT SPAM?
       → Keywords: "verify account", "click immediately", "unusual activity", "congratulations you won", 
         "prize", "unsubscribe", "marketing", "newsletter", "promotional"
       → Sender: Unknown/suspicious sender asking to click/verify
       → Rule: If it's unsolicited marketing, phishing, or lottery → SPAM (priority: LOW)
       Example: "Your account has been compromised - verify now" = SPAM, LOW
       Example: "Newsletter: Top 10 productivity hacks" = SPAM, LOW
       Example: "Congratulations you won $1000" = SPAM, LOW

    2. DOES IT REQUIRE URGENT IMMEDIATE ACTION (TODAY/NOW)?
       → Keywords: "urgent", "immediate", "today", "now", "immediately", "down", "broken", "escalating", "vendor will cancel"
       → Context: Production outage, security breach, legal hold, angry client, revenue impact, hard deadline
       → Rule: Revenue/ops impact, hard deadline TODAY, client escalation, or "vendor will cancel" → URGENT
       → Special case: If an approval request includes a clear immediate deadline, treat it as URGENT even if it also mentions approval or sign-off.
       Example: "Server down - revenue impact $5k/min" = URGENT, HIGH
       Example: "Budget decision needed TODAY or vendor cancels" = URGENT, HIGH
       Example: "Enterprise client escalating - war room call TODAY" = URGENT, HIGH

    3. DOES IT NEED SOMEONE ELSE'S SIGN-OFF OR SPECIALIST?
       → Keywords: "contract", "approval", "legal review", "sign-off", "redlines", "vendor", 
         "billing", "invoice", "accounting", "CFO", "legal team"
       → Context: Financial decisions, legal/contract review, external vendor negotiation
       → Rule: Requires SIGN-OFF from finance/legal/leadership, or needs specialist expertise → DELEGATE
       → Special case: Overdue invoice/payment requests are DELEGATE if they ask to arrange payment, contact billing, or resolve an overdue balance.
       Example: "Vendor contract - legal redlines need review" = DELEGATE, HIGH
       Example: "Invoice overdue - accounting needs to follow up" = DELEGATE, MEDIUM
       Example: "Budget approval - CFO sign-off needed" = DELEGATE, HIGH

    4. DOES IT REQUEST YOUR ACTION/REVIEW?
       → Keywords: "please review", "feedback", "your input", "by Friday", "RSVP", "vote", "response"
       → Context: You need to provide feedback, attend, respond, or take action
       → Rule: Requires YOUR participation/response but no urgent deadline → FOLLOW_UP
       Example: "Q3 report - please review by Friday" = FOLLOW_UP, MEDIUM
       Example: "Team lunch tomorrow - please confirm attendance" = FOLLOW_UP, LOW
       Example: "Budget survey - please vote on venue options" = FOLLOW_UP, LOW

    5. IS IT INFORMATIONAL ONLY?
       → Keywords: "FYI", "for your information", "meeting notes", "status", "update"
       → Context: No action requested, just sharing info
       → Rule: No action needed → ARCHIVE
       Example: "Meeting notes from yesterday - FYI" = ARCHIVE, LOW
       Example: "Team status update" = ARCHIVE, LOW

    ═══════════════════════════════════════════════════════════════════════════
    PRIORITY MAPPING:
    ═══════════════════════════════════════════════════════════════════════════
    
    HIGH:
    - Urgent emails (production down, legal hold, security, escalation)
    - Financial/budget approvals or overdue payments when they require your decision or carry a same-day deadline
    - High-stakes client proposals and requests that require a timely business response
    - Hard deadline TODAY
    
    MEDIUM:
    - Important internal or client requests without an immediate emergency
    - Client check-ins, review requests, or status updates due soon
    - Routine business requests or follow-up without urgent escalation
    
    LOW:
    - Spam/phishing (always low priority)
    - Informational (FYI, meeting notes)
    - Voluntary event planning, team lunch invites, and optional RSVP requests
    - Internal coordination that is non-urgent

    ═══════════════════════════════════════════════════════════════════════════
    CRITICAL RULES:
    ═══════════════════════════════════════════════════════════════════════════

    - SPAM is ALWAYS priority LOW (even if it claims to be urgent)
    - Phishing emails that say "urgent" or "unusual activity" are SPAM + LOW, not urgent
    - "Today/now" in deadline = URGENT (unless it's spam)
    - Budget approval requests with same-day risk or vendor cancellation are URGENT, HIGH
    - Overdue invoice/payment requests are DELEGATE, MEDIUM unless they include a hard deadline or escalation
    - Financial/legal decisions that require specialist review can be DELEGATE if not urgent
    - Newsletters/marketing = SPAM (not archive, not follow_up)
    - Team coordination (lunch, meeting RSVP, venue voting) = FOLLOW_UP, LOW
    - Client check-ins and review requests = FOLLOW_UP, MEDIUM
    - Proposal feedback requiring a business response = FOLLOW_UP, HIGH

    ═══════════════════════════════════════════════════════════════════════════
    REPLY RULES:
    ═══════════════════════════════════════════════════════════════════════════
    - Only draft a reply when the email explicitly requests a response
    - Keep replies concise (≤ 150 words)
    - If no reply is needed, set reply_draft to null

    Output ONLY the JSON object. No markdown, no extra text.
""").strip()


def build_user_prompt(obs) -> str:
    """Build the user message from the current observation."""
    attachment_note = "📎 Has attachment" if obs.has_attachment else "No attachment"
    return textwrap.dedent(f"""
        Task: {obs.task_description}
        Emails remaining after this one: {obs.emails_remaining - 1}

        === EMAIL ===
        From:    {obs.sender}
        Subject: {obs.subject}
        Thread:  {obs.thread_length} message(s)
        {attachment_note}

        {obs.body}
        === END EMAIL ===

        Respond with the JSON triage decision.
    """).strip()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_response(text: str) -> EmailTriageAction:
    """Extract JSON from the model response and build an action."""
    if not text:
        return EmailTriageAction()

    match = JSON_RE.search(text)
    if not match:
        return EmailTriageAction()

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return EmailTriageAction()

    label = str(data.get("label", "archive")).lower().strip()
    priority = str(data.get("priority", "low")).lower().strip()

    if label not in {"urgent", "follow_up", "archive", "spam", "delegate"}:
        label = "archive"
    if priority not in {"high", "medium", "low"}:
        priority = "low"

    return EmailTriageAction(
        label=label,
        priority=priority,
        reply_draft=data.get("reply_draft") or None,
        reasoning=data.get("reasoning") or None,
    )


def call_model(client: OpenAI, messages: List[dict], model_name: str) -> Tuple[Optional[str], Optional[Exception]]:
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content or "", None
    except Exception as exc:
        return None, exc


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_task(
    client: OpenAI,
    task_name: str,
) -> Tuple[float, int]:
    """
    Run one complete task episode.
    Returns (mean_score, num_emails).
    """
    print(f"\n{'='*55}")
    print(f"  Task: {task_name.upper()}")
    print(f"{'='*55}")

    rewards: List[float] = []

    with EmailTriageEnv(base_url=ENV_BASE_URL, task_name=task_name).sync() as env:
        result = env.reset()
        obs = result.observation
        print(f"  Episode started | emails: {obs.emails_remaining}")

        # ── [START] structured log ──────────────────────────────────────
        print(f"[START] task={task_name} emails={obs.emails_remaining}")

        step = 0
        while not result.done:
            step += 1
            user_prompt = build_user_prompt(obs)

            # --- LLM call ---
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            response_text, exc = call_model(client, messages, MODEL_NAME)
            if exc is not None:
                if FALLBACK_MODEL_NAME:
                    print(f"  [Step {step}] LLM error with {MODEL_NAME}: {exc} — trying fallback {FALLBACK_MODEL_NAME}")
                    response_text, exc = call_model(client, messages, FALLBACK_MODEL_NAME)
                if exc is not None:
                    print(f"  [Step {step}] LLM error: {exc} — using archive fallback")
                    response_text = '{"label":"archive","priority":"low","reply_draft":null}'
            elif response_text is None:
                response_text = '{"label":"archive","priority":"low","reply_draft":null}'

            action = parse_model_response(response_text)

            print(
                f"  [Step {step}] {obs.subject[:40]!r} "
                f"→ label={action.label}, priority={action.priority}"
            )

            result = env.step(action)
            obs = result.observation
            rewards.append(result.reward)

            print(f"           reward={result.reward:.2f} | {obs.feedback[:60]}")

            # ── [STEP] structured log ──────────────────────────────────
            print(
                f"[STEP] step={step} "
                f"action={{\"label\": \"{action.label}\", "
                f"\"priority\": \"{action.priority}\", "
                f"\"reply_draft\": {json.dumps(action.reply_draft)}}} "
                f"reward={result.reward:.4f} done={result.done}"
            )

    mean_score = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\n  Task '{task_name}' complete | mean score: {mean_score:.4f}")

    # ── [END] structured log ────────────────────────────────────────
    print(
        f"[END] task={task_name} "
        f"steps={len(rewards)} "
        f"mean_reward={mean_score:.4f}"
    )

    return mean_score, len(rewards)


def main() -> None:
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is not set. "
            "Export it before running: export HF_TOKEN=hf_..."
        )
    if not MODEL_NAME:
        raise EnvironmentError("MODEL_NAME is not set.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print("\n" + "="*55)
    print("  Email Triage OpenEnv — Baseline Inference")
    print(f"  Model:   {MODEL_NAME}")
    if FALLBACK_MODEL_NAME:
        print(f"  Fallback: {FALLBACK_MODEL_NAME}")
    print(f"  Server:  {ENV_BASE_URL}")
    print("="*55)

    results: Dict[str, float] = {}

    for task_name in TASKS_TO_RUN:
        mean_score, n = run_task(client, task_name)
        results[task_name] = mean_score

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SCORE SUMMARY")
    print("="*55)
    total = 0.0
    for task, score in results.items():
        bar = "█" * int(score * 20)
        print(f"  {task:8s}  {score:.4f}  [{bar:<20}]")
        total += score
    overall = total / len(results) if results else 0.0
    print(f"\n  Overall mean: {overall:.4f}")
    print("="*55 + "\n")

    # Machine-readable output (useful for CI / auto-graders)
    print("SCORES_JSON:", json.dumps({"tasks": results, "overall": overall}))


if __name__ == "__main__":
    main()