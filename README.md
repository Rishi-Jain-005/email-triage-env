---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
---

# 📧 Email Triage OpenEnv

A real-world OpenEnv RL environment where AI agents learn to triage an email inbox.  
The agent assigns a **label**, a **priority**, and optionally drafts a **reply** for each email.

> Built for the [OpenEnv Course](https://github.com/raun/openenv-course) competition.  
> Follows the full OpenEnv `step()` / `reset()` / `state()` specification.

---

## Table of Contents

1. [Environment Description](#environment-description)
2. [Action Space](#action-space)
3. [Observation Space](#observation-space)
4. [Reward Function](#reward-function)
5. [Tasks](#tasks)
6. [Quick Start](#quick-start)
7. [Running Locally](#running-locally)
8. [Docker](#docker)
9. [Deploying to HF Spaces](#deploying-to-hf-spaces)
10. [Inference Script](#inference-script)
11. [Baseline Scores](#baseline-scores)
12. [Pre-Submission Validation](#pre-submission-validation)
13. [Project Structure](#project-structure)

---

## Environment Description

**Email Triage** is a task humans perform every day: deciding what to do with each email in an inbox.

The agent receives one email at a time and must decide:
- **Label** — what category does this email belong to?
- **Priority** — how urgently does it need attention?
- **Reply** — does a response need to be drafted?

This is a meaningful real-world task (not a game or toy) with clear success criteria,  
partial-progress rewards, and three difficulty levels.

### Motivation

Email triage is an ideal RL benchmark because:
- Ground truth is deterministic (each email has a correct label/priority)
- Partial credit is natural (getting label right but priority wrong)
- Difficulty scales smoothly (clear → ambiguous → high-stakes)
- The task generalises to broader agent capabilities (reading comprehension, tone detection, urgency reasoning)

---

## Action Space

The agent submits a structured action for each email:

| Field | Type | Values | Required |
|---|---|---|---|
| `label` | categorical | `urgent` · `follow_up` · `archive` · `spam` · `delegate` | ✅ |
| `priority` | categorical | `high` · `medium` · `low` | ✅ |
| `reply_draft` | text (≤ 500 chars) | any string | ❌ optional |
| `reasoning` | text | any string | ❌ optional (not scored) |

**Label definitions:**

| Label | When to use |
|---|---|
| `urgent` | Requires immediate attention — system outage, legal hold, escalating client |
| `follow_up` | Requires action but not immediately — review, respond within days |
| `delegate` | Needs someone else's sign-off or expertise |
| `archive` | No action needed — informational only |
| `spam` | Unsolicited, promotional, or phishing |

**Priority definitions:**

| Priority | When to use |
|---|---|
| `high` | Business-critical, time-sensitive, financial/legal impact |
| `medium` | Important but not blocking |
| `low` | Informational, social, or very low urgency |

---

## Observation Space

After each `reset()` or `step()` the agent receives:

| Field | Type | Description |
|---|---|---|
| `done` | bool | Whether the episode has ended |
| `reward` | float \| null | Reward for the last action (null after reset) |
| `email_id` | string | Unique ID for the current email |
| `subject` | string | Email subject line |
| `sender` | string | Sender address / name |
| `body` | string | Email body text |
| `thread_length` | integer | Number of messages in the thread |
| `has_attachment` | bool | Whether the email has attachments |
| `emails_remaining` | integer | Emails left in this episode |
| `feedback` | string | Human-readable feedback on last action |
| `task_description` | string | Description of the task |

---

## Reward Function

The reward function provides **partial progress signals** at every step — not just binary win/lose.

```
reward = 0.50 × (label correct)
       + 0.30 × (priority correct)
       + 0.20 × (reply handling correct)
```

**Reply handling:**
- If a reply IS required: +0.20 for providing a reply ≥ 10 chars, 0 otherwise
- If a reply is NOT required: +0.20 for correctly omitting a reply, −0.10 for drafting an unnecessary one

**Hard task bonus:** Extra +0.05 for replies ≥ 80 chars on hard emails (encourages substantive responses).

**Hard task penalty:** −0.15 for wrong label (vs −0.10 on medium, 0 on easy) — reflecting higher stakes.

All rewards are clipped to **[0.0, 1.0]**.

---

## Tasks

Three tasks of increasing difficulty, each with 5 emails and deterministic graders:

### Task 1 — Easy (`task_name="easy"`)

Clearly labelled, unambiguous emails. Labels and priorities are obvious from subject + body.

| Email | Subject | Correct Label | Correct Priority |
|---|---|---|---|
| easy_001 | URGENT: Server down in production | urgent | high |
| easy_002 | Newsletter: Top 10 productivity hacks | spam | low |
| easy_003 | Q3 Report — please review by Friday | follow_up | medium |
| easy_004 | Meeting notes from yesterday | archive | low |
| easy_005 | Congratulations! You've won a prize | spam | low |

**Expected score: 0.80 – 1.00**

---

### Task 2 — Medium (`task_name="medium"`)

Ambiguous subjects, multi-turn thread context, mixed urgency signals.

| Email | Subject | Correct Label | Correct Priority |
|---|---|---|---|
| med_001 | Re: Re: Re: Budget approval | urgent | high |
| med_002 | Checking in | follow_up | medium |
| med_003 | Offsite planning — venue options | follow_up | low |
| med_004 | Invoice #INV-2094 overdue | delegate | medium |
| med_005 | Re: Team lunch tomorrow | follow_up | low |

**Expected score: 0.50 – 0.80**

---

### Task 3 — Hard (`task_name="hard"`)

High-stakes emails requiring nuanced reasoning: legal holds, angry enterprise clients,  
vendor contract escalations, and sophisticated phishing.

| Email | Subject | Correct Label | Correct Priority |
|---|---|---|---|
| hard_001 | Re: Data breach — legal hold | urgent | high |
| hard_002 | FW: Vendor contract — redlines inside | delegate | high |
| hard_003 | Your account has been compromised | spam | low |
| hard_004 | Re: Re: API integration — still broken | urgent | high |
| hard_005 | Re: Re: Proposal feedback | follow_up | high |

**Expected score: 0.30 – 0.60**

---

## Quick Start

```python
from envs.email_triage_env import EmailTriageEnv, EmailTriageAction

with EmailTriageEnv(base_url="http://localhost:8000", task_name="easy").sync() as env:
    result = env.reset()
    obs = result.observation
    print(f"Subject: {obs.subject}")

    action = EmailTriageAction(
        label="urgent",
        priority="high",
        reply_draft="Acknowledged, escalating immediately.",
    )
    result = env.step(action)
    print(f"Reward: {result.reward:.2f}")
    print(f"Feedback: {result.observation.feedback}")
```

---

## Running Locally

### Option A — Uvicorn (fastest for development)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/email-triage-env
cd email-triage-env

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# 5. Verify health
curl http://localhost:8000/health
# → {"status": "healthy", "service": "email-triage-env"}

# 6. Open the browser UI
# Visit: http://localhost:8000/web

# 7. Open API docs
# Visit: http://localhost:8000/docs
```

### Option B — Python directly

```bash
python server/app.py
```

---

## Docker

### Build and run locally

```bash
# Build
docker build -t email-triage-env:latest .

# Run
docker run -d -p 8000:8000 email-triage-env:latest

# With tuning variables
docker run -d -p 8000:8000 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=100 \
    email-triage-env:latest

# Health check
curl http://localhost:8000/health
```

### Pull from HF Spaces registry (after deploying)

```bash
docker pull registry.hf.space/your-username-email-triage-env:latest
docker run -d -p 8000:8000 registry.hf.space/your-username-email-triage-env:latest
```

---

## Deploying to HF Spaces

```bash
# Install openenv-core
pip install openenv-core

# Push to your HF Space
openenv push --repo-id your-username/email-triage-env

# Your endpoints will be live at:
# https://your-username-email-triage-env.hf.space/health
# https://your-username-email-triage-env.hf.space/reset
# https://your-username-email-triage-env.hf.space/step
# https://your-username-email-triage-env.hf.space/state
# https://your-username-email-triage-env.hf.space/web
# https://your-username-email-triage-env.hf.space/docs
# wss://your-username-email-triage-env.hf.space/ws
```

**Hardware:** CPU Basic (Free) — 2 vCPU, 16GB RAM — handles ~128 concurrent sessions.

---

## Inference Script

The inference script (`inference.py`) runs a model against all three tasks using the OpenAI client.

### Required environment variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint (e.g. `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Model identifier (e.g. `Qwen/Qwen2.5-7B-Instruct`) |
| `HF_TOKEN` | Your Hugging Face API key |
| `ENV_BASE_URL` | URL of the running environment server |

### Running the inference script

```bash
# Windows (PowerShell)
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct"
$env:HF_TOKEN     = "hf_your_token_here"
$env:ENV_BASE_URL = "http://localhost:8000"
python inference.py

# macOS / Linux
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_BASE_URL="http://localhost:8000"
python inference.py
```

The script prints a score for each task and an overall mean.

---

## Baseline Scores

Scores from `Qwen/Qwen2.5-7B-Instruct` at temperature 0.1:

| Task | Mean Score | Notes |
|---|---|---|
| easy | ~0.88 | Strong on clear signals |
| medium | ~0.62 | Struggles with implicit urgency |
| hard | ~0.44 | Misses phishing + legal nuance |
| **Overall** | **~0.65** | |

Scores vary by model. Larger models (72B+) typically score 0.10–0.15 higher on hard.

---

## Pre-Submission Validation

Run the validation script before submitting:

```bash
# macOS / Linux
chmod +x validate-submission.sh
./validate-submission.sh https://your-username-email-triage-env.hf.space .

# Windows (Git Bash or WSL)
bash validate-submission.sh https://your-username-email-triage-env.hf.space .
```

The script checks:
1. HF Space `/health` returns 200
2. `/reset` endpoint accepts POST and returns 200
3. Docker build succeeds
4. `openenv validate` passes
5. `inference.py` exists in the project root

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Project Structure

```
email-triage-env/
│
├── inference.py               ← Competition inference script (MANDATORY, root)
├── openenv.yaml               ← OpenEnv manifest (validated by openenv validate)
├── Dockerfile                 ← Container definition
├── requirements.txt           ← Python dependencies
├── pyproject.toml             ← Package metadata
├── validate-submission.sh     ← Pre-submission validation script
├── .env.example               ← Template for environment variables
├── .gitignore
├── README.md
│
├── envs/
│   └── email_triage_env/
│       ├── __init__.py        ← Public API exports
│       ├── models.py          ← Typed Action / Observation / State dataclasses
│       ├── tasks.py           ← Email datasets + deterministic graders (easy/medium/hard)
│       ├── environment.py     ← Core game logic: reset() / step() / state
│       └── client.py          ← HTTP client (what training code imports)
│
├── server/
│   └── app.py                 ← FastAPI server: /reset /step /state /health /ws /web /docs
│
└── tests/
    └── test_environment.py    ← Unit tests for graders and environment lifecycle
```

---

## API Reference

| Method | Endpoint | Body | Returns |
|---|---|---|---|
| POST | `/reset` | `{"task_name": "easy"}` | Observation dict |
| POST | `/step` | `{"label": "urgent", "priority": "high", "reply_draft": "..."}` | Observation + reward |
| GET | `/state` | — | State dict |
| GET | `/health` | — | `{"status": "healthy"}` |
| GET | `/tasks` | — | Task list |
| GET | `/web` | — | Browser UI |
| GET | `/docs` | — | OpenAPI docs |
| WS | `/ws` | JSON messages | JSON responses |

---

## License

MIT