"""
server/app.py — FastAPI server for the Email Triage OpenEnv environment.

Endpoints
---------
POST /reset          → start new episode, return first email
POST /step           → submit action, get next email + reward
GET  /state          → current episode metadata
GET  /health         → liveness probe
GET  /tasks          → list available tasks
GET  /docs           → auto-generated OpenAPI docs (built-in FastAPI)
GET  /web            → simple browser UI

WebSocket /ws        → persistent session (reset → step loop over WS)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Ensure project root is on the path regardless of how this file is invoked
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from envs.email_triage_env.environment import EmailTriageEnvironment
from envs.email_triage_env.models import EmailTriageAction
from envs.email_triage_env.tasks import TASKS


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic)
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_name: str = Field(default="easy", description="One of: easy, medium, hard")


class StepRequest(BaseModel):
    label: str = Field(default="archive")
    priority: str = Field(default="low")
    reply_draft: Optional[str] = Field(default=None)
    reasoning: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# In-process session store (one env per WebSocket session / HTTP session-id)
# ---------------------------------------------------------------------------

# For simplicity the HTTP endpoints share ONE environment per worker.
# For production, add session-id based routing.
_env: EmailTriageEnvironment = EmailTriageEnvironment(task_name="easy")


def _obs_to_dict(obs) -> dict:
    return {
        "done": obs.done,
        "reward": obs.reward,
        "email_id": obs.email_id,
        "subject": obs.subject,
        "sender": obs.sender,
        "body": obs.body,
        "thread_length": obs.thread_length,
        "has_attachment": obs.has_attachment,
        "emails_remaining": obs.emails_remaining,
        "feedback": obs.feedback,
        "task_description": obs.task_description,
    }


def _state_to_dict(state) -> dict:
    return {
        "episode_id": state.episode_id,
        "step_count": state.step_count,
        "task_name": state.task_name,
        "total_emails": state.total_emails,
        "cumulative_score": state.cumulative_score,
        "max_steps": state.max_steps,
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "A real-world RL environment where agents triage emails "
        "by assigning labels, priorities, and optional reply drafts. "
        "Implements the full OpenEnv step()/reset()/state() interface."
    ),
    version="1.0.0",
)


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    """Start a new episode for the specified task."""
    global _env
    task_name = req.task_name if req.task_name in TASKS else "easy"
    _env = EmailTriageEnvironment(task_name=task_name)
    obs = _env.reset()
    return _obs_to_dict(obs)


@app.post("/step")
def step(req: StepRequest) -> dict:
    """Submit the agent's triage decision for the current email."""
    action = EmailTriageAction(
        label=req.label,
        priority=req.priority,
        reply_draft=req.reply_draft,
        reasoning=req.reasoning,
    )
    obs = _env.step(action)
    result = _obs_to_dict(obs)
    result["reward"] = obs.reward  # ensure reward is in response
    return result


@app.get("/state")
def state() -> dict:
    """Return current episode metadata."""
    return _state_to_dict(_env.state)


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "healthy", "service": "email-triage-env"}


@app.get("/tasks")
def list_tasks() -> dict:
    """List all available tasks with descriptions."""
    return {
        name: {
            "difficulty": task.difficulty,
            "description": task.description,
            "num_emails": len(task.emails),
        }
        for name, task in TASKS.items()
    }


# ---------------------------------------------------------------------------
# Simple browser UI
# ---------------------------------------------------------------------------

WEB_UI = """
<!DOCTYPE html>
<html>
<head>
  <title>Email Triage OpenEnv</title>
  <style>
    body { font-family: monospace; background:#111; color:#eee; padding:2rem; max-width:800px; margin:auto; }
    h1 { color:#4ade80; }
    pre { background:#1e1e1e; padding:1rem; border-radius:6px; overflow-x:auto; }
    button { background:#4ade80; color:#111; border:none; padding:0.5rem 1.2rem;
             border-radius:4px; cursor:pointer; font-size:1rem; margin:0.3rem; }
    input, select { background:#1e1e1e; color:#eee; border:1px solid #555;
                    padding:0.4rem; border-radius:4px; margin:0.2rem; }
    .label { color:#86efac; font-weight:bold; }
    .section { margin:1.5rem 0; padding:1rem; border:1px solid #333; border-radius:8px; }
  </style>
</head>
<body>
  <h1>📧 Email Triage OpenEnv</h1>
  <div class="section">
    <b>Task:</b>
    <select id="task">
      <option value="easy">Easy</option>
      <option value="medium">Medium</option>
      <option value="hard">Hard</option>
    </select>
    <button onclick="reset()">Reset Episode</button>
  </div>
  <div class="section" id="email-panel" style="display:none">
    <p class="label">📨 Current Email</p>
    <p><b>From:</b> <span id="sender"></span></p>
    <p><b>Subject:</b> <span id="subject"></span></p>
    <p><b>Body:</b></p>
    <pre id="body"></pre>
    <p><b>Thread length:</b> <span id="thread"></span> &nbsp;
       <b>Attachment:</b> <span id="attach"></span> &nbsp;
       <b>Remaining:</b> <span id="remaining"></span></p>
    <hr>
    <b>Label:</b>
    <select id="label">
      <option>urgent</option><option selected>archive</option>
      <option>follow_up</option><option>spam</option><option>delegate</option>
    </select>
    <b>Priority:</b>
    <select id="priority">
      <option>high</option><option selected>low</option><option>medium</option>
    </select>
    <br><br>
    <b>Reply draft (optional):</b><br>
    <textarea id="reply" rows="3" style="width:100%;background:#1e1e1e;color:#eee;border:1px solid #555;border-radius:4px;padding:0.4rem"></textarea>
    <br><br>
    <button onclick="doStep()">Submit Triage Decision</button>
  </div>
  <div class="section">
    <p class="label">📋 Last Response</p>
    <pre id="output">— press Reset to start —</pre>
  </div>
  <script>
    async function reset() {
      const task = document.getElementById('task').value;
      const r = await fetch('/reset', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_name:task})});
      const d = await r.json();
      showEmail(d);
      document.getElementById('output').textContent = JSON.stringify(d, null, 2);
    }
    async function doStep() {
      const body = {
        label: document.getElementById('label').value,
        priority: document.getElementById('priority').value,
        reply_draft: document.getElementById('reply').value || null,
      };
      const r = await fetch('/step', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      const d = await r.json();
      if (d.done) {
        document.getElementById('email-panel').style.display='none';
        document.getElementById('output').textContent = 'Episode complete!\n\n' + JSON.stringify(d, null, 2);
      } else {
        showEmail(d);
        document.getElementById('output').textContent = JSON.stringify(d, null, 2);
      }
    }
    function showEmail(d) {
      document.getElementById('email-panel').style.display='block';
      document.getElementById('sender').textContent = d.sender;
      document.getElementById('subject').textContent = d.subject;
      document.getElementById('body').textContent = d.body;
      document.getElementById('thread').textContent = d.thread_length;
      document.getElementById('attach').textContent = d.has_attachment ? 'Yes' : 'No';
      document.getElementById('remaining').textContent = d.emails_remaining;
      document.getElementById('reply').value = '';
    }
  </script>
</body>
</html>
"""


@app.get("/web", response_class=HTMLResponse)
def web_ui() -> str:
    """Browser-based interactive UI."""
    return WEB_UI


# ---------------------------------------------------------------------------
# WebSocket endpoint — persistent session
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket session.

    Client sends JSON messages:
        {"type": "reset", "task_name": "easy"}
        {"type": "step",  "label": "urgent", "priority": "high", "reply_draft": "..."}
        {"type": "state"}

    Server replies with JSON observation / state dicts.
    """
    await websocket.accept()
    ws_env: EmailTriageEnvironment = EmailTriageEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                task_name = msg.get("task_name", "easy")
                if task_name not in TASKS:
                    task_name = "easy"
                ws_env = EmailTriageEnvironment(task_name=task_name)
                obs = ws_env.reset()
                await websocket.send_json(_obs_to_dict(obs))

            elif msg_type == "step":
                action = EmailTriageAction(
                    label=msg.get("label", "archive"),
                    priority=msg.get("priority", "low"),
                    reply_draft=msg.get("reply_draft"),
                    reasoning=msg.get("reasoning"),
                )
                obs = ws_env.step(action)
                result = _obs_to_dict(obs)
                result["reward"] = obs.reward
                await websocket.send_json(result)

            elif msg_type == "state":
                await websocket.send_json(_state_to_dict(ws_env.state))

            else:
                await websocket.send_json({"error": f"Unknown message type '{msg_type}'"})

    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Entry-point (python server/app.py)
# ---------------------------------------------------------------------------

def main():
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        reload=False,
    )

if __name__ == "__main__":
    main()