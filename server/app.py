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
def reset(req: Optional[ResetRequest] = None) -> dict:
    """Start a new episode for the specified task."""
    global _env
    task_name = req.task_name if req and req.task_name in TASKS else "easy"
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
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Email Triage — AI Workspace</title>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&family=Fira+Code&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --glass: rgba(255, 255, 255, 0.04);
      --border: rgba(255, 255, 255, 0.1);
      --accent: #6366f1;
      --accent-glow: rgba(99, 102, 241, 0.5);
      --text: #e2e8f0;
      --muted: #94a3b8;
    }
    body {
      font-family: 'Outfit', sans-serif;
      margin: 0; padding: 0;
      background-color: var(--bg);
      background-image: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                        radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.1) 0%, transparent 40%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }
    .container {
      width: 100%; max-width: 900px;
      margin: 3rem auto;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }
    h1 {
      font-weight: 700;
      font-size: 2.2rem;
      text-align: center;
      margin-bottom: 0.5rem;
      background: linear-gradient(135deg, #a5b4fc, #818cf8, #34d399);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-transform: uppercase;
      letter-spacing: 2px;
    }
    .glass-card {
      background: var(--glass);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.5rem 2rem;
      box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
      transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover { border-color: rgba(255, 255, 255, 0.2); }
    
    .header-controls { display: flex; justify-content: space-between; align-items: center; }
    
    select, input, textarea {
      font-family: 'Outfit', sans-serif;
      background: rgba(0, 0, 0, 0.3);
      color: #fff;
      border: 1px solid var(--border);
      padding: 0.6rem 1rem;
      border-radius: 8px;
      font-size: 1rem;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    select:focus, textarea:focus { border-color: var(--accent); box-shadow: 0 0 10px var(--accent-glow); }
    
    button {
      background: linear-gradient(135deg, #6366f1, #4f46e5);
      color: white;
      border: none;
      padding: 0.7rem 1.5rem;
      border-radius: 8px;
      font-weight: 700;
      font-size: 1rem;
      cursor: pointer;
      box-shadow: 0 4px 15px var(--accent-glow);
      transition: all 0.2s;
    }
    button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px var(--accent-glow); }
    button:active { transform: translateY(1px); }

    .email-header { display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid var(--border); padding-bottom: 1rem; margin-bottom: 1rem; }
    .email-meta { font-size: 0.9rem; color: var(--muted); }
    .email-meta span { display: inline-block; background: rgba(255,255,255,0.05); padding: 0.2rem 0.6rem; border-radius: 12px; margin-right: 0.5rem; border: 1px solid var(--border); }
    
    .email-subject { font-size: 1.5rem; font-weight: 500; color: #fff; margin: 0.5rem 0; }
    .email-sender { color: #a5b4fc; font-weight: 500; }
    
    .email-body {
      font-family: 'Fira Code', monospace;
      font-size: 0.95rem;
      line-height: 1.6;
      background: rgba(0, 0, 0, 0.4);
      padding: 1.5rem;
      border-radius: 12px;
      border: 1px solid var(--border);
      white-space: pre-wrap;
      max-height: 400px;
      overflow-y: auto;
      margin-bottom: 1.5rem;
      color: #cbd5e1;
    }

    .triage-controls { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
    .triage-group label { display: block; margin-bottom: 0.5rem; color: var(--muted); font-weight: 500; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
    .triage-group select { width: 100%; appearance: none; }

    .full-width { grid-column: 1 / -1; }
    textarea { width: 100%; resize: vertical; box-sizing: border-box; }

    .slide-in { animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    #json-output { font-family: 'Fira Code', monospace; font-size: 0.85rem; color: #34d399; }
    
    /* Progress styling */
    .progress-container { width: 100%; background: rgba(0,0,0,0.3); border-radius: 8px; height: 8px; overflow: hidden; margin-top: 1rem; }
    .progress-bar { height: 100%; background: linear-gradient(90deg, #34d399, #10b981); width: 0%; transition: width 0.4s ease; box-shadow: 0 0 10px rgba(52, 211, 153, 0.5); }
  </style>
</head>
<body>
  <div class="container">
    <h1>📧 AI Triage Environment</h1>
    
    <div class="glass-card header-controls">
      <div>
        <label style="color:var(--muted); margin-right: 1rem; font-weight: 500;">SELECT WORKLOAD</label>
        <select id="task">
          <option value="easy">Level 1 - Easy</option>
          <option value="medium">Level 2 - Medium</option>
          <option value="hard">Level 3 - Hard</option>
        </select>
      </div>
      <button onclick="reset()">Initialize Engine</button>
    </div>

    <div class="glass-card slide-in" id="email-panel" style="display:none">
      <div class="email-header">
        <div>
          <div class="email-sender" id="sender"></div>
          <div class="email-subject" id="subject"></div>
        </div>
        <div class="email-meta">
          <span title="Thread Length">🧵 <b id="thread"></b></span>
          <span title="Attachment">📎 <b id="attach"></b></span>
        </div>
      </div>

      <div class="email-body" id="body"></div>

      <div class="triage-controls">
        <div class="triage-group">
          <label>AI Routing Label</label>
          <select id="label">
            <option value="urgent">🔴 Urgent</option>
            <option value="archive" selected>📦 Archive</option>
            <option value="follow_up">⭐ Follow Up</option>
            <option value="spam">🚫 Spam</option>
            <option value="delegate">🤝 Delegate</option>
          </select>
        </div>
        
        <div class="triage-group">
          <label>Assign Priority</label>
          <select id="priority">
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low" selected>Low</option>
          </select>
        </div>

        <div class="triage-group full-width">
          <label>Auto-Generated Reply Draft (Optional)</label>
          <textarea id="reply" rows="3" placeholder="If required, type an automated reply draft here..."></textarea>
        </div>
      </div>

      <button style="width: 100%" onclick="doStep()">Execute Triage Decision &rarr;</button>
      
      <div class="progress-container">
        <div class="progress-bar" id="progress-bar"></div>
      </div>
      <p style="text-align: right; margin: 0.5rem 0 0 0; font-size: 0.8rem; color: var(--muted);"><span id="remaining"></span> items left in queue</p>
    </div>

    <div class="glass-card">
      <p style="color:var(--muted); font-weight:700; font-size:0.9rem; text-transform:uppercase; margin-top:0;">Terminal Output</p>
      <pre id="output" id="json-output">>> Waiting for engine initialization...</pre>
    </div>
  </div>

  <script>
    let totalEmails = 0;

    async function reset() {
      const task = document.getElementById('task').value;
      const r = await fetch('/reset', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_name:task})});
      const d = await r.json();
      totalEmails = d.emails_remaining;
      showEmail(d);
      document.getElementById('output').textContent = JSON.stringify(d, null, 2);
    }
    async function doStep() {
      const body = {
        label: document.getElementById('label').value,
        priority: document.getElementById('priority').value,
        reply_draft: document.getElementById('reply').value || null,
      };
      
      // Animate out
      document.getElementById('email-panel').classList.remove('slide-in');
      document.getElementById('email-panel').style.opacity = '0.5';

      const r = await fetch('/step', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      const d = await r.json();
      
      document.getElementById('email-panel').style.opacity = '1';
      document.getElementById('email-panel').classList.add('slide-in');

      if (d.done) {
        document.getElementById('email-panel').style.display='none';
        document.getElementById('output').textContent = 'WORKSPACE COMPLETE!\nFinal Stats:\n\n' + JSON.stringify(d, null, 2);
      } else {
        showEmail(d);
        document.getElementById('output').textContent = JSON.stringify(d, null, 2);
      }
    }
    function showEmail(d) {
      document.getElementById('email-panel').style.display='block';
      
      // Force reflow for animation
      const panel = document.getElementById('email-panel');
      panel.style.animation = 'none';
      panel.offsetHeight; 
      panel.style.animation = null;

      document.getElementById('sender').textContent = d.sender;
      document.getElementById('subject').textContent = d.subject;
      document.getElementById('body').textContent = d.body;
      document.getElementById('thread').textContent = d.thread_length;
      document.getElementById('attach').textContent = d.has_attachment ? 'Yes' : 'None';
      document.getElementById('remaining').textContent = d.emails_remaining;
      document.getElementById('reply').value = '';
      
      // Update progress bar
      if(totalEmails > 0) {
        const completed = totalEmails - d.emails_remaining;
        document.getElementById('progress-bar').style.width = ((completed / totalEmails) * 100) + '%';
      }
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