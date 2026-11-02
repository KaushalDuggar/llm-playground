"""HTTP layer for the chat UI. No LLM logic lives here, only calls into core.

    uv run uvicorn app.server:app --reload
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from core import config, gateway

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # exported variables take precedence over .env

app = FastAPI(title="LLM Playground")

# Which HTTP status each gateway stop reason maps to.
_STATUS = {
    "not_allowed": 403,
    "prompt_too_long": 413,
    "budget": 429,
    "bad_request": 400,
    "deadline": 504,
    "all_routes_failed": 502,
}


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    user: str
    model: str
    messages: list[Message]


class PolicyChange(BaseModel):
    field: str
    value: int | None   # None: back to the tier default


def _user_or_404(user_id: str) -> None:
    if user_id not in config.load().users:
        raise HTTPException(404, f"unknown user '{user_id}'")


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/users")
def users():
    cfg = config.load()
    return [{"id": u.id, "name": u.name, "tier": cfg.tiers[u.tier].label} for u in cfg.users.values()]


@app.get("/api/users/{user_id}")
def user(user_id: str):
    _user_or_404(user_id)
    return gateway.describe(user_id)


@app.put("/api/users/{user_id}/policy")
def change_policy(user_id: str, change: PolicyChange):
    _user_or_404(user_id)
    try:
        gateway.set_policy(user_id, change.field, change.value)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return gateway.describe(user_id)


@app.post("/api/chat")
def chat(req: ChatRequest):
    _user_or_404(req.user)
    try:
        result = gateway.call(req.user, req.model, [m.model_dump() for m in req.messages])
    except gateway.GatewayError as exc:
        return JSONResponse(
            status_code=_STATUS[exc.reason],
            content={"reason": exc.reason, "detail": str(exc), "attempts": [asdict(a) for a in exc.attempts]},
        )
    return asdict(result)
