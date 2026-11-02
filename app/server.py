"""HTTP layer for the chat UI. No LLM logic lives here, only calls into core.

    uv run uvicorn app.server:app --reload
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core import client

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # exported variables take precedence over .env

app = FastAPI(title="LLM Playground")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    provider: str
    model: str
    messages: list[Message]


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/api/providers")
def providers():
    """Every provider, whether it can be used right now, and its models."""
    out = []
    for p in client.load_providers().values():
        entry = {"name": p.name, "label": p.label, "ready": False, "reason": None, "models": []}
        if not p.api_key:
            entry["reason"] = f"{p.key_env} not set"
        else:
            try:
                entry["models"] = client.list_models(p)
                entry["ready"] = bool(entry["models"])
                if not entry["ready"]:
                    entry["reason"] = "no models found"
            except Exception as exc:
                entry["reason"] = f"could not list models: {type(exc).__name__}"
        out.append(entry)
    return out


@app.post("/api/chat")
def chat(req: ChatRequest):
    provider = client.load_providers().get(req.provider)
    if provider is None:
        raise HTTPException(404, f"unknown provider '{req.provider}'")
    try:
        result = client.chat(provider, req.model, [m.model_dump() for m in req.messages])
    except Exception as exc:
        # Surface the provider's own error: with no retries or fallbacks,
        # seeing exactly what failed is the point.
        raise HTTPException(502, f"{type(exc).__name__}: {exc}")
    return {**asdict(result), "tokens_per_sec": result.tokens_per_sec}
