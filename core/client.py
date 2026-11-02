"""One way to call any LLM: pick a provider and a model, get an LLMResult back.

Providers and their free models live in providers.yaml; this file turns them
into LangChain chat models. A call goes to exactly the provider and model you
picked. There are no retries and no fallbacks: if it fails, you see the real
error.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

PROVIDERS_FILE = Path(__file__).with_name("providers.yaml")


@dataclass
class Provider:
    name: str
    label: str
    base_url: str
    key_env: str | None          # None: no key needed (ollama)
    langchain_provider: str
    base_url_override: bool
    supports_max_retries: bool
    models: list[str]            # curated list, empty when `discover` is set
    discover: str | None         # "openrouter_free" or "ollama"

    @property
    def api_key(self) -> str | None:
        if self.key_env is None:
            # The OpenAI client refuses an empty key even when the server
            # ignores it, as Ollama does.
            return "unused"
        return os.getenv(self.key_env)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: float      # wall-clock time for the whole call, as the user feels it
    input_tokens: int
    output_tokens: int

    @property
    def tokens_per_sec(self) -> float:
        # Includes network and time to first token, so it understates the
        # model's raw generation speed.
        return self.output_tokens / (self.latency_ms / 1000) if self.latency_ms > 0 else 0.0


def load_providers() -> dict[str, Provider]:
    raw = yaml.safe_load(PROVIDERS_FILE.read_text())["providers"]
    providers = {}
    for name, spec in raw.items():
        lc = spec["langchain"]
        providers[name] = Provider(
            name=name,
            label=spec["label"],
            base_url=spec["base_url"],
            key_env=spec["key_env"],
            langchain_provider=lc["provider"],
            base_url_override=lc.get("base_url_override", False),
            supports_max_retries=lc.get("supports_max_retries", True),
            models=spec.get("models", []),
            discover=spec.get("discover"),
        )
    return providers


def list_models(provider: Provider) -> list[str]:
    """The models this provider offers. Raises if a live listing fails."""
    if provider.discover == "openrouter_free":
        resp = httpx.get(f"{provider.base_url}/models", timeout=10)
        resp.raise_for_status()
        ids = (m["id"] for m in resp.json()["data"])
        return sorted(i for i in ids if i.endswith(":free"))
    if provider.discover == "ollama":
        root = provider.base_url.removesuffix("/v1")
        resp = httpx.get(f"{root}/api/tags", timeout=2)
        resp.raise_for_status()
        names = (m["name"] for m in resp.json()["models"])
        return sorted(n for n in names if "embed" not in n and "minilm" not in n)
    return provider.models


def get_chat_model(provider: Provider, model: str) -> BaseChatModel:
    """The only place provider quirks get translated into a chat model.

    max_retries=0 because the SDKs retry rate limits on their own, which
    would hide failures and stretch latency. Retrying is a decision for a
    later lab, not something to inherit by accident.
    """
    key = provider.api_key
    if not key:
        raise RuntimeError(f"{provider.key_env} is not set")

    kwargs = dict(model=model, model_provider=provider.langchain_provider, api_key=key)
    if provider.supports_max_retries:
        kwargs["max_retries"] = 0
    if provider.base_url_override:
        kwargs["base_url"] = provider.base_url
    return init_chat_model(**kwargs)


def chat(provider: Provider, model: str, messages: list[dict]) -> LLMResult:
    """Send a conversation, as [{"role": ..., "content": ...}], to one model."""
    llm = get_chat_model(provider, model)

    start = time.perf_counter()
    resp = llm.invoke(messages)
    latency_ms = (time.perf_counter() - start) * 1000

    usage = resp.usage_metadata or {}
    return LLMResult(
        text=resp.text,  # .text flattens block content (Gemini returns a list)
        provider=provider.name,
        model=model,
        latency_ms=latency_ms,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )
