"""One attempt at one model on one host: build the LangChain model, call it, time it.

This layer makes exactly one attempt. Retries, fallback, budgets and health
all live in gateway.py, which calls this.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from core.config import Provider


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: float      # wall-clock time for the whole call, as the user feels it
    input_tokens: int
    output_tokens: int
    truncated: bool = False  # stopped at max_tokens rather than finishing

    @property
    def tokens_per_sec(self) -> float:
        # Includes network and time to first token, so it understates the
        # model's raw generation speed.
        return self.output_tokens / (self.latency_ms / 1000) if self.latency_ms > 0 else 0.0


def get_chat_model(provider: Provider, model: str, timeout: float | None = None,
                   max_tokens: int | None = None) -> BaseChatModel:
    """The only place provider quirks get translated into a chat model.

    max_retries=0 because the SDKs retry rate limits on their own, which
    would hide failures and stretch latency. Retrying is the gateway's job.
    """
    key = provider.api_key
    if not key:
        raise RuntimeError(f"{provider.key_env} is not set")

    kwargs = dict(model=model, model_provider=provider.langchain_provider, api_key=key)
    if provider.supports_max_retries:
        kwargs["max_retries"] = 0
    if provider.base_url_override:
        kwargs["base_url"] = provider.base_url
    if timeout is not None:
        # Every class here accepts `timeout`; for most it is how long the
        # connection may sit silent, not a cap on the whole call.
        kwargs["timeout"] = timeout
    if max_tokens is not None:
        # Set here, not with .bind() per call: Google's class only accepts
        # it at construction and rejects it as a call argument.
        kwargs["max_tokens"] = max_tokens
    return init_chat_model(**kwargs)


def chat(provider: Provider, model: str, messages: list[dict], timeout: float | None = None,
         max_tokens: int | None = None) -> LLMResult:
    """Send a conversation, as [{"role": ..., "content": ...}], to one model."""
    llm = get_chat_model(provider, model, timeout, max_tokens)

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
        # "length" from OpenAI-style APIs, "MAX_TOKENS" from Google.
        truncated=str(resp.response_metadata.get("finish_reason", "")).lower() in ("length", "max_tokens"),
    )
