"""One error vocabulary for every provider's SDK.

Every provider raises its own exception types for the same handful of
situations. This module is the only place that looks at a raw exception;
the gateway branches on ErrorCategory and never on provider exception
classes.

    TIMEOUT          no reply in time. Might work next time: retry.
    TRANSIENT        5xx, the provider is struggling. Retry.
    RATE_LIMITED     too fast, not too much. Retry after a pause.
    QUOTA_EXHAUSTED  used up until a reset (per day, credits, or a plan that
                     grants zero of this model). Don't retry; try another route.
    AUTH             bad key, no billing, no access. Don't retry; another route.
    NOT_FOUND        the model is retired (410) or this account can't reach it
                     (NVIDIA's "not found for account" 404). Don't retry.
    BAD_REQUEST      the request itself is wrong. Don't retry, and don't try
                     another route either: it would fail there too.
    CONTEXT_LENGTH   too long for this model. Another route with a bigger
                     window might take it.

Ported from llm-factory's factory/errors.py, which was checked against real
failures. Changed here: TIMEOUT and NOT_FOUND are their own categories
(timeouts were TRANSIENT, 404s were AUTH), because the 410s and timeouts seen
on 2026-11-02 need different handling from the errors they were lumped with.
"""

from __future__ import annotations

import re
from enum import Enum


class ErrorCategory(str, Enum):
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH = "auth"
    NOT_FOUND = "not_found"
    BAD_REQUEST = "bad_request"
    CONTEXT_LENGTH = "context_length"


# Not every SDK hands back a clean status code (NVIDIA raises a bare
# Exception("[503] ...")), so extraction tries attributes first, then the
# message text. Only real status codes are matched, so a port number like
# 443 in "HTTPSConnectionPool(port=443)" isn't mistaken for one.
_LEADING_BRACKET = re.compile(r"^\[(\d{3})\]")
_HTTP_CODE = re.compile(r"\b(400|401|402|403|404|408|410|413|422|429|500|502|503|504)\b")
# Google puts the wait in the message: "Please retry in 23.4s" or
# "'retryDelay': '23s'".
_RETRY_IN_MESSAGE = re.compile(r"retry(?:Delay)?['\"]?\s*(?:in|:)\s*['\"]?(\d+(?:\.\d+)?)s", re.I)


def _status(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    response = getattr(exc, "response", None)
    if isinstance(getattr(response, "status_code", None), int):
        return response.status_code
    text = str(exc)
    m = _LEADING_BRACKET.match(text) or _HTTP_CODE.search(text)
    return int(m.group(1)) if m else None


def _is_context_length(message: str) -> bool:
    lowered = message.lower()
    return any(
        phrase in lowered
        for phrase in ("context length", "context window", "maximum context",
                       "too many tokens", "reduce the length")
    )


def _is_quota_exhausted(message: str) -> bool:
    """429s that waiting a moment won't fix: daily caps, used-up credits,
    or a plan that grants zero of this model."""
    lowered = message.lower().replace("_", "").replace(" ", "")
    return any(
        marker in lowered
        for marker in ("limit:0", "perday", "quotaexceeded", "dailylimit", "insufficientquota", "credit")
    )


def classify(exc: Exception) -> ErrorCategory:
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return ErrorCategory.TIMEOUT

    status = _status(exc)
    message = str(exc)

    if status == 400:
        if _is_context_length(message):
            return ErrorCategory.CONTEXT_LENGTH
        # Anthropic reports used-up credits as a 400.
        return ErrorCategory.QUOTA_EXHAUSTED if "credit balance" in message.lower() else ErrorCategory.BAD_REQUEST
    if status == 413:
        return ErrorCategory.CONTEXT_LENGTH
    if status == 422:
        return ErrorCategory.BAD_REQUEST
    if status in (401, 402, 403):
        return ErrorCategory.AUTH
    if status in (404, 410):
        return ErrorCategory.NOT_FOUND
    if status == 408:
        return ErrorCategory.TIMEOUT
    if status == 429:
        return ErrorCategory.QUOTA_EXHAUSTED if _is_quota_exhausted(message) else ErrorCategory.RATE_LIMITED

    # 5xx, or a shape we don't recognise. Treating the unknown as TRANSIENT
    # keeps it bounded by the retry count instead of failing on the spot.
    return ErrorCategory.TRANSIENT


def retry_after(exc: Exception) -> float | None:
    """Seconds the provider asked us to wait, if it said."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            return float(headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
    m = _RETRY_IN_MESSAGE.search(str(exc))
    return float(m.group(1)) if m else None
