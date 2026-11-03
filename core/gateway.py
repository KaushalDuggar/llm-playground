"""The one door every chat request goes through.

    call(user, model, messages)
      1. policy   the tier's defaults, with the user's own overrides
      2. checks   is the model in the user's tier? is the prompt within
                  max_prompt_tokens? is any of today's token budget left?
      3. routes   the first route only, or every route if the tier has fallback
      4. each route is skipped if its context window is too small or it is
         cooling down; otherwise it is attempted, and failures that might
         heal (timeout, 5xx, rate limit) are retried with backoff
      5. it stops at the first success, an error no route can fix, a used-up
         budget, or the deadline

Every attempt and skip goes into the trace returned with the answer (or the
error), so the UI can show exactly what happened and which limit stopped it.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from core import client, config, store
from core.config import Route
from core.errors import ErrorCategory, classify, retry_after

RETRYABLE = {ErrorCategory.TIMEOUT, ErrorCategory.TRANSIENT, ErrorCategory.RATE_LIMITED}

# Google resets daily quotas at midnight Pacific. Other providers' daily caps
# vary; this is the best single guess for "used up for today".
QUOTA_RESET_TZ = ZoneInfo("America/Los_Angeles")


@dataclass
class Attempt:
    provider: str
    model: str
    outcome: str            # "ok", "skipped", or an ErrorCategory value
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class GatewayResult:
    text: str
    named_model: str
    provider: str           # the route that answered
    model: str
    latency_ms: float       # the whole request: every attempt and wait
    input_tokens: int
    output_tokens: int
    max_output_tokens: int  # the cap this answer ran under
    truncated: bool         # stopped at that cap rather than finishing
    attempts: list[Attempt] = field(default_factory=list)


class GatewayError(Exception):
    """A request the gateway gave up on.

    reason says which rule stopped it: not_allowed, prompt_too_long, budget,
    deadline, bad_request or all_routes_failed.
    """

    def __init__(self, reason: str, message: str, attempts: list[Attempt] | None = None):
        super().__init__(message)
        self.reason = reason
        self.attempts = attempts or []


# --- policy -------------------------------------------------------------------

def policy_for(user: config.User, tier: config.Tier) -> dict[str, int]:
    """The tier's defaults, replaced by the user's overrides within limits."""
    own = store.overrides(user.id)
    return {
        name: min(max(own.get(name, f.default), f.min), f.max)
        for name, f in tier.policy.items()
    }


def set_policy(user_id: str, name: str, value: int | None) -> None:
    """Change one of a user's policy values. None goes back to the default."""
    cfg = config.load()
    tier = cfg.tiers[cfg.users[user_id].tier]
    if name not in tier.policy:
        raise ValueError(f"unknown policy field '{name}'")
    f = tier.policy[name]
    if value is not None and not f.min <= value <= f.max:
        raise ValueError(f"{name} must be between {f.min} and {f.max} for the {tier.label} tier")
    store.set_override(user_id, name, value)


# --- the call -----------------------------------------------------------------

def call(user_id: str, named_model: str, messages: list[dict]) -> GatewayResult:
    cfg = config.load()
    user = cfg.users[user_id]
    tier = cfg.tiers[user.tier]
    policy = policy_for(user, tier)

    if named_model not in tier.models:
        raise GatewayError("not_allowed", f"{named_model} is not available on the {tier.label} tier")

    prompt_tokens = estimate_tokens(messages)
    if prompt_tokens > policy["max_prompt_tokens"]:
        raise GatewayError(
            "prompt_too_long",
            f"Conversation is ~{prompt_tokens:,} tokens; your limit is "
            f"{policy['max_prompt_tokens']:,}. Start a new chat.",
        )

    if store.usage(user.id)["tokens"] >= policy["daily_tokens"]:
        raise GatewayError("budget", f"Used all {policy['daily_tokens']:,} tokens for today. {_resets()}")

    model = cfg.models[named_model]
    routes = model.routes if tier.fallback else model.routes[:1]
    attempts: list[Attempt] = []
    out_of_time = False  # a retry was dropped because its wait would pass the deadline
    start = time.monotonic()  # a stopwatch: never jumps when the clock is corrected

    def time_left() -> float:
        return policy["deadline_s"] - (time.monotonic() - start)

    for route in routes:
        skip = _skip_reason(route, prompt_tokens)
        if skip:
            attempts.append(Attempt(route.provider.name, route.model, "skipped", skip))
            continue

        max_tokens = min(policy["max_output_tokens"], route.max_output_tokens or policy["max_output_tokens"])
        for attempt_no in range(policy["retries"] + 1):
            if time_left() <= 0:
                raise GatewayError(
                    "deadline",
                    f"Deadline of {policy['deadline_s']}s reached after {_count(attempts)} attempts. "
                    f"Last error: {_last_error(attempts)}",
                    attempts,
                )
            if not store.take_call(user.id, policy["daily_calls"]):
                raise GatewayError(
                    "budget", f"Used all {policy['daily_calls']} calls for today. {_resets()}", attempts
                )

            t0 = time.monotonic()
            try:
                # Never let one attempt run past the deadline.
                result = client.chat(route.provider, route.model, messages,
                                     timeout=min(policy["timeout_s"], time_left()),
                                     max_tokens=max_tokens)
            except Exception as exc:
                category = classify(exc)
                detail = _short(exc)
                attempts.append(Attempt(route.provider.name, route.model, category.value, detail, _ms(t0)))
                store.log_attempt(user.id, named_model, route.provider.name, route.model,
                                  category.value, _ms(t0), error=detail)
                wait = retry_after(exc)
                _record_health(route, category, wait, detail)

                if category is ErrorCategory.BAD_REQUEST:
                    # The request itself is wrong; another route would reject it too.
                    raise GatewayError("bad_request", detail, attempts)
                if category not in RETRYABLE or attempt_no == policy["retries"]:
                    break  # on to the next route, if there is one
                pause = wait if wait is not None else _backoff(attempt_no)
                if pause >= time_left():
                    out_of_time = True
                    break  # waiting would pass the deadline; the next route might not need to
                time.sleep(pause)
                continue

            store.add_tokens(user.id, result.input_tokens + result.output_tokens)
            store.clear_cooldowns([route.key, route.meter])
            store.log_attempt(user.id, named_model, route.provider.name, route.model, "ok",
                              result.latency_ms, result.input_tokens, result.output_tokens)
            attempts.append(Attempt(route.provider.name, route.model, "ok", "", result.latency_ms))
            return GatewayResult(
                text=result.text,
                named_model=named_model,
                provider=route.provider.name,
                model=route.model,
                latency_ms=_ms(start),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                max_output_tokens=max_tokens,
                truncated=result.truncated,
                attempts=attempts,
            )

    if not _count(attempts):
        reasons = "; ".join(f"{a.provider}: {a.detail}" for a in attempts)
        raise GatewayError("all_routes_failed", f"No route could take this request. {reasons}", attempts)
    if out_of_time:
        raise GatewayError(
            "deadline",
            f"Deadline of {policy['deadline_s']}s would pass before the next retry, after "
            f"{_count(attempts)} attempts. Last error: {_last_error(attempts)}",
            attempts,
        )
    raise GatewayError(
        "all_routes_failed",
        f"Gave up after {_count(attempts)} attempts (retries used up). Last error: {_last_error(attempts)}"
        f"{_hint(attempts)}",
        attempts,
    )


def _skip_reason(route: Route, prompt_tokens: int) -> str | None:
    if prompt_tokens > route.context_window:
        return f"prompt ~{prompt_tokens:,} tokens exceeds its {route.context_window:,} window"
    if not route.provider.api_key:
        return f"{route.provider.key_env} not set"
    cooldown = store.active_cooldown([route.key, route.meter])
    if cooldown:
        return f"cooling down after {cooldown['category']} until {_clock(cooldown['until'])}"
    return None


def _record_health(route: Route, category: ErrorCategory, wait: float | None, detail: str) -> None:
    """Remember failures that say something about the route, not this request.

    Rate limits and quota belong to the meter (the whole account, for
    providers that meter per account). Auth and not-found belong to the one
    route: NVIDIA refusing one model says nothing about its others.
    Timeouts, 5xx and bad requests change nothing here.
    """
    if category is ErrorCategory.RATE_LIMITED:
        store.cool_down(route.meter, category.value, wait or 10, detail)
    elif category is ErrorCategory.QUOTA_EXHAUSTED:
        # Ignore retry-after here: Google sends "retry in 40s" even for a
        # daily cap of zero, and retrying every 40s would only waste calls.
        store.cool_down(route.meter, category.value, _seconds_until_quota_reset(), detail)
    elif category is ErrorCategory.AUTH:
        store.cool_down(route.key, category.value, 3600, detail)
    elif category is ErrorCategory.NOT_FOUND:
        store.cool_down(route.key, category.value, 86400, detail)


def _backoff(attempt_no: int) -> float:
    """1s, 2s, 4s… with jitter: each wait lands somewhere between half and
    all of that, so parallel callers that failed together don't retry together."""
    base = min(2 ** attempt_no, 30)
    return base / 2 + random.uniform(0, base / 2)


def estimate_tokens(messages: list[dict]) -> int:
    """Rough count: about 4 characters per token in English, plus a little
    per message for the role markers. Real tokenizers differ by model; this
    only needs to be close enough to catch a conversation that won't fit."""
    return sum(len(m["content"]) // 4 + 4 for m in messages)


# --- for the UI ---------------------------------------------------------------

def describe(user_id: str) -> dict:
    """Everything the UI shows for one user: tier, models and their routes'
    health, policy values and limits, and today's usage."""
    cfg = config.load()
    user = cfg.users[user_id]
    tier = cfg.tiers[user.tier]
    policy = policy_for(user, tier)
    models = []
    for name in tier.models:
        m = cfg.models[name]
        routes = m.routes if tier.fallback else m.routes[:1]
        models.append({
            "name": name,
            "label": m.label,
            # The biggest prompt any usable route takes; smaller routes are
            # skipped, so this is the model's real ceiling.
            "context_window": max(r.context_window for r in routes),
            "routes": [
                {"provider": r.provider.name, "model": r.model, "context_window": r.context_window,
                 "unavailable": _skip_reason(r, 0)}
                for r in routes
            ],
        })
    return {
        "user": {"id": user.id, "name": user.name},
        "tier": {"name": tier.name, "label": tier.label, "fallback": tier.fallback},
        "models": models,
        "policy": {
            name: {"value": policy[name], "default": f.default, "min": f.min, "max": f.max}
            for name, f in tier.policy.items()
        },
        "usage": {**store.usage(user.id), "resets_at": store.next_reset().strftime("%I:%M %p IST, %d %b")},
    }


# --- small helpers ------------------------------------------------------------

def _count(attempts: list[Attempt]) -> int:
    return sum(a.outcome != "skipped" for a in attempts)


def _last_error(attempts: list[Attempt]) -> str:
    failed = [a for a in attempts if a.outcome not in ("ok", "skipped")]
    if not failed:
        return "none"
    a = failed[-1]
    return f"{a.provider} {a.outcome}: {a.detail}"


_HINTS = {
    ErrorCategory.TOO_LARGE.value: " The answer limit is bigger than this host allows per minute: "
                                   "lower Max output tokens in your policy.",
    ErrorCategory.CONTEXT_LENGTH.value: " The conversation is too long for this model: start a new chat.",
}


def _hint(attempts: list[Attempt]) -> str:
    """What the user can do about the last failure, when there is something."""
    failed = [a for a in attempts if a.outcome not in ("ok", "skipped")]
    return _HINTS.get(failed[-1].outcome, "") if failed else ""


def _short(exc: Exception) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return f"{type(exc).__name__}: {text}"[:300]


def _ms(since: float) -> float:
    return (time.monotonic() - since) * 1000


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts, store.BUDGET_TZ).strftime("%I:%M %p IST, %d %b")


def _resets() -> str:
    return f"Resets at {store.next_reset().strftime('%I:%M %p IST')}."


def _seconds_until_quota_reset() -> float:
    now = datetime.now(QUOTA_RESET_TZ)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (midnight - now).total_seconds()
