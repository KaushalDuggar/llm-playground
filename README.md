# LLM Playground

One app for learning LLM concepts, one lab at a time. So far: a client layer
and a gateway in front of free models, with user tiers and per-user policies
(retries, timeouts, deadline, daily budgets, context limits), and a chat UI
that shows every attempt the gateway made.

## Run

```bash
export GROQ_API_KEY=... NVIDIA_API_KEY=... GOOGLE_API_KEY=... OPENROUTER_API_KEY=...
# or: cp .env.example .env and fill it in
uv run uvicorn app.server:app --reload
```

Open http://localhost:8000 and pick who to act as: Asha (Normal) or Ravi
(Premium).

## Layout

```
core/
  config/providers.yaml  hosts: base URL, key env var, how they meter quota
  config/models.yaml     named models -> routes (host + model id + context window)
  config/tiers.yaml      tier -> allowed models, fallback on/off, policy defaults and maxes
  config/users.yaml      demo users and their tier
  config.py              loads and cross-checks the YAML
  client.py              one attempt at one route -> LLMResult
  errors.py              any provider exception -> one ErrorCategory
  store.py               SQLite (data/playground.db): daily usage, cooldowns, overrides, call log
  gateway.py             call(user, model, messages): every rule, in order
app/
  server.py              FastAPI: users, policy changes, chat
  index.html             the UI
```

`core` knows nothing about the UI.

## What the gateway does, in order

1. **Policy**: the tier's defaults, replaced by the user's own values (never above the tier max).
2. **Checks**: model allowed on the tier, prompt under `max_prompt_tokens`, tokens left today.
3. **Routes**: first route only (Normal) or each in turn (Premium, fallback on).
4. **Per route**: skipped if its context window is too small or it is cooling down.
   Otherwise attempted with a timeout; timeouts, 5xx and rate limits are retried
   with jittered backoff (or the provider's `retry-after`).
5. **Stops** at the first success, a bad request (no other route would help),
   a used-up daily budget, or the deadline. The error names which one.

Failures that say something about a route are remembered in SQLite so later
requests skip it: rate limits for a few seconds, used-up quota until the
provider's daily reset, auth errors for an hour, retired models for a day.
