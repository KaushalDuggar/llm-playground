"""Everything the gateway must remember between requests, in one SQLite file.

    usage       calls and tokens per user per day (the daily budget)
    cooldowns   routes or meters to skip until a time (the health store)
    overrides   a user's own policy values, within their tier's limits
    attempts    one row per attempt sent to a provider (the call log)

SQLite rather than memory because a used-up daily quota or budget has to
survive a restart. Times saved here are wall-clock (time.time()), never
time.monotonic(): the monotonic clock restarts with the machine, so its
values mean nothing tomorrow.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "playground.db"

# Daily budgets reset at midnight in this zone. Named explicitly so a server
# whose clock runs on UTC still resets at midnight IST.
BUDGET_TZ = ZoneInfo("Asia/Kolkata")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    user TEXT, day TEXT, calls INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user, day));
CREATE TABLE IF NOT EXISTS cooldowns (
    key TEXT PRIMARY KEY, category TEXT, until REAL, error TEXT);
CREATE TABLE IF NOT EXISTS overrides (
    user TEXT, field TEXT, value INTEGER, PRIMARY KEY (user, field));
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY, ts REAL, user TEXT, named_model TEXT, provider TEXT, model TEXT,
    outcome TEXT, latency_ms REAL, input_tokens INTEGER, output_tokens INTEGER, error TEXT);
"""

_initialised = False


def _connect() -> sqlite3.Connection:
    global _initialised
    DB_PATH.parent.mkdir(exist_ok=True)
    # timeout: wait this long for another thread's write to finish instead
    # of failing with "database is locked".
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    if not _initialised:
        conn.executescript(_SCHEMA)
        _initialised = True
    return conn


def _execute(sql: str, params: tuple = ()) -> int:
    """Run a write and return how many rows it changed."""
    conn = _connect()
    with conn:  # commits on success
        changed = conn.execute(sql, params).rowcount
    conn.close()
    return changed


def _query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn = _connect()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


# --- daily budget -------------------------------------------------------------

def today() -> str:
    return datetime.now(BUDGET_TZ).date().isoformat()


def next_reset() -> datetime:
    """Next midnight in BUDGET_TZ, when today's counters stop applying."""
    now = datetime.now(BUDGET_TZ)
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def take_call(user: str, limit: int) -> bool:
    """Count one call against today's budget, if one is left.

    Check and count happen in one UPDATE, so two parallel requests can never
    both take the last call: the database runs them one after the other, and
    the second finds calls already at the limit.
    """
    day = today()
    _execute("INSERT OR IGNORE INTO usage (user, day) VALUES (?, ?)", (user, day))
    changed = _execute(
        "UPDATE usage SET calls = calls + 1 WHERE user = ? AND day = ? AND calls < ?",
        (user, day, limit),
    )
    return changed == 1


def add_tokens(user: str, tokens: int) -> None:
    # Only known once the reply arrives, so tokens are added after the call
    # and the day's last call can go slightly over the limit.
    _execute("UPDATE usage SET tokens = tokens + ? WHERE user = ? AND day = ?", (tokens, user, today()))


def usage(user: str) -> dict:
    rows = _query("SELECT calls, tokens FROM usage WHERE user = ? AND day = ?", (user, today()))
    return dict(rows[0]) if rows else {"calls": 0, "tokens": 0}


# --- cooldowns (health) -------------------------------------------------------

def cool_down(key: str, category: str, seconds: float, error: str) -> None:
    _execute(
        "INSERT OR REPLACE INTO cooldowns (key, category, until, error) VALUES (?, ?, ?, ?)",
        (key, category, time.time() + seconds, error[:500]),
    )


def clear_cooldowns(keys: list[str]) -> None:
    _execute(f"DELETE FROM cooldowns WHERE key IN ({','.join('?' * len(keys))})", tuple(keys))


def active_cooldown(keys: list[str]) -> dict | None:
    """The cooldown blocking any of these keys right now, if there is one."""
    rows = _query(
        f"SELECT * FROM cooldowns WHERE key IN ({','.join('?' * len(keys))}) AND until > ? "
        "ORDER BY until DESC LIMIT 1",
        (*keys, time.time()),
    )
    return dict(rows[0]) if rows else None


# --- policy overrides ---------------------------------------------------------

def overrides(user: str) -> dict[str, int]:
    rows = _query("SELECT field, value FROM overrides WHERE user = ?", (user,))
    return {r["field"]: r["value"] for r in rows}


def set_override(user: str, field: str, value: int | None) -> None:
    if value is None:
        _execute("DELETE FROM overrides WHERE user = ? AND field = ?", (user, field))
    else:
        _execute("INSERT OR REPLACE INTO overrides (user, field, value) VALUES (?, ?, ?)", (user, field, value))


# --- call log -----------------------------------------------------------------

def log_attempt(user: str, named_model: str, provider: str, model: str, outcome: str,
                latency_ms: float, input_tokens: int = 0, output_tokens: int = 0,
                error: str | None = None) -> None:
    _execute(
        "INSERT INTO attempts (ts, user, named_model, provider, model, outcome, latency_ms, "
        "input_tokens, output_tokens, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (time.time(), user, named_model, provider, model, outcome, latency_ms,
         input_tokens, output_tokens, error[:500] if error else None),
    )
