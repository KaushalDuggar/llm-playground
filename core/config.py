"""Reads core/config/*.yaml into plain objects, and checks they agree.

The files are re-read on every load() so edits apply without a restart; they
are small enough that this costs nothing noticeable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).with_name("config")

POLICY_FIELDS = ("retries", "timeout_s", "deadline_s", "daily_calls", "daily_tokens",
                 "max_prompt_tokens", "max_output_tokens")


@dataclass
class Provider:
    name: str
    base_url: str
    key_env: str | None          # None: no key needed (ollama)
    quota_per_model: bool
    langchain_provider: str
    base_url_override: bool
    supports_max_retries: bool

    @property
    def api_key(self) -> str | None:
        if self.key_env is None:
            # The OpenAI client refuses an empty key even when the server
            # ignores it, as Ollama does.
            return "unused"
        return os.getenv(self.key_env)


@dataclass
class Route:
    """One host serving a named model."""

    provider: Provider
    model: str                   # the provider's own spelling of the model id
    context_window: int
    max_output_tokens: int | None = None   # host's own cap, if lower than policy

    @property
    def key(self) -> str:
        return f"{self.provider.name}:{self.model}"

    @property
    def meter(self) -> str:
        """What the provider counts rate limits and quota against.

        Providers that meter per account share one meter across all their
        models, so a 429 on one Groq model means every Groq model is limited.
        """
        return self.key if self.provider.quota_per_model else self.provider.name


@dataclass
class NamedModel:
    name: str
    label: str
    routes: list[Route]


@dataclass
class PolicyField:
    default: int
    max: int
    min: int


@dataclass
class Tier:
    name: str
    label: str
    fallback: bool
    models: list[str]
    policy: dict[str, PolicyField]


@dataclass
class User:
    id: str
    name: str
    tier: str


@dataclass
class Config:
    providers: dict[str, Provider]
    models: dict[str, NamedModel]
    tiers: dict[str, Tier]
    users: dict[str, User]


def _read(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def load() -> Config:
    providers = {}
    for name, spec in _read("providers.yaml")["providers"].items():
        lc = spec["langchain"]
        providers[name] = Provider(
            name=name,
            base_url=spec["base_url"],
            key_env=spec["key_env"],
            quota_per_model=spec["quota_per_model"],
            langchain_provider=lc["provider"],
            base_url_override=lc.get("base_url_override", False),
            supports_max_retries=lc.get("supports_max_retries", True),
        )

    models = {}
    for name, spec in _read("models.yaml")["models"].items():
        routes = []
        for r in spec["routes"]:
            if r["provider"] not in providers:
                raise ValueError(f"model '{name}': unknown provider '{r['provider']}'")
            routes.append(Route(providers[r["provider"]], r["model"], r["context_window"],
                                r.get("max_output_tokens")))
        models[name] = NamedModel(name, spec["label"], routes)

    tiers = {}
    for name, spec in _read("tiers.yaml")["tiers"].items():
        unknown = [m for m in spec["models"] if m not in models]
        if unknown:
            raise ValueError(f"tier '{name}': unknown models {unknown}")
        missing = [f for f in POLICY_FIELDS if f not in spec["policy"]]
        if missing:
            raise ValueError(f"tier '{name}': policy is missing {missing}")
        policy = {
            # Zero retries is a valid choice; zero of anything else would
            # make every request fail.
            f: PolicyField(spec["policy"][f]["default"], spec["policy"][f]["max"], 0 if f == "retries" else 1)
            for f in POLICY_FIELDS
        }
        tiers[name] = Tier(name, spec["label"], spec["fallback"], spec["models"], policy)

    users = {}
    for uid, spec in _read("users.yaml")["users"].items():
        if spec["tier"] not in tiers:
            raise ValueError(f"user '{uid}': unknown tier '{spec['tier']}'")
        users[uid] = User(uid, spec["name"], spec["tier"])

    return Config(providers, models, tiers, users)
