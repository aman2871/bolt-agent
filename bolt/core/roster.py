"""Load the agent roster: YAML defaults, plus whatever the dashboard saved.

The YAML file is the shipped roster — the seven agents everyone starts with. On
top of it sit user edits and user-created agents, stored as one JSON blob in the
settings table so a person can add an agent from the UI without editing files in
a container.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from bolt.core.loop import AgentSpec

ROSTER_PATH = Path(os.environ.get(
    "BOLT_ROSTER", Path(__file__).resolve().parent.parent / "config" / "agents.yaml"))

# Settings key holding {name: {field: value}} — edits to shipped agents and
# whole agents that only exist here.
OVERRIDES_KEY = "roster.agents"

_cache: dict[str, Any] = {"stamp": None, "specs": None}


def _overrides() -> dict[str, dict[str, Any]]:
    try:
        from bolt.core import store
        raw = store.get_setting(OVERRIDES_KEY)
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _stamp() -> str:
    try:
        from bolt.core import store
        return store.get_setting(OVERRIDES_KEY) or ""
    except Exception:
        return ""


def _spec(name: str, merged: dict[str, Any]) -> AgentSpec:
    return AgentSpec(
        name=name,
        model=merged.get("model", "auto"),
        blurb=str(merged.get("blurb") or "").strip(),
        system_prompt=str(merged.get("system_prompt", "")).strip(),
        tools=list(merged.get("tools") or []),
        max_iterations=int(merged.get("max_iterations", 8)),
        max_tokens=merged.get("max_tokens"),
        temperature=float(merged.get("temperature", 0.2)),
        effort=str(merged.get("effort", "default")),
        can_delegate_to=list(merged.get("can_delegate_to") or []),
        requires=list(merged.get("requires") or []),
        prefers=list(merged.get("prefers") or []),
    )


def defaults() -> dict[str, dict[str, Any]]:
    """The shipped roster, as plain dicts."""
    data = yaml.safe_load(ROSTER_PATH.read_text(encoding="utf-8")) or {}
    base = data.get("defaults") or {}
    return {name: {**base, **cfg} for name, cfg in (data.get("agents") or {}).items()}


def load() -> dict[str, AgentSpec]:
    stamp = _stamp()
    if _cache["specs"] is not None and _cache["stamp"] == stamp:
        return _cache["specs"]

    merged = defaults()
    for name, patch in _overrides().items():
        if not isinstance(patch, dict):
            continue
        merged[name] = {**merged.get(name, {}), **patch}

    specs = {name: _spec(name, cfg) for name, cfg in merged.items()
             if cfg.get("system_prompt")}
    _cache.update(stamp=stamp, specs=specs)
    return specs


def is_builtin(name: str) -> bool:
    return name in defaults()


def save(name: str, patch: dict[str, Any]) -> AgentSpec:
    """Create or edit an agent. Only the fields given are changed."""
    from bolt.core import store
    data = _overrides()
    data[name] = {**data.get(name, {}), **patch}
    store.set_setting(OVERRIDES_KEY, json.dumps(data))
    _cache["specs"] = None
    return get(name)


def remove(name: str) -> None:
    """Drop a user-created agent, or reset a shipped one to its defaults."""
    from bolt.core import store
    data = _overrides()
    data.pop(name, None)
    store.set_setting(OVERRIDES_KEY, json.dumps(data))
    _cache["specs"] = None


def get(name: str) -> AgentSpec:
    specs = load()
    key = name.strip().lower()
    if key not in specs:
        raise KeyError(f"unknown agent {name!r}. Known agents: {', '.join(sorted(specs))}")
    return specs[key]


def names() -> list[str]:
    return sorted(load())
