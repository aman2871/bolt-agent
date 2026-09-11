"""The model catalogue: what each model can do, declared and observed.

Declared capabilities come from config/models.yaml. Observed capabilities come
from `probe()`, which sends a real request and records what actually happened —
that always wins, because several models advertise support they then reject.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pathlib import Path
from typing import Any

import yaml

from bolt.core import store

CATALOGUE_PATH = Path(os.environ.get(
    "BOLT_MODELS", Path(__file__).resolve().parent.parent / "config" / "models.yaml"))

TIER_RANK = {"fast": 0, "balanced": 1, "deep": 2}

# Measured output rate, in tokens/sec, that separates the speed bands.
FAST_TOKENS_SEC = 80.0
SLOW_TOKENS_SEC = 50.0


@dataclass
class ModelInfo:
    name: str
    tier: str = "balanced"              # declared; measured_tier overrides when known
    vision: bool = False
    context: int = 128_000
    cost: float = 0.5
    strengths: list[str] = field(default_factory=list)
    notes: str = ""
    probed: dict[str, bool] = field(default_factory=dict)     # capability -> observed
    speed: float | None = None                                # measured tokens/sec

    @property
    def vision_probed(self) -> bool | None:
        return self.probed.get("vision")

    @property
    def vision_confirmed(self) -> bool:
        """Declared vision, unless a probe proved otherwise."""
        return self.vision if self.vision_probed is None else self.vision_probed

    def supports(self, capability: str) -> bool:
        """Whether this model has a capability, preferring observation to claims."""
        if capability in self.probed:
            return self.probed[capability]
        if capability == "vision":
            return self.vision
        return capability in self.strengths

    @property
    def speed_band(self) -> str:
        """How quick it is, measured. Deliberately not the same thing as tier:
        being slow is not evidence of being clever, and the capability probes
        found no ability gap between these models to justify conflating them."""
        if self.speed is None:
            return "unmeasured"
        if self.speed >= FAST_TOKENS_SEC:
            return "quick"
        if self.speed >= SLOW_TOKENS_SEC:
            return "steady"
        return "slow"

    @property
    def tier_rank(self) -> int:
        return TIER_RANK.get(self.tier, 1)

    @property
    def passed(self) -> list[str]:
        return sorted(c for c, ok in self.probed.items() if ok)

    @property
    def failed(self) -> list[str]:
        return sorted(c for c, ok in self.probed.items() if not ok)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "tier": self.tier,
                "speed_band": self.speed_band, "vision": self.vision_confirmed,
                "vision_probed": self.vision_probed, "context": self.context,
                "cost": self.cost, "strengths": self.strengths, "notes": self.notes,
                "probed": self.probed, "speed": self.speed,
                "passed": self.passed, "failed": self.failed}


# Models the user added from the dashboard, on top of the shipped catalogue.
EXTRA_KEY = "models.extra"


def _extra() -> dict[str, dict[str, Any]]:
    import json
    try:
        raw = store.get_setting(EXTRA_KEY)
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def add(name: str, cfg: dict[str, Any] | None = None) -> None:
    """Register a model the shipped catalogue does not know about."""
    import json
    data = _extra()
    data[name] = {**data.get(name, {}), **(cfg or {})}
    store.set_setting(EXTRA_KEY, json.dumps(data))


def forget(name: str) -> None:
    import json
    data = _extra()
    data.pop(name, None)
    store.set_setting(EXTRA_KEY, json.dumps(data))


def is_builtin(name: str) -> bool:
    data = yaml.safe_load(CATALOGUE_PATH.read_text(encoding="utf-8")) or {}
    return name in (data.get("models") or {})


async def discovered() -> list[str]:
    """What the configured Ollama endpoint actually serves, right now."""
    from bolt.core import llm
    response = await llm.client().get("/models")
    response.raise_for_status()
    return sorted(m.get("id", "") for m in (response.json().get("data") or []) if m.get("id"))


def _declared() -> dict[str, ModelInfo]:
    data = yaml.safe_load(CATALOGUE_PATH.read_text(encoding="utf-8")) or {}
    declared = {**(data.get("models") or {}), **_extra()}
    out: dict[str, ModelInfo] = {}
    for name, cfg in declared.items():
        cfg = cfg or {}
        out[name] = ModelInfo(
            name=name, tier=cfg.get("tier", "balanced"), vision=bool(cfg.get("vision")),
            context=int(cfg.get("context", 128_000)), cost=float(cfg.get("cost", 0.5)),
            strengths=list(cfg.get("strengths") or []), notes=cfg.get("notes", ""))
    return out


def catalogue() -> dict[str, ModelInfo]:
    """Declared catalogue with every probe result merged in."""
    observed = store.probes()
    out: dict[str, ModelInfo] = {}
    for name, info in _declared().items():
        rows = observed.get(name, {})
        probed = {cap: row["supported"] for cap, row in rows.items() if cap != "speed"}
        speed_row = rows.get("speed")
        out[name] = ModelInfo(**{**vars(info), "probed": probed,
                                 "speed": speed_row.get("metric") if speed_row else None})
    return out


def capabilities_seen() -> list[str]:
    """Every capability any probe has recorded, in a stable order."""
    from bolt.core import probes
    seen = {cap for rows in store.probes().values() for cap in rows if cap != "speed"}
    ordered = [c for c in probes.CAPABILITIES if c in seen]
    return ordered + sorted(seen - set(ordered))


def supporting(capability: str) -> list[str]:
    """Models observed to have a capability, cheapest first."""
    return [m.name for m in sorted(catalogue().values(), key=lambda m: m.cost)
            if m.supports(capability)]


def discriminating() -> list[str]:
    """Capabilities that actually separate the models.

    If every model passes a probe, that probe tells the router nothing; surfacing
    which capabilities are uniform stops us pretending a ranking exists.
    """
    entries = list(catalogue().values())
    out = []
    for cap in capabilities_seen():
        results = {m.supports(cap) for m in entries if cap in m.probed}
        if len(results) > 1:
            out.append(cap)
    return out


def get(name: str) -> ModelInfo:
    """Catalogue entry, or a neutral default for a model we know nothing about."""
    return catalogue().get(name) or ModelInfo(name=name)


def names() -> list[str]:
    return sorted(_declared())


def vision_models() -> list[str]:
    """Models that can accept images, cheapest first."""
    return [m.name for m in sorted(catalogue().values(), key=lambda m: m.cost)
            if m.vision_confirmed]


async def probe_all(models: list[str] | None = None,
                    capabilities: list[str] | None = None,
                    on_result=None) -> dict[str, list]:
    """Run the full capability suite. See bolt.core.probes for the tests."""
    from bolt.core import probes
    return await probes.run_all(models or names(), capabilities, on_result=on_result)
