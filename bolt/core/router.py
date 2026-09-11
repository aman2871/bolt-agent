"""Choosing which model runs a given task.

Three levels of authority, highest first:

1. A manual override set from the dashboard or CLI (`model_override:<agent>`).
2. An explicit model on the agent in agents.yaml.
3. Automatic routing, when the agent's model is `auto`.

Routing is deliberately rule-based rather than an LLM call: it runs on every
task, so it has to be free and predictable. Hard requirements (vision) filter the
candidates; everything else only scores them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from bolt.core import models, store

OVERRIDE_PREFIX = "model_override:"
AUTO = "auto"

# Prompts that justify the slowest, strongest model.
_DEEP = re.compile(r"\b(design|architect|plan|refactor|debug|why|trade[- ]?off|"
                   r"strategy|investigate|root cause|compare|decide)\b", re.I)
# Prompts that a cheap fast model handles fine.
_SHALLOW = re.compile(r"\b(list|count|show|read|find|summari[sz]e|rename|"
                      r"what is|how many|fetch|lookup)\b", re.I)

LONG_PROMPT_CHARS = 1500
# A declared strength is worth roughly one tier step: a specialist one notch
# below the ideal tier should beat a generalist at it.
TIER_WEIGHT = 3.0
STRENGTH_WEIGHT = 4.0
VERIFIED_WEIGHT = 2.0
COST_WEIGHT = 1.0
SPEED_WEIGHT = 1.0


@dataclass
class Choice:
    model: str
    reason: str
    source: str          # "override" | "explicit" | "auto"
    candidates: list[tuple[str, float]] | None = None


def override_key(agent: str) -> str:
    return f"{OVERRIDE_PREFIX}{agent}"


def set_override(agent: str, model: str | None) -> None:
    """Pin an agent to a model, or clear the pin with None."""
    if model is None:
        store.delete_setting(override_key(agent))
    else:
        store.set_setting(override_key(agent), model)


def get_override(agent: str) -> str | None:
    return store.get_setting(override_key(agent))


def overrides() -> dict[str, str]:
    return {key[len(OVERRIDE_PREFIX):]: value
            for key, value in store.all_settings(OVERRIDE_PREFIX).items()}


def wanted_tier(prompt: str, requires: list[str]) -> str:
    """How much model the task looks like it needs."""
    if "deep" in requires:
        return "deep"
    if len(prompt) > LONG_PROMPT_CHARS or _DEEP.search(prompt):
        return "deep"
    if _SHALLOW.search(prompt) and len(prompt) < 400:
        return "fast"
    return "balanced"


def score(info: models.ModelInfo, tier: str, prefers: list[str]) -> float:
    """Higher is better.

    Capability probing found every model here equally able at tools, JSON, code,
    reasoning and long context — only vision and speed separate them. So hard
    requirements are filtered out before scoring, and what is left is decided on
    measured speed and cost. Declared strengths only break ties, because nothing
    has verified them.
    """
    target = models.TIER_RANK.get(tier, 1)
    # Overshooting the needed tier is wasteful; undershooting is worse.
    gap = info.tier_rank - target
    tier_score = -abs(gap) * (1.0 if gap > 0 else 1.6)

    overlap = len(set(prefers) & set(info.strengths))
    # A probed strength is worth more than a claimed one.
    verified = len([p for p in prefers if p in info.probed and info.probed[p]])
    cheapness = 1.0 - min(info.cost, 2.0) / 2.0
    # Among equally suitable models, prefer the one measured to be quicker.
    quickness = min((info.speed or 0.0) / 100.0, 1.0)

    return (TIER_WEIGHT * tier_score
            + STRENGTH_WEIGHT * overlap
            + VERIFIED_WEIGHT * verified
            + COST_WEIGHT * cheapness
            + SPEED_WEIGHT * quickness)


def missing_capabilities(info: models.ModelInfo, requires: list[str]) -> list[str]:
    """Required capabilities this model is not known to have."""
    return [c for c in requires if c not in ("deep", "fast") and not info.supports(c)]


def choose(agent: str, prompt: str = "", declared_model: str = AUTO,
           requires: list[str] | None = None, prefers: list[str] | None = None,
           needs_vision: bool = False) -> Choice:
    """Resolve the model for one run."""
    requires = list(requires or [])
    prefers = list(prefers or [])
    needs_vision = needs_vision or "vision" in requires

    hard = list(requires)
    if needs_vision and "vision" not in hard:
        hard.append("vision")

    def capable_fallback(why: str, source: str) -> Choice | None:
        """The cheapest model that has every hard requirement."""
        options = [m for m in sorted(models.catalogue().values(), key=lambda m: m.cost)
                   if not missing_capabilities(m, hard)]
        return Choice(options[0].name, why.format(model=options[0].name), source) if options else None

    pinned = get_override(agent)
    if pinned:
        info = models.get(pinned)
        gaps = missing_capabilities(info, hard)
        if gaps:
            # Never let a pin silently break a task that needs a capability.
            alt = capable_fallback(
                f"pinned {pinned} lacks {', '.join(gaps)}; using {{model}}", "override")
            if alt:
                return alt
        return Choice(pinned, f"pinned to {pinned}", "override")

    if declared_model and declared_model != AUTO:
        info = models.get(declared_model)
        gaps = missing_capabilities(info, hard)
        if gaps:
            alt = capable_fallback(
                f"{declared_model} lacks {', '.join(gaps)}; using {{model}}", "explicit")
            if alt:
                return alt
        return Choice(declared_model, "set in agents.yaml", "explicit")

    candidates = [m for m in models.catalogue().values()
                  if not missing_capabilities(m, hard)]
    if not candidates:
        raise RuntimeError(
            f"no configured model has: {', '.join(hard)}. "
            f"Run `bolt models --probe` to test them.")

    tier = wanted_tier(prompt, requires)
    ranked = sorted(((m, score(m, tier, prefers)) for m in candidates),
                    key=lambda pair: pair[1], reverse=True)
    best, best_score = ranked[0]

    bits = [f"{tier} tier"]
    if hard:
        bits.append("needs " + ", ".join(hard))
    if matched := set(prefers) & set(best.strengths):
        bits.append("good at " + ", ".join(sorted(matched)))
    return Choice(best.name, "auto: " + ", ".join(bits), "auto",
                  [(m.name, round(s, 2)) for m, s in ranked[:4]])
