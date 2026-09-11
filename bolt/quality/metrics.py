"""Aggregate views over the runs and tasks tables."""

from __future__ import annotations

import time
from typing import Any

from bolt.core import store

# Rough per-million-token prices, only used to give cost a sense of scale.
COST_PER_MTOK: dict[str, float] = {
    "kimi-k3:cloud": 1.0,
    "kimi-k2.7-code:cloud": 1.0,
    "glm-5.2:cloud": 0.6,
    "deepseek-v4-pro:cloud": 0.8,
    "deepseek-v4-flash:cloud": 0.2,
    "nemotron-3-super:cloud": 0.4,
}
DEFAULT_COST_PER_MTOK = 0.5


def estimate_cost(model: str, tokens: int) -> float:
    return tokens / 1_000_000 * COST_PER_MTOK.get(model, DEFAULT_COST_PER_MTOK)


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def agent_stats(since_hours: float | None = None) -> list[dict[str, Any]]:
    """Per-agent latency, token and success figures."""
    conn = store.connect()
    cutoff = time.time() - since_hours * 3600 if since_hours else 0

    rows = conn.execute(
        "SELECT agent, model, latency_ms, attempts, ok, prompt_tokens, completion_tokens"
        " FROM runs WHERE created_at >= ?", (cutoff,)).fetchall()

    by_agent: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = by_agent.setdefault(row["agent"], {
            "agent": row["agent"], "model": row["model"], "calls": 0, "failed_calls": 0,
            "retries": 0, "tokens": 0, "latencies": [], "cost_usd": 0.0,
        })
        tokens = row["prompt_tokens"] + row["completion_tokens"]
        entry["calls"] += 1
        entry["failed_calls"] += 0 if row["ok"] else 1
        entry["retries"] += max(0, row["attempts"] - 1)
        entry["tokens"] += tokens
        entry["latencies"].append(row["latency_ms"])
        entry["cost_usd"] += estimate_cost(row["model"], tokens)

    task_rows = conn.execute(
        "SELECT agent, status, COUNT(*) AS n FROM tasks WHERE created_at >= ?"
        " GROUP BY agent, status", (cutoff,)).fetchall()
    tasks: dict[str, dict[str, int]] = {}
    for row in task_rows:
        tasks.setdefault(row["agent"], {})[row["status"]] = row["n"]

    out = []
    for agent, entry in by_agent.items():
        latencies = entry.pop("latencies")
        counts = tasks.get(agent, {})
        done = counts.get("done", 0)
        total = sum(counts.values())
        out.append({
            **entry,
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "tasks": total,
            "tasks_done": done,
            "success_rate": round(done / total, 3) if total else None,
            "cost_usd": round(entry["cost_usd"], 4),
        })
    return sorted(out, key=lambda e: e["agent"])


def totals(since_hours: float | None = None) -> dict[str, Any]:
    stats = agent_stats(since_hours)
    tokens = sum(s["tokens"] for s in stats)
    calls = sum(s["calls"] for s in stats)
    tasks = sum(s["tasks"] for s in stats)
    done = sum(s["tasks_done"] for s in stats)
    return {
        "agents": len(stats), "calls": calls, "tokens": tokens, "tasks": tasks,
        "tasks_done": done,
        "success_rate": round(done / tasks, 3) if tasks else None,
        "retries": sum(s["retries"] for s in stats),
        "cost_usd": round(sum(s["cost_usd"] for s in stats), 4),
    }


def task_tree(root_id: str) -> dict[str, Any] | None:
    """The task DAG for one request, with per-node token counts."""
    conn = store.connect()
    rows = conn.execute(
        "SELECT t.*, COALESCE(SUM(r.prompt_tokens + r.completion_tokens), 0) AS tokens,"
        " COUNT(r.id) AS calls FROM tasks t LEFT JOIN runs r ON r.task_id = t.id"
        " WHERE t.root_id = ? GROUP BY t.id ORDER BY t.created_at", (root_id,)).fetchall()
    if not rows:
        return None

    nodes = {row["id"]: {**dict(row), "children": []} for row in rows}
    root: dict[str, Any] | None = None
    for node in nodes.values():
        parent = nodes.get(node["parent_id"])
        if parent is None:
            root = root or node
        else:
            parent["children"].append(node)
    return root


def recent_tasks(limit: int = 50, roots_only: bool = True) -> list[dict[str, Any]]:
    clause = "WHERE t.parent_id IS NULL" if roots_only else ""
    rows = store.connect().execute(
        f"SELECT t.id, t.agent, t.status, t.depth, t.prompt, t.created_at, t.finished_at,"
        f" COALESCE(SUM(r.prompt_tokens + r.completion_tokens), 0) AS tokens"
        f" FROM tasks t LEFT JOIN runs r ON r.task_id = t.id {clause}"
        f" GROUP BY t.id ORDER BY t.created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
