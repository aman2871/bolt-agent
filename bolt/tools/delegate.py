"""The delegate tool: the only path by which one agent gets work done by another.

Routing through here is what enforces depth limits, roster permissions and cycle
detection. Agents never call each other directly.
"""

from __future__ import annotations

import asyncio
import os

from bolt.core import bus, roster, store
from bolt.core.tools import tool

MAX_DEPTH = int(os.environ.get("BOLT_MAX_DEPTH", "3"))
BACKEND = os.environ.get("BOLT_BUS", "inproc")  # "inproc" or "redis"
RESULT_TIMEOUT = int(os.environ.get("BOLT_DELEGATE_TIMEOUT", "600"))


@tool(description=(
        "Hand one clearly scoped subtask to a named specialist agent and wait for its "
        "answer. Include all context the specialist needs; it cannot see your conversation."),
      agent="Specialist name, one of: scout, forge, sentinel, archivist, ops.",
      task="The complete, self-contained instruction for that specialist.")
async def delegate(agent: str, task: str, ctx=None) -> str:
    if ctx is None:
        return "ERROR: delegate is only available inside an agent run"

    target = agent.strip().lower()
    try:
        spec = roster.get(target)
    except KeyError as exc:
        return f"ERROR: {exc}"

    allowed = ctx.spec.can_delegate_to
    if allowed and target not in allowed:
        return (f"ERROR: {ctx.spec.name} may not delegate to {target}. "
                f"Allowed: {', '.join(allowed)}")
    if target == ctx.spec.name:
        return f"ERROR: {target} cannot delegate to itself"
    if ctx.depth + 1 > MAX_DEPTH:
        return f"ERROR: delegation depth limit of {MAX_DEPTH} reached; do this work yourself"

    remaining = ctx.budget_tokens - store.tokens_used_tree(ctx.root_id)
    if remaining <= 0:
        return "ERROR: the token budget for this request is exhausted"

    child_id = store.create_task(target, task, parent_id=ctx.task_id,
                                 root_id=ctx.root_id, depth=ctx.depth + 1,
                                 budget_tokens=ctx.budget_tokens)

    if BACKEND == "redis":
        result = await _via_bus(target, child_id, task, ctx)
    else:
        result = await _in_process(spec, child_id, task, ctx)

    return result


async def _in_process(spec, child_id: str, task: str, ctx) -> str:
    from bolt.core.loop import run_agent  # imported here to avoid a circular import

    outcome = await run_agent(spec, task, task_id=child_id, root_id=ctx.root_id,
                              depth=ctx.depth + 1, budget_tokens=ctx.budget_tokens,
                              on_event=ctx.extra.get("on_event"))
    if not outcome.ok:
        return f"[{spec.name} FAILED] {outcome.error}"
    return f"[{spec.name} result]\n{outcome.content}"


async def _via_bus(target: str, child_id: str, task: str, ctx) -> str:
    await bus.submit(target, {
        "task_id": child_id, "root_id": ctx.root_id, "agent": target, "prompt": task,
        "depth": ctx.depth + 1, "budget_tokens": ctx.budget_tokens,
        "reply_to": ctx.task_id,
    })
    result = await bus.await_result(child_id, timeout=RESULT_TIMEOUT)
    if result is None:
        store.set_status(child_id, "failed", error="timed out waiting for a worker")
        return (f"[{target} TIMEOUT] no worker answered within {RESULT_TIMEOUT}s. "
                f"Is a worker running? (`bolt worker {target}`)")
    if not result.get("ok"):
        return f"[{target} FAILED] {result.get('error')}"
    return f"[{target} result]\n{result.get('content')}"
