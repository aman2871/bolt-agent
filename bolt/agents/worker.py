"""Generic worker daemon: consume one agent's inbox, run the loop, publish results.

There is no per-agent worker code. Which agent a worker embodies is a CLI argument.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any

from bolt.core import bus, roster, store
from bolt.core.loop import run_agent


async def handle(envelope: dict[str, Any], on_event=None) -> dict[str, Any]:
    agent = envelope["agent"]
    spec = roster.get(agent)
    task_id = envelope["task_id"]

    if store.get_task(task_id) is None:
        # Redelivery after a crash on the producer side, or a task created elsewhere.
        store.create_task(agent, envelope["prompt"], parent_id=envelope.get("reply_to"),
                          root_id=envelope.get("root_id"), depth=envelope.get("depth", 0),
                          budget_tokens=envelope.get("budget_tokens", 200_000))

    result = await run_agent(
        spec, envelope["prompt"], task_id=task_id, root_id=envelope.get("root_id"),
        depth=envelope.get("depth", 0), budget_tokens=envelope.get("budget_tokens", 200_000),
        extra={"on_event": on_event}, on_event=on_event,
    )
    return {"task_id": task_id, "agent": agent, "ok": result.ok,
            "content": result.content, "error": result.error, "tokens": result.tokens}


async def serve(agent: str, on_event=None) -> None:
    """Consume this agent's inbox until interrupted."""
    roster.get(agent)  # fail fast on an unknown name
    await bus.ensure_group(agent)
    stopping = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopping.set)

    if on_event:
        on_event("worker_ready", {"agent": agent})

    while not stopping.is_set():
        try:
            batch = await bus.consume(agent, block_ms=2000)
        except Exception as exc:
            if on_event:
                on_event("worker_error", {"agent": agent, "error": str(exc)})
            await asyncio.sleep(1)
            continue

        for entry_id, envelope in batch:
            try:
                result = await handle(envelope, on_event)
            except Exception as exc:
                result = {"task_id": envelope.get("task_id"), "agent": agent, "ok": False,
                          "content": "", "error": f"{type(exc).__name__}: {exc}"}
                store.set_status(envelope["task_id"], "failed", error=result["error"])
            await bus.publish_result(envelope["task_id"], result)
            await bus.ack(agent, entry_id)

    await bus.aclose()
