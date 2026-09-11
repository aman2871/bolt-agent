"""Redis Streams message bus.

One durable inbox stream per agent, read through a consumer group so a crashed
worker's task is redelivered rather than lost. Results go to a per-task reply
list (cheap blocking wait) and to a shared audit stream.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from typing import Any

import redis.asyncio as aioredis

REDIS_URL = os.environ.get("BOLT_REDIS_URL", "redis://localhost:6379/0")
GROUP = "workers"
AUDIT_STREAM = "bolt:results"
INBOX_MAXLEN = 10_000
REPLY_TTL_SECONDS = 3600
# A worker that died leaves entries pending under its own consumer name; another
# worker reclaims them once they have been idle this long.
RECLAIM_IDLE_MS = int(os.environ.get("BOLT_RECLAIM_IDLE_MS", "60000"))

_redis: aioredis.Redis | None = None
_redis_loop: asyncio.AbstractEventLoop | None = None


def redis() -> aioredis.Redis:
    """Connection for the *current* event loop.

    A client caches connections bound to the loop that created them, so a client
    built under one `asyncio.run` blows up with "Event loop is closed" under the
    next. Rebuild whenever the running loop changes.
    """
    global _redis, _redis_loop
    loop = asyncio.get_running_loop()
    if _redis is None or _redis_loop is not loop:
        # health_check_interval pings a pooled connection that has been idle,
        # which is what catches a socket stranded by a network change; without it
        # a blocked read can sit there until TCP keepalive expires, hours later.
        _redis = aioredis.from_url(
            REDIS_URL, decode_responses=True,
            socket_keepalive=True,
            socket_connect_timeout=5,
            health_check_interval=30,
            retry_on_timeout=True,
        )
        _redis_loop = loop
    return _redis


async def aclose() -> None:
    global _redis, _redis_loop
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception:
            pass
        _redis = None
        _redis_loop = None


def inbox(agent: str) -> str:
    return f"bolt:inbox:{agent}"


def reply_key(task_id: str) -> str:
    return f"bolt:reply:{task_id}"


def consumer_name() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


async def ensure_group(agent: str) -> None:
    """Create the consumer group, tolerating the case where it already exists."""
    try:
        await redis().xgroup_create(inbox(agent), GROUP, id="0", mkstream=True)
    except aioredis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def submit(agent: str, envelope: dict[str, Any]) -> str:
    """Publish a task to an agent's inbox. Returns the stream entry id."""
    await ensure_group(agent)
    return await redis().xadd(
        inbox(agent), {"payload": json.dumps(envelope)}, maxlen=INBOX_MAXLEN, approximate=True
    )


def _decode(entries: Any) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for entry_id, fields in entries or []:
        if fields and "payload" in fields:
            out.append((entry_id, json.loads(fields["payload"])))
    return out


async def reclaim(agent: str, count: int = 1) -> list[tuple[str, dict[str, Any]]]:
    """Take over entries abandoned by a worker that died before acking."""
    _next_id, entries, _deleted = await redis().xautoclaim(
        inbox(agent), GROUP, consumer_name(), min_idle_time=RECLAIM_IDLE_MS,
        start_id="0-0", count=count,
    )
    return _decode(entries)


async def consume(agent: str, block_ms: int = 5000, count: int = 1) -> list[tuple[str, dict[str, Any]]]:
    """Return work for this consumer as (entry_id, envelope).

    Order of preference: this consumer's own un-acked backlog, then entries
    abandoned by a dead worker, then new messages.
    """
    await ensure_group(agent)
    name = consumer_name()

    own = await redis().xreadgroup(GROUP, name, {inbox(agent): "0"}, count=count)
    for _stream, entries in own or []:
        if decoded := _decode(entries):
            return decoded

    if stolen := await reclaim(agent, count):
        return stolen

    fresh = await redis().xreadgroup(
        GROUP, name, {inbox(agent): ">"}, count=count, block=block_ms)
    for _stream, entries in fresh or []:
        if decoded := _decode(entries):
            return decoded
    return []


async def ack(agent: str, entry_id: str) -> None:
    await redis().xack(inbox(agent), GROUP, entry_id)


async def publish_result(task_id: str, result: dict[str, Any]) -> None:
    """Deliver a result to whoever is waiting, and record it in the audit stream."""
    payload = json.dumps(result)
    pipe = redis().pipeline()
    pipe.rpush(reply_key(task_id), payload)
    pipe.expire(reply_key(task_id), REPLY_TTL_SECONDS)
    pipe.xadd(AUDIT_STREAM, {"task_id": task_id, "payload": payload},
              maxlen=INBOX_MAXLEN, approximate=True)
    await pipe.execute()


async def await_result(task_id: str, timeout: int = 600) -> dict[str, Any] | None:
    """Block until a result for this task arrives, or the timeout elapses."""
    item = await redis().blpop([reply_key(task_id)], timeout=timeout)
    if item is None:
        return None
    return json.loads(item[1])


async def ping() -> bool:
    """Health check. Closes the client so it does not outlive a short-lived loop."""
    try:
        return bool(await redis().ping())
    except Exception:
        return False
    finally:
        await aclose()
