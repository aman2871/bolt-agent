"""In-process fan-out of agent events to connected websocket clients."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

MAX_BUFFERED = 200

_subscribers: set[asyncio.Queue] = set()
_history: deque[dict[str, Any]] = deque(maxlen=MAX_BUFFERED)


def publish(kind: str, data: dict[str, Any]) -> None:
    """Called from the agent loop. Never blocks; a slow client drops events."""
    event = {"kind": kind, **data}
    _history.append(event)
    for queue in list(_subscribers):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


def history() -> list[dict[str, Any]]:
    return list(_history)


def subscribe() -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_BUFFERED)
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    _subscribers.discard(queue)
