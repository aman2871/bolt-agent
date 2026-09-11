"""Cross-process voice control.

The always-on daemon and the web server are separate processes that share one
microphone and one pair of speakers. Without coordination they both answer the
same utterance and talk over each other, and nothing the browser does can stop
the daemon mid-sentence.

State lives in one small JSON file so any process — CLI, API, daemon — can read
and change it without a running server.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

STATE_PATH = Path(os.environ.get(
    "BOLT_VOICE_STATE", Path(os.environ.get("BOLT_STATE_DIR", Path.home() / ".bolt"))
    / "voice-control.json"))

# How long a single "pause the daemon" request holds, unless renewed or released.
# Bounded so a browser tab that dies cannot mute the daemon forever.
PAUSE_SECONDS = float(os.environ.get("BOLT_VOICE_PAUSE_SECONDS", "120"))

# stop_seq is a counter, not a timestamp: two calls to time.time() can return the
# same value, and an equal timestamp cannot be distinguished from a stale one.
_DEFAULT: dict[str, Any] = {"stop_seq": 0, "paused_until": 0.0, "speaking_since": 0.0,
                            "tab_seen": 0.0, "enabled": True}


def _read() -> dict[str, Any]:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return {**_DEFAULT, **data} if isinstance(data, dict) else dict(_DEFAULT)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(_DEFAULT)


def _write(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(STATE_PATH)          # atomic, so a reader never sees half a file


def _update(**changes: Any) -> dict[str, Any]:
    state = _read()
    state.update(changes)
    _write(state)
    return state


# --- stopping speech ----------------------------------------------------------

def request_stop() -> int:
    """Tell whoever is speaking to stop now. Returns the new counter value."""
    state = _read()
    seq = int(state.get("stop_seq", 0)) + 1
    _update(stop_seq=seq)
    return seq


def stop_seq() -> int:
    return int(_read().get("stop_seq", 0))


def stop_requested_since(marker: int) -> bool:
    """True if a stop was requested after `marker` — the counter read when playback
    began, so an earlier stop cannot cancel a later reply."""
    return stop_seq() > marker


# --- pausing the daemon -------------------------------------------------------

def pause(seconds: float = PAUSE_SECONDS) -> float:
    """Stop the daemon listening, so it does not answer speech meant for the
    dashboard mic. Expires on its own if nobody releases it."""
    until = time.time() + seconds
    _update(paused_until=until)
    return until


def resume() -> None:
    _update(paused_until=0.0)


def is_paused() -> bool:
    return time.time() < float(_read().get("paused_until", 0.0))


# --- turning voice off altogether ------------------------------------------
#
# Distinct from pause(): pause is a short lease the browser takes while it has
# the microphone, and expires on its own. This is the user's own switch and
# stays off until they turn it back on.

def set_enabled(enabled: bool) -> bool:
    _update(enabled=bool(enabled))
    return bool(enabled)


def is_enabled() -> bool:
    return bool(_read().get("enabled", True))


# --- who is talking -----------------------------------------------------------

def mark_speaking(active: bool) -> None:
    _update(speaking_since=time.time() if active else 0.0)


def is_speaking() -> bool:
    since = float(_read().get("speaking_since", 0.0))
    # A crashed speaker would otherwise leave this set forever.
    return bool(since) and (time.time() - since) < 300


# --- is the dashboard actually open? ------------------------------------------
#
# The daemon listens only while a browser tab is watching. The API server
# refreshes this while a websocket client is connected; if the browser dies the
# timestamp simply goes stale, so nothing is left listening by accident.
# The dashboard sends a heartbeat every 10s, so a tab that dies without closing
# its socket goes stale within this window and the daemon stops listening.
TAB_TTL_SECONDS = float(os.environ.get("BOLT_TAB_TTL", "30"))


def mark_tab_open() -> None:
    _update(tab_seen=time.time())


def mark_tab_closed() -> None:
    _update(tab_seen=0.0)


def tab_open() -> bool:
    seen = float(_read().get("tab_seen", 0.0))
    return bool(seen) and (time.time() - seen) < TAB_TTL_SECONDS


def snapshot() -> dict[str, Any]:
    state = _read()
    return {"enabled": is_enabled(), "paused": is_paused(), "speaking": is_speaking(),
            "tab_open": tab_open(), "paused_until": state["paused_until"],
            "stop_seq": state["stop_seq"]}
