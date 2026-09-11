"""Holding pen for file changes an agent wants to make.

The code gate already protects `bolt code`, but chat had no equivalent: an agent
could write a file directly and you would only find out afterwards. When
collecting is active, writes are recorded here instead of hitting the disk, and a
person approves or discards them.
"""

from __future__ import annotations

import difflib
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from bolt.core import store

MAX_DIFF_LINES = 400


@dataclass
class Collector:
    conversation_id: str | None = None
    agent: str | None = None
    ids: list[str] = field(default_factory=list)


_collector: ContextVar[Collector | None] = ContextVar("bolt_proposals", default=None)


def active() -> Collector | None:
    return _collector.get()


@contextmanager
def collecting(conversation_id: str | None = None,
               agent: str | None = None) -> Iterator[Collector]:
    collector = Collector(conversation_id, agent)
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)


def record(path: Path, new_content: str) -> str:
    """Store a write instead of performing it. Returns the proposal id."""
    collector = active()
    existed = path.is_file()
    old = path.read_text(encoding="utf-8", errors="replace") if existed else None
    proposal_id = store.add_proposal(
        str(path), new_content, old, existed,
        collector.conversation_id if collector else None,
        collector.agent if collector else None)
    if collector:
        collector.ids.append(proposal_id)
    return proposal_id


def diff_for(proposal: dict[str, Any], root: Path | None = None) -> str:
    """Unified diff between what is on disk and what the agent wants there."""
    path = Path(proposal["path"])
    name = str(path)
    if root:
        try:
            name = str(path.relative_to(root))
        except ValueError:
            pass

    old = (proposal["old_content"] or "").splitlines(keepends=True)
    new = (proposal["new_content"] or "").splitlines(keepends=True)
    lines = list(difflib.unified_diff(
        old, new,
        fromfile=f"a/{name}" if proposal["existed"] else "/dev/null",
        tofile=f"b/{name}", n=3))
    if len(lines) > MAX_DIFF_LINES:
        omitted = len(lines) - MAX_DIFF_LINES
        lines = lines[:MAX_DIFF_LINES] + [f"\n... {omitted} more lines not shown\n"]
    return "".join(lines) or "(no textual change)"


class Conflict(RuntimeError):
    """The file on disk is not what we expected, so writing would lose work."""


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def apply(proposal_id: str, force: bool = False) -> str:
    """Write an approved change to disk, recording enough to undo it.

    Refuses if the file has changed since the agent read it. A proposal is a
    diff against a specific version, and applying it over newer content silently
    discards whatever was written in between.
    """
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        raise KeyError(f"no proposal {proposal_id}")
    if proposal["status"] == "applied":
        return f"{proposal['path']} was already applied"

    path = Path(proposal["path"])
    if not force:
        current = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
        expected = proposal["old_content"] if proposal["existed"] else None
        if current != expected:
            raise Conflict(
                f"{path.name} has changed since this was proposed"
                f"{' (it now exists)' if expected is None else ''}. Applying would "
                f"overwrite those edits — review the diff again, or force it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(proposal["new_content"], encoding="utf-8")
    store.mark_applied(proposal_id, digest(proposal["new_content"]))
    return f"wrote {path}"


def undo(proposal_id: str, force: bool = False) -> str:
    """Put a file back the way it was before this change was applied.

    Refuses when the file no longer matches what we wrote: something else has
    edited it since, and restoring the old content would silently throw that
    away. `force` overrides, because sometimes that is exactly what you want.
    """
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        raise KeyError(f"no proposal {proposal_id}")
    if proposal["status"] != "applied":
        return f"{proposal['path']} is {proposal['status']}, nothing to undo"

    path = Path(proposal["path"])
    current = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None

    if not force and proposal["applied_hash"]:
        if current is None:
            raise Conflict(f"{path.name} has been deleted since it was applied")
        if digest(current) != proposal["applied_hash"]:
            raise Conflict(
                f"{path.name} has been edited since Bolt wrote it. Undoing would "
                f"discard those edits — pass force to do it anyway.")

    if proposal["existed"]:
        path.write_text(proposal["old_content"] or "", encoding="utf-8")
        outcome = f"restored {path}"
    else:
        # The change created this file, so undoing it means removing it again.
        path.unlink(missing_ok=True)
        outcome = f"removed {path}"

    store.mark_undone(proposal_id)
    return outcome


def undo_turn(turn_id: str, force: bool = False) -> dict[str, Any]:
    """Undo every file change made by one reply, newest first."""
    results, conflicts = [], []
    for proposal in reversed(store.proposals_in_turn(turn_id)):
        if proposal["status"] != "applied":
            continue
        try:
            results.append(undo(proposal["id"], force))
        except Conflict as exc:
            conflicts.append(str(exc))
    return {"undone": results, "conflicts": conflicts}


def checkpoints(limit: int = 50) -> list[dict[str, Any]]:
    """Applied changes that can still be reverted, newest first."""
    out = []
    for proposal in store.applied_proposals(limit):
        path = Path(proposal["path"])
        current = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
        drifted = bool(proposal["applied_hash"]) and (
            current is None or digest(current) != proposal["applied_hash"])
        out.append({**summarise(proposal), "turn_id": proposal["turn_id"],
                    "source": proposal["source"] or "chat",
                    "applied_at": proposal["decided_at"], "drifted": drifted})
    return out


def reject(proposal_id: str) -> str:
    store.set_proposal_status(proposal_id, "rejected")
    return f"discarded {proposal_id}"


def summarise(proposal: dict[str, Any]) -> dict[str, Any]:
    old_lines = (proposal["old_content"] or "").splitlines()
    new_lines = (proposal["new_content"] or "").splitlines()
    added = sum(1 for line in difflib.ndiff(old_lines, new_lines) if line.startswith("+ "))
    removed = sum(1 for line in difflib.ndiff(old_lines, new_lines) if line.startswith("- "))
    return {"id": proposal["id"], "path": proposal["path"], "agent": proposal["agent"],
            "status": proposal["status"], "existed": bool(proposal["existed"]),
            "added": added, "removed": removed}
