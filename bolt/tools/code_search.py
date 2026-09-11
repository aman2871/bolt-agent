"""Search tools over the cross-project index.

These exist so an agent can reuse what already works elsewhere instead of
writing a fifth implementation of the same thing.
"""

from __future__ import annotations

from pathlib import Path

from bolt.core import index
from bolt.core.tools import tool

MAX_SNIPPET = 160


def _display(path: str) -> str:
    """The path as an agent must actually use it.

    Shortening it to the last few segments reads nicely but produces something
    that does not resolve: relative paths resolve against the workspace, and
    these files live under other roots. Absolute is longer and correct.
    """
    return path


@tool(description=("Search code across every indexed project. Use this before writing "
                   "anything non-trivial — the same problem has often been solved in "
                   "another project already."),
      query="What to look for, e.g. 'razorpay signature verify' or 'celery beat schedule'.",
      project="Restrict to one project name (see list_projects).",
      language="Restrict to a file extension, e.g. py or ts.")
def search_code(query: str, project: str = "", language: str = "") -> str:
    rows = index.search(query, project or None, language or None, limit=12)
    if not rows:
        return (f"no matches for {query!r}. The index may be stale — "
                f"`bolt index` rebuilds it.")
    if "error" in rows[0]:
        return f"ERROR: {rows[0]['error']}"

    lines = [f"{len(rows)} match(es) for {query!r}:"]
    for row in rows:
        snippet = " ".join((row.get("snippet") or "").split())[:MAX_SNIPPET]
        lines.append(f"- [{row['project']}] {_display(row['path'])}\n    {snippet}")
    lines.append("\nThese paths are absolute — pass them to read_file as they are.")
    return "\n".join(lines)


@tool(description=("Find where a function, class or table is defined, across every "
                   "project. Falls back to a partial match when the exact name is absent."),
      symbol="Exact or partial symbol name, e.g. send_sms.")
def find_symbol(symbol: str) -> str:
    rows = index.find_symbol(symbol)
    if not rows:
        return f"no symbol matching {symbol!r} in the index."
    lines = [f"{len(rows)} definition(s) of {symbol!r}:"]
    for row in rows:
        lines.append(f"- [{row['project']}] {_display(row['path'])}:{row['line']}"
                     f"  {row['kind']}  {row['signature'] or ''}".rstrip())
    return "\n".join(lines)


@tool(description="List the indexed projects, with how much code each has.")
def list_projects(limit: int = 40) -> str:
    rows = index.projects()
    if not rows:
        return "nothing indexed yet — run `bolt index`."
    lines = [f"{len(rows)} indexed projects (largest first):"]
    for row in rows[:limit]:
        languages = ", ".join(sorted(set((row["languages"] or "").split(",")))[:5])
        lines.append(f"- {row['project']}  ({row['files']} files: {languages})")
    if len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more")
    return "\n".join(lines)
