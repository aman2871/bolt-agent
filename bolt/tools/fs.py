"""Filesystem tools, confined to a workspace root."""

from __future__ import annotations

import subprocess

from bolt.core.tools import tool
from bolt.core import proposals, workspace
from bolt.core.workspace import resolve, root

MAX_READ_BYTES = 200_000


@tool(description=("Read a UTF-8 text file from the workspace. Large files are "
                   "truncated, so read a range when you only need part of one."),
      path="Workspace-relative path to the file.",
      start_line="First line to return, 1-based. Omit to start at the beginning.",
      max_lines="How many lines to return. Omit for the whole file.")
def read_file(path: str, start_line: int = 1, max_lines: int = 0) -> str:
    try:
        target = resolve(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not target.is_file():
        return f"ERROR: {path} is not a file"

    raw = target.read_bytes()
    text = raw[:MAX_READ_BYTES].decode("utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)

    start = max(1, int(start_line or 1))
    if start > 1 or max_lines:
        end = start - 1 + int(max_lines) if max_lines else total
        window = lines[start - 1:end]
        header = (f"[{target.name} lines {start}-{min(end, total)} of "
                  f"{total}{'+' if len(raw) > MAX_READ_BYTES else ''}]\n")
        return header + "\n".join(window)

    # Silent truncation is a correctness hazard: an agent that does not know it
    # got half a file will happily answer as though it saw all of it.
    if len(raw) > MAX_READ_BYTES:
        return (text + f"\n\n[TRUNCATED: {target.name} is {len(raw)} bytes; "
                f"only the first {MAX_READ_BYTES} were read ({total} lines). "
                f"Use start_line and max_lines to read the rest, or grep for "
                f"what you need.]")
    return text


@tool(description="Write a UTF-8 text file in the workspace, creating parent directories.",
      path="Workspace-relative path.", content="Full file contents.")
def write_file(path: str, content: str) -> str:
    try:
        target = resolve(path)
    except ValueError as exc:
        return f"ERROR: {exc}"

    # In chat the change is proposed, not performed: a person sees the diff and
    # decides. Direct writes still happen inside the code gate's snapshot, which
    # is already isolated from the real workspace.
    if proposals.active() is not None:
        proposals.record(target, content)
        return (f"proposed a change to {target.name} ({len(content)} chars). "
                f"It is waiting for the user to approve it — do not assume it "
                f"has been written.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {target.relative_to(root())}"


@tool(description="List directory entries in the workspace.",
      path="Workspace-relative directory, defaults to the workspace root.")
def list_dir(path: str = ".") -> str:
    try:
        target = resolve(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not target.is_dir():
        return f"ERROR: {path} is not a directory"
    lines = []
    for entry in sorted(target.iterdir()):
        if entry.name.startswith(".") and entry.name not in (".env.example",):
            continue
        lines.append(f"{'dir ' if entry.is_dir() else 'file'}  {entry.name}")
    return "\n".join(lines) or "(empty)"


@tool(description="Search file contents in the workspace with a regular expression.",
      pattern="Regular expression.", path="Directory to search under.")
def grep(pattern: str, path: str = ".") -> str:
    try:
        target = resolve(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    proc = subprocess.run(
        ["grep", "-rnI", "--exclude-dir=.git", "--exclude-dir=.venv",
         "--exclude-dir=node_modules", "--exclude-dir=__pycache__",
         "-E", pattern, str(target)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode not in (0, 1):
        return f"ERROR: grep failed: {proc.stderr[:400]}"

    kept, hidden = [], 0
    for line in proc.stdout.splitlines():
        # A match line carries the file's contents, so a credentials file would
        # leak through grep even though read_file refuses it.
        filename = line.split(":", 1)[0]
        if workspace.is_secret(filename):
            hidden += 1
            continue
        kept.append(line.replace(f"{root()}/", ""))
        if len(kept) >= 200:
            break

    if not kept:
        return f"no matches{f' ({hidden} in credential files, not shown)' if hidden else ''}"
    note = f"\n[{hidden} match(es) in credential files not shown]" if hidden else ""
    return "\n".join(kept) + note


@tool(description="List the projects and directories an agent may reach.")
def list_roots() -> str:
    lines = []
    for i, base in enumerate(workspace.roots()):
        label = "workspace" if i == 0 else "mounted"
        entries = sorted(p.name for p in base.iterdir()
                         if p.is_dir() and not p.name.startswith(".")) if base.is_dir() else []
        lines.append(f"{label}: {base}")
        if entries:
            lines.append("  " + ", ".join(entries[:60]))
    return "\n".join(lines)
