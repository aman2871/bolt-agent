"""Filesystem tools, confined to a workspace root."""

from __future__ import annotations

import subprocess

from bolt.core.tools import tool
from bolt.core import proposals, workspace
from bolt.core.workspace import resolve, root

MAX_READ_BYTES = 200_000
# A ranged read may legitimately skip past the byte budget to reach its start
# line, but not forever: without a ceiling, read_file on a huge file walks the
# whole thing.
MAX_SCAN_BYTES = 20_000_000

# Formats that are not text. Decoding one yields a screenful of replacement
# characters — tens of thousands of tokens of noise that tell the agent nothing
# — so refuse before reading rather than after.
BINARY_SUFFIXES = {
    ".xlsx", ".xls", ".xlsm", ".docx", ".doc", ".pptx", ".ppt", ".pdf",
    ".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tiff",
    ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg",
    ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a", ".pyc",
    ".db", ".sqlite", ".sqlite3", ".woff", ".woff2", ".ttf", ".otf",
}
# What to do instead, where there is something to do.
BINARY_HINT = {
    ".xlsx": "Save it as CSV and read that.",
    ".xls": "Save it as CSV and read that.",
    ".xlsm": "Save it as CSV and read that.",
    ".png": "Use look_at_image.",
    ".jpg": "Use look_at_image.",
    ".jpeg": "Use look_at_image.",
    ".gif": "Use look_at_image.",
    ".webp": "Use look_at_image.",
    ".mp4": "Use watch_video.",
    ".mov": "Use watch_video.",
    ".mkv": "Use watch_video.",
    ".webm": "Use watch_video.",
}
SNIFF_BYTES = 8192


def _size(n: int) -> str:
    return f"{n / 1e6:.1f}MB" if n >= 1e6 else f"{n:,} bytes"


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

    size = target.stat().st_size
    suffix = target.suffix.lower()
    if suffix in BINARY_SUFFIXES:
        hint = BINARY_HINT.get(suffix, "There is nothing here to read as text.")
        return (f"ERROR: {target.name} is {_size(size)} of {suffix} data, which is "
                f"not text. {hint}")

    # Never pull a whole file into memory to throw almost all of it away: at most
    # MAX_READ_BYTES can ever be returned, so at most that much is ever read. A
    # 1GB file used to cost 1GB of resident memory to answer with 200KB.
    with target.open("rb") as handle:
        head = handle.read(SNIFF_BYTES)
    if b"\x00" in head:
        return (f"ERROR: {target.name} is {_size(size)} of binary data (NUL bytes "
                f"in the first {SNIFF_BYTES}), not text.")

    start = max(1, int(start_line or 1))
    want = max(0, int(max_lines or 0))
    lines: list[str] = []
    used = scanned = 0
    number = 0           # the line most recently completed
    partial = False      # the last chunk ended mid-line
    truncated = False

    with target.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            # Bounded: one "line" of a minified file can be the entire file.
            chunk = handle.readline(MAX_READ_BYTES)
            if not chunk:
                break
            scanned += len(chunk)
            if partial:                      # a continuation, not a new line
                continuation = True
            else:
                continuation = False
                number += 1
            partial = not chunk.endswith("\n")

            if number < start or continuation:
                if scanned > MAX_SCAN_BYTES:
                    return (f"ERROR: gave up looking for line {start} of "
                            f"{target.name} after {_size(scanned)}. Narrow it down "
                            f"with grep.")
                continue
            if want and len(lines) >= want:
                truncated = True
                break
            if used + len(chunk) > MAX_READ_BYTES:
                truncated = True
                break
            lines.append(chunk.rstrip("\n"))
            used += len(chunk)

    text = "\n".join(lines)
    if start > 1 or want:
        last = start + len(lines) - 1
        header = f"[{target.name} lines {start}-{max(last, start - 1)}, {_size(size)} total]\n"
        return header + text
    # Silent truncation is a correctness hazard: an agent that does not know it
    # got half a file will happily answer as though it saw all of it.
    if truncated:
        return (text + f"\n\n[TRUNCATED: {target.name} is {_size(size)}; only the "
                f"first {len(lines)} lines were read. Read on with "
                f"start_line={len(lines) + 1}, or grep for what you need.]")
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
