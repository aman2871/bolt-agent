"""Files handed to Bolt through the browser.

Agents address everything by path, so an upload is really a save: the file lands
somewhere inside the workspace and the tools that already exist work unchanged.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from bolt.core import workspace

UPLOAD_DIR_NAME = os.environ.get("BOLT_UPLOAD_DIR", "uploads")
MAX_UPLOAD_BYTES = int(os.environ.get("BOLT_MAX_UPLOAD_MB", "500")) * 1_000_000

# What an agent can actually do something with. Anything else is refused rather
# than saved and ignored.
ALLOWED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi",
    ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".log",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".sql", ".sh", ".html", ".css",
    ".pdf", ".wav", ".mp3", ".m4a",
}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class UploadRejected(ValueError):
    """The file cannot be accepted, with a reason worth showing the user."""


def safe_name(name: str) -> str:
    """A filename that cannot escape the directory it is written to.

    Path separators and traversal are stripped rather than escaped: the name is
    only ever a label, so there is nothing to preserve by being clever.
    """
    base = Path(name).name                      # drops any directory component
    cleaned = _UNSAFE.sub("-", base).strip("-.") or "upload"
    stem, dot, suffix = cleaned.rpartition(".")
    if not dot:
        return cleaned[:120]
    return f"{stem[:100]}.{suffix[:12]}"


def target_dir(conversation_id: str | None = None) -> Path:
    """Uploads live inside the workspace so existing tools can read them."""
    root = workspace.DEFAULT / UPLOAD_DIR_NAME
    return root / conversation_id if conversation_id else root


def check_name(name: str) -> str:
    """Vet the filename before a single byte of the body is read.

    A 1GB drop that will be refused for its extension should cost nothing to
    refuse.
    """
    filename = safe_name(name)
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise UploadRejected(f"{Path(filename).suffix or 'that file type'} is not accepted")
    return filename


def _destination(filename: str, conversation_id: str | None) -> Path:
    directory = target_dir(conversation_id)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if destination.exists():
        destination = directory / f"{destination.stem}-{int(time.time())}{destination.suffix}"
    return destination


async def save_stream(name: str, chunks, conversation_id: str | None = None) -> tuple[Path, int]:
    """Write an upload to disk as it arrives, refusing it the moment it is too big.

    Buffering the body first means holding the whole file in memory only to
    decide it is over the limit, which is the most expensive possible way to say
    no.
    """
    filename = check_name(name)
    destination = _destination(filename, conversation_id)

    total = 0
    try:
        with destination.open("wb") as handle:
            async for chunk in chunks:
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise UploadRejected(
                        f"that file is over the {MAX_UPLOAD_BYTES // 1_000_000}MB limit")
                handle.write(chunk)
        if not total:
            raise UploadRejected("that file is empty")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    # Belt and braces: confirm the write landed where we meant it to.
    workspace.resolve(str(destination))
    return destination, total


def save(name: str, data: bytes, conversation_id: str | None = None) -> Path:
    if not data:
        raise UploadRejected("that file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"{len(data) // 1_000_000}MB is over the "
            f"{MAX_UPLOAD_BYTES // 1_000_000}MB limit")

    filename = safe_name(name)
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise UploadRejected(f"{Path(filename).suffix or 'that file type'} is not accepted")

    directory = target_dir(conversation_id)
    directory.mkdir(parents=True, exist_ok=True)

    destination = directory / filename
    if destination.exists():
        stem = destination.stem
        destination = directory / f"{stem}-{int(time.time())}{destination.suffix}"

    destination.write_bytes(data)
    # Belt and braces: confirm the write landed where we meant it to.
    workspace.resolve(str(destination))
    return destination


def relative(path: Path) -> str:
    """How an agent should refer to the file."""
    try:
        return str(path.relative_to(workspace.DEFAULT))
    except ValueError:
        return str(path)
