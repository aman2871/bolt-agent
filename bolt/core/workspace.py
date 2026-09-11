"""The directories agents are allowed to touch.

The primary workspace is held in a ContextVar so a sandboxed run can point an
agent at an isolated copy without disturbing anything running concurrently.
Additional roots can be mounted alongside it — other projects you want agents to
read — without making the whole filesystem reachable.
"""

from __future__ import annotations

import fnmatch
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

DEFAULT = Path(os.environ.get("BOLT_WORKSPACE", Path.home() / "Projects" / "Bolt")).resolve()

# Extra directories agents may reach, colon-separated, e.g. mounted projects.
EXTRA_ROOTS = tuple(
    Path(p).resolve() for p in os.environ.get("BOLT_EXTRA_ROOTS", "").split(":") if p.strip()
)

# Files whose contents are never handed to a model. Agents summarise code by
# sending it to a cloud API, and a mounted project can carry hundreds of
# credential files that nobody meant to share. Names are matched, not contents,
# so this is a cheap guard rather than a scanner.
SECRET_PATTERNS = tuple(
    p for p in os.environ.get(
        "BOLT_SECRET_PATTERNS",
        ".env:.env.*:*.pem:*.key:*.p12:*.pfx:*.jks:id_rsa*:id_ed25519*:"
        "*credentials*.json:*service-account*.json:.npmrc:.pypirc:.netrc:"
        "*.kdbx:*secret*.yaml:*secret*.yml:credentials"
    ).split(":") if p.strip()
)
# Sample files are templates, not secrets, and are often the most useful thing to read.
SECRET_EXCEPTIONS = ("*.sample", "*.example", "*.template", "*.dist", "*.md")

ALLOW_SECRETS = os.environ.get("BOLT_ALLOW_SECRETS", "0") == "1"

_workspace: ContextVar[Path] = ContextVar("bolt_workspace", default=DEFAULT)


def root() -> Path:
    return _workspace.get()


def roots() -> tuple[Path, ...]:
    """Every directory an agent may reach, primary first."""
    return (root(), *EXTRA_ROOTS)


@contextmanager
def using(path: Path | str) -> Iterator[Path]:
    token = _workspace.set(Path(path).resolve())
    try:
        yield root()
    finally:
        _workspace.reset(token)


# Whole directories whose contents are credentials regardless of filename.
SECRET_DIRS = tuple(
    d for d in os.environ.get("BOLT_SECRET_DIRS", ".aws:.ssh:.gnupg:.docker").split(":")
    if d.strip()
)


def is_secret(path: Path | str) -> bool:
    """Whether a path looks like it holds credentials."""
    if ALLOW_SECRETS:
        return False
    candidate = Path(path)
    # The AWS CLI reads ~/.aws itself; agents have no reason to read it directly,
    # and its files are not named like secrets ("config", "credentials").
    if any(part in SECRET_DIRS for part in candidate.parts):
        return True
    name = candidate.name
    if any(fnmatch.fnmatch(name, ok) for ok in SECRET_EXCEPTIONS):
        return False
    return any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_PATTERNS)


def contains(base: Path, candidate: Path) -> bool:
    return candidate == base or base in candidate.parents


def resolve(path: str, allow_secret: bool = False) -> Path:
    """Resolve a user-supplied path inside an allowed root, refusing escapes.

    A relative path is taken against the primary workspace; an absolute path may
    land in any allowed root.
    """
    candidate = Path(path).resolve() if Path(path).is_absolute() else (root() / path).resolve()

    if not any(contains(base, candidate) for base in roots()):
        allowed = ", ".join(str(b) for b in roots())
        raise ValueError(f"path {path!r} is outside the allowed roots ({allowed})")

    if not allow_secret and is_secret(candidate):
        raise ValueError(
            f"{candidate.name} looks like a credentials file, so its contents are not "
            f"read. Set BOLT_ALLOW_SECRETS=1 to override."
        )
    return candidate
