"""Run generated code in an isolated copy of the workspace.

Isolation has two tiers. With Docker available a test command runs with no
network, capped memory and CPU, and a read-only root filesystem. Without it we
fall back to a plain subprocess in the snapshot directory, which still protects
the real workspace but offers no network or resource isolation — the result says
which tier was used so nobody mistakes one for the other.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from bolt.core.workspace import DEFAULT as DEFAULT_WORKSPACE

SNAPSHOT_ROOT = Path(os.environ.get(
    "BOLT_SNAPSHOT_ROOT", DEFAULT_WORKSPACE / ".worktrees"))
IMAGE = os.environ.get("BOLT_SANDBOX_IMAGE", "python:3.13-slim")
TEST_TIMEOUT = int(os.environ.get("BOLT_TEST_TIMEOUT", "300"))
MEMORY = os.environ.get("BOLT_SANDBOX_MEMORY", "2g")
CPUS = os.environ.get("BOLT_SANDBOX_CPUS", "2")

EXCLUDES = (
    ".git", ".venv", "__pycache__", "*.pyc", ".worktrees",
    "bolt.db", "bolt.db-wal", "bolt.db-shm", "*.egg-info",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".DS_Store",
)
IGNORE = shutil.ignore_patterns(*EXCLUDES)


@dataclass
class SandboxResult:
    ok: bool
    exit_code: int
    output: str
    duration_ms: int
    isolation: str  # "docker" or "subprocess"
    workdir: Path


def docker_available() -> bool:
    try:
        proc = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                              capture_output=True, text=True, timeout=10)
        return proc.returncode == 0 and bool(proc.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


MANIFEST_NAME = ".bolt-manifest.json"


def _excluded(rel: Path) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(part, pattern) for part in rel.parts for pattern in EXCLUDES)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(rootdir: Path) -> dict[str, str]:
    """sha256 of every non-excluded file, keyed by path relative to rootdir."""
    out: dict[str, str] = {}
    for path in rootdir.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(rootdir)
        if _excluded(rel) or rel.name == MANIFEST_NAME:
            continue
        out[str(rel)] = _digest(path)
    return out


def snapshot(task_id: str, source: Path | None = None) -> Path:
    """Copy the workspace into an isolated directory for this task.

    A manifest of file hashes is written alongside it so we can later tell what
    the agent changed, as distinct from what drifted in the real workspace.
    """
    source = Path(source or DEFAULT_WORKSPACE).resolve()
    target = SNAPSHOT_ROOT / task_id
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=IGNORE, symlinks=True)
    (SNAPSHOT_ROOT / f"{task_id}{MANIFEST_NAME}").write_text(
        json.dumps(_manifest(target)), encoding="utf-8")
    return target


def load_manifest(workdir: Path) -> dict[str, str] | None:
    path = Path(workdir).parent / f"{Path(workdir).name}{MANIFEST_NAME}"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discard(workdir: Path) -> None:
    """Remove a snapshot and its manifest. Only touches paths under SNAPSHOT_ROOT."""
    workdir = Path(workdir).resolve()
    if SNAPSHOT_ROOT.resolve() not in workdir.parents:
        return
    if workdir.exists():
        shutil.rmtree(workdir)
    manifest = workdir.parent / f"{workdir.name}{MANIFEST_NAME}"
    manifest.unlink(missing_ok=True)


def diff(workdir: Path, source: Path | None = None) -> str:
    """Unified diff of the snapshot against the real workspace."""
    source = Path(source or DEFAULT_WORKSPACE).resolve()
    argv = ["diff", "-ruN"] + [f"--exclude={pattern}" for pattern in EXCLUDES]
    proc = subprocess.run(argv + [str(source), str(workdir)],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode not in (0, 1):
        return f"ERROR: diff failed: {proc.stderr[:400]}"
    return proc.stdout[:60_000] or "(no changes)"


def run_tests(workdir: Path, command: str) -> SandboxResult:
    """Execute a test command against the snapshot."""
    workdir = Path(workdir).resolve()
    started = time.monotonic()

    if docker_available():
        argv = [
            "docker", "run", "--rm",
            "--network=none",
            f"--memory={MEMORY}", f"--cpus={CPUS}",
            "--pids-limit=256",
            "--read-only",
            "--tmpfs", "/tmp:rw,exec,size=256m",
            "-v", f"{workdir}:/work:rw",
            "-w", "/work",
            IMAGE, "sh", "-c", command,
        ]
        isolation = "docker"
        kwargs = {}
    else:
        argv = ["sh", "-c", command]
        isolation = "subprocess"
        kwargs = {"cwd": str(workdir)}

    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=TEST_TIMEOUT, **kwargs)
    except subprocess.TimeoutExpired:
        return SandboxResult(False, 124, f"test command timed out after {TEST_TIMEOUT}s",
                             int((time.monotonic() - started) * 1000), isolation, workdir)

    output = ((proc.stdout or "") + (proc.stderr or "")).strip()[:20_000]
    return SandboxResult(
        ok=proc.returncode == 0,
        exit_code=proc.returncode,
        output=output or "(no output)",
        duration_ms=int((time.monotonic() - started) * 1000),
        isolation=isolation,
        workdir=workdir,
    )


def changed_files(workdir: Path, source: Path | None = None) -> tuple[list[str], list[str]]:
    """What the agent changed inside the snapshot.

    Compares the snapshot against the manifest taken when it was created, so
    edits made in the real workspace meanwhile are never mistaken for the
    agent's work. Returns (changed_or_added, deleted).
    """
    workdir = Path(workdir).resolve()
    before = load_manifest(workdir)
    if before is None:
        raise RuntimeError(f"no manifest for {workdir}; cannot tell what the agent changed")

    after = _manifest(workdir)
    changed = sorted(rel for rel, digest in after.items() if before.get(rel) != digest)
    deleted = sorted(rel for rel in before if rel not in after)
    return changed, deleted


def sync_into(workdir: Path, source: Path | None = None, delete: bool = False) -> list[str]:
    """Copy the snapshot's changes into the real workspace.

    Copying beats `patch` here: the diff carries absolute paths at two different
    depths, so no single -p level applies cleanly.
    """
    source = Path(source or DEFAULT_WORKSPACE).resolve()
    workdir = Path(workdir).resolve()
    changed, deleted = changed_files(workdir, source)
    applied: list[str] = []

    for rel in changed:
        src, dst = workdir / rel, source / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True, symlinks=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        applied.append(f"updated {rel}")

    if delete:
        for rel in deleted:
            target = source / rel
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            applied.append(f"deleted {rel}")

    return applied


SNAPSHOT_TTL_SECONDS = float(os.environ.get("BOLT_SNAPSHOT_TTL", str(24 * 3600)))
SNAPSHOT_KEEP = int(os.environ.get("BOLT_SNAPSHOT_KEEP", "3"))


def sweep_snapshots(ttl: float = SNAPSHOT_TTL_SECONDS, keep: int = SNAPSHOT_KEEP) -> int:
    """Delete old sandbox snapshots.

    Every gate run copies the whole workspace, so these are the largest thing
    Bolt writes and nothing removed them — four runs had accumulated ~1GB.
    The most recent are kept so a result can still be inspected or applied.
    """
    if not SNAPSHOT_ROOT.exists():
        return 0
    snapshots = sorted((p for p in SNAPSHOT_ROOT.glob("t_*") if p.is_dir()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    cutoff = time.time() - ttl
    removed = 0
    for i, path in enumerate(snapshots):
        if i < keep or path.stat().st_mtime > cutoff:
            continue
        discard(path)
        removed += 1
    return removed


def snapshot_usage() -> dict[str, Any]:
    if not SNAPSHOT_ROOT.exists():
        return {"count": 0, "bytes": 0}
    total = sum(f.stat().st_size for f in SNAPSHOT_ROOT.rglob("*") if f.is_file())
    return {"count": sum(1 for p in SNAPSHOT_ROOT.glob("t_*") if p.is_dir()),
            "bytes": total, "mb": round(total / 1_000_000, 1)}
