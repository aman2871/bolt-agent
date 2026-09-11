"""Allowlisted shell access. Anything not on the list is refused, not sanitised."""

from __future__ import annotations

import os
import shlex
import subprocess

from bolt.core.tools import tool
from bolt.core import workspace
from bolt.core.workspace import root

# AWS is read-only by default. These are the verbs that only observe; anything
# else is refused, because an agent acting on a misread instruction should not be
# able to change real infrastructure.
AWS_READ_PREFIXES = (
    "describe", "list", "get", "lookup", "search", "scan", "query",
    "head", "batch-get", "estimate", "preview", "validate", "simulate",
    "generate-credential-report", "test-",
)
AWS_READ_EXACT = {
    ("sts", "get-caller-identity"), ("s3", "ls"), ("configure", "list"),
    ("configure", "list-profiles"), ("iam", "get-account-summary"),
}
# Verbs that read in most services but write in these, so never assume.
AWS_ALWAYS_BLOCKED = ("get-session-token", "get-federation-token")
AWS_WRITE_ALLOWED = os.environ.get("BOLT_AWS_WRITE", "0") == "1"


def check_aws(parts: list[str]) -> str | None:
    """Return an error if this aws command is not read-only. None if allowed."""
    if AWS_WRITE_ALLOWED:
        return None

    args = [a for a in parts[1:] if not a.startswith("-")]
    if not args:
        return None                      # bare `aws` prints help

    service = args[0]
    operation = args[1] if len(args) > 1 else ""

    if (service, operation) in AWS_READ_EXACT:
        return None
    if operation in AWS_ALWAYS_BLOCKED:
        return f"ERROR: 'aws {service} {operation}' issues credentials and is refused."
    if operation.startswith(AWS_READ_PREFIXES):
        return None

    return (f"ERROR: 'aws {service} {operation or '<no operation>'}' is not a read-only "
            f"command, so it is refused. Bolt has read-only AWS access; run it "
            f"yourself, or set BOLT_AWS_WRITE=1 to allow changes.")


ALLOWED = {
    "ls", "cat", "head", "tail", "wc", "grep", "find", "file", "stat",
    "git", "python", "python3", "pytest", "ruff", "uv", "pip",
    "echo", "date", "uname", "df", "du", "which", "sort", "uniq", "sed", "awk",
    "curl", "jq", "docker", "redis-cli", "sqlite3", "ollama", "aws",
}
BLOCKED_GIT = {"push", "reset", "clean", "rebase"}
TIMEOUT = int(os.environ.get("BOLT_SHELL_TIMEOUT", "120"))


@tool(description="Run an allowlisted shell command in the workspace. No pipes or redirection.",
      command="The command line, for example 'git status --short'.")
def run_shell(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return f"ERROR: could not parse command: {exc}"
    if not parts:
        return "ERROR: empty command"

    program = os.path.basename(parts[0])
    if program not in ALLOWED:
        return f"ERROR: {program!r} is not allowlisted. Allowed: {', '.join(sorted(ALLOWED))}"
    if program == "git" and len(parts) > 1 and parts[1] in BLOCKED_GIT:
        return f"ERROR: 'git {parts[1]}' is blocked"
    if program == "aws" and (refusal := check_aws(parts)):
        return refusal
    if any(tok in (">", ">>", "|", "&&", ";", "<") for tok in parts):
        return "ERROR: shell operators are not supported, run one command at a time"

    # read_file refuses credentials files; `cat`, `head` and friends would walk
    # straight around that, so the same rule applies to shell arguments.
    for token in parts[1:]:
        if token.startswith("-"):
            continue
        if workspace.is_secret(token):
            return (f"ERROR: {token} looks like a credentials file. "
                    f"Set BOLT_ALLOW_SECRETS=1 to override.")

    try:
        proc = subprocess.run(parts, cwd=root(), capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {TIMEOUT}s"
    except FileNotFoundError:
        return f"ERROR: {program} not found"

    out = (proc.stdout or "") + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
    return f"[exit {proc.returncode}]\n{out.strip() or '(no output)'}"
