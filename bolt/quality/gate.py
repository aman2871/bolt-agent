"""The code-change gate: Forge writes, the sandbox runs the tests, Sentinel reviews.

Nothing reaches the user unless the tests pass and the reviewer approves. A
failure on either count goes back to Forge with the evidence, up to a retry cap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bolt.core import proposals, roster, store, workspace
from bolt.core.loop import run_agent
from bolt.quality import reviewer
from bolt.sandbox import runner

MAX_ATTEMPTS = int(os.environ.get("BOLT_GATE_ATTEMPTS", "3"))  # 1 try + 2 retries
DEFAULT_TEST_COMMAND = os.environ.get("BOLT_TEST_COMMAND", "python3 -m pytest -q")

FORGE_BRIEF = """\
{task}

You are working inside an isolated snapshot of the repository, so change files
freely — nothing here touches the real workspace until the change is approved.

Requirements:
- Write or update tests that would fail without your change.
- The command `{test_command}` must pass when you are done.
- Use `run_shell` to actually run it. Do not claim it passes without running it.
- Finish by listing the files you changed and why.
"""

RETRY_BRIEF = """\
Your previous attempt was not accepted. Fix it in place.

## Test command
{test_command}

## Test result
status: {status}
```
{test_output}
```

## Reviewer findings
{findings}

Address every point above, re-run the tests, and confirm they pass.
"""


@dataclass
class Attempt:
    number: int
    tests_passed: bool
    test_output: str
    isolation: str
    verdict: str
    findings: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class GateResult:
    ok: bool
    task_id: str
    workdir: str
    diff: str
    attempts: list[Attempt] = field(default_factory=list)
    artifact_id: str | None = None
    forge_report: str = ""
    error: str | None = None


async def run_code_task(
    task: str,
    test_command: str = DEFAULT_TEST_COMMAND,
    budget_tokens: int = 400_000,
    keep_snapshot: bool = True,
    on_event=None,
) -> GateResult:
    """Produce a reviewed, test-passing change in an isolated snapshot."""
    spec = roster.get("forge")
    task_id = store.create_task("forge", task, budget_tokens=budget_tokens)
    workdir = runner.snapshot(task_id)

    def emit(kind: str, data: dict[str, Any]) -> None:
        if on_event:
            on_event(kind, {"task_id": task_id, "agent": "gate", **data})

    emit("sandbox_ready", {"workdir": str(workdir)})

    attempts: list[Attempt] = []
    prompt = FORGE_BRIEF.format(task=task, test_command=test_command)
    forge_report = ""
    diff_text = ""

    for number in range(1, MAX_ATTEMPTS + 1):
        emit("attempt", {"number": number, "of": MAX_ATTEMPTS})

        # Forge only ever sees the snapshot, never the real workspace.
        with workspace.using(workdir):
            outcome = await run_agent(
                spec, prompt, root_id=task_id, parent_id=task_id, depth=1,
                budget_tokens=budget_tokens, extra={"on_event": on_event}, on_event=on_event,
            )
        if not outcome.ok:
            return GateResult(False, task_id, str(workdir), diff_text, attempts,
                              error=f"forge failed: {outcome.error}")
        forge_report = outcome.content

        test = runner.run_tests(workdir, test_command)
        emit("tests", {"passed": test.ok, "exit_code": test.exit_code,
                       "isolation": test.isolation, "duration_ms": test.duration_ms})

        diff_text = runner.diff(workdir)
        review = await reviewer.review(
            diff_text, test.output, test.ok, task,
            root_id=task_id, parent_id=task_id,
            budget_tokens=budget_tokens, on_event=on_event,
        )
        emit("review", {"verdict": review.verdict, "summary": review.summary,
                        "findings": review.findings})

        attempts.append(Attempt(number, test.ok, test.output, test.isolation,
                                review.verdict, review.findings, review.summary))

        if test.ok and review.approved:
            artifact_id = store.add_artifact(task_id, "diff", content=diff_text, path=str(workdir))
            store.set_review(artifact_id, "approved", review.summary)
            store.set_status(task_id, "done", result=forge_report)
            return GateResult(True, task_id, str(workdir), diff_text, attempts,
                              artifact_id, forge_report)

        if number == MAX_ATTEMPTS:
            break

        prompt = RETRY_BRIEF.format(
            test_command=test_command,
            status="PASSED" if test.ok else f"FAILED (exit {test.exit_code})",
            test_output=test.output[:6000],
            findings="\n".join(f"- {f}" for f in review.findings) or "- (none given)",
        )

    artifact_id = store.add_artifact(task_id, "diff", content=diff_text, path=str(workdir))
    store.set_review(artifact_id, "rejected",
                     attempts[-1].summary if attempts else "no attempts completed")
    error = f"not accepted after {MAX_ATTEMPTS} attempts"
    store.set_status(task_id, "rejected", error=error)
    if not keep_snapshot:
        runner.discard(workdir)
    return GateResult(False, task_id, str(workdir), diff_text, attempts,
                      artifact_id, forge_report, error)


def apply_snapshot(result: GateResult, delete: bool = False) -> str:
    """Copy an approved snapshot's changes into the real workspace.

    Each file is recorded as an applied change first, so a gate result can be
    undone the same way a chat proposal can.
    """
    if not result.ok:
        return "ERROR: refusing to apply a change that was not approved"

    workdir = Path(result.workdir)
    try:
        changed, _deleted = runner.changed_files(workdir)
    except Exception as exc:
        changed = []
        print(f"could not enumerate changes for undo: {exc}")

    for relative in changed:
        source = workdir / relative
        target = workspace.DEFAULT / relative
        if not source.is_file():
            continue
        try:
            new_content = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue          # binary or unreadable: still applied, just not undoable
        existed = target.is_file()
        old_content = target.read_text(encoding="utf-8", errors="replace") if existed else None
        proposal_id = store.add_proposal(
            str(target), new_content, old_content, existed,
            agent="forge", source="gate", status="pending")
        store.mark_applied(proposal_id, proposals.digest(new_content))

    try:
        applied = runner.sync_into(workdir, delete=delete)
    except Exception as exc:
        return f"ERROR: could not apply snapshot: {type(exc).__name__}: {exc}"
    if not applied:
        return "nothing to apply, the snapshot matches the workspace"
    return "\n".join(applied) + f"\n[{len(changed)} change(s) recorded, undoable]"
