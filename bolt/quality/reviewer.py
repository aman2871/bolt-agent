"""Sentinel: the review gate every code change passes through."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from bolt.core import roster
from bolt.core.loop import run_agent

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

REVIEW_INSTRUCTIONS = """\
Review the change below.

Judge only whether the change is correct and safe: logic errors, unhandled
failures, broken contracts, security problems, tests that do not actually test
the behaviour. Ignore style.

Reply with a single JSON object and nothing else:
{"verdict": "approve" | "reject", "findings": ["path:line: problem. fix.", ...],
 "summary": "one sentence"}

Reject only for defects that change behaviour or hide a failure. If the tests
passed and the code is sound, approve.
"""


@dataclass
class Review:
    verdict: str
    summary: str
    findings: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def approved(self) -> bool:
        return self.verdict == "approve"


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model reply, fenced or bare."""
    for candidate in [m.group(1) for m in _FENCE.finditer(text)] + [text]:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start == -1 or end <= start:
                continue
            try:
                parsed = json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


async def review(diff: str, test_output: str, tests_passed: bool,
                 task_prompt: str, root_id: str | None = None,
                 parent_id: str | None = None,
                 depth: int = 1, budget_tokens: int = 200_000,
                 on_event=None) -> Review:
    """Ask Sentinel to pass or fail a change. An unparseable reply is a rejection."""
    spec = roster.get("sentinel")
    prompt = (
        f"{REVIEW_INSTRUCTIONS}\n"
        f"## The task the change was meant to accomplish\n{task_prompt}\n\n"
        f"## Test run\nstatus: {'PASSED' if tests_passed else 'FAILED'}\n"
        f"```\n{test_output[:8000]}\n```\n\n"
        f"## Diff\n```diff\n{diff[:40000]}\n```\n"
    )
    result = await run_agent(spec, prompt, root_id=root_id, parent_id=parent_id,
                             depth=depth, budget_tokens=budget_tokens, on_event=on_event)
    if not result.ok:
        return Review("reject", f"review could not run: {result.error}", [], result.content)

    parsed = extract_json(result.content)
    if parsed is None:
        return Review("reject", "reviewer did not return parseable JSON", [], result.content)

    verdict = str(parsed.get("verdict", "reject")).strip().lower()
    if verdict not in ("approve", "reject"):
        verdict = "reject"
    findings = [str(f) for f in (parsed.get("findings") or [])]
    return Review(verdict, str(parsed.get("summary", "")), findings, result.content)
