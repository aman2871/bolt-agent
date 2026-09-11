"""Replay a golden task set so agent and model changes are measurable.

Assertions are deliberately cheap and literal where they can be (contains,
used_tool, max_chars); an LLM judge is used only where correctness is a matter
of meaning rather than string matching.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import bolt.registry  # noqa: F401  (registers tools)
from bolt.core import roster, store
from bolt.core.loop import run_agent
from bolt.quality.reviewer import extract_json

TASKS_DIR = Path(__file__).parent / "tasks"
JUDGE_AGENT = "sentinel"

JUDGE_PROMPT = """\
You are grading one answer against one criterion. Be strict but fair.

## Criterion
{criterion}

## The answer
{answer}

Reply with only this JSON: {{"pass": true|false, "why": "one short sentence"}}
"""


@dataclass
class EvalCase:
    id: str
    prompt: str
    agent: str
    assertions: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None


@dataclass
class EvalOutcome:
    case: EvalCase
    passed: bool
    score: float
    failures: list[str]
    answer: str
    tokens: int
    latency_ms: int
    model: str
    task_id: str


def load_suite(name: str) -> tuple[str, list[EvalCase]]:
    path = TASKS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no suite {name!r} in {TASKS_DIR}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = [
        EvalCase(id=item["id"], prompt=item["prompt"], agent=item.get("agent", "manager"),
                 assertions=list(item.get("assert") or []), model=item.get("model"))
        for item in data.get("evals") or []
    ]
    return data.get("suite", name), cases


def suites() -> list[str]:
    return sorted(path.stem for path in TASKS_DIR.glob("*.yaml"))


def _tools_used(task_id: str) -> set[str]:
    rows = store.connect().execute(
        "SELECT DISTINCT name FROM messages WHERE task_id = ? AND role = 'tool'"
        " UNION SELECT DISTINCT t.name FROM messages m"
        " JOIN tasks c ON c.parent_id = m.task_id JOIN messages t ON t.task_id = c.id"
        " WHERE m.task_id = ? AND t.role = 'tool'", (task_id, task_id)).fetchall()
    return {row["name"] for row in rows if row["name"]}


async def _judge(criterion: str, answer: str) -> tuple[bool, str]:
    spec = roster.get(JUDGE_AGENT)
    graded = await run_agent(
        # The judge reads text only; giving it tools invites it to go exploring.
        type(spec)(name="judge", model=spec.model, system_prompt="You grade answers.",
                   tools=[], max_iterations=1, temperature=0.0),
        JUDGE_PROMPT.format(criterion=criterion, answer=answer[:6000]),
    )
    parsed = extract_json(graded.content) if graded.ok else None
    if parsed is None:
        return False, "judge did not return parseable JSON"
    return bool(parsed.get("pass")), str(parsed.get("why", ""))


async def check(assertion: dict[str, Any], answer: str, tools_used: set[str]) -> tuple[bool, str]:
    kind, value = next(iter(assertion.items()))
    lowered = answer.lower()

    if kind == "contains":
        return str(value) in answer, f"missing {value!r}"
    if kind == "contains_all":
        missing = [v for v in value if str(v).lower() not in lowered]
        return not missing, f"missing {missing}"
    if kind == "contains_any":
        return any(str(v).lower() in lowered for v in value), f"none of {value} present"
    if kind == "not_contains":
        return str(value).lower() not in lowered, f"should not contain {value!r}"
    if kind == "used_tool":
        return value in tools_used, f"did not use {value} (used: {sorted(tools_used) or 'none'})"
    if kind == "max_chars":
        return len(answer) <= int(value), f"{len(answer)} chars exceeds {value}"
    if kind == "judge":
        ok, why = await _judge(str(value), answer)
        return ok, why
    return False, f"unknown assertion {kind!r}"


async def run_case(case: EvalCase, suite: str, run_label: str | None = None) -> EvalOutcome:
    spec = roster.get(case.agent)
    if case.model:
        spec = type(spec)(**{**vars(spec), "model": case.model})

    started = time.monotonic()
    result = await run_agent(spec, case.prompt)
    latency_ms = int((time.monotonic() - started) * 1000)

    failures: list[str] = []
    if not result.ok:
        failures.append(f"agent failed: {result.error}")
        passed_count = 0
    else:
        tools_used = _tools_used(result.task_id)
        checks = await asyncio.gather(*[
            check(assertion, result.content, tools_used) for assertion in case.assertions])
        passed_count = sum(1 for ok, _ in checks if ok)
        failures += [reason for ok, reason in checks if not ok]

    total = len(case.assertions) or 1
    score = passed_count / total
    passed = not failures

    # Record the model that actually ran, not "auto" — otherwise the scoreboard
    # cannot tell you which model produced which score.
    ran_model = result.model or spec.model
    store.record_eval(suite, case.id, case.agent, ran_model, passed, score,
                      failures, result.tokens, latency_ms, result.task_id, run_label)

    return EvalOutcome(case, passed, score, failures, result.content,
                       result.tokens, latency_ms, ran_model, result.task_id)


async def run_suite(name: str, model: str | None = None, only: str | None = None,
                    run_label: str | None = None, concurrency: int = 2,
                    on_result=None) -> list[EvalOutcome]:
    suite, cases = load_suite(name)
    if only:
        cases = [c for c in cases if c.id == only]
        if not cases:
            raise KeyError(f"no eval {only!r} in suite {name!r}")
    if model:
        for case in cases:
            case.model = model

    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(case: EvalCase) -> EvalOutcome:
        async with semaphore:
            outcome = await run_case(case, suite, run_label)
            if on_result:
                on_result(outcome)
            return outcome

    return list(await asyncio.gather(*[guarded(case) for case in cases]))
