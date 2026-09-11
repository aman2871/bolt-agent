"""Empirical capability tests.

Every strength the router acts on should be something we watched a model do, not
something a config file claims. Each probe sends one real request and checks the
answer, so "good at code" means it wrote working code here, on this account,
today — not that a table said so.

Probes are deliberately small and checkable. A probe that cannot be checked
mechanically is not worth running.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from bolt.core import llm, store

# A 420x220 PNG containing a red square, a green circle and the text BOLT-7492.
# Reading the text back proves the model actually saw the image rather than
# merely accepting it.
_VISION_PNG = Path(__file__).resolve().parent.parent / "assets" / "vision_probe.png"


@dataclass
class ProbeResult:
    model: str
    capability: str
    supported: bool
    detail: str
    metric: float | None = None
    latency_ms: int = 0
    conclusive: bool = True
    """False when the request never completed.

    A timeout or a dropped connection says nothing about the model's ability, and
    the router uses probe results as hard filters — so recording an unreachable
    model as incapable would silently bench it until someone re-probed. Only
    conclusive results are stored.
    """


@dataclass
class Probe:
    name: str
    description: str
    run: Callable[[str], Awaitable[ProbeResult]]
    order: int = 50


def _ok(model: str, cap: str, detail: str, metric: float | None = None,
        latency_ms: int = 0) -> ProbeResult:
    return ProbeResult(model, cap, True, detail, metric, latency_ms)


def _no(model: str, cap: str, detail: str, latency_ms: int = 0) -> ProbeResult:
    return ProbeResult(model, cap, False, detail, None, latency_ms)


def _unknown(model: str, cap: str, exc: Exception) -> ProbeResult:
    """The request failed to complete. Not evidence about the model."""
    return ProbeResult(model, cap, False, f"inconclusive: {str(exc)[:140]}",
                       None, 0, conclusive=False)


def _extract_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for candidate in ([fenced.group(1)] if fenced else []) + [text]:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start:end + 1])
                except json.JSONDecodeError:
                    continue
    return None


def _extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (fenced[0] if fenced else text).strip()


# --- individual probes --------------------------------------------------------

async def probe_vision(model: str) -> ProbeResult:
    """Does it accept an image, and can it read text out of one?"""
    if not _VISION_PNG.is_file():
        return _no(model, "vision", "probe image missing")
    message = llm.vision_message(
        "What alphanumeric code is written in this image? Reply with only that code.",
        [_VISION_PNG])
    try:
        result = await llm.complete(model, [message], temperature=0)
    except llm.LLMError as exc:
        detail = str(exc)
        if "does not support image" in detail.lower():
            return _no(model, "vision", "refuses image input")   # a real refusal
        return _unknown(model, "vision", exc)

    read = "BOLT-7492" in result.content.replace(" ", "").upper()
    return ProbeResult(model, "vision", read,
                       "read the code correctly" if read
                       else f"accepted the image but misread it: {result.content[:80]!r}",
                       None, result.latency_ms)


async def probe_tools(model: str) -> ProbeResult:
    """Does it emit a well-formed tool call with the right arguments?"""
    tools = [{"type": "function", "function": {
        "name": "get_stock", "description": "Look up a stock level for a part.",
        "parameters": {"type": "object",
                       "properties": {"part_id": {"type": "string"},
                                      "warehouse": {"type": "integer"}},
                       "required": ["part_id", "warehouse"]}}}]
    messages = [{"role": "user",
                 "content": "How many of part XR-19 are in warehouse 4? Use the tool."}]
    try:
        result = await llm.complete(model, messages, tools, temperature=0)
    except llm.LLMError as exc:
        return _unknown(model, "tools", exc)

    if not result.tool_calls:
        return _no(model, "tools", "did not call the tool", result.latency_ms)

    call = result.tool_calls[0]["function"]
    try:
        args = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError:
        return _no(model, "tools", "tool arguments were not valid JSON", result.latency_ms)

    correct = (call["name"] == "get_stock"
               and str(args.get("part_id", "")).upper() == "XR-19"
               and int(args.get("warehouse", -1)) == 4)
    return ProbeResult(model, "tools", correct,
                       "called correctly" if correct else f"called with {args}",
                       None, result.latency_ms)


async def probe_structured_output(model: str) -> ProbeResult:
    """Can it return strict JSON matching a requested shape, and nothing else?"""
    messages = [{"role": "user", "content":
        'Reply with only this JSON object, no prose and no code fence: '
        '{"city": "Pune", "country": "India", "population_millions": <a number>}'}]
    try:
        result = await llm.complete(model, messages, temperature=0)
    except llm.LLMError as exc:
        return _unknown(model, "structured_output", exc)

    parsed = _extract_json(result.content)
    if not isinstance(parsed, dict):
        return _no(model, "structured_output", "did not return parseable JSON",
                   result.latency_ms)
    shape_ok = ({"city", "country", "population_millions"} <= set(parsed)
                and isinstance(parsed.get("population_millions"), (int, float)))
    clean = result.content.strip().startswith("{")   # no prose, no fence
    return ProbeResult(model, "structured_output", shape_ok,
                       ("exact JSON" if clean and shape_ok
                        else "correct shape but wrapped in extra text" if shape_ok
                        else f"wrong shape: {list(parsed)[:4]}"),
                       None, result.latency_ms)


async def probe_reasoning(model: str) -> ProbeResult:
    """A short multi-step problem with one checkable answer."""
    messages = [{"role": "user", "content":
        "A depot has 3 bus routes. Route A runs every 12 minutes, route B every 18 "
        "minutes, route C every 30 minutes. All three leave together at 06:00. "
        "At what time do all three next leave together? Reply with only the time "
        "in HH:MM."}]
    try:
        result = await llm.complete(model, messages, temperature=0)
    except llm.LLMError as exc:
        return _unknown(model, "reasoning", exc)
    # lcm(12,18,30) = 180 minutes -> 09:00
    correct = "09:00" in result.content or "9:00" in result.content
    return ProbeResult(model, "reasoning", correct,
                       "solved it" if correct else f"answered {result.content.strip()[:60]!r}",
                       None, result.latency_ms)


async def probe_code(model: str) -> ProbeResult:
    """Write a function, then actually run it against hidden tests."""
    messages = [{"role": "user", "content":
        "Write a Python function `merge_ranges(ranges)` that takes a list of "
        "(start, end) integer tuples and returns them merged and sorted, combining "
        "any that overlap or touch. Reply with only the function in one code block."}]
    try:
        result = await llm.complete(model, messages, temperature=0)
    except llm.LLMError as exc:
        return _unknown(model, "code", exc)

    code = _extract_code(result.content)
    harness = code + """

_cases = [
    ([(1, 3), (2, 6), (8, 10)], [(1, 6), (8, 10)]),
    ([(1, 4), (4, 5)],          [(1, 5)]),
    ([(5, 6), (1, 2)],          [(1, 2), (5, 6)]),
    ([],                        []),
]
for _in, _want in _cases:
    _got = [tuple(x) for x in merge_ranges(list(_in))]
    assert _got == _want, f"{_in} -> {_got}, wanted {_want}"
print("PASS")
"""
    with tempfile.TemporaryDirectory(prefix="bolt-probe-") as tmp:
        script = Path(tmp) / "probe.py"
        script.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(script)],
                                  capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return _no(model, "code", "generated code timed out", result.latency_ms)

    passed = proc.returncode == 0 and "PASS" in proc.stdout
    why = "code ran and passed every case" if passed else (
        (proc.stderr or proc.stdout).strip().splitlines()[-1][:160] if (proc.stderr or proc.stdout)
        else "no output")
    return ProbeResult(model, "code", passed, why, None, result.latency_ms)


async def probe_instruction_following(model: str) -> ProbeResult:
    """Does it obey an exact, easily-checked format constraint?"""
    messages = [{"role": "user", "content":
        "Reply with exactly three words, lowercase, separated by single spaces, "
        "naming three primary colours. No punctuation, no other text."}]
    try:
        result = await llm.complete(model, messages, temperature=0)
    except llm.LLMError as exc:
        return _unknown(model, "instruction_following", exc)
    text = result.content.strip()
    obeyed = bool(re.fullmatch(r"[a-z]+ [a-z]+ [a-z]+", text))
    return ProbeResult(model, "instruction_following", obeyed,
                       "followed the format exactly" if obeyed else f"returned {text[:60]!r}",
                       None, result.latency_ms)


async def probe_speed(model: str) -> ProbeResult:
    """Measured output rate. This is what sets the tier, instead of a guess."""
    messages = [{"role": "user", "content":
        "Write two plain sentences about why bus depots need scheduled maintenance."}]
    started = time.monotonic()
    try:
        result = await llm.complete(model, messages, temperature=0.3)
    except llm.LLMError as exc:
        return _unknown(model, "speed", exc)
    elapsed = max(time.monotonic() - started, 0.001)
    rate = result.completion_tokens / elapsed
    return ProbeResult(model, "speed", True,
                       f"{rate:.1f} output tokens/sec over {elapsed:.1f}s",
                       round(rate, 2), result.latency_ms)


async def probe_code_hard(model: str) -> ProbeResult:
    """A harder algorithmic task with edge cases the easy probe does not reach."""
    messages = [{"role": "user", "content":
        "Write a Python function `schedule(jobs)` where jobs is a list of "
        "(name, duration, deadline) tuples. Return the largest-size list of job "
        "names that can all be completed by their deadlines, running one job at a "
        "time starting at time 0, ordered by the sequence you would run them. If "
        "several sets tie on size, return the one with the smallest total duration. "
        "Reply with only the function in one code block."}]
    try:
        result = await llm.complete(model, messages, temperature=0)
    except llm.LLMError as exc:
        return _unknown(model, "code_hard", exc)

    harness = _extract_code(result.content) + """

def _check(jobs, want_names, want_len):
    got = list(schedule(list(jobs)))
    assert len(got) == want_len, f"{jobs} -> {got}, wanted {want_len} jobs"
    if want_names is not None:
        assert sorted(got) == sorted(want_names), f"{jobs} -> {got}, wanted {want_names}"
    t = 0
    for name in got:
        job = next(j for j in jobs if j[0] == name)
        t += job[1]
        assert t <= job[2], f"{name} finishes at {t}, past deadline {job[2]}"

# all three fit
_check([("a", 2, 2), ("b", 1, 3), ("c", 3, 9)], ["a", "b", "c"], 3)
# one must be dropped
_check([("a", 4, 4), ("b", 3, 5), ("c", 2, 6)], None, 2)
# deadline ties, ordering matters
_check([("x", 1, 1), ("y", 1, 2), ("z", 1, 3)], ["x", "y", "z"], 3)
# impossible job excluded
_check([("big", 10, 3), ("ok", 1, 2)], ["ok"], 1)
_check([], [], 0)
print("PASS")
"""
    with tempfile.TemporaryDirectory(prefix="bolt-probe-") as tmp:
        script = Path(tmp) / "hard.py"
        script.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(script)],
                                  capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return _no(model, "code_hard", "generated code timed out", result.latency_ms)

    passed = proc.returncode == 0 and "PASS" in proc.stdout
    why = "passed every case including edge cases" if passed else (
        (proc.stderr or proc.stdout).strip().splitlines()[-1][:160] or "no output")
    return ProbeResult(model, "code_hard", passed, why, None, result.latency_ms)


async def probe_reasoning_hard(model: str) -> ProbeResult:
    """A constraint puzzle that cannot be pattern-matched to a common answer."""
    messages = [{"role": "user", "content":
        "Five depots (A-E) each dispatch one bus. Facts:\n"
        "1. The bus from A leaves before the bus from C.\n"
        "2. No bus leaves between C and E, and C leaves first.\n"
        "3. B leaves last.\n"
        "4. D leaves immediately after A.\n"
        "Give the dispatch order, first to last, as five letters with no spaces "
        "or punctuation, and nothing else."}]
    try:
        result = await llm.complete(model, messages, temperature=0)
    except llm.LLMError as exc:
        return _unknown(model, "reasoning_hard", exc)
    # Brute-forced over all 120 orders: these four facts admit exactly one, ADCEB.
    answer = re.sub(r"[^A-E]", "", result.content.upper())
    correct = answer == "ADCEB"
    return ProbeResult(model, "reasoning_hard", correct,
                       "solved it" if correct else f"answered {answer[:12] or result.content[:40]!r}",
                       None, result.latency_ms)


async def probe_long_context(model: str) -> ProbeResult:
    """Can it find one planted fact in a long, boring document?"""
    filler = ("Depot maintenance log entry. Routine inspection completed, no faults "
              "recorded, vehicle returned to service on schedule. ")
    needle = ("Bus 4417 was withdrawn from service after inspector Raghavan found a "
              "cracked axle housing on 14 March. ")
    # ~12k words of filler with the needle two thirds of the way in.
    before = filler * 900
    after = filler * 450
    document = before + needle + after
    messages = [{"role": "user", "content":
        f"{document}\n\nUsing only the log above: which bus number was withdrawn "
        f"from service, and who found the fault? Reply as 'NUMBER NAME'."}]
    try:
        result = await llm.complete(model, messages, temperature=0)
    except llm.LLMError as exc:
        detail = str(exc)
        if "context" in detail.lower() or "too long" in detail.lower():
            return _no(model, "long_context", "rejected the document as too long")
        return _unknown(model, "long_context", exc)

    text = result.content.upper()
    found = "4417" in text and "RAGHAVAN" in text
    approx_tokens = result.prompt_tokens
    return ProbeResult(model, "long_context", found,
                       f"found the fact in a ~{approx_tokens // 1000}k-token document" if found
                       else f"missed it: {result.content.strip()[:60]!r}",
                       float(approx_tokens), result.latency_ms)


PROBES: list[Probe] = [
    Probe("speed", "Measured output tokens per second", probe_speed, order=10),
    Probe("tools", "Emits a correct tool call", probe_tools, order=20),
    Probe("structured_output", "Returns strict, parseable JSON", probe_structured_output, order=30),
    Probe("instruction_following", "Obeys an exact output format",
          probe_instruction_following, order=40),
    Probe("reasoning", "Solves a multi-step problem", probe_reasoning, order=50),
    Probe("code", "Writes code that passes hidden tests", probe_code, order=60),
    Probe("reasoning_hard", "Solves a constraint puzzle", probe_reasoning_hard, order=65),
    Probe("code_hard", "Handles a harder algorithm with edge cases", probe_code_hard, order=70),
    Probe("long_context", "Finds a planted fact in a long document",
          probe_long_context, order=80),
    Probe("vision", "Reads text out of an image", probe_vision, order=90),
]

CAPABILITIES = [p.name for p in sorted(PROBES, key=lambda p: p.order)]


def get(name: str) -> Probe | None:
    return next((p for p in PROBES if p.name == name), None)


async def run_for_model(model: str, capabilities: list[str] | None = None,
                        on_result: Callable[[ProbeResult], None] | None = None
                        ) -> list[ProbeResult]:
    """Run probes for one model, in order, recording each result."""
    wanted = capabilities or CAPABILITIES
    results: list[ProbeResult] = []
    for name in wanted:
        probe = get(name)
        if probe is None:
            continue
        try:
            result = await probe.run(model)
        except Exception as exc:                     # a probe must never take the run down
            result = _unknown(model, name, exc)
        # Only store results that actually tell us something.
        if result.conclusive:
            store.record_probe(result.model, result.capability, result.supported,
                               result.detail, result.metric)
        results.append(result)
        if on_result:
            on_result(result)
    return results


async def run_all(models: list[str], capabilities: list[str] | None = None,
                  concurrency: int = 2,
                  on_result: Callable[[ProbeResult], None] | None = None
                  ) -> dict[str, list[ProbeResult]]:
    """Probe several models. Models run concurrently, probes within one serially."""
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(model: str) -> tuple[str, list[ProbeResult]]:
        async with semaphore:
            return model, await run_for_model(model, capabilities, on_result)

    return dict(await asyncio.gather(*[guarded(m) for m in models]))
