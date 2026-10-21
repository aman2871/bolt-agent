"""The ReAct loop every agent shares. Specialisation lives in AgentSpec, not here."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from bolt.core import llm, router, store, tools

MAX_TOOL_RESULT_CHARS = 12_000


@dataclass
class AgentSpec:
    name: str
    model: str                      # a model name, or "auto" to let the router pick
    system_prompt: str
    blurb: str = ""                 # one line, shown wherever a person picks an agent
    tools: list[str] = field(default_factory=list)
    max_iterations: int = 8
    max_tokens: int | None = None
    temperature: float = 0.2
    # How hard the model should think before answering. "default" sends nothing
    # and leaves the model to decide; anything else needs Ollama's native API,
    # because the OpenAI-compatible endpoint silently drops it.
    effort: str = "default"
    can_delegate_to: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)   # hard needs, e.g. "vision"
    prefers: list[str] = field(default_factory=list)    # strengths to favour


@dataclass
class RunContext:
    """Passed to tools and carried through one task's execution."""
    task_id: str
    root_id: str
    spec: AgentSpec
    depth: int = 0
    budget_tokens: int = 1_000_000
    extra: dict[str, Any] = field(default_factory=dict)
    model: str = ""      # the model actually resolved for this run


@dataclass
class AgentResult:
    task_id: str
    content: str
    ok: bool
    iterations: int
    tokens: int
    error: str | None = None
    model: str = ""      # the model the router actually resolved for this run


# Every iteration resends the whole conversation, so a long tool-using run costs
# roughly the square of its context. Old tool output is the bulk of it and is
# rarely needed verbatim once the model has acted on it.
KEEP_TOOL_RESULTS_VERBATIM = int(os.environ.get("BOLT_KEEP_TOOL_RESULTS", "3"))
COMPACT_ABOVE_CHARS = int(os.environ.get("BOLT_COMPACT_ABOVE_CHARS", "24000"))
TRIMMED_TOOL_CHARS = 400


def _compact(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Shorten tool results the model has already had a chance to use.

    Returns the messages and how many characters were saved. System and user
    turns, and every assistant message, are left alone — losing those changes
    what the model is being asked.
    """
    size = sum(len(str(m.get("content") or "")) for m in messages)
    if size < COMPACT_ABOVE_CHARS:
        return messages, 0

    tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    keep = set(tool_positions[-KEEP_TOOL_RESULTS_VERBATIM:])

    compacted, saved = [], 0
    for i, message in enumerate(messages):
        content = message.get("content") or ""
        if (message.get("role") == "tool" and i not in keep
                and len(content) > TRIMMED_TOOL_CHARS):
            short = (content[:TRIMMED_TOOL_CHARS]
                     + f"\n… [{len(content) - TRIMMED_TOOL_CHARS} characters trimmed "
                       f"from this earlier result]")
            saved += len(content) - len(short)
            compacted.append({**message, "content": short})
        else:
            compacted.append(message)
    return compacted, saved


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    keep = MAX_TOOL_RESULT_CHARS // 2
    omitted = len(text) - 2 * keep
    return f"{text[:keep]}\n\n...[{omitted} chars omitted]...\n\n{text[-keep:]}"


async def run_agent(
    spec: AgentSpec,
    prompt: str,
    task_id: str | None = None,
    root_id: str | None = None,
    parent_id: str | None = None,
    depth: int = 0,
    budget_tokens: int = 1_000_000,
    extra: dict[str, Any] | None = None,
    images: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
    stream: bool = False,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> AgentResult:
    """Drive one agent to a final answer, persisting every message and run."""
    task_id = task_id or store.create_task(
        spec.name, prompt, parent_id=parent_id, root_id=root_id,
        depth=depth, budget_tokens=budget_tokens)
    task = store.get_task(task_id) or {}
    root_id = root_id or task.get("root_id") or task_id
    def emit(kind: str, data: dict[str, Any]) -> None:
        if on_event:
            on_event(kind, {"task_id": task_id, "agent": spec.name, **data})

    # Resolve the model once per task: a manual pin, an explicit model, or the
    # router's pick. Recorded per run so the dashboard shows what actually ran.
    # Streaming is for the agent being spoken to, not for everything it delegates.
    stream_output = stream and depth == 0
    choice = router.choose(spec.name, prompt, spec.model, spec.requires, spec.prefers,
                           needs_vision=bool(images))
    model = choice.model
    ctx = RunContext(task_id, root_id, spec, depth, budget_tokens, extra or {}, model)
    emit("model", {"model": model, "reason": choice.reason, "source": choice.source})

    store.set_status(task_id, "running")
    user_message: dict[str, Any] = (
        llm.vision_message(prompt, [llm.user_image(str(i)) for i in images])
        if images else {"role": "user", "content": prompt})
    # Earlier turns of this conversation, so "what about the second one?" means
    # something. Tool calls are deliberately left out — the agent needs what was
    # said, not how it got there.
    prior = [{"role": m["role"], "content": m["content"]}
             for m in (history or []) if m.get("content")]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": spec.system_prompt},
        *prior,
        user_message,
    ]
    store.add_message(task_id, 0, "system", spec.system_prompt)
    store.add_message(task_id, 1, "user",
                      prompt + (f"\n[{len(images)} image(s) attached]" if images else ""))
    seq = 2
    tool_schemas = tools.schemas(spec.tools)

    for iteration in range(1, spec.max_iterations + 1):
        if store.tokens_used_tree(root_id) > budget_tokens:
            used = store.tokens_used_tree(root_id)
            error = (f"token budget exhausted: {used:,} of {budget_tokens:,} tokens. "
                     f"This counts every token billed across the whole request, and "
                     f"each step resends the conversation — so a long tool-using "
                     f"search costs far more than its context size. Raise it with "
                     f"--budget, or ask something narrower.")
            salvaged = next((m["content"] for m in reversed(messages)
                             if m.get("role") == "assistant" and m.get("content")), "")
            if salvaged:
                salvaged = ("[incomplete — the token budget ran out before this "
                            "request finished. What was established so far:]\n\n"
                            + salvaged)
            store.set_status(task_id, "failed", error=error)
            return AgentResult(task_id, salvaged, False, iteration,
                               store.tokens_used(task_id), error, model)

        emit("iteration", {"iteration": iteration})
        request_messages, saved = _compact(messages)
        if saved:
            emit("compacted", {"chars_saved": saved})

        try:
            if stream_output:
                # Only the agent the user is talking to streams. Delegated agents
                # run at depth > 0, and their text interleaved into the reply
                # would be unreadable — their progress shows as tool events.
                emit("stream_start", {"iteration": iteration})
                completion = await llm.stream(
                    model, request_messages, tool_schemas or None,
                    temperature=spec.temperature, max_tokens=spec.max_tokens,
                    effort=spec.effort,
                    on_text=lambda piece: emit("delta", {"text": piece}),
                    on_reasoning=lambda piece: emit("reasoning", {"text": piece}),
                )
                emit("stream_end", {"had_text": bool(completion.content)})
            else:
                completion = await llm.complete(
                    model, request_messages, tool_schemas or None,
                    temperature=spec.temperature, max_tokens=spec.max_tokens,
                    effort=spec.effort,
                )
        except llm.LLMError as exc:
            store.record_run(task_id, spec.name, model, iteration, 0, 0, 0, ok=False, error=str(exc))
            store.set_status(task_id, "failed", error=str(exc))
            return AgentResult(task_id, "", False, iteration, store.tokens_used(task_id),
                               str(exc), model)

        store.record_run(task_id, spec.name, model, iteration,
                         completion.prompt_tokens, completion.completion_tokens,
                         completion.latency_ms, completion.attempts)

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": completion.content}
        if completion.tool_calls:
            assistant_msg["tool_calls"] = completion.tool_calls
        messages.append(assistant_msg)
        store.add_message(task_id, seq, "assistant", completion.content, completion.tool_calls)
        seq += 1

        if not completion.tool_calls:
            store.set_status(task_id, "done", result=completion.content)
            emit("final", {"content": completion.content})
            return AgentResult(task_id, completion.content, True, iteration,
                               store.tokens_used(task_id), None, model)

        # Independent tool calls in one turn run concurrently; this is what lets the
        # manager fan out to several specialists at once.
        for call in completion.tool_calls:
            fn = call["function"]
            emit("tool_call", {"tool": fn["name"], "arguments": fn["arguments"]})

        results = await asyncio.gather(*[
            tools.dispatch(call["function"]["name"], call["function"]["arguments"], ctx)
            for call in completion.tool_calls
        ])

        for call, result in zip(completion.tool_calls, results):
            fn = call["function"]
            result = _truncate(result)
            emit("tool_result", {"tool": fn["name"], "result": result[:500]})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "name": fn["name"], "content": result})
            store.add_message(task_id, seq, "tool", result, name=fn["name"])
            seq += 1

    # Out of steps, but the work is not worthless: everything found so far is in
    # `messages`. Ask once more with no tools available, so the model has to
    # answer from what it has rather than start another search.
    emit("wrapping_up", {"iterations": spec.max_iterations})
    messages.append({
        "role": "user",
        "content": ("You have run out of research steps. Answer now using only what "
                    "you have already found. Be concrete about what you know, and say "
                    "plainly what you did not manage to check."),
    })
    try:
        final = await llm.complete(model, messages, None, temperature=spec.temperature,
                                   max_tokens=spec.max_tokens, effort=spec.effort)
    except llm.LLMError as exc:
        error = (f"hit the {spec.max_iterations}-iteration cap, and the closing "
                 f"summary also failed: {exc}")
        store.set_status(task_id, "failed", error=error)
        return AgentResult(task_id, "", False, spec.max_iterations,
                           store.tokens_used(task_id), error, model)

    store.record_run(task_id, spec.name, model, spec.max_iterations + 1,
                     final.prompt_tokens, final.completion_tokens, final.latency_ms,
                     final.attempts)
    store.add_message(task_id, seq, "assistant", final.content)
    store.set_status(task_id, "done", result=final.content)
    emit("final", {"content": final.content, "truncated": True})
    return AgentResult(task_id, final.content, True, spec.max_iterations,
                       store.tokens_used(task_id), None, model)
