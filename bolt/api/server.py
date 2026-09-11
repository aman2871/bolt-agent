"""HTTP and websocket front end for Bolt."""

from __future__ import annotations

import asyncio
import httpx
import os
from enum import Enum
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import bolt.registry  # noqa: F401  (registers tools)
from bolt.api import events
from dataclasses import replace

from bolt.core import (backgrounds, bus, llm, models as models_mod, proposals,
                       roster, router, store, uploads, workspace)
from bolt.core.loop import run_agent
from bolt.quality import gate, metrics
from bolt.sandbox import runner

app = FastAPI(title="Bolt", version="0.1.0")


@app.on_event("startup")
async def _startup_housekeeping() -> None:
    """Clean up what a previous process left behind.

    Tasks stay 'running' forever if the process died mid-flight, and sandbox
    snapshots are full copies of the workspace that nothing else removes.
    """
    orphaned = store.reconcile_running()
    if orphaned:
        print(f"marked {orphaned} interrupted task(s) as failed")
    freed = runner.sweep_snapshots()
    if freed:
        print(f"removed {freed} old sandbox snapshot(s)")
# A worker polls its inbox every 2s, so anything quiet for longer is gone.
WORKER_IDLE_MS = 15_000


def _consumer_idle_ms(consumer: dict[str, Any]) -> int:
    """Milliseconds since a consumer last did anything.

    Redis reports `inactive` as -1 when it cannot tell, which would read as
    "just active" on a naive comparison, so fall back to `idle` in that case.
    """
    inactive = consumer.get("inactive", -1)
    if isinstance(inactive, int) and inactive >= 0:
        return inactive
    return int(consumer.get("idle", 0))
DASHBOARD = Path(__file__).parent / "dashboard" / "index.html"


class ChatRequest(BaseModel):
    prompt: str
    conversation_id: str | None = None
    agent: str = "manager"
    budget_tokens: int = Field(400_000, ge=1_000)
    reply_to: str | None = None      # turn id being replied to


class FeedbackRequest(BaseModel):
    rating: str | None = None        # "up", "down", or null to clear
    note: str | None = None
    retry: bool = True               # a thumbs-down reworks the answer by default


class TaskRequest(BaseModel):
    prompt: str
    agent: str = "manager"
    budget_tokens: int = Field(400_000, ge=1_000)
    images: list[str] | None = None


class CodeRequest(BaseModel):
    task: str
    test_command: str = gate.DEFAULT_TEST_COMMAND
    budget_tokens: int = Field(400_000, ge=1_000)
    apply: bool = False


# The only directory served over HTTP. The workspace deliberately is not — but
# these are wallpapers the fetcher put here, and the browser has to load them.
if backgrounds.BACKGROUND_DIR.is_dir():
    app.mount("/backgrounds",
              StaticFiles(directory=str(backgrounds.BACKGROUND_DIR)), name="backgrounds")


@app.get("/api/backgrounds")
async def list_backgrounds() -> dict[str, Any]:
    """Photo backgrounds on disk, with where each came from."""
    return {"photos": backgrounds.available(),
            "licence": backgrounds.LICENCE,
            "catalogue": len(backgrounds.CATALOGUE)}


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    return DASHBOARD.read_text(encoding="utf-8")


class OllamaSettings(BaseModel):
    base_url: str | None = None
    api_key: str | None = None        # omitted means "keep what is stored"


class AgentPatch(BaseModel):
    blurb: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    max_iterations: int | None = None
    temperature: float | None = None
    can_delegate_to: list[str] | None = None
    prefers: list[str] | None = None
    requires: list[str] | None = None


class ModelEntry(BaseModel):
    name: str
    tier: str = "balanced"
    context: int = 128_000
    cost: float = 0.5
    vision: bool = False
    notes: str = ""


class ModelPin(BaseModel):
    model: str | None = None      # None or "auto" releases the pin


def _masked(key: str) -> str:
    """Never send a stored key back to the browser — only enough to recognise it."""
    if not key or key == "ollama":
        return ""
    return f"{key[:4]}…{key[-4:]}" if len(key) > 12 else "•" * len(key)


@app.get("/api/settings/ollama")
async def get_ollama_settings() -> dict[str, Any]:
    url, key = llm.endpoint()
    stored = store.get_setting("ollama.api_key") or ""
    return {"base_url": url, "key_masked": _masked(stored),
            "key_set": bool(stored),
            "from_env": not stored and bool(os.environ.get("OLLAMA_API_KEY")),
            "default_url": llm.BASE_URL}


@app.post("/api/settings/ollama")
async def set_ollama_settings(body: OllamaSettings) -> dict[str, Any]:
    if body.base_url is not None:
        url = body.base_url.strip().rstrip("/")
        if url:
            store.set_setting("ollama.base_url", url)
        else:
            store.delete_setting("ollama.base_url")
    if body.api_key is not None:
        key = body.api_key.strip()
        if key:
            store.set_setting("ollama.api_key", key)
        else:
            store.delete_setting("ollama.api_key")     # fall back to the environment
    await llm.aclose()                                  # the pooled client holds the old key
    return await test_ollama()


@app.post("/api/settings/ollama/test")
async def test_ollama() -> dict[str, Any]:
    """Ask the endpoint what it serves. That is the whole connection check."""
    url, _ = llm.endpoint()
    try:
        found = await models_mod.discovered()
    except Exception as exc:
        return {"ok": False, "base_url": url, "error": f"{type(exc).__name__}: {exc}",
                "models": [], "known": models_mod.names()}
    return {"ok": True, "base_url": url, "models": found, "known": models_mod.names()}


@app.get("/api/tools")
async def list_tools() -> list[dict[str, Any]]:
    from bolt.core import tools as tools_mod
    return [{"name": name, "description": entry.description}
            for name, entry in sorted(tools_mod.REGISTRY.items())]


@app.put("/api/agents/{name}")
async def upsert_agent(name: str, patch: AgentPatch) -> dict[str, Any]:
    key = name.strip().lower()
    if not key.isidentifier():
        raise HTTPException(status_code=400,
                            detail="an agent name must be a single word: letters, digits, underscore")
    fields = {k: v for k, v in patch.model_dump().items() if v is not None}
    new = key not in roster.load()
    if new and not fields.get("system_prompt"):
        raise HTTPException(status_code=400, detail="a new agent needs a system prompt")
    spec = roster.save(key, fields)
    return {"name": spec.name, "blurb": spec.blurb, "model": spec.model,
            "tools": spec.tools, "created": new, "builtin": roster.is_builtin(key)}


@app.delete("/api/agents/{name}")
async def delete_agent(name: str) -> dict[str, Any]:
    key = name.strip().lower()
    if key not in roster.load():
        raise HTTPException(status_code=404, detail=f"unknown agent {name!r}")
    roster.remove(key)
    # A shipped agent cannot be deleted, only reset — it comes back from the YAML.
    return {"removed": key, "reset_to_default": roster.is_builtin(key)}


@app.post("/api/models/catalogue")
async def add_model(entry: ModelEntry) -> dict[str, Any]:
    name = entry.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="a model name is required")
    models_mod.add(name, entry.model_dump(exclude={"name"}))
    return {"added": name, "known": models_mod.names()}


@app.delete("/api/models/catalogue/{name:path}")
async def forget_model(name: str) -> dict[str, Any]:
    if models_mod.is_builtin(name):
        raise HTTPException(status_code=400, detail="that model ships with Bolt")
    models_mod.forget(name)
    return {"removed": name, "known": models_mod.names()}


@app.get("/api/agents")
async def list_agents() -> list[dict[str, Any]]:
    out = []
    for name in roster.names():
        spec = roster.get(name)
        choice = router.choose(name, "", spec.model, spec.requires, spec.prefers)
        out.append({"name": name, "blurb": spec.blurb,
                    "builtin": roster.is_builtin(name),
                    "system_prompt": spec.system_prompt,
                    "temperature": spec.temperature,
                    "model": spec.model, "tools": spec.tools,
                    "max_iterations": spec.max_iterations,
                    "can_delegate_to": spec.can_delegate_to,
                    "requires": spec.requires, "prefers": spec.prefers,
                    "resolved_model": choice.model, "model_source": choice.source,
                    "model_reason": choice.reason})
    return out


@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    from bolt.core import probes as probes_mod
    return {"models": [m.as_dict() for m in models_mod.catalogue().values()],
            "capabilities": models_mod.capabilities_seen(),
            "discriminating": models_mod.discriminating(),
            "descriptions": {p.name: p.description for p in probes_mod.PROBES},
            "vision": models_mod.supporting("vision"),
            "overrides": router.overrides()}


@app.post("/api/models/probe")
async def probe_models(capability: str | None = None,
                       model: str | None = None) -> dict[str, Any]:
    """Run the capability suite for real and store what actually happened."""
    targets = [model] if model else models_mod.names()
    caps = [capability] if capability else None
    results = await models_mod.probe_all(targets, caps, on_result=lambda r: events.publish(
        "probe", {"agent": "probe", "model": r.model, "capability": r.capability,
                  "supported": r.supported, "detail": r.detail, "metric": r.metric}))
    return {"results": {name: [{"capability": r.capability, "supported": r.supported,
                                "detail": r.detail, "metric": r.metric} for r in rows]
                        for name, rows in results.items()},
            "discriminating": models_mod.discriminating()}


@app.post("/api/agents/{name}/model")
async def set_agent_model(name: str, pin: ModelPin) -> dict[str, Any]:
    try:
        spec = roster.get(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    wanted = (pin.model or "auto").strip()
    if wanted in ("auto", "clear", "none", ""):
        router.set_override(name, None)
    else:
        if wanted not in models_mod.names():
            raise HTTPException(status_code=400, detail=f"unknown model {wanted!r}")
        gaps = router.missing_capabilities(models_mod.get(wanted), spec.requires)
        if gaps:
            raise HTTPException(
                status_code=400,
                detail=f"{name} requires {', '.join(gaps)} and {wanted} is not "
                       f"known to have {'it' if len(gaps) == 1 else 'them'}")
        router.set_override(name, wanted)

    choice = router.choose(name, "", spec.model, spec.requires, spec.prefers)
    return {"agent": name, "resolved_model": choice.model,
            "model_source": choice.source, "model_reason": choice.reason}


@app.post("/api/task")
async def create_task(request: TaskRequest) -> dict[str, Any]:
    try:
        spec = roster.get(request.agent)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # A bad image path is the caller's mistake, not a server fault: say which
    # file and why, instead of a bare 500.
    for image in request.images or []:
        if str(image).startswith("data:"):
            continue
        try:
            llm.user_image(str(image))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    result = await run_agent(spec, request.prompt, budget_tokens=request.budget_tokens,
                             images=request.images,
                             extra={"on_event": events.publish}, on_event=events.publish)
    return {"task_id": result.task_id, "ok": result.ok, "content": result.content,
            "error": result.error, "tokens": result.tokens, "iterations": result.iterations}


@app.post("/api/code")
async def create_code_task(request: CodeRequest) -> dict[str, Any]:
    result = await gate.run_code_task(
        request.task, test_command=request.test_command,
        budget_tokens=request.budget_tokens, on_event=events.publish)
    applied = gate.apply_snapshot(result) if (request.apply and result.ok) else None
    return {
        "task_id": result.task_id, "ok": result.ok, "error": result.error,
        "workdir": result.workdir, "artifact_id": result.artifact_id,
        "forge_report": result.forge_report, "applied": applied,
        "attempts": [vars(attempt) for attempt in result.attempts],
    }


# In-flight replies, so one can be stopped. Aborting the browser request alone
# would leave the agent running, still burning tokens and still calling tools.
_running: dict[str, asyncio.Task] = {}


@app.post("/api/chat/stop")
async def chat_stop(conversation_id: str = Form(None)) -> dict[str, Any]:
    """Stop a reply that is being generated."""
    targets = ([k for k in _running if k.startswith(f"{conversation_id}:")]
               if conversation_id else list(_running))
    stopped = []
    for key in targets:
        task = _running.get(key)
        if task and not task.done():
            task.cancel()
            stopped.append(key.split(":", 1)[0])
    events.publish("chat_stopped", {"agent": "chat", "conversations": stopped})
    return {"stopped": stopped}


class RewindRequest(BaseModel):
    seq: int | None = None       # None rewinds to the last user turn


@app.post("/api/conversations/{conversation_id}/rewind")
async def rewind(conversation_id: str, request: RewindRequest) -> dict[str, Any]:
    """Drop a turn and everything after it, so the last message can be redone."""
    if store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="no such conversation")

    seq = request.seq
    if seq is None:
        last = store.last_user_turn(conversation_id)
        if last is None:
            return {"removed": 0, "content": None, "turns": []}
        seq = last["seq"]

    removed = store.turns_from(conversation_id, seq)
    content = next((t["content"] for t in removed if t["role"] == "user"), None)
    store.delete_turns_from(conversation_id, seq)
    return {"removed": len(removed), "content": content,
            "turns": store.conversation_turns(conversation_id)}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    """One turn of a conversation, with the previous turns as context."""
    try:
        spec = roster.get(request.agent)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    conversation_id = request.conversation_id
    if not conversation_id or store.get_conversation(conversation_id) is None:
        conversation_id = store.create_conversation(request.agent)

    history = store.conversation_history(conversation_id)

    prompt = request.prompt
    if request.reply_to:
        quoted = store.get_turn(request.reply_to)
        if quoted:
            # Naming the message being answered beats hoping the model infers it
            # from position, which is exactly what breaks in a long thread.
            prompt = (f"Replying to your earlier message:\n"
                      f"---\n{quoted['content'][:4000]}\n---\n\n{request.prompt}")

    # Corrections left on thumbs-down replies. Prompt-level guidance, not training.
    lessons = store.agent_lessons(request.agent)
    if lessons:
        spec = replace(spec, system_prompt=spec.system_prompt + "\n\nCorrections from "
                       "earlier feedback, apply them:\n"
                       + "\n".join(f"- {lesson}" for lesson in lessons))

    store.add_turn(conversation_id, "user", request.prompt)
    events.publish("chat_turn", {"agent": request.agent, "conversation_id": conversation_id,
                                 "role": "user", "content": request.prompt})

    # The collecting context must be entered before the task is created: a task
    # copies the current context when it is scheduled, so a ContextVar set
    # afterwards is invisible inside it and writes would go straight to disk.
    with proposals.collecting(conversation_id, request.agent) as collector:
        task = asyncio.ensure_future(
            run_agent(spec, prompt, budget_tokens=request.budget_tokens,
                      history=history, stream=True, extra={"on_event": events.publish},
                      on_event=events.publish))
    # Keyed per request: two chats in one conversation would otherwise share a
    # slot, so the second would orphan the first and stop would miss it.
    request_key = f"{conversation_id}:{id(task)}"
    _running[request_key] = task
    try:
        result = await task
    except asyncio.CancelledError:
        # The user pressed stop. Keep their message so it can be edited or
        # retried, and record that the reply did not finish.
        store.add_turn(conversation_id, "assistant", "(stopped)", agent=request.agent)
        events.publish("chat_turn", {"agent": request.agent, "role": "assistant",
                                     "conversation_id": conversation_id,
                                     "content": "(stopped)", "stopped": True})
        return {"conversation_id": conversation_id, "ok": False, "stopped": True,
                "reply": "(stopped)", "agent": request.agent, "tokens": 0}
    finally:
        _running.pop(request_key, None)

    reply = result.content if result.ok else f"That failed. {result.error}"
    turn_id = store.add_turn(conversation_id, "assistant", reply, agent=request.agent,
                             task_id=result.task_id, tokens=result.tokens)
    store.attach_proposals(collector.ids, turn_id)
    pending = [proposals.summarise(store.get_proposal(i)) for i in collector.ids]
    events.publish("chat_turn", {"agent": request.agent, "conversation_id": conversation_id,
                                 "role": "assistant", "content": reply,
                                 "task_id": result.task_id})

    return {"conversation_id": conversation_id, "ok": result.ok, "reply": reply,
            "agent": request.agent, "task_id": result.task_id, "turn_id": turn_id,
            "tokens": result.tokens, "model": result.model,
            "lessons_applied": len(lessons), "proposals": pending}


# How many times a question is reworked before asking the person to rephrase.
# A model that has missed three times is usually missing context, not effort.
MAX_RETRIES = int(os.environ.get("BOLT_MAX_RETRIES", "3"))

RETRY_BRIEF = """\
You answered the question below and the user rejected the answer. Work out what
was actually being asked and answer it properly this time.

## The question
{question}

## Answers already rejected
{rejected}

Do not repeat the rejected approach or restate it in different words. If the
question is genuinely ambiguous, say which part is unclear rather than guessing
again."""

GIVE_UP = """\
I've tried this {attempts} times and still not got it right, so guessing again \
probably won't help.

Could you tell me more about what you're after? What was wrong with the answers \
above, what you expected to see, or an example of a good answer would each be \
enough to get me there."""


def _rejected_block(attempts: list[dict[str, Any]]) -> str:
    lines = []
    for i, attempt in enumerate(attempts, 1):
        lines.append(f"### Attempt {i}\n{(attempt['content'] or '')[:1500]}")
        if attempt.get("note"):
            lines.append(f"The user said this was wrong because: {attempt['note']}")
    return "\n\n".join(lines)


@app.post("/api/turns/{turn_id}/feedback")
async def rate_turn(turn_id: str, request: FeedbackRequest) -> dict[str, Any]:
    turn = store.get_turn(turn_id)
    if turn is None:
        raise HTTPException(status_code=404, detail="no such turn")
    if request.rating not in (None, "up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up', 'down' or null")

    agent = turn.get("agent") or "manager"
    conversation_id = turn["conversation_id"]
    store.set_feedback(turn_id, request.rating, request.note, agent)

    base = {"turn_id": turn_id, "rating": request.rating, "note": request.note,
            "lessons": store.agent_lessons(agent), "retried": False}
    if request.rating != "down" or not request.retry:
        return base

    question = store.preceding_user_turn(conversation_id, turn["seq"])
    if question is None:
        return base

    rejected = store.rejected_attempts(conversation_id, question["seq"])
    if len(rejected) >= MAX_RETRIES:
        message = GIVE_UP.format(attempts=len(rejected))
        new_turn = store.add_turn(conversation_id, "assistant", message, agent=agent)
        return {**base, "needs_explanation": True, "attempts": len(rejected),
                "reply": message, "new_turn_id": new_turn, "agent": agent}

    spec = roster.get(agent)
    lessons = store.agent_lessons(agent)
    if lessons:
        spec = replace(spec, system_prompt=spec.system_prompt + "\n\nCorrections from "
                       "earlier feedback, apply them:\n"
                       + "\n".join(f"- {lesson}" for lesson in lessons))

    prompt = RETRY_BRIEF.format(question=question["content"],
                                rejected=_rejected_block(rejected))
    # History stops before the question itself, so the retry sees the conversation
    # that led here without the answers it is meant to replace.
    history = [m for m in store.conversation_history(conversation_id)][:-1]

    result = await run_agent(spec, prompt, history=history,
                             extra={"on_event": events.publish}, on_event=events.publish)
    reply = result.content if result.ok else f"That failed. {result.error}"
    new_turn = store.add_turn(conversation_id, "assistant", reply, agent=agent,
                              task_id=result.task_id, tokens=result.tokens)
    events.publish("chat_turn", {"agent": agent, "conversation_id": conversation_id,
                                 "role": "assistant", "content": reply, "retry": True})

    return {**base, "retried": True, "attempt": len(rejected) + 1,
            "remaining": MAX_RETRIES - len(rejected) - 1,
            "reply": reply, "new_turn_id": new_turn, "agent": agent,
            "model": result.model, "tokens": result.tokens}


@app.delete("/api/feedback/lessons/{agent}")
async def clear_agent_lessons(agent: str) -> dict[str, Any]:
    """Drop the corrections being fed into this agent's prompt."""
    return {"agent": agent, "cleared": store.clear_lessons(agent),
            "lessons": store.agent_lessons(agent)}


@app.get("/api/feedback")
async def feedback_overview() -> dict[str, Any]:
    from bolt.core import roster as roster_mod
    return {"summary": store.feedback_summary(),
            "lessons": {name: store.agent_lessons(name) for name in roster_mod.names()}}


@app.get("/api/proposals/{proposal_id}")
async def get_proposal(proposal_id: str) -> dict[str, Any]:
    proposal = store.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail="no such proposal")
    return {**proposals.summarise(proposal),
            "diff": proposals.diff_for(proposal, workspace.DEFAULT)}


@app.post("/api/proposals/{proposal_id}/undo")
async def undo_proposal(proposal_id: str, force: bool = False) -> dict[str, Any]:
    try:
        message = proposals.undo(proposal_id, force)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such proposal")
    except proposals.Conflict as exc:
        # 409: the caller can retry with force once they know what they'd lose.
        raise HTTPException(status_code=409, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not undo: {exc}")
    events.publish("proposal", {"agent": "gate", "proposal_id": proposal_id,
                                "decision": "undo"})
    return {"proposal_id": proposal_id, "status": "undone", "message": message}


class Decision(str, Enum):
    apply = "apply"
    reject = "reject"


@app.post("/api/proposals/{proposal_id}/{decision}")
async def decide_proposal(proposal_id: str, decision: Decision,
                          force: bool = False) -> dict[str, Any]:
    if store.get_proposal(proposal_id) is None:
        raise HTTPException(status_code=404, detail="no such proposal")
    try:
        message = proposals.apply(proposal_id, force) if decision is Decision.apply \
            else proposals.reject(proposal_id)
    except proposals.Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not write: {exc}")
    events.publish("proposal", {"agent": "gate", "proposal_id": proposal_id,
                                "decision": decision.value})
    return {"proposal_id": proposal_id, "status": decision.value, "message": message}


@app.post("/api/files")
async def upload_file(file: UploadFile = File(...),
                      conversation_id: str = Form(None)) -> dict[str, Any]:
    """Accept a dropped file and put it where agents can read it.

    Agents address everything by path, so this saves into the workspace and hands
    back the path — `watch_video`, `look_at_image` and the rest need no changes.
    """
    data = await file.read()
    try:
        saved = uploads.save(file.filename or "upload", data, conversation_id)
    except uploads.UploadRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not save: {exc}")

    relative = uploads.relative(saved)
    kind = ("video" if saved.suffix.lower() in {".mp4", ".mov", ".m4v", ".webm",
                                                ".mkv", ".avi"}
            else "image" if saved.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif",
                                                     ".webp", ".bmp"}
            else "file")
    events.publish("upload", {"agent": "you", "path": relative, "kind": kind,
                              "bytes": len(data)})
    return {"path": relative, "name": saved.name, "kind": kind, "bytes": len(data)}


@app.get("/api/checkpoints")
async def list_checkpoints(limit: int = 50) -> list[dict[str, Any]]:
    """File changes on disk that can still be undone, newest first."""
    return proposals.checkpoints(limit)


@app.post("/api/turns/{turn_id}/undo")
async def undo_turn(turn_id: str, force: bool = False) -> dict[str, Any]:
    """Undo every file change made by one reply."""
    outcome = proposals.undo_turn(turn_id, force)
    events.publish("proposal", {"agent": "gate", "turn_id": turn_id,
                                "decision": "undo_turn"})
    return outcome


@app.get("/api/conversations")
async def list_conversations(limit: int = 50) -> list[dict[str, Any]]:
    return store.conversations(limit)


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="no such conversation")
    return {"conversation": conversation,
            "turns": store.conversation_turns(conversation_id),
            "feedback": store.feedback_for(conversation_id)}


@app.delete("/api/conversations/{conversation_id}")
async def remove_conversation(conversation_id: str) -> dict[str, Any]:
    store.delete_conversation(conversation_id)
    return {"deleted": conversation_id}


@app.get("/api/tasks")
async def list_tasks(limit: int = 50, roots_only: bool = True) -> list[dict[str, Any]]:
    return metrics.recent_tasks(limit, roots_only)


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="no such task")
    rows = store.connect().execute(
        "SELECT seq, role, name, content, tool_calls FROM messages WHERE task_id = ? ORDER BY seq",
        (task_id,)).fetchall()
    return {"task": task, "tree": metrics.task_tree(task["root_id"]),
            "messages": [dict(row) for row in rows]}


@app.get("/api/metrics")
async def get_metrics(since_hours: float | None = None) -> dict[str, Any]:
    return {"totals": metrics.totals(since_hours), "agents": metrics.agent_stats(since_hours)}


@app.get("/api/usage/cloud")
async def get_cloud_usage() -> dict[str, Any]:
    """Ollama's own session and weekly limits, straight from the account.

    ollama.com/api/usage accepts the same bearer key the chat API does, and
    returns exactly what the website's Cloud usage panel shows. It carries no
    reset times, so we do not invent any.
    """
    url, key = llm.endpoint()
    if not key or key == "ollama":
        return {"ok": False, "reason": "no_key",
                "error": "No Ollama API key saved — add one under Connection."}
    account = "https://ollama.com"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as http:
            response = await http.get(f"{account}/api/usage",
                                      headers={"Authorization": f"Bearer {key}"})
        if response.status_code == 401:
            return {"ok": False, "reason": "unauthorized",
                    "error": "Ollama rejected the saved key."}
        response.raise_for_status()
        return {"ok": True, **response.json()}
    except Exception as exc:
        return {"ok": False, "reason": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/usage")
async def get_usage(days: int = 30) -> dict[str, Any]:
    return store.usage_summary(max(1, min(days, 365)))


@app.get("/api/evals")
async def get_evals(suite: str | None = None, limit: int = 200) -> dict[str, Any]:
    rows = store.connect().execute(
        "SELECT suite, eval_id, agent, model, passed, score, failures, tokens,"
        " latency_ms, run_label, task_id, created_at FROM evals"
        " ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return {"scoreboard": store.eval_scoreboard(suite, limit),
            "runs": [dict(row) for row in rows]}


@app.get("/api/artifacts/{task_id}")
async def get_artifacts(task_id: str) -> list[dict[str, Any]]:
    rows = store.connect().execute(
        "SELECT * FROM artifacts WHERE task_id = ? ORDER BY created_at DESC",
        (task_id,)).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/health")
async def health() -> dict[str, Any]:
    from bolt.voice import control
    """What the system can actually do right now, not what it is configured to do."""
    redis_up = await bus.ping()

    # Redis keeps a consumer registered after its process dies, so counting
    # group members would report ghosts. Count only recently-active consumers.
    workers: dict[str, int] = {}
    if redis_up:
        for name in roster.names():
            try:
                consumers = await bus.redis().xinfo_consumers(bus.inbox(name), bus.GROUP)
                workers[name] = sum(1 for c in consumers if _consumer_idle_ms(c) < WORKER_IDLE_MS)
            except Exception:
                workers[name] = 0
        await bus.aclose()

    docker_up = runner.docker_available()
    return {
        "voice": control.snapshot(),
        "viewers": len(_viewers),
        "redis": redis_up,
        "workers": workers,
        "docker": docker_up,
        "sandbox_tier": "docker" if docker_up else "subprocess",
        "bus_mode": __import__("os").environ.get("BOLT_BUS", "inproc"),
        "ollama_url": __import__("bolt.core.llm", fromlist=["BASE_URL"]).BASE_URL,
        "snapshots": sorted(
            p.name for p in runner.SNAPSHOT_ROOT.glob("t_*") if p.is_dir()
        ) if runner.SNAPSHOT_ROOT.exists() else [],
    }


@app.post("/api/voice/stop")
async def voice_stop() -> dict[str, Any]:
    """Stop whatever is speaking, in this process or the daemon."""
    from bolt.voice import control
    seq = control.request_stop()
    events.publish("voice_stop", {"agent": "voice"})
    return {"stopped": True, "stop_seq": seq}


@app.post("/api/voice/pause")
async def voice_pause(seconds: float = Form(None)) -> dict[str, Any]:
    """Stop the daemon listening while the browser mic has the room.

    Without this both pipelines hear the same sentence and answer it twice.
    """
    from bolt.voice import control
    until = control.pause(seconds) if seconds else control.pause()
    return {"paused": True, "until": until}


@app.post("/api/voice/resume")
async def voice_resume() -> dict[str, Any]:
    from bolt.voice import control
    control.resume()
    return {"paused": False}


@app.post("/api/voice/enabled")
async def voice_enabled(enabled: bool = Form(...)) -> dict[str, Any]:
    """The user's own switch. Unlike /pause it does not expire."""
    from bolt.voice import control
    control.set_enabled(enabled)
    if not enabled:
        control.request_stop()          # anything mid-sentence stops now
    return control.snapshot()


@app.get("/api/voice/control")
async def voice_control() -> dict[str, Any]:
    from bolt.voice import control
    return control.snapshot()


@app.get("/api/voice/status")
async def voice_status() -> dict[str, Any]:
    """Whether this process can transcribe and speak. The dashboard mic button
    needs the first; the browser handles playback itself."""
    from bolt.voice import stt, tts
    stt_ok, stt_why = stt.available()
    tts_ok, tts_why = tts.available()
    return {"stt": stt_ok, "stt_detail": stt_why,
            "tts": tts_ok, "tts_detail": tts_why,
            "ffmpeg": shutil.which("ffmpeg") is not None}


@app.post("/api/voice/ask")
async def voice_ask(audio: UploadFile = File(...),
                    agent: str = Form("manager"),
                    run: bool = Form(True),
                    conversation_id: str = Form(None)) -> dict[str, Any]:
    """Transcribe recorded audio, work out who it is for, and optionally run it.

    Transcription is local — the same whisper.cpp model the voice loop uses — so
    recorded speech never leaves the machine.
    """
    from bolt.voice import stt, wake

    ok, why = stt.available()
    if not ok:
        raise HTTPException(status_code=503, detail=f"speech to text unavailable: {why}")
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=503, detail="ffmpeg is needed to decode browser audio")

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty recording")

    with tempfile.TemporaryDirectory(prefix="bolt-mic-") as tmp:
        source = Path(tmp) / "clip.webm"
        wav = Path(tmp) / "clip.wav"
        source.write_bytes(raw)
        # Browsers record webm/opus; whisper wants 16 kHz mono PCM.
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(source), "-ac", "1", "-ar", "16000",
             "-f", "wav", str(wav)],
            capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not wav.is_file():
            raise HTTPException(status_code=400,
                                detail=f"could not decode the recording: {proc.stderr[:200]}")
        transcript = stt.transcribe_file(wav)

    if not transcript:
        return {"transcript": "", "addressed": False, "agent": None,
                "reply": None, "detail": "nothing recognisable in the recording"}

    address = wake.resolve(transcript, agent)
    events.publish("transcript", {"agent": "mic", "text": transcript,
                                  "addressed": address.addressed,
                                  "routed_to": address.agent})

    if not address.addressed:
        return {"transcript": transcript, "addressed": False, "agent": None, "reply": None,
                "detail": 'not addressed to Bolt — start with "Bolt" or an agent name'}
    if not run or not address.text:
        return {"transcript": transcript, "addressed": True, "agent": address.agent,
                "reply": None, "detail": "heard, not run"}

    # Spoken and typed turns share one thread, so you can start a question by
    # voice and follow it up by typing.
    if not conversation_id or store.get_conversation(conversation_id) is None:
        conversation_id = store.create_conversation(address.agent)
    history = store.conversation_history(conversation_id)
    store.add_turn(conversation_id, "user", address.text)
    events.publish("chat_turn", {"agent": address.agent, "conversation_id": conversation_id,
                                 "role": "user", "content": address.text, "spoken": True})

    result = await run_agent(roster.get(address.agent), address.text, history=history,
                             extra={"on_event": events.publish}, on_event=events.publish)
    reply = result.content if result.ok else f"That failed. {result.error}"
    voice_turn_id = store.add_turn(conversation_id, "assistant", reply, agent=address.agent,
                                   task_id=result.task_id, tokens=result.tokens)
    events.publish("chat_turn", {"agent": address.agent, "conversation_id": conversation_id,
                                 "role": "assistant", "content": reply})

    return {"transcript": transcript, "addressed": True, "agent": address.agent,
            "text": address.text, "ok": result.ok, "reply": reply,
            "conversation_id": conversation_id, "turn_id": voice_turn_id,
            "task_id": result.task_id, "tokens": result.tokens}


@app.post("/api/voice/speak")
async def voice_speak(text: str = Form(...)) -> Response:
    """Synthesise a reply as WAV so the browser can play it back."""
    import io
    import wave

    import numpy as np

    from bolt.voice import tts

    ok, why = tts.available()
    if not ok:
        raise HTTPException(status_code=503, detail=f"text to speech unavailable: {why}")

    spoken = tts.speakable(text)[:1200]
    if not spoken:
        raise HTTPException(status_code=400, detail="nothing speakable in that text")

    chunks, rate = [], 24_000
    for samples, sample_rate in tts.synthesize_stream(spoken):
        chunks.append(samples)
        rate = sample_rate
    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm.tobytes())
    return Response(content=buffer.getvalue(), media_type="audio/wav")


# Open dashboard sockets. The voice daemon listens only while this is non-empty,
# so closing the last tab stops the microphone.
_viewers: set[WebSocket] = set()


def _refresh_tab_state() -> None:
    from bolt.voice import control
    if _viewers:
        control.mark_tab_open()
    else:
        control.mark_tab_closed()


@app.websocket("/ws")
async def stream(websocket: WebSocket) -> None:
    from bolt.voice import control

    await websocket.accept()
    queue = events.subscribe()
    _viewers.add(websocket)
    _refresh_tab_state()

    # Waiting only on the event queue means a closed tab goes unnoticed until the
    # next send — up to the timeout below. Reading from the socket in parallel
    # makes the disconnect land immediately, which is what stops the microphone.
    async def watch_for_close() -> None:
        try:
            while True:
                await websocket.receive()
                # Proof the browser is still alive, from the browser itself. A
                # tab that dies without closing its socket simply stops sending,
                # and the freshness window below expires it.
                control.mark_tab_open()
        except Exception:
            return

    closed = asyncio.create_task(watch_for_close())
    try:
        for event in events.history()[-50:]:
            await websocket.send_json(event)
        while not closed.done():
            pending = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({pending, closed},
                                         timeout=20,
                                         return_when=asyncio.FIRST_COMPLETED)
            if closed in done:
                pending.cancel()
                break
            if pending in done:
                await websocket.send_json(pending.result())
                continue
            pending.cancel()
            # Deliberately does not refresh the tab's liveness: this side keeps
            # working even when the browser is gone, so only the client's own
            # messages count as proof it is still there.
            await websocket.send_json({"kind": "ping"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        closed.cancel()
        events.unsubscribe(queue)
        _viewers.discard(websocket)
        _refresh_tab_state()
        # A closing tab should never leave the daemon muted.
        control.resume()
