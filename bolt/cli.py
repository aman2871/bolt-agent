"""Bolt command line."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

import bolt.registry  # noqa: F401  (registers tools)
from bolt.agents import worker as worker_mod
from bolt.core import bus, loop, models as models_mod, roster, router, store
from bolt.quality import gate as gate_mod

app = typer.Typer(add_completion=False, help="Bolt multi-agent system")
console = Console()




def _printer(verbose: bool):
    def on_event(kind: str, data: dict[str, Any]) -> None:
        if not verbose or kind == "final":
            return
        agent = data.get("agent", "?")
        if kind == "model":
            colour = {"override": "yellow", "explicit": "dim", "auto": "green"}.get(
                data.get("source", ""), "dim")
            console.print(f"[{colour}]{agent} uses {data['model']}[/{colour}] "
                          f"[dim]({data['reason']})[/dim]")
        elif kind == "iteration":
            console.print(f"[dim]{agent} · step {data['iteration']}[/dim]")
        elif kind == "tool_call":
            console.print(f"[cyan]{agent} → {data['tool']}[/cyan] {data['arguments'][:160]}")
        elif kind == "tool_result":
            console.print(f"[dim green]{agent} ← {data['tool']}[/dim green] "
                          f"{data['result'][:160].replace(chr(10), ' ')}")
        elif kind == "sandbox_ready":
            console.print(f"[yellow]sandbox:[/yellow] {data['workdir']}")
        elif kind == "attempt":
            console.rule(f"[yellow]attempt {data['number']}/{data['of']}[/yellow]")
        elif kind == "tests":
            mark = "[green]PASSED[/green]" if data["passed"] else "[red]FAILED[/red]"
            console.print(f"tests {mark} (exit {data['exit_code']}, "
                          f"{data['isolation']} isolation, {data['duration_ms']}ms)")
        elif kind == "review":
            colour = "green" if data["verdict"] == "approve" else "red"
            console.print(f"review [{colour}]{data['verdict']}[/{colour}]: {data['summary']}")
            for finding in data["findings"]:
                console.print(f"  [red]·[/red] {finding}")
        elif kind == "worker_ready":
            console.print(f"[green]worker ready:[/green] {agent}")
        elif kind == "worker_error":
            console.print(f"[red]worker error:[/red] {data.get('error')}")
    return on_event


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="What you want done."),
    agent: str = typer.Option("manager", "--agent", "-a", help="Which agent to address."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q"),
    budget: int = typer.Option(200_000, "--budget", help="Token budget for the whole task tree."),
) -> None:
    """Send a request to an agent."""
    try:
        spec = roster.get(agent)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    printer = _printer(verbose)
    result = asyncio.run(loop.run_agent(spec, prompt, budget_tokens=budget,
                                        extra={"on_event": printer}, on_event=printer))
    console.rule(f"[bold]{spec.name}[/bold]")
    console.print(result.content if result.ok else f"[red]failed: {result.error}[/red]")
    console.print(f"[dim]task {result.task_id} · {result.iterations} steps · "
                  f"{result.tokens} tokens[/dim]")
    if not result.ok:
        raise typer.Exit(1)


@app.command()
def code(
    task: str = typer.Argument(..., help="The change you want made."),
    test_command: str = typer.Option(gate_mod.DEFAULT_TEST_COMMAND, "--test", "-t"),
    apply: bool = typer.Option(False, "--apply", help="Apply the approved diff to the workspace."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q"),
    budget: int = typer.Option(400_000, "--budget"),
) -> None:
    """Make a code change behind the sandbox and review gate."""
    result = asyncio.run(gate_mod.run_code_task(
        task, test_command=test_command, budget_tokens=budget, on_event=_printer(verbose)))

    console.rule("[bold]result[/bold]")
    if result.ok:
        console.print(f"[green]approved[/green] after {len(result.attempts)} attempt(s)")
        console.print(result.forge_report)
        console.print(f"[dim]snapshot: {result.workdir}[/dim]")
        if apply:
            console.print(gate_mod.apply_snapshot(result))
        else:
            console.print("[dim]re-run with --apply to land it in the workspace[/dim]")
    else:
        console.print(f"[red]{result.error}[/red]")
        for attempt in result.attempts:
            console.print(f"  attempt {attempt.number}: tests "
                          f"{'passed' if attempt.tests_passed else 'failed'}, "
                          f"review {attempt.verdict} — {attempt.summary}")
        console.print(f"[dim]snapshot kept for inspection: {result.workdir}[/dim]")
        raise typer.Exit(1)


@app.command()
def worker(
    agent: str = typer.Argument(..., help="Which agent this worker embodies."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q"),
) -> None:
    """Run a worker daemon that consumes one agent's inbox from the bus."""
    try:
        roster.get(agent)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    if not asyncio.run(bus.ping()):
        console.print("[red]cannot reach Redis. Start it with `brew services start redis`.[/red]")
        raise typer.Exit(1)
    try:
        asyncio.run(worker_mod.serve(agent, _printer(verbose)))
    except KeyboardInterrupt:
        pass
    console.print(f"[dim]worker {agent} stopped[/dim]")


@app.command()
def listen(
    agent: str = typer.Option("manager", "--agent", "-a",
                              help="Agent used when no name is spoken."),
    input_device: int = typer.Option(None, "--input", help="Input device index."),
    output_device: int = typer.Option(None, "--output", help="Output device index."),
    silent: bool = typer.Option(False, "--silent", help="Print replies instead of speaking."),
    always: bool = typer.Option(False, "--always",
                                help="Listen even when no dashboard tab is open."),
    verbose: bool = typer.Option(False, "--verbose/--quiet", "-v/-q"),
) -> None:
    """Start the voice loop. Address an agent by name, for example "Bolt, ask Forge to..."."""
    from bolt.voice.session import VoiceConfig, run_session

    def report(kind: str, data: dict) -> None:
        if kind == "ready":
            console.print(f"[green]voice loop started[/green] · agents: {data['agents']}")
            console.print("[dim]the microphone is opened only while a dashboard tab "
                          "is open, and released as soon as it closes.[/dim]")
            console.print("[dim]say \"Bolt, ...\" or an agent name to address it. "
                          "ctrl-c to stop.[/dim]")
        elif kind == "problem":
            console.print(f"[red]{data['detail']}[/red]")
        elif kind == "listening":
            console.print("[dim]· listening[/dim]")
        elif kind == "heard":
            console.print(f"[dim]· captured {data['seconds']}s[/dim]")
        elif kind == "transcript":
            console.print(f"[cyan]you:[/cyan] {data['text']}")
        elif kind == "ignored":
            console.print(f"[dim]not addressed to bolt, ignored: {data['text']}[/dim]")
        elif kind == "routed":
            console.print(f"[yellow]→ {data['agent']}[/yellow] {data['text']}")
        elif kind == "reply":
            console.rule(f"[bold]{data['agent']}[/bold]")
            console.print(data["text"])
        elif kind == "barge_in":
            console.print(f"[dim]· stopped ({data.get('by', 'interrupted')})[/dim]")
        elif kind == "gated":
            console.print(f"[dim]· not listening — {data['reason']}[/dim]")
        elif kind == "mic_open":
            console.print(f"[green]· microphone opened[/green] "
                          f"[dim](noise floor {data['noise_floor']})[/dim]")
        elif kind == "mic_released":
            console.print("[yellow]· microphone released[/yellow]")
        elif kind == "ungated":
            console.print("[green]· listening again[/green]")

    config = VoiceConfig(default_agent=agent, input_device=input_device,
                         output_device=output_device, speak_replies=not silent,
                         require_tab=not always)
    try:
        asyncio.run(run_session(config, report, _printer(verbose)))
    except KeyboardInterrupt:
        console.print("[dim]stopped[/dim]")


@app.command()
def hush() -> None:
    """Stop whatever is speaking right now, wherever it is speaking from."""
    from bolt.voice import control
    control.request_stop()
    console.print("[yellow]told every speaker to stop[/yellow]")


@app.command()
def voice() -> None:
    """Show what the voice system is doing."""
    from bolt.voice import control
    state = control.snapshot()
    listening = ("[green]listening[/green]" if state["tab_open"] and not state["paused"]
                 else "[yellow]paused[/yellow]" if state["paused"]
                 else "[dim]idle — no dashboard tab open[/dim]")
    console.print(f"state:    {listening}")
    console.print(f"speaking: {'[yellow]yes[/yellow]' if state['speaking'] else 'no'}")
    console.print(f"[dim]{control.STATE_PATH}[/dim]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8090, "--port"),
) -> None:
    """Run the HTTP API and dashboard."""
    import uvicorn
    console.print(f"[green]dashboard:[/green] http://{host}:{port}")
    uvicorn.run("bolt.api.server:app", host=host, port=port, log_level="warning")


@app.command()
def agents() -> None:
    """List the agent roster and the model each one would use."""
    table = Table("agent", "configured", "resolves to", "why", "tools")
    for name in roster.names():
        spec = roster.get(name)
        choice = router.choose(name, "", spec.model, spec.requires, spec.prefers)
        colour = {"override": "yellow", "explicit": "white", "auto": "green"}[choice.source]
        table.add_row(name, spec.model, f"[{colour}]{choice.model}[/{colour}]",
                      choice.reason, ", ".join(spec.tools) or "-")
    console.print(table)
    console.print("[dim]pin one with: bolt pin <agent> <model>   ·   release: bolt pin <agent> auto[/dim]")


@app.command()
def checkpoints(limit: int = typer.Option(20, "--limit", "-n")) -> None:
    """File changes Bolt has written that can still be undone."""
    from bolt.core import proposals as proposals_mod

    rows = proposals_mod.checkpoints(limit)
    if not rows:
        console.print("[dim]nothing applied yet — no changes to undo[/dim]")
        return

    table = Table("id", "when", "file", "change", "by", "state")
    for row in rows:
        when = datetime.fromtimestamp(row["applied_at"]).strftime("%H:%M:%S") \
            if row["applied_at"] else "-"
        change = ("new file" if not row["existed"]
                  else f"+{row['added']} -{row['removed']}")
        state = "[yellow]edited since[/yellow]" if row["drifted"] else "[green]clean[/green]"
        table.add_row(row["id"], when, row["path"].split("/")[-1], change,
                      f"{row['agent'] or '?'} · {row['source']}", state)
    console.print(table)
    console.print("[dim]undo one with: bolt undo <id>   ·   most recent: bolt undo[/dim]")


@app.command()
def undo(
    proposal_id: str = typer.Argument(None, help="Which change; omit for the most recent."),
    force: bool = typer.Option(False, "--force",
                               help="Undo even if the file changed afterwards."),
) -> None:
    """Put a file back the way it was before Bolt changed it."""
    from bolt.core import proposals as proposals_mod

    if proposal_id is None:
        recent = proposals_mod.checkpoints(1)
        if not recent:
            console.print("[dim]nothing to undo[/dim]")
            return
        proposal_id = recent[0]["id"]
        console.print(f"[dim]undoing the most recent change: {recent[0]['path']}[/dim]")

    try:
        console.print(f"[green]{proposals_mod.undo(proposal_id, force)}[/green]")
    except KeyError:
        console.print(f"[red]no such change: {proposal_id}[/red]")
        raise typer.Exit(1)
    except proposals_mod.Conflict as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        console.print("[dim]re-run with --force to discard those edits[/dim]")
        raise typer.Exit(1)


@app.command()
def wallpapers(
    fetch: bool = typer.Option(False, "--fetch", help="Download any that are missing."),
) -> None:
    """Photographic chat backgrounds."""
    from bolt.core import backgrounds as bg

    if fetch:
        console.print(f"[dim]fetching up to {len(bg.CATALOGUE)} photos…[/dim]")
        counts = bg.fetch(on_progress=lambda photo, note: console.print(
            f"  [dim]{photo.category:<13}[/dim] {photo.name:<14} "
            f"{'[green]' + note + '[/green]' if 'KB' in note else '[red]' + note + '[/red]'}"))
        console.print(f"[green]downloaded {counts['downloaded']}[/green] · "
                      f"skipped {counts['skipped']} · failed {counts['failed']}")

    have = bg.available()
    if not have:
        console.print("[yellow]no photo backgrounds yet.[/yellow] "
                      "Run: bolt wallpapers --fetch")
        return

    table = Table("category", "name", "size", "source")
    for photo in have:
        table.add_row(photo["category"], photo["name"],
                      f"{photo['bytes'] // 1024}KB", photo["source"])
    console.print(table)
    console.print(f"[dim]{bg.LICENCE}[/dim]")
    console.print(f"[dim]stored in {bg.BACKGROUND_DIR}[/dim]")


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="Reindex from scratch."),
) -> None:
    """Build or inspect the cross-project code index."""
    from bolt.core import index as index_mod

    if rebuild:
        console.print("[dim]indexing every reachable project…[/dim]")
        stats = index_mod.build(
            on_progress=lambda s: console.print(
                f"[dim]  {s.files} files, {s.symbols} symbols…[/dim]"))
        console.print(f"[green]indexed[/green] {stats.files} files, {stats.symbols} "
                      f"symbols across {stats.projects} projects in {stats.seconds:.1f}s "
                      f"[dim]({stats.skipped} skipped)[/dim]")

    state = index_mod.status()
    if not state.get("built"):
        console.print("[yellow]no index yet.[/yellow] Run: bolt index --rebuild")
        return

    table = Table("files", "symbols", "projects", "size", "location")
    table.add_row(f"{state['files']:,}", f"{state['symbols']:,}",
                  str(state["projects"]), f"{state['size_mb']} MB", state["path"])
    console.print(table)

    top = index_mod.projects()[:8]
    if top:
        console.print("[dim]largest: " +
                      ", ".join(f"{p['project']} ({p['files']})" for p in top) + "[/dim]")


@app.command()
def find(
    query: str = typer.Argument(..., help="What to search for."),
    project: str = typer.Option(None, "--project", "-p"),
    language: str = typer.Option(None, "--lang", "-l"),
) -> None:
    """Search code across every indexed project."""
    from bolt.core import index as index_mod

    rows = index_mod.search(query, project, language, limit=15)
    if not rows:
        console.print(f"[dim]no matches for {query!r}[/dim]")
        return
    if "error" in rows[0]:
        console.print(f"[red]{rows[0]['error']}[/red]")
        raise typer.Exit(1)
    for row in rows:
        snippet = " ".join((row.get("snippet") or "").split())[:110]
        console.print(f"[yellow]{row['project']}[/yellow] "
                      f"[dim]{row['path']}[/dim]\n  {snippet}")


@app.command()
def models(
    probe: bool = typer.Option(False, "--probe", help="Run the full capability suite."),
    only: str = typer.Option(None, "--only", help="Probe one capability, e.g. vision."),
    model: str = typer.Option(None, "--model", help="Probe one model."),
) -> None:
    """Show the model catalogue and what each model has been observed to do."""
    from bolt.core import probes as probes_mod

    if probe:
        targets = [model] if model else models_mod.names()
        caps = [only] if only else None
        console.print(f"[dim]probing {len(targets)} model(s) with real requests…[/dim]")

        def show(r) -> None:
            mark = "[green]pass[/green]" if r.supported else "[red]fail[/red]"
            metric = f" [dim]{r.metric}[/dim]" if r.metric is not None else ""
            console.print(f"  {mark}  {r.model:<24} {r.capability:<22} {r.detail[:60]}{metric}")

        asyncio.run(probes_mod.run_all(targets, caps, on_result=show))
        console.rule()

    catalogue = models_mod.catalogue()
    seen = models_mod.capabilities_seen()
    in_use: dict[str, list[str]] = {}
    for agent in roster.names():
        spec = roster.get(agent)
        choice = router.choose(agent, "", spec.model, spec.requires, spec.prefers)
        in_use.setdefault(choice.model, []).append(agent)

    if not seen:
        console.print("[yellow]nothing probed yet.[/yellow] Run: bolt models --probe")

    # Only capabilities that differ between models earn a column; the rest go in
    # one "probes passed" count, otherwise the table is unreadably wide.
    split = models_mod.discriminating()
    table = Table("model", "tok/s", "$/Mtok", "ctx", "probes", *split, "used by")
    for info in sorted(catalogue.values(), key=lambda m: -(m.speed or 0)):
        cells = ["[green]yes[/green]" if info.probed.get(cap) else "[red]no[/red]"
                 if cap in info.probed else "[dim]·[/dim]" for cap in split]
        passed = len(info.passed)
        total = len(info.probed)
        table.add_row(info.name,
                      f"{info.speed:.0f}" if info.speed else "-",
                      f"{info.cost:.2f}", f"{info.context // 1000}k",
                      f"{passed}/{total}" if total else "[dim]none[/dim]",
                      *cells, ", ".join(in_use.get(info.name, [])) or "-")
    console.print(table)
    if seen:
        uniform = [c for c in seen if c not in split]
        if uniform:
            console.print(f"[dim]every model passed: {', '.join(uniform)} — these tell "
                          f"the router nothing.[/dim]")
        console.print(f"[bold]actually separates them:[/bold] "
                      f"{', '.join(split) if split else 'nothing but speed and cost'}")

    if pinned := router.overrides():
        console.print("[yellow]pinned:[/yellow] " +
                      ", ".join(f"{a}={m}" for a, m in sorted(pinned.items())))


@app.command()
def pin(
    agent: str = typer.Argument(..., help="Agent to pin."),
    model: str = typer.Argument(..., help="Model name, or 'auto' to release the pin."),
) -> None:
    """Pin an agent to a specific model, overriding its config and the router."""
    try:
        spec = roster.get(agent)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if model in ("auto", "clear", "none"):
        router.set_override(agent, None)
        choice = router.choose(agent, "", spec.model, spec.requires, spec.prefers)
        console.print(f"[green]released[/green] {agent} — now resolves to "
                      f"{choice.model} ({choice.reason})")
        return

    known = models_mod.names()
    if model not in known:
        console.print(f"[red]unknown model {model!r}[/red]. Known: {', '.join(known)}")
        raise typer.Exit(1)

    info = models_mod.get(model)
    gaps = router.missing_capabilities(info, spec.requires)
    if gaps:
        console.print(f"[red]{agent} requires {', '.join(gaps)} and {model} is not "
                      f"known to have {'it' if len(gaps) == 1 else 'them'}.[/red]")
        for gap in gaps:
            console.print(f"[dim]{gap}: {', '.join(models_mod.supporting(gap)) or 'no model'}[/dim]")
        raise typer.Exit(1)

    router.set_override(agent, model)
    console.print(f"[yellow]pinned[/yellow] {agent} → {model}")


@app.command()
def look(
    path: str = typer.Argument(..., help="Image or video in the workspace."),
    question: str = typer.Argument("Describe this in detail.", help="What to ask about it."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q"),
) -> None:
    """Ask the vision agent about an image or a video."""
    printer = _printer(verbose)
    prompt = f"Look at the file {path}. {question}"
    result = asyncio.run(loop.run_agent(roster.get("iris"), prompt,
                                        extra={"on_event": printer}, on_event=printer))
    console.rule("[bold]iris[/bold]")
    console.print(result.content if result.ok else f"[red]{result.error}[/red]")


@app.command()
def eval(
    suite: str = typer.Argument("baseline", help="Suite name from bolt/quality/evals/tasks."),
    model: str = typer.Option(None, "--model", help="Override every agent's model."),
    only: str = typer.Option(None, "--only", help="Run a single eval by id."),
    label: str = typer.Option(None, "--label", help="Tag this run in the scoreboard."),
    concurrency: int = typer.Option(2, "--concurrency", "-c"),
    board: bool = typer.Option(False, "--board", help="Only print the stored scoreboard."),
) -> None:
    """Replay a golden task set and score it."""
    from bolt.quality.evals import run_evals

    if board:
        _print_scoreboard(suite)
        return

    def show(outcome) -> None:
        mark = "[green]pass[/green]" if outcome.passed else "[red]fail[/red]"
        console.print(f"{mark}  {outcome.case.id:<28} {outcome.case.agent:<10} "
                      f"{outcome.score:.0%}  {outcome.tokens:>6} tok  {outcome.latency_ms:>6}ms")
        for failure in outcome.failures:
            console.print(f"      [red]·[/red] {failure}")

    try:
        outcomes = asyncio.run(run_evals.run_suite(
            suite, model=model, only=only, run_label=label,
            concurrency=concurrency, on_result=show))
    except (FileNotFoundError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    passed = sum(1 for o in outcomes if o.passed)
    console.rule()
    console.print(f"[bold]{passed}/{len(outcomes)} passed[/bold] · "
                  f"{sum(o.tokens for o in outcomes)} tokens")
    _print_scoreboard(suite)
    if passed < len(outcomes):
        raise typer.Exit(1)


def _print_scoreboard(suite: str) -> None:
    rows = store.eval_scoreboard(suite)
    if not rows:
        return
    table = Table("agent", "model", "label", "passed", "avg score", "tokens", "avg latency")
    for row in rows:
        table.add_row(row["agent"], row["model"], row["run_label"] or "-",
                      f"{row['passed']}/{row['n']}", f"{row['avg_score']:.0%}",
                      str(row["tokens"]), f"{row['avg_latency']:.0f}ms")
    console.print(table)


@app.command()
def tasks(limit: int = typer.Option(20, "--limit", "-n")) -> None:
    """Show recent tasks."""
    rows = store.connect().execute(
        "SELECT t.id, t.agent, t.status, t.depth, substr(t.prompt, 1, 60) AS prompt,"
        " COALESCE(SUM(r.prompt_tokens + r.completion_tokens), 0) AS tokens"
        " FROM tasks t LEFT JOIN runs r ON r.task_id = t.id"
        " GROUP BY t.id ORDER BY t.created_at DESC LIMIT ?", (limit,)).fetchall()
    table = Table("id", "agent", "status", "depth", "tokens", "prompt")
    for row in rows:
        table.add_row(row["id"], row["agent"], row["status"], str(row["depth"]),
                      str(row["tokens"]), row["prompt"])
    console.print(table)


@app.command()
def logs(task_id: str = typer.Argument(..., help="Task id from `bolt tasks`.")) -> None:
    """Replay the full transcript of a task."""
    rows = store.connect().execute(
        "SELECT * FROM messages WHERE task_id = ? ORDER BY seq", (task_id,)).fetchall()
    if not rows:
        console.print("[red]no such task[/red]")
        raise typer.Exit(1)
    for row in rows:
        label = row["role"] if not row["name"] else f"tool:{row['name']}"
        console.print(f"[bold cyan]{label}[/bold cyan]")
        console.print((row["content"] or "")[:4000])
        if row["tool_calls"]:
            console.print(f"[dim]{row['tool_calls'][:800]}[/dim]")
        console.rule(style="dim")


if __name__ == "__main__":
    app()
