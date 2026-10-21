"""SQLite persistence for tasks, runs, messages and artifacts."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(os.environ.get("BOLT_DB", Path.home() / "Projects" / "Bolt" / "bolt.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    parent_id     TEXT REFERENCES tasks(id),
    root_id       TEXT NOT NULL,
    agent         TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    result        TEXT,
    error         TEXT,
    depth         INTEGER NOT NULL DEFAULT 0,
    budget_tokens INTEGER NOT NULL DEFAULT 1000000,
    created_at    REAL NOT NULL,
    finished_at   REAL
);
CREATE INDEX IF NOT EXISTS tasks_root ON tasks(root_id);
CREATE INDEX IF NOT EXISTS tasks_parent ON tasks(parent_id);

CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL REFERENCES tasks(id),
    agent             TEXT NOT NULL,
    model             TEXT NOT NULL,
    iteration         INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    attempts          INTEGER NOT NULL DEFAULT 1,
    ok                INTEGER NOT NULL DEFAULT 1,
    error             TEXT,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_task ON runs(task_id);

CREATE TABLE IF NOT EXISTS messages (
    id         TEXT PRIMARY KEY,
    task_id    TEXT NOT NULL REFERENCES tasks(id),
    seq        INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT,
    tool_calls TEXT,
    name       TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_task ON messages(task_id, seq);

CREATE TABLE IF NOT EXISTS artifacts (
    id            TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL REFERENCES tasks(id),
    kind          TEXT NOT NULL,
    path          TEXT,
    content       TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending',
    review_notes  TEXT,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_task ON artifacts(task_id);

CREATE TABLE IF NOT EXISTS evals (
    id          TEXT PRIMARY KEY,
    suite       TEXT NOT NULL,
    eval_id     TEXT NOT NULL,
    agent       TEXT NOT NULL,
    model       TEXT NOT NULL,
    task_id     TEXT REFERENCES tasks(id),
    passed      INTEGER NOT NULL,
    score       REAL NOT NULL DEFAULT 0,
    failures    TEXT,
    tokens      INTEGER NOT NULL DEFAULT 0,
    latency_ms  INTEGER NOT NULL DEFAULT 0,
    run_label   TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS evals_suite ON evals(suite, created_at);

CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    agent      TEXT NOT NULL DEFAULT 'manager',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- One user turn and its reply. Kept separately from `messages`, which is the
-- raw model transcript including tool calls; this is what a person reads back.
CREATE TABLE IF NOT EXISTS turns (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    agent           TEXT,
    task_id         TEXT,
    tokens          INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_conversation ON turns(conversation_id, seq);

-- File changes an agent wants to make, held until a person approves them.
CREATE TABLE IF NOT EXISTS proposals (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT,
    turn_id         TEXT,
    agent           TEXT,
    path            TEXT NOT NULL,
    old_content     TEXT,
    new_content     TEXT NOT NULL,
    existed         INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    applied_hash    TEXT,          -- what we wrote, to spot later external edits
    undone_at       REAL,
    source          TEXT,          -- 'chat' | 'gate'
    created_at      REAL NOT NULL,
    decided_at      REAL
);
CREATE INDEX IF NOT EXISTS proposals_conversation ON proposals(conversation_id, status);

CREATE TABLE IF NOT EXISTS feedback (
    turn_id    TEXT PRIMARY KEY REFERENCES turns(id),
    agent      TEXT,
    rating     TEXT NOT NULL,          -- 'up' | 'down'
    note       TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_agent ON feedback(agent, rating);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- Empirically observed model capabilities, which override anything declared
-- in models.yaml.
CREATE TABLE IF NOT EXISTS model_probes (
    model      TEXT NOT NULL,
    capability TEXT NOT NULL,
    supported  INTEGER NOT NULL,
    detail     TEXT,
    metric     REAL,          -- probe-specific number, e.g. tokens/sec or latency
    checked_at REAL NOT NULL,
    PRIMARY KEY (model, capability)
);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# One connection per thread, not one shared connection.
#
# A single sqlite3 connection is not safe to use concurrently even with
# check_same_thread=False: two threads interleaving statements on it produce
# "bad parameter or other API misuse" and spurious FOREIGN KEY failures, because
# each tx() commit lands in the middle of the other's transaction. Agents run
# concurrently whenever the manager delegates, so this is the normal case.
_local = threading.local()
_schema_ready = threading.Event()
_schema_lock = threading.Lock()


def connect() -> sqlite3.Connection:
    """This thread's connection, with WAL enabled and the schema applied."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # WAL lets readers and one writer coexist; this waits rather than failing
    # instantly when two writers collide.
    conn.execute("PRAGMA busy_timeout=30000")

    if not _schema_ready.is_set():
        with _schema_lock:
            if not _schema_ready.is_set():
                conn.executescript(SCHEMA)
                _migrate(conn)
                conn.commit()
                _schema_ready.set()

    _local.conn = conn
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that older databases predate. CREATE TABLE IF NOT EXISTS will
    not add them to a table that already exists."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(model_probes)")}
    if "metric" not in existing:
        conn.execute("ALTER TABLE model_probes ADD COLUMN metric REAL")

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(proposals)")}
    if columns:
        for name, ddl in (("applied_hash", "TEXT"), ("undone_at", "REAL"),
                          ("source", "TEXT")):
            if name not in columns:
                conn.execute(f"ALTER TABLE proposals ADD COLUMN {name} {ddl}")


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def create_task(
    agent: str,
    prompt: str,
    parent_id: str | None = None,
    root_id: str | None = None,
    depth: int = 0,
    budget_tokens: int = 1_000_000,
) -> str:
    task_id = new_id("t")
    with tx() as conn:
        conn.execute(
            "INSERT INTO tasks (id, parent_id, root_id, agent, prompt, depth, budget_tokens, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task_id, parent_id, root_id or task_id, agent, prompt, depth, budget_tokens, time.time()),
        )
    return task_id


def set_status(task_id: str, status: str, result: str | None = None, error: str | None = None) -> None:
    finished = time.time() if status in ("done", "failed", "rejected") else None
    with tx() as conn:
        conn.execute(
            "UPDATE tasks SET status = ?, result = COALESCE(?, result),"
            " error = COALESCE(?, error), finished_at = COALESCE(?, finished_at) WHERE id = ?",
            (status, result, error, finished, task_id),
        )


def get_task(task_id: str) -> dict[str, Any] | None:
    row = connect().execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


def children(task_id: str) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT * FROM tasks WHERE parent_id = ? ORDER BY created_at", (task_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def record_run(
    task_id: str,
    agent: str,
    model: str,
    iteration: int,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    attempts: int = 1,
    ok: bool = True,
    error: str | None = None,
) -> str:
    run_id = new_id("r")
    with tx() as conn:
        conn.execute(
            "INSERT INTO runs (id, task_id, agent, model, iteration, prompt_tokens,"
            " completion_tokens, latency_ms, attempts, ok, error, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, task_id, agent, model, iteration, prompt_tokens, completion_tokens,
             latency_ms, attempts, int(ok), error, time.time()),
        )
    return run_id


def tokens_used(task_id: str) -> int:
    row = connect().execute(
        "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS n FROM runs WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    return int(row["n"])


def tokens_used_tree(root_id: str) -> int:
    row = connect().execute(
        "SELECT COALESCE(SUM(r.prompt_tokens + r.completion_tokens), 0) AS n"
        " FROM runs r JOIN tasks t ON t.id = r.task_id WHERE t.root_id = ?",
        (root_id,),
    ).fetchone()
    return int(row["n"])


def add_message(task_id: str, seq: int, role: str, content: str | None,
                tool_calls: Any = None, name: str | None = None) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO messages (id, task_id, seq, role, content, tool_calls, name, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (new_id("m"), task_id, seq, role, content,
             json.dumps(tool_calls) if tool_calls else None, name, time.time()),
        )


def add_artifact(task_id: str, kind: str, content: str | None = None, path: str | None = None) -> str:
    art_id = new_id("a")
    with tx() as conn:
        conn.execute(
            "INSERT INTO artifacts (id, task_id, kind, path, content, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (art_id, task_id, kind, path, content, time.time()),
        )
    return art_id


def set_review(artifact_id: str, status: str, notes: str | None = None) -> None:
    with tx() as conn:
        conn.execute(
            "UPDATE artifacts SET review_status = ?, review_notes = ? WHERE id = ?",
            (status, notes, artifact_id),
        )


def record_eval(suite: str, eval_id: str, agent: str, model: str, passed: bool,
                score: float, failures: list[str], tokens: int, latency_ms: int,
                task_id: str | None = None, run_label: str | None = None) -> str:
    row_id = new_id("e")
    with tx() as conn:
        conn.execute(
            "INSERT INTO evals (id, suite, eval_id, agent, model, task_id, passed, score,"
            " failures, tokens, latency_ms, run_label, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row_id, suite, eval_id, agent, model, task_id, int(passed), score,
             json.dumps(failures), tokens, latency_ms, run_label, time.time()),
        )
    return row_id


def eval_scoreboard(suite: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Pass rate and cost per (agent, model), most recent runs first."""
    clause = "WHERE suite = ?" if suite else ""
    params: tuple[Any, ...] = (suite, limit) if suite else (limit,)
    rows = connect().execute(
        f"SELECT agent, model, run_label, COUNT(*) AS n, SUM(passed) AS passed,"
        f" AVG(score) AS avg_score, SUM(tokens) AS tokens, AVG(latency_ms) AS avg_latency"
        f" FROM (SELECT * FROM evals {clause} ORDER BY created_at DESC LIMIT ?)"
        f" GROUP BY agent, model, run_label ORDER BY passed * 1.0 / COUNT(*) DESC",
        params).fetchall()
    return [dict(row) for row in rows]


def set_setting(key: str, value: str) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " updated_at = excluded.updated_at",
            (key, value, time.time()),
        )


def get_setting(key: str, default: str | None = None) -> str | None:
    row = connect().execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def delete_setting(key: str) -> None:
    with tx() as conn:
        conn.execute("DELETE FROM settings WHERE key = ?", (key,))


def all_settings(prefix: str = "") -> dict[str, str]:
    rows = connect().execute(
        "SELECT key, value FROM settings WHERE key LIKE ?", (f"{prefix}%",)).fetchall()
    return {row["key"]: row["value"] for row in rows}


def record_probe(model: str, capability: str, supported: bool, detail: str = "",
                 metric: float | None = None) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO model_probes (model, capability, supported, detail, metric, checked_at)"
            " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(model, capability) DO UPDATE SET"
            " supported = excluded.supported, detail = excluded.detail,"
            " metric = excluded.metric, checked_at = excluded.checked_at",
            (model, capability, int(supported), detail[:500], metric, time.time()),
        )


def probes() -> dict[str, dict[str, Any]]:
    """{model: {capability: {"supported": bool, "detail": str, "checked_at": float}}}"""
    out: dict[str, dict[str, Any]] = {}
    for row in connect().execute("SELECT * FROM model_probes").fetchall():
        out.setdefault(row["model"], {})[row["capability"]] = {
            "supported": bool(row["supported"]),
            "detail": row["detail"],
            "metric": row["metric"],
            "checked_at": row["checked_at"],
        }
    return out


# --- conversations ------------------------------------------------------------

def create_conversation(agent: str = "manager", title: str | None = None) -> str:
    conversation_id = new_id("c")
    now = time.time()
    with tx() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, agent, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)", (conversation_id, title, agent, now, now))
    return conversation_id


def add_turn(conversation_id: str, role: str, content: str, agent: str | None = None,
             task_id: str | None = None, tokens: int = 0) -> str:
    turn_id = new_id("u")
    now = time.time()
    with tx() as conn:
        seq = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 AS n FROM turns WHERE conversation_id = ?",
            (conversation_id,)).fetchone()["n"]
        conn.execute(
            "INSERT INTO turns (id, conversation_id, seq, role, content, agent, task_id,"
            " tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (turn_id, conversation_id, seq, role, content, agent, task_id, tokens, now))
        # The first thing said becomes the conversation's name in the sidebar.
        conn.execute(
            "UPDATE conversations SET updated_at = ?,"
            " title = COALESCE(title, CASE WHEN ? = 'user' THEN substr(?, 1, 60) END)"
            " WHERE id = ?", (now, role, content, conversation_id))
    return turn_id


def conversation_turns(conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT * FROM turns WHERE conversation_id = ? ORDER BY seq LIMIT ?",
        (conversation_id, limit)).fetchall()
    return [dict(r) for r in rows]


def conversation_history(conversation_id: str, max_turns: int = 20) -> list[dict[str, str]]:
    """Recent turns as chat messages, for continuing a conversation."""
    rows = connect().execute(
        "SELECT role, content FROM turns WHERE conversation_id = ?"
        " ORDER BY seq DESC LIMIT ?", (conversation_id, max_turns)).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def conversations(limit: int = 50) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT c.*, COUNT(t.id) AS turns FROM conversations c"
        " LEFT JOIN turns t ON t.conversation_id = c.id"
        " GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    row = connect().execute(
        "SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return dict(row) if row else None


def delete_conversation(conversation_id: str) -> None:
    with tx() as conn:
        # feedback references turns, so it has to go first or the foreign key
        # constraint refuses the delete.
        conn.execute(
            "DELETE FROM feedback WHERE turn_id IN"
            " (SELECT id FROM turns WHERE conversation_id = ?)", (conversation_id,))
        conn.execute("DELETE FROM turns WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def turns_from(conversation_id: str, seq: int) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT * FROM turns WHERE conversation_id = ? AND seq >= ? ORDER BY seq",
        (conversation_id, seq)).fetchall()
    return [dict(r) for r in rows]


def delete_turns_from(conversation_id: str, seq: int) -> int:
    """Drop a turn and everything after it, so a conversation can be rewound and
    taken in a different direction."""
    with tx() as conn:
        conn.execute(
            "DELETE FROM feedback WHERE turn_id IN"
            " (SELECT id FROM turns WHERE conversation_id = ? AND seq >= ?)",
            (conversation_id, seq))
        cursor = conn.execute(
            "DELETE FROM turns WHERE conversation_id = ? AND seq >= ?",
            (conversation_id, seq))
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                     (time.time(), conversation_id))
        return cursor.rowcount


def last_user_turn(conversation_id: str) -> dict[str, Any] | None:
    row = connect().execute(
        "SELECT * FROM turns WHERE conversation_id = ? AND role = 'user'"
        " ORDER BY seq DESC LIMIT 1", (conversation_id,)).fetchone()
    return dict(row) if row else None


# --- feedback -----------------------------------------------------------------

def set_feedback(turn_id: str, rating: str | None, note: str | None = None,
                 agent: str | None = None) -> None:
    """Record thumbs up/down on a reply. `rating=None` clears it."""
    with tx() as conn:
        if rating is None:
            conn.execute("DELETE FROM feedback WHERE turn_id = ?", (turn_id,))
            return
        conn.execute(
            "INSERT INTO feedback (turn_id, agent, rating, note, created_at)"
            " VALUES (?, ?, ?, ?, ?) ON CONFLICT(turn_id) DO UPDATE SET"
            " rating = excluded.rating, note = COALESCE(excluded.note, feedback.note),"
            " created_at = excluded.created_at",
            (turn_id, agent, rating, note, time.time()))


def feedback_for(conversation_id: str) -> dict[str, dict[str, Any]]:
    rows = connect().execute(
        "SELECT f.* FROM feedback f JOIN turns t ON t.id = f.turn_id"
        " WHERE t.conversation_id = ?", (conversation_id,)).fetchall()
    return {r["turn_id"]: dict(r) for r in rows}


def feedback_summary() -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT agent, rating, COUNT(*) AS n FROM feedback"
        " GROUP BY agent, rating ORDER BY agent").fetchall()
    return [dict(r) for r in rows]


def agent_lessons(agent: str, limit: int = 5) -> list[str]:
    """Notes left on replies that were marked down, newest first.

    These are folded into the agent's prompt on later turns. It is prompt-level
    correction, not training — nothing about the model itself changes.
    """
    rows = connect().execute(
        "SELECT note FROM feedback WHERE agent = ? AND rating = 'down'"
        " AND note IS NOT NULL AND TRIM(note) != ''"
        " ORDER BY created_at DESC LIMIT ?", (agent, limit)).fetchall()
    return [r["note"] for r in rows]


def get_turn(turn_id: str) -> dict[str, Any] | None:
    row = connect().execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
    return dict(row) if row else None


def clear_lessons(agent: str) -> int:
    """Forget the corrections stored for an agent.

    Lessons are injected into every later prompt for that agent, so a mistaken or
    stale one needs a way out.
    """
    with tx() as conn:
        cursor = conn.execute(
            "UPDATE feedback SET note = NULL WHERE agent = ? AND rating = 'down'", (agent,))
        return cursor.rowcount


def preceding_user_turn(conversation_id: str, seq: int) -> dict[str, Any] | None:
    """The user message a given reply was answering."""
    row = connect().execute(
        "SELECT * FROM turns WHERE conversation_id = ? AND role = 'user' AND seq < ?"
        " ORDER BY seq DESC LIMIT 1", (conversation_id, seq)).fetchone()
    return dict(row) if row else None


def replies_to(conversation_id: str, user_seq: int) -> list[dict[str, Any]]:
    """Every assistant turn answering one user message.

    Retries append rather than replace, so all assistant turns between this user
    message and the next one are attempts at the same question.
    """
    next_user = connect().execute(
        "SELECT MIN(seq) AS n FROM turns WHERE conversation_id = ? AND role = 'user'"
        " AND seq > ?", (conversation_id, user_seq)).fetchone()["n"]
    limit = next_user if next_user is not None else 10 ** 9
    rows = connect().execute(
        "SELECT * FROM turns WHERE conversation_id = ? AND role = 'assistant'"
        " AND seq > ? AND seq < ? ORDER BY seq",
        (conversation_id, user_seq, limit)).fetchall()
    return [dict(r) for r in rows]


def rejected_attempts(conversation_id: str, user_seq: int) -> list[dict[str, Any]]:
    """Replies to this question that were marked down, with any notes left."""
    attempts = replies_to(conversation_id, user_seq)
    if not attempts:
        return []
    marks = feedback_for(conversation_id)
    return [{**a, "note": marks[a["id"]].get("note")}
            for a in attempts if marks.get(a["id"], {}).get("rating") == "down"]


# --- proposed file changes ----------------------------------------------------

def add_proposal(path: str, new_content: str, old_content: str | None,
                 existed: bool, conversation_id: str | None = None,
                 agent: str | None = None, source: str = "chat",
                 status: str = "pending") -> str:
    proposal_id = new_id("p")
    with tx() as conn:
        conn.execute(
            "INSERT INTO proposals (id, conversation_id, turn_id, agent, path,"
            " old_content, new_content, existed, source, status, created_at)"
            " VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
            (proposal_id, conversation_id, agent, path, old_content, new_content,
             int(existed), source, status, time.time()))
    return proposal_id


def mark_applied(proposal_id: str, applied_hash: str) -> None:
    with tx() as conn:
        conn.execute(
            "UPDATE proposals SET status = 'applied', applied_hash = ?, decided_at = ?,"
            " undone_at = NULL WHERE id = ?", (applied_hash, time.time(), proposal_id))


def mark_undone(proposal_id: str) -> None:
    with tx() as conn:
        conn.execute("UPDATE proposals SET status = 'undone', undone_at = ? WHERE id = ?",
                     (time.time(), proposal_id))


def applied_proposals(limit: int = 50) -> list[dict[str, Any]]:
    """Changes written to disk that could still be undone, newest first."""
    rows = connect().execute(
        "SELECT * FROM proposals WHERE status = 'applied'"
        " ORDER BY decided_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def proposals_in_turn(turn_id: str) -> list[dict[str, Any]]:
    rows = connect().execute(
        "SELECT * FROM proposals WHERE turn_id = ? ORDER BY created_at",
        (turn_id,)).fetchall()
    return [dict(r) for r in rows]


def attach_proposals(ids: list[str], turn_id: str) -> None:
    if not ids:
        return
    with tx() as conn:
        conn.executemany("UPDATE proposals SET turn_id = ? WHERE id = ?",
                         [(turn_id, i) for i in ids])


def get_proposal(proposal_id: str) -> dict[str, Any] | None:
    row = connect().execute("SELECT * FROM proposals WHERE id = ?",
                            (proposal_id,)).fetchone()
    return dict(row) if row else None


def set_proposal_status(proposal_id: str, status: str) -> None:
    with tx() as conn:
        conn.execute("UPDATE proposals SET status = ?, decided_at = ? WHERE id = ?",
                     (status, time.time(), proposal_id))


def proposals_for(conversation_id: str, status: str | None = None) -> list[dict[str, Any]]:
    query = "SELECT * FROM proposals WHERE conversation_id = ?"
    params: list[Any] = [conversation_id]
    if status:
        query += " AND status = ?"
        params.append(status)
    rows = connect().execute(query + " ORDER BY created_at", params).fetchall()
    return [dict(r) for r in rows]


def reconcile_running(older_than_seconds: float = 60.0) -> int:
    """Close out tasks left 'running' by a process that died.

    Nothing else will ever finish them: the asyncio task that owned them is gone
    with the process, so they would sit in the dashboard as permanently busy.
    """
    cutoff = time.time() - older_than_seconds
    with tx() as conn:
        cursor = conn.execute(
            "UPDATE tasks SET status = 'failed', finished_at = ?,"
            " error = COALESCE(error, 'interrupted: the process restarted while this ran')"
            " WHERE status = 'running' AND created_at < ?", (time.time(), cutoff))
        return cursor.rowcount


def usage_summary(days: int = 30) -> dict[str, Any]:
    """What Bolt has spent, from our own accounting.

    Ollama exposes no credit or quota endpoint — its cloud requests are signed
    with the daemon's key, and a session limit only shows up as a 429 at call
    time. So this is what we billed ourselves: every run we recorded, plus the
    rate-limit refusals, which are the only signal the account has run out.
    """
    since = time.time() - days * 86400
    conn = connect()

    def rows(sql: str) -> list[dict[str, Any]]:
        return [dict(r) for r in conn.execute(sql, (since,)).fetchall()]

    totals = dict(conn.execute(
        "SELECT COUNT(*) AS calls,"
        " COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,"
        " COALESCE(SUM(completion_tokens), 0) AS completion_tokens,"
        " COALESCE(SUM(attempts), 0) AS attempts,"
        " COALESCE(SUM(1 - ok), 0) AS failures,"
        " COALESCE(AVG(latency_ms), 0) AS avg_latency_ms"
        " FROM runs WHERE created_at >= ?", (since,)).fetchone())

    limited = conn.execute(
        "SELECT COUNT(*) AS n, MAX(created_at) AS last_at FROM runs"
        " WHERE created_at >= ? AND ok = 0 AND ("
        " error LIKE '%usage limit%' OR error LIKE '%429%'"
        " OR error LIKE '%rate limit%')", (since,)).fetchone()

    return {
        "days": days,
        "since": since,
        "totals": totals,
        "by_model": rows(
            "SELECT model, COUNT(*) AS calls,"
            " SUM(prompt_tokens) AS prompt_tokens,"
            " SUM(completion_tokens) AS completion_tokens,"
            " AVG(latency_ms) AS avg_latency_ms"
            " FROM runs WHERE created_at >= ? GROUP BY model"
            " ORDER BY prompt_tokens + completion_tokens DESC"),
        "by_agent": rows(
            "SELECT agent, COUNT(*) AS calls,"
            " SUM(prompt_tokens + completion_tokens) AS tokens"
            " FROM runs WHERE created_at >= ? GROUP BY agent ORDER BY tokens DESC"),
        "by_day": rows(
            "SELECT date(created_at, 'unixepoch', 'localtime') AS day,"
            " COUNT(*) AS calls,"
            " SUM(prompt_tokens + completion_tokens) AS tokens"
            " FROM runs WHERE created_at >= ? GROUP BY day ORDER BY day"),
        "rate_limited": {"count": limited["n"], "last_at": limited["last_at"]},
    }
