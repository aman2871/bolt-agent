"""A searchable index over every project an agent can reach.

Grep does not scale across 95 repositories: an agent either misses the file or
burns thousands of tokens looking for it. This builds a full-text index plus a
symbol table, so "how did we do refunds elsewhere" is one query rather than a
directory crawl.

SQLite FTS5 rather than embeddings, deliberately: it needs no model, indexes
24k files in under a minute, and ranks exact identifiers — which is what code
search is mostly made of. Semantic search can sit on top later.
"""

from __future__ import annotations

import ast
import os
import re
import sqlite3
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from bolt.core import workspace

INDEX_PATH = Path(os.environ.get(
    "BOLT_INDEX_DB", Path(os.environ.get("BOLT_STATE_DIR", Path.home() / ".bolt"))
    / "code-index.db"))

SOURCE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rb", ".rs", ".php",
    ".sql", ".sh", ".yaml", ".yml", ".tf", ".toml", ".ini", ".cfg",
    ".html", ".css", ".scss", ".md", ".proto", ".dockerfile",
}
SKIP_DIRS = {
    "node_modules", ".git", ".venv", "venv", "env", "__pycache__", "dist", "build",
    ".next", ".nuxt", "vendor", "target", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "migrations", "static", "staticfiles", "site-packages", ".terraform", "coverage",
}
MAX_FILE_BYTES = 250_000
MAX_LINE_LENGTH = 2_000        # a proxy for minified or generated files

SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS files USING fts5(
    project, path, language, content,
    tokenize = "unicode61 tokenchars '_'"
);

CREATE TABLE IF NOT EXISTS symbols (
    name       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    project    TEXT NOT NULL,
    path       TEXT NOT NULL,
    line       INTEGER NOT NULL,
    signature  TEXT
);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS symbols_project ON symbols(project);

CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Enough to find definitions in the languages actually present here. A real
# parser per language would be better; this is accurate for Python and good
# enough elsewhere to point an agent at the right file.
_PATTERNS = {
    ".js": [(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
            (r"^\s*(?:export\s+)?class\s+(\w+)", "class"),
            (r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(", "function")],
    ".go": [(r"^func\s+(?:\([^)]*\)\s*)?(\w+)", "function"),
            (r"^type\s+(\w+)\s+struct", "struct")],
    ".java": [(r"^\s*(?:public|private|protected).*?\bclass\s+(\w+)", "class"),
              (r"^\s*(?:public|private|protected).*?\s(\w+)\s*\([^)]*\)\s*\{", "method")],
    ".sql": [(r"(?i)^\s*create\s+(?:or\s+replace\s+)?table\s+(?:if\s+not\s+exists\s+)?[`\"]?(\w+)", "table"),
             (r"(?i)^\s*create\s+(?:or\s+replace\s+)?function\s+[`\"]?(\w+)", "function")],
}
_PATTERNS[".jsx"] = _PATTERNS[".ts"] = _PATTERNS[".tsx"] = _PATTERNS[".js"]


@dataclass
class IndexStats:
    files: int = 0
    symbols: int = 0
    skipped: int = 0
    seconds: float = 0.0
    projects: int = 0


def connect() -> sqlite3.Connection:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(INDEX_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _walk(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in SOURCE_SUFFIXES:
                yield path


def _readable(path: Path) -> str | None:
    """File text, or None if it is too big, binary, or machine-generated."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError):
        return None
    if any(len(line) > MAX_LINE_LENGTH for line in text.splitlines()[:50]):
        return None            # minified bundle or a data blob
    return text


def extract_symbols(path: Path, text: str) -> list[tuple[str, str, int, str]]:
    """(name, kind, line, signature) for definitions in one file."""
    suffix = path.suffix.lower()
    out: list[tuple[str, str, int, str]] = []

    if suffix == ".py":
        try:
            # Old files with invalid escapes emit SyntaxWarning on every parse,
            # which floods the output when indexing thousands of them.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            return out
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = ", ".join(a.arg for a in node.args.args)
                out.append((node.name, "function", node.lineno, f"def {node.name}({args})"))
            elif isinstance(node, ast.ClassDef):
                out.append((node.name, "class", node.lineno, f"class {node.name}"))
        return out

    for pattern, kind in _PATTERNS.get(suffix, []):
        for match in re.finditer(pattern, text, re.MULTILINE):
            line = text.count("\n", 0, match.start()) + 1
            out.append((match.group(1), kind, line, match.group(0).strip()[:120]))
    return out


def build(roots: list[Path] | None = None, on_progress=None) -> IndexStats:
    """Index every source file under the given roots, replacing what was there."""
    roots = roots or list(workspace.roots())
    started = time.monotonic()
    stats = IndexStats()
    conn = connect()

    try:
        conn.execute("DELETE FROM files")
        conn.execute("DELETE FROM symbols")

        for root in roots:
            if not root.is_dir():
                continue
            for path in _walk(root):
                if workspace.is_secret(path):
                    stats.skipped += 1
                    continue
                text = _readable(path)
                if text is None:
                    stats.skipped += 1
                    continue

                relative = path.relative_to(root)
                # The first directory under a root is the project it belongs to.
                project = relative.parts[0] if len(relative.parts) > 1 else root.name
                conn.execute(
                    "INSERT INTO files (project, path, language, content) VALUES (?, ?, ?, ?)",
                    (project, str(path), path.suffix.lstrip(".").lower(), text))

                symbols = extract_symbols(path, text)
                if symbols:
                    conn.executemany(
                        "INSERT INTO symbols (name, kind, project, path, line, signature)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        [(n, k, project, str(path), line, sig) for n, k, line, sig in symbols])
                    stats.symbols += len(symbols)

                stats.files += 1
                if on_progress and stats.files % 500 == 0:
                    on_progress(stats)

        stats.projects = conn.execute(
            "SELECT COUNT(DISTINCT project) AS n FROM files").fetchone()["n"]
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES ('built_at', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(time.time()),))
        conn.commit()
    finally:
        conn.close()

    stats.seconds = time.monotonic() - started
    return stats


def _terms(query: str) -> list[str]:
    """Quoted FTS5 terms. Punctuation is syntax in FTS5, so everything is quoted."""
    out = []
    for token in re.findall(r'"[^"]*"|\S+', query):
        token = token.strip('"').strip()
        if token:
            out.append('"' + token.replace('"', "") + '"')
    return out


def _run(conn: sqlite3.Connection, match: str, project: str | None,
         language: str | None, limit: int) -> list[dict[str, Any]]:
    sql = ("SELECT project, path, language,"
           " snippet(files, 3, '>>', '<<', ' … ', 18) AS snippet, rank"
           " FROM files WHERE files MATCH ?")
    params: list[Any] = [match]
    if project:
        sql += " AND project = ?"
        params.append(project)
    if language:
        sql += " AND language = ?"
        params.append(language)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def search(query: str, project: str | None = None, language: str | None = None,
           limit: int = 12) -> list[dict[str, Any]]:
    """Full-text search across every indexed project.

    FTS5 requires every term by default, which returns nothing for a natural
    phrase like "jwt decode token middleware" — no single file holds all four.
    So: all terms first for precision, then any term, ranked, for recall.
    """
    terms = _terms(query)
    if not terms:
        return []
    conn = connect()
    try:
        results = _run(conn, " ".join(terms), project, language, limit)
        if not results and len(terms) > 1:
            results = _run(conn, " OR ".join(terms), project, language, limit)
        return results
    except sqlite3.OperationalError as exc:
        return [{"error": f"search failed: {exc}"}]
    finally:
        conn.close()


def find_symbol(name: str, limit: int = 20) -> list[dict[str, Any]]:
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT * FROM symbols WHERE name = ? ORDER BY project, path LIMIT ?",
            (name, limit)).fetchall()
        if not rows:
            rows = conn.execute(
                "SELECT * FROM symbols WHERE name LIKE ? ORDER BY project, path LIMIT ?",
                (f"%{name}%", limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def projects() -> list[dict[str, Any]]:
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT project, COUNT(*) AS files,"
            " GROUP_CONCAT(DISTINCT language) AS languages"
            " FROM files GROUP BY project ORDER BY files DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def status() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        return {"built": False, "path": str(INDEX_PATH)}
    conn = connect()
    try:
        built_at = conn.execute(
            "SELECT value FROM index_meta WHERE key = 'built_at'").fetchone()
        return {
            "built": built_at is not None,
            "built_at": float(built_at["value"]) if built_at else None,
            "files": conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"],
            "symbols": conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"],
            "projects": conn.execute(
                "SELECT COUNT(DISTINCT project) AS n FROM files").fetchone()["n"],
            "size_mb": round(INDEX_PATH.stat().st_size / 1_000_000, 1),
            "path": str(INDEX_PATH),
        }
    finally:
        conn.close()
