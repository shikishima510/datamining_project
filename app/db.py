import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import CONFIG


def ensure_db_dir(path: str) -> None:
    db_dir = os.path.dirname(path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


def connect_db() -> sqlite3.Connection:
    ensure_db_dir(CONFIG.db_path)
    conn = sqlite3.connect(CONFIG.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=3000")
    return conn


@contextmanager
def db_cursor(commit: bool = False):
    conn = connect_db()
    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    finally:
        cur.close()
        conn.close()


def init_db() -> None:
    with db_cursor(commit=True) as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS papers(
              id INTEGER PRIMARY KEY,
              arxiv_id TEXT UNIQUE,
              title TEXT,
              abstract TEXT,
              authors_json TEXT,
              primary_category TEXT,
              categories_json TEXT,
              cross_categories_json TEXT,
              published_at TEXT,
              updated_at TEXT,
              citations_count INTEGER,
              citations_updated_at TEXT,
              pagerank_score REAL,
              pagerank_updated_at TEXT,
              doi TEXT,
              journal_ref TEXT,
              comment TEXT,
              abs_url TEXT,
              pdf_url TEXT,
              version_count INTEGER,
              cross_list_count INTEGER,
              ingested_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
              title,
              abstract,
              content='papers',
              content_rowid='id',
              tokenize='unicode61'
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_state(
              key TEXT PRIMARY KEY,
              value TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_events(
              id INTEGER PRIMARY KEY,
              user_id TEXT,
              arxiv_id TEXT,
              action TEXT,
              ts TEXT,
              context_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profile(
              user_id TEXT PRIMARY KEY,
              profile_json TEXT,
              updated_at TEXT,
              version INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_files(
              id INTEGER PRIMARY KEY,
              user_id TEXT,
              path TEXT,
              file_mtime REAL,
              ingested_at TEXT,
              UNIQUE(user_id, path)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_edges(
              id INTEGER PRIMARY KEY,
              from_arxiv_id TEXT,
              to_arxiv_id TEXT,
              source TEXT,
              ingested_at TEXT,
              UNIQUE(from_arxiv_id, to_arxiv_id, source)
            )
            """
        )
        _ensure_column(cur, "papers", "citations_count", "INTEGER")
        _ensure_column(cur, "papers", "citations_updated_at", "TEXT")
        _ensure_column(cur, "papers", "pagerank_score", "REAL")
        _ensure_column(cur, "papers", "pagerank_updated_at", "TEXT")


def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, col_type: str) -> None:
    cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
    names = {row["name"] for row in cols}
    if column not in names:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_sync_state(cur: sqlite3.Cursor, key: str, value: str) -> None:
    cur.execute(
        "INSERT INTO sync_state(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_sync_state(cur: sqlite3.Cursor, key: str) -> str | None:
    row = cur.execute("SELECT value FROM sync_state WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def load_profile(cur: sqlite3.Cursor, user_id: str) -> dict:
    row = cur.execute(
        "SELECT profile_json FROM user_profile WHERE user_id=?", (user_id,)
    ).fetchone()
    if not row:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


def save_profile(cur: sqlite3.Cursor, user_id: str, profile: dict, version: int) -> None:
    cur.execute(
        "INSERT INTO user_profile(user_id, profile_json, updated_at, version) "
        "VALUES(?, ?, ?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json, "
        "updated_at=excluded.updated_at, version=excluded.version",
        (user_id, json.dumps(profile, ensure_ascii=True), now_utc(), version),
    )
