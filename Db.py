"""
Local SQLite cache for external API responses (arXiv, Semantic Scholar, etc).

Why this exists: Contradiction Hunter and Hypothesis Evolution (built later)
re-query the same sources repeatedly per research session. arXiv and
Semantic Scholar both throttle free/unauthenticated requests, so every
query result is cached here first. A cache hit costs nothing and never
fails; a cache miss falls back to the live API.
"""

import sqlite3
import json
import hashlib
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "research_cache.db"

# How long a cached result stays valid before it's treated as stale.
DEFAULT_MAX_AGE_SECONDS = 7 * 24 * 3600  # 1 week


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            query TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    return conn


def _make_key(source: str, query: str) -> str:
    normalized = f"{source}:{query.lower().strip()}"
    return hashlib.sha256(normalized.encode()).hexdigest()


def get_cached(source: str, query: str, max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS):
    """Returns cached data (already parsed from JSON) or None on a miss/stale entry."""
    conn = _connect()
    key = _make_key(source, query)
    row = conn.execute(
        "SELECT response_json, created_at FROM api_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    conn.close()

    if row is None:
        return None

    response_json, created_at = row
    if time.time() - created_at > max_age_seconds:
        return None  # stale -> treat as a miss, caller will re-fetch

    return json.loads(response_json)


def set_cached(source: str, query: str, data) -> None:
    """Stores data (any JSON-serializable object: list/dict) keyed by source+query."""
    conn = _connect()
    key = _make_key(source, query)
    conn.execute(
        """
        INSERT OR REPLACE INTO api_cache (cache_key, source, query, response_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (key, source, query, json.dumps(data), time.time()),
    )
    conn.commit()
    conn.close()


def cache_stats() -> dict:
    """Quick visibility into how much the cache is being used — useful when you
    write up Phase 9 evaluation later (shows how many API calls were avoided)."""
    conn = _connect()
    total = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
    by_source = conn.execute(
        "SELECT source, COUNT(*) FROM api_cache GROUP BY source"
    ).fetchall()
    conn.close()
    return {"total_entries": total, "by_source": dict(by_source)}