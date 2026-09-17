"""Caches judge results in a local SQLite db to avoid re-calling the LLM."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class JudgeCache:
    """SQLite-backed store mapping a content hash to a judge's raw text response."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS judge_cache (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def make_key(prompt: str, judge_model: str, prompt_version: str) -> str:
        """Hash the inputs that determine the judge's output into a stable key."""
        payload = f"{judge_model}::{prompt_version}::{prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        """Return the cached response for this key, or None if not present."""
        row = self._conn.execute(
            "SELECT response FROM judge_cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, response: str) -> None:
        """Store a response under its key (overwrites if the key already exists)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO judge_cache (key, response) VALUES (?, ?)",
            (key, response),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()