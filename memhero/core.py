"""MemoryStore: Postgres + pgvector persistence for discrete facts."""

import json
import math
import uuid
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


def _vec(v):
    return np.asarray(v, dtype=np.float32)


@dataclass
class Memory:
    id: str
    user_id: str
    content: str
    sim: float = 0.0
    access_count: int = 0
    updated_at: str = ""


class MemoryStore:
    def __init__(self, dsn: str):
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("SELECT 1")
        register_vector(self._conn)

    # -- write ops -----------------------------------------------------------

    def add(self, user_id: str, content: str, vec, slot: str | None = None) -> str:
        mid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO memories (id, user_id, slot, content, embedding) VALUES (%s, %s, %s, %s, %s)",
            (mid, user_id, slot, content, _vec(vec)),
        )
        self._log(mid, user_id, "ADD", {"content": content, "slot": slot})
        return mid

    def update(self, memory_id: str, user_id: str, content: str, vec) -> None:
        self._conn.execute(
            """UPDATE memories SET content = %s, embedding = %s,
               updated_at = now() WHERE id = %s""",
            (content, vec, memory_id),
        )
        self._log(memory_id, user_id, "UPDATE", {"content": content})

    def delete(self, memory_id: str, user_id: str, action: str, detail: dict) -> None:
        row = self._conn.execute(
            "DELETE FROM memories WHERE id = %s RETURNING content", (memory_id,)
        ).fetchone()
        detail = {**detail, "deleted_content": row[0] if row else None}
        self._log(memory_id, user_id, action, detail)

    def touch(self, ids: list[str]) -> None:
        if not ids:
            return
        self._conn.execute(
            """UPDATE memories SET access_count = access_count + 1,
               last_accessed_at = now() WHERE id = ANY(%s)""",
            (ids,),
        )

    def enforce_cap(self, user_id: str, cap: int) -> int:
        """Evict lowest-scoring memories beyond cap. Score = recency × access_count.
        Old-but-frequently-accessed facts survive; stale facts go first."""
        row = self._conn.execute(
            """WITH over AS (
                 SELECT id FROM memories WHERE user_id = %s
                 ORDER BY
                   (EXTRACT(EPOCH FROM (now() - updated_at)) / 86400.0) DESC
                   - (access_count * 10) DESC
                 OFFSET %s
               )
               DELETE FROM memories WHERE id IN (SELECT id FROM over) RETURNING 1""",
            (user_id, cap),
        ).fetchall()
        n = len(row)
        if n:
            self._log(None, user_id, "EVICT", {"count": n, "reason": "cap"})
        return n

    # -- read ops ------------------------------------------------------------

    def search(self, user_id: str, query_vec, k: int, min_sim: float) -> list[Memory]:
        rows = self._conn.execute(
            """SELECT id, content, access_count,
                      to_char(updated_at, 'YYYY-MM-DD'),
                      1 - (embedding <=> %s) AS sim
               FROM memories WHERE user_id = %s
               ORDER BY embedding <=> %s LIMIT 50""",
            (_vec(query_vec), user_id, _vec(query_vec)),
        ).fetchall()
        scored = []
        for mid, content, cnt, updated, sim in rows:
            mid = str(mid)
            if sim < min_sim:
                continue
            days_old = _days_since(updated)
            decay = 1.0 if days_old <= 90 else 0.3
            score = sim * (1 + math.log1p(cnt)) * decay
            scored.append((score, Memory(id=mid, user_id=user_id, content=content,
                                         sim=sim, access_count=cnt, updated_at=updated)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored[:k]]

    def slot_search(self, user_id: str, slots: list[str]) -> list[Memory]:
        """Direct slot lookup — zero embedding cost. For known-slot queries like 'where do i live?'."""
        rows = self._conn.execute(
            """SELECT id, slot, content, access_count, to_char(updated_at, 'YYYY-MM-DD')
               FROM memories WHERE user_id = %s AND slot = ANY(%s)
               ORDER BY updated_at DESC""",
            (user_id, slots),
        ).fetchall()
        return [Memory(id=str(r[0]), user_id=user_id, content=r[2], access_count=r[3], updated_at=r[4])
                for r in rows]

    def has_slot(self, user_id: str, slot: str) -> str | None:
        """Return current value for a slot if it exists."""
        row = self._conn.execute(
            "SELECT content FROM memories WHERE user_id = %s AND slot = %s ORDER BY updated_at DESC LIMIT 1",
            (user_id, slot),
        ).fetchone()
        return row[0] if row else None

    def near_duplicates(self, user_id: str, vec, min_sim: float, k: int = 4) -> list[Memory]:
        rows = self._conn.execute(
            """SELECT id, content, 1 - (embedding <=> %s) AS sim
               FROM memories WHERE user_id = %s AND 1 - (embedding <=> %s) >= %s
               ORDER BY embedding <=> %s LIMIT %s""",
            (_vec(vec), user_id, _vec(vec), min_sim, _vec(vec), k),
        ).fetchall()
        return [Memory(id=str(r[0]), user_id=user_id, content=r[1], sim=r[2]) for r in rows]

    def list_memories(self, user_id: str, limit: int = 200) -> list[Memory]:
        rows = self._conn.execute(
            """SELECT id, content, access_count, to_char(updated_at, 'YYYY-MM-DD')
               FROM memories WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s""",
            (user_id, limit),
        ).fetchall()
        return [Memory(id=str(r[0]), user_id=user_id, content=r[1],
                       access_count=r[2], updated_at=r[3]) for r in rows]

    def count(self, user_id: str) -> int:
        return self._conn.execute(
            "SELECT count(*) FROM memories WHERE user_id = %s", (user_id,)
        ).fetchone()[0]

    def clear_user(self, user_id: str) -> int:
        n = self._conn.execute(
            "DELETE FROM memories WHERE user_id = %s RETURNING 1", (user_id,)
        ).fetchall()
        # purge pending extractions so they don't resurrect after reset
        self._conn.execute("DELETE FROM pending_extractions WHERE user_id = %s", (user_id,))
        return len(n)

    # -- extraction queue (crash-safe) ---------------------------------------

    def enqueue_turn(self, user_id: str, conversation: str, user_msg: str, assistant_msg: str) -> None:
        self._conn.execute(
            "INSERT INTO pending_extractions (user_id, conversation, turn) VALUES (%s, %s, %s)",
            (user_id, conversation, json.dumps({"user": user_msg, "assistant": assistant_msg})),
        )

    def drain(self, user_id: str | None = None, limit: int = 5, claimant: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        rows = self._conn.execute(
            """WITH picked AS (
                   SELECT id FROM pending_extractions
                   WHERE (claimed_by IS NULL OR claimed_at < now() - interval '5 minutes')
                     AND (user_id = %s OR %s::text IS NULL)
                   ORDER BY created_at LIMIT %s
               )
               UPDATE pending_extractions SET claimed_by = %s, claimed_at = now()
               WHERE id IN (SELECT id FROM picked) RETURNING *""",
            (user_id, user_id, limit, claimant),
        ).fetchall()
        if not rows:
            return []
        cols = [d[0] for d in self._conn.execute("SELECT * FROM pending_extractions LIMIT 0").description]
        return [dict(zip(cols, r)) for r in rows]

    def dequeued(self, ids: list[int]) -> None:
        self._conn.execute("DELETE FROM pending_extractions WHERE id = ANY(%s)", (ids,))

    # -- audit ---------------------------------------------------------------

    def _log(self, memory_id: str | None, user_id: str, action: str, detail: dict) -> None:
        self._conn.execute(
            "INSERT INTO memory_events (memory_id, user_id, action, detail) VALUES (%s, %s, %s, %s)",
            (memory_id, user_id, action, json.dumps(detail)),
        )


def _days_since(date_str: str) -> float:
    from datetime import datetime, timezone
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400
