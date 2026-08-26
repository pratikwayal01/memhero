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
    importance: float = 0.5
    access_count: int = 0
    updated_at: str = ""


class MemoryStore:
    def __init__(self, dsn: str, org_id: str = ""):
        self.org_id = org_id
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("SELECT 1")
        register_vector(self._conn)

    # -- write ops -----------------------------------------------------------

    def add(self, user_id: str, content: str, vec, slot: str | None = None,
            importance: float = 0.5, ttl_days: int | None = None) -> str:
        mid = str(uuid.uuid4())
        expires = f"now() + interval '{int(ttl_days)} days'" if ttl_days else "NULL"
        self._conn.execute(
            f"""INSERT INTO memories (id, org_id, user_id, slot, content, embedding, importance, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, {expires})""",
            (mid, self.org_id, user_id, slot, content, _vec(vec), max(0.0, min(1.0, importance))),
        )
        self._log(mid, user_id, "ADD", {"content": content, "slot": slot, "importance": importance})
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
        """Archive lowest-scoring memories beyond cap. Score = age penalty − access bonus."""
        row = self._conn.execute(
            """WITH over AS (
                 SELECT id FROM memories WHERE org_id = %s AND user_id = %s AND status = 'active'
                 ORDER BY
                   (EXTRACT(EPOCH FROM (now() - updated_at)) / 86400.0 - access_count * 10) DESC
                 OFFSET %s
               )
               UPDATE memories SET status = 'archived', updated_at = now()
               WHERE id IN (SELECT id FROM over) RETURNING 1""",
            (self.org_id, user_id, cap),
        ).fetchall()
        n = len(row)
        if n:
            self._log(None, user_id, "EVICT", {"count": n, "reason": "cap"})
        return n

    # -- read ops (all filter by org_id, status='active', non-expired) ----------

    def search(self, user_id: str, query_vec, k: int, min_sim: float) -> list[Memory]:
        rows = self._conn.execute(
            """SELECT id, content, importance, access_count,
                      to_char(updated_at, 'YYYY-MM-DD'),
                      1 - (embedding <=> %s) AS sim
               FROM memories
               WHERE org_id = %s AND user_id = %s
                 AND status = 'active'
                 AND (expires_at IS NULL OR expires_at > now())
               ORDER BY embedding <=> %s LIMIT 50""",
            (_vec(query_vec), self.org_id, user_id, _vec(query_vec)),
        ).fetchall()
        scored = []
        for mid, content, importance, cnt, updated, sim in rows:
            mid = str(mid)
            if sim < min_sim:
                continue
            days_old = _days_since(updated)
            decay = 1.0 if days_old <= 90 else 0.3
            score = sim * importance * (1 + math.log1p(cnt)) * decay
            scored.append((score, Memory(id=mid, user_id=user_id, content=content,
                                          sim=sim, importance=importance,
                                          access_count=cnt, updated_at=updated)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored[:k]]

    def slot_search(self, user_id: str, slots: list[str]) -> list[Memory]:
        rows = self._conn.execute(
            """SELECT id, content, importance, access_count, to_char(updated_at, 'YYYY-MM-DD')
               FROM memories WHERE org_id = %s AND user_id = %s AND slot = ANY(%s)
                 AND status = 'active' AND (expires_at IS NULL OR expires_at > now())
               ORDER BY updated_at DESC""",
            (self.org_id, user_id, slots),
        ).fetchall()
        return [Memory(id=str(r[0]), user_id=user_id, content=r[1],
                       importance=r[2], access_count=r[3], updated_at=r[4])
                for r in rows]

    def has_slot(self, user_id: str, slot: str) -> str | None:
        row = self._conn.execute(
            """SELECT content FROM memories WHERE org_id = %s AND user_id = %s AND slot = %s
               AND status = 'active' ORDER BY updated_at DESC LIMIT 1""",
            (self.org_id, user_id, slot),
        ).fetchone()
        return row[0] if row else None

    def near_duplicates(self, user_id: str, vec, min_sim: float, k: int = 4) -> list[Memory]:
        rows = self._conn.execute(
            """SELECT id, content, 1 - (embedding <=> %s) AS sim
               FROM memories WHERE org_id = %s AND user_id = %s
                 AND status = 'active' AND 1 - (embedding <=> %s) >= %s
               ORDER BY embedding <=> %s LIMIT %s""",
            (_vec(vec), self.org_id, user_id, _vec(vec), min_sim, _vec(vec), k),
        ).fetchall()
        return [Memory(id=str(r[0]), user_id=user_id, content=r[1], sim=r[2]) for r in rows]

    def list_memories(self, user_id: str, limit: int = 200) -> list[Memory]:
        rows = self._conn.execute(
            """SELECT id, content, importance, access_count, to_char(updated_at, 'YYYY-MM-DD')
               FROM memories WHERE org_id = %s AND user_id = %s AND status = 'active'
               ORDER BY updated_at DESC LIMIT %s""",
            (self.org_id, user_id, limit),
        ).fetchall()
        return [Memory(id=str(r[0]), user_id=user_id, content=r[1],
                       importance=r[2], access_count=r[3], updated_at=r[4]) for r in rows]

    def count(self, user_id: str) -> int:
        return self._conn.execute(
            "SELECT count(*) FROM memories WHERE org_id = %s AND user_id = %s AND status = 'active'",
            (self.org_id, user_id),
        ).fetchone()[0]

    def clear_user(self, user_id: str) -> int:
        """Soft-delete: archive all active memories for a user."""
        n = self._conn.execute(
            """UPDATE memories SET status = 'archived', updated_at = now()
               WHERE org_id = %s AND user_id = %s AND status = 'active' RETURNING 1""",
            (self.org_id, user_id),
        ).fetchall()
        self._conn.execute(
            "DELETE FROM pending_extractions WHERE org_id = %s AND user_id = %s",
            (self.org_id, user_id),
        )
        return len(n)

    # -- extraction queue (crash-safe) ---------------------------------------

    def enqueue_turn(self, user_id: str, conversation: str, user_msg: str, assistant_msg: str) -> None:
        self._conn.execute(
            "INSERT INTO pending_extractions (org_id, user_id, conversation, turn) VALUES (%s, %s, %s, %s)",
            (self.org_id, user_id, conversation, json.dumps({"user": user_msg, "assistant": assistant_msg})),
        )

    def drain(self, user_id: str | None = None, limit: int = 5, claimant: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        rows = self._conn.execute(
            """WITH picked AS (
                   SELECT id FROM pending_extractions
                   WHERE org_id = %s
                     AND (claimed_by IS NULL OR claimed_at < now() - interval '5 minutes')
                     AND (user_id = %s OR %s::text IS NULL)
                   ORDER BY created_at LIMIT %s
               )
               UPDATE pending_extractions SET claimed_by = %s, claimed_at = now()
               WHERE id IN (SELECT id FROM picked) RETURNING *""",
            (self.org_id, user_id, user_id, limit, claimant),
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
            "INSERT INTO memory_events (memory_id, org_id, user_id, action, detail) VALUES (%s, %s, %s, %s, %s)",
            (memory_id, self.org_id, user_id, action, json.dumps(detail)),
        )


def _days_since(date_str: str) -> float:
    from datetime import datetime, timezone
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400