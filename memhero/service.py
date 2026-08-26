"""ChatService: the turn flow — retrieve -> reply -> background extract+reconcile."""

import threading
import time
from datetime import datetime

from .config import get_config
from .core import MemoryStore
from . import llm

_store: MemoryStore | None = None


def store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore(get_config().database_url)
    return _store


class ChatService:
    def turn(self, user_id: str, message: str, use_memory: bool = True,
             conversation: str = "", background: bool = True) -> dict:
        """Blocking: retrieval (embedding + slot-direct) -> reply. Extraction enqueued for async drain."""
        cfg = get_config()
        t0 = time.perf_counter()
        retrieved = []
        if use_memory:
            # drain any pending extractions from previous turns first
            # (closes the async consistency gap — enqueued facts are stored before retrieval)
            self.drain_pending(user_id)
            # slot-direct first: zero embedding cost for known-slot queries
            slots = llm.detect_slots(message)
            if slots:
                retrieved = store().slot_search(user_id, slots)
            # fallback: semantic search for non-slot or additional results
            word_count = len(message.split())
            if cfg.skip_retrieval_under_words == 0 or word_count >= cfg.skip_retrieval_under_words:
                vec = llm.embed([message])[0]
                semantic = store().search(user_id, vec, cfg.retrieve_k, cfg.retrieve_min_sim)
                # merge: semantic results append after slot-direct, dedup by id
                seen = {m.id for m in retrieved}
                for m in semantic:
                    if m.id not in seen:
                        retrieved.append(m)

        memory_ms = (time.perf_counter() - t0) * 1000

        memory_block = ""
        if retrieved:
            lines = "\n".join(f"- {m.content}" for m in retrieved)
            memory_block = (
                f"\n\n[WHAT YOU REMEMBER ABOUT THIS USER]\n{lines}\n"
                "Use these memories when relevant. If they conflict with what "
                "the user just said, trust the user."
            )
        today = datetime.now().strftime("%Y-%m-%d")
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Today is {today}.{memory_block}"},
            {"role": "user", "content": message},
        ]
        reply = llm.chat(messages)
        inline_ms = (time.perf_counter() - t0) * 1000

        if use_memory:
            store().touch([m.id for m in retrieved])
            # ponytail: enqueue turn for crash-safe async extraction.
            # drain_pending() processes these in background or on next turn.
            store().enqueue_turn(user_id, conversation, message, reply)
            if background:
                threading.Thread(
                    target=self._drain_pending, args=(user_id,), daemon=True,
                ).start()
            else:
                self.drain_pending(user_id)

        return {
            "reply": reply,
            "retrieved": [m.content for m in retrieved],
            "inline_ms": round(inline_ms, 1),
            "memory_ms": round(memory_ms, 1),
        }

    def learn(self, user_id: str, message: str, reply: str) -> dict:
        """Extract facts from an exchange and reconcile with the store. Blocking."""
        cfg = get_config()
        candidates = llm.extract_facts(message, reply)
        if not candidates:
            return {"ops": []}

        # secret guard: drop anything that slipped past the prompt rule
        contents = [c["content"] for c in candidates]
        contents = [c for c in contents if llm.guard_secret(c)]
        if not contents:
            return {"ops": [{"op": "GUARD_DROPPED_ALL"}]}

        slots = [c.get("slot") for c in candidates]
        importances = [c.get("importance", 0.5) for c in candidates]
        ttl_days_list = [c.get("ttl_days") for c in candidates]
        vecs = llm.embed(contents)
        matches = [
            [(m.id, m.content) for m in store().near_duplicates(user_id, v, cfg.dup_sim)]
            for v in vecs
        ]
        if not any(matches):
            ops = [{"op": "ADD", "content": c, "slot": s} for c, s in zip(contents, slots)]
        else:
            ops = llm.reconcile(contents, matches)
        applied = self._apply_ops(user_id, contents, ops, vecs, slots, importances, ttl_days_list)
        store().enforce_cap(user_id, cfg.cap_per_user)
        return {"ops": ops, "applied": applied}

    def drain_pending(self, user_id: str | None = None) -> dict:
        """Process queued turns — extraction → reconcile → write. Crash-safe, multi-worker."""
        import uuid as _uuid
        claimant = _uuid.uuid4().hex
        stats = {"extracted": 0}
        for row in store().drain(user_id, claimant=claimant):
            try:
                turn = row["turn"]
                self.learn(user_id, turn["user"], turn["assistant"])
                store().dequeued([row["id"]])
                stats["extracted"] += 1
            except Exception as e:
                print(f"[drain] dropping turn {row['id']}: {type(e).__name__}: {e}")
                store().dequeued([row["id"]])
        return stats

    def _drain_pending(self, user_id: str):
        try:
            self.drain_pending(user_id)
        except Exception as e:
            print(f"[memhero] background drain failed: {e}")

    def _apply_ops(self, user_id: str, candidates: list[str], ops: list[dict], vecs,
                   slots=None, importances=None, ttl_days_list=None) -> int:
        s = store()
        n = 0
        slots = slots or [None] * len(candidates)
        importances = importances or [0.5] * len(candidates)
        ttl_days_list = ttl_days_list or [None] * len(candidates)
        for i, op in enumerate(ops):
            try:
                cand_idx = min(i, len(vecs) - 1)
                slot = slots[cand_idx] if cand_idx < len(slots) else None
                imp = importances[cand_idx]
                ttl = ttl_days_list[cand_idx]
                if op["op"] == "ADD":
                    content = op.get("content") or candidates[i]
                    s.add(user_id, content, vecs[cand_idx], slot=slot, importance=imp, ttl_days=ttl)
                elif op["op"] in ("UPDATE", "SUPERSEDE"):
                    mid = op["id"]
                    exists = any(m.id == mid for m in s.list_memories(user_id, limit=10_000))
                    if not exists:
                        continue
                    if op["op"] == "SUPERSEDE":
                        s.delete(mid, user_id, "SUPERSEDE", {"by": op.get("content")})
                        s.add(user_id, op["content"], vecs[cand_idx], slot=slot)
                    else:
                        s.update(mid, user_id, op["content"], vecs[cand_idx])
                elif op["op"] == "DELETE":
                    s.delete(op["id"], user_id, "DELETE", {"reason": op.get("reason")})
                else:  # SKIP
                    continue
                n += 1
            except Exception as e:
                print(f"[memhero] op failed ({op}): {e}")
        return n

    def remember_fact(self, user_id: str, fact: str) -> dict:
        """Store an explicitly-provided fact through the same dedupe/conflict path."""
        cfg = get_config()
        if not llm.guard_secret(fact):
            return {"ops": [{"op": "GUARD_REJECTED"}], "applied": 0}
        vec = llm.embed([fact])[0]
        matches = [
            [(m.id, m.content) for m in store().near_duplicates(user_id, vec, cfg.dup_sim)]
        ]
        ops = llm.reconcile([fact], matches)
        applied = self._apply_ops(user_id, [fact], ops, [vec])
        return {"ops": ops, "applied": applied}

    def remember_many(self, user_id: str, facts: list[str]) -> dict:
        """Batch variant of remember_fact; one reconcile call for the whole batch."""
        cfg = get_config()
        facts = [f for f in facts if llm.guard_secret(f)]
        if not facts:
            return {"ops": [{"op": "GUARD_REJECTED_ALL"}], "applied": 0}
        vecs = llm.embed(facts)
        matches = [
            [(m.id, m.content) for m in store().near_duplicates(user_id, v, cfg.dup_sim)]
            for v in vecs
        ]
        ops = llm.reconcile(facts, matches)
        applied = self._apply_ops(user_id, facts, ops, vecs)
        return {"ops": ops, "applied": applied}

    # -- explicit forgetting --------------------------------------------------

    def forget(self, user_id: str, query: str) -> list[str]:
        """Delete memories matching a natural-language request. Code-enforced."""
        vec = llm.embed([query])[0]
        hits = store().search(user_id, vec, k=5, min_sim=0.35)
        # LLM picks which of the candidates actually match the forget request
        listing = "\n".join(f"[{m.id}] {m.content}" for m in hits) or "(none)"
        out = llm.aux_chat(
            [
                {"role": "system", "content":
                    "User asks to forget something. Given candidate memories, return ONLY "
                    "a JSON array of ids that should be deleted per the request. [] if none match."},
                {"role": "user", "content": f"REQUEST: {query}\nCANDIDATES:\n{listing}"},
            ],
            temperature=0,
        )
        import json
        import re
        try:
            ids = json.loads(out)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", out, re.DOTALL)
            ids = json.loads(m.group()) if m else []
        deleted = []
        for mid in ids:
            if isinstance(mid, str):
                store().delete(mid, user_id, "FORGET", {"request": query})
                deleted.append(mid)
        return deleted
