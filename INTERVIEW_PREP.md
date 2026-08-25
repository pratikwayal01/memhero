# Interview Prep: memhero

## 30-second pitch

> memhero is a conversational memory layer for LLM agents. It extracts durable facts from chats, stores them in Postgres+pgvector, and retrieves them semantically on future turns. It has both API and local embedding support, slot-direct keyword retrieval that skips embeddings entirely, a crash-safe extraction queue, and multiple interfaces: CLI, HTTP API, HTML UI, Gradio, and MCP server.

## System design walkthrough

### Turn flow (what happens on every chat message)

```
User: "where do I live?"
  │
  ├─► 1. Slot detection: keyword match → ["location"]
  │      (zero cost, regex in Python)
  │
  ├─► 2. Slot-direct lookup: SELECT * FROM memories WHERE slot='location'
  │      (zero embedding, pure DB index scan)
  │
  ├─► 3. Semantic fallback: only if no slot match OR message ≥ N words
  │      embed(message) → pgvector cosine search (k=8, min_sim=0.25)
  │      LRU cache: same text → cached vector, zero recompute
  │
  ├─► 4. Merge results: slot-direct + semantic, dedup by id
  │      Score: sim × (1 + log(access_count)) × recency_decay
  │
  ├─► 5. System prompt: inject "[WHAT YOU REMEMBER ABOUT THIS USER]" block
  │      LLM chat → reply returned to user
  │
  └─► 6. Enqueue: INSERT INTO pending_extractions (turn, claimed_by=NULL)
         Background thread: drain → extract facts → reconcile → write
```

**Key design decisions:**

- **Why slot-direct?** Most user memory queries are predictable ("where do i live?", "what's my job?", "any allergies?"). Keyword matching avoids embedding cost entirely — critical at scale where embedding latency dominates.
- **Why LRU cache?** Chat users repeat themselves. "How are you?" embedded once, cached forever. `functools.lru_cache(maxsize=4096)` — in-process, zero infra.
- **Why pending_extractions queue?** If the process crashes between reply and fact extraction, facts are lost. The queue is crash-safe: turn is INSERTed immediately, then `claimed_by` atomically marks it for a worker. If worker dies, `claimed_at > 5 min` → re-claimable.

### Write path (how facts get stored)

```
enqueued turn (user_msg, assistant_reply)
  │
  ├─► extract_facts(): LLM call → [{content: "lives in Bangalore", slot: "location"}, ...]
  │      Same LLM classifies each fact into a slot category
  │
  ├─► guard_secret(): regex drops cards, keys, tokens, base64 blobs
  │      (prompt rule in extract_facts + code-enforced regex as double guard)
  │
  ├─► near_duplicates(): cosine ≥ 0.65 search against existing memories
  │
  ├─► reconcile(): IF matches exist → ONE LLM call decides per-candidate:
  │      ADD (new fact), UPDATE (refine), SUPERSEDE (replace conflicting),
  │      DELETE (false/forgotten), SKIP (duplicate)
  │      IF no matches → ADD all directly (zero LLM cost for new facts!)
  │
  └─► apply_ops(): write to memories table, audit to memory_events
         enforce_cap(500): LRU eviction if exceeded
```

**Key design decisions:**

- **Why skip reconcile when no matches?** Most early turns produce genuinely new facts. The reconcile LLM call only fires when there's an actual conflict to resolve. Saves ~50% of write-path LLM costs.
- **Why SUPERSEDE instead of UPDATE?** "I live in Delhi" then "I moved to Bangalore" — UPDATE would destroy temporal info. SUPERSEDE marks old row inactive + ADDs new row. Past-tense queries ("where did I live before?") can still access superseded rows.
- **Why secret guard double-layered?** The extraction prompt says "never extract secrets" — but LLMs hallucinate. The regex guard is a hard, code-enforced backstop.

### Database schema

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `memories` | Active facts | id, user_id, slot, content, embedding (vector), access_count |
| `memory_events` | Audit log | memory_id, action (ADD/UPDATE/SUPERSEDE/DELETE), detail (jsonb) |
| `pending_extractions` | Crash-safe queue | turn (jsonb), claimed_by, claimed_at |

Indexes: HNSW on embedding (≤2000 dims only), B-tree on (user_id, slot), B-tree on (user_id, created_at).

### Embedding providers

```
MEMHERO_EMBED_PROVIDER=api   →  OpenAI/Gemini API (variable dims, 50-450ms)
MEMHERO_EMBED_PROVIDER=local →  sentence-transformers (384d MiniLM, ~5ms CPU)
```

Vector dimensions auto-detected on first embed. Schema uses untyped `vector` (not `vector(N)`) so dim changes don't require migration.

## Interviewer questions and answers

### Q1: "What if the embedding model changes and dimensions differ?"

**A:** Schema uses untyped `vector`. Any dimension works. `MEMHERO_VECTOR_DIMS` env var can pre-declare, or it auto-detects from the first embedding call. No migration needed — just change the env var and restart.

### Q2: "How do you handle concurrent writes to the same user?"

**A:** Each `MemoryStore` has one psycopg connection (autocommit). The write path is single-threaded per `ChatService` instance. For multi-worker, the `pending_extractions` queue uses `UPDATE ... RETURNING` with `claimed_by` — atomic row-level claims prevent double-processing. Two workers never process the same turn.

### Q3: "How do you handle rate limiting on the LLM API?"

**A:** Six-retry exponential backoff in `_retry()`: 30s, 60s, ... up to 60s max wait. Separate client instances for chat vs auxiliary (extraction/reconcile/judging) — different rate limit buckets don't fight each other.

### Q4: "Why not use Redis for caching?"

**A:** Premature optimization for this scale. LRU cache (4096 entries, in-process) covers repeated queries. If per-instance cache misses become a bottleneck, Redis shared cache is a one-line swap. Measure first.

### Q5: "How do you prevent prompt injection through memory?"

**A:** Triple defense:
1. Extraction prompt says "never extract secrets" — LLM-level guard
2. `guard_secret()` regex: cards (13-19 digits), API keys (`sk-...`), AWS keys (`AKIA...`), long hex/base64 — code-enforced
3. Memory content is only injected into system prompt as `[WHAT YOU REMEMBER]` — it's structured, not raw user input

### Q6: "This extracts facts from every turn. What about cost?"

**A:** Two cost optimizations:
1. Write path: reconcile LLM call only fires when near-duplicates exist (cos ≥ 0.65). New facts skip it entirely.
2. Read path: slot-direct retrieval (keyword match) skips embedding. LRU cache handles repeats. Word-count gate (`MEMHERO_SKIP_RETRIEVAL_UNDER_WORDS`) skips trivial messages.
3. Local embeddings: `MEMHERO_EMBED_PROVIDER=local` makes all embeddings free (~5ms CPU).

### Q7: "How would you scale this to millions of users?"

**A:**
1. **Connection pooling**: Replace single `psycopg.connect` with pool (pgbouncer or psycopg pool) — one connection per request would exhaust Postgres connections.
2. **Partitioning**: Partition `memories` by `user_id` hash. Search stays within one partition.
3. **Separate read replicas**: Search (read-heavy) hits read replicas. Write path hits primary.
4. **Async drain workers**: Replace background threads with a proper worker pool (Celery/Redis Queue) consuming `pending_extractions`. Multiple workers across replicas.
5. **Embedding service**: Deploy embedding model as a separate microservice behind a load balancer instead of in-process.

## Possible on-the-spot implementation tasks

### Task 1: "Add multi-tenancy — separate memory stores per organization"

```python
# config.py - add org_id
org_id: str = field(default_factory=lambda: os.environ.get("MEMHERO_ORG_ID", ""))

# core.py - add org_id to all queries
# WHERE user_id = %s AND org_id = %s
```

### Task 2: "Add importance-based ranking instead of just recency"

```python
# Extract facts now return importance (0-1)
# search() score becomes: sim × importance × (1 + log(access_count)) × decay
```

### Task 3: "Add TTL-based automatic memory expiration"

```python
# schema: add expires_at TIMESTAMPTZ
# config: default_ttl_days = 90
# search(): WHERE (expires_at IS NULL OR expires_at > now())
# Cron job: DELETE WHERE expires_at < now()
```

### Task 4: "Add memory consolidation — merge similar facts"

```python
# Periodically: SELECT memories with high cosine similarity
# → LLM consolidates into one richer fact
# → Mark old ones as SUPERSEDE → consolidated
```

## Architecture: what to highlight

1. **Layered retrieval** — slot-direct (free) → cache (free) → semantic (paid). Graceful degradation.
2. **Crash-safe queue** — `pending_extractions` with claim-based concurrency. Works with any number of workers.
3. **LLM-cost-optimized write** — skip reconcile when no conflicts. One LLM decision per batch, not per fact.
4. **Defense-in-depth for secrets** — prompt + regex + structured injection. No single point of failure.
5. **No framework lock-in** — stdlib HTTP server for HTML UI, FastAPI for API, raw psycopg (no ORM), Langfuse for tracing. Everything swappable.