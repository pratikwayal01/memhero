# memhero architecture

## Design goal

Long-term memory for LLM agents. Extract facts from conversations, store in
Postgres+pgvector, retrieve semantically on later turns. Single Python package
with multiple interfaces (CLI, HTTP API, HTML UI, Gradio UI, MCP).

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Chat LLM | OpenAI-compatible (Gemini/OpenRouter) | Swap with env vars; split chat/aux models |
| Embeddings | API (Gemini/OpenAI) or local (sentence-transformers) | `MEMHERO_EMBED_PROVIDER=local` for zero API cost |
| Vector DB | Postgres + pgvector | No new infra |
| Tracing | Langfuse (local docker) | Every LLM call + embedding traced |
| DB migration | Raw SQL in `schema.sql` | No Alembic overhead for 3 tables |

## Code layout

```
memhero/
  __init__.py
  config.py        # Config dataclass from env vars
  core.py          # MemoryStore (Postgres+pgvector), slot_search, queue, eviction
  llm.py           # chat, embed (API + local + LRU cache), extract_facts, reconcile, guard_secret
  service.py       # ChatService — layered retrieval, queue drain, apply_ops
  api.py           # FastAPI server (:8000)
  http_server.py   # Stdlib HTTP server + HTML UI (:8765) + startup DB check
  ui.py            # Gradio UI (:7860)
  cli.py           # Interactive REPL
  mcp_server.py    # MCP stdio server (remember/recall/forget/memories tools)
  templates/
    index.html     # Dark-themed HTML UI with memory toggle, click-to-delete
schema.sql         # Table definitions
eval/
  run_eval.py      # 9-scenario Langfuse eval runner
docs/
  architecture.md  # This file
```

## Turn flow (ChatService.turn)

```
user_msg
  │
  ├─► Drain pending extractions (close async consistency gap)
  │
  ├─► Slot detection: keyword match → ["location", "job", ...]
  │     │
  │     ├─ slot found → slot_search() DB query (0ms, zero embedding)
  │     └─ no match + word_count ≥ gate → embed(msg) via LRU cache
  │           → pgvector HNSW search (k=8, min_sim=0.25)
  │           → score: sim × importance × (1+log(access)) × recency_decay
  │
  ├─► Merge: slot-direct results + semantic results, dedup by id
  │
  ├─► [WHAT YOU REMEMBER ABOUT THIS USER] block injected into system prompt
  │
  ├─► chat(messages) ──► reply returned to user
  │
  └─► Enqueue turn → pending_extractions (crash-safe queue)
        Background thread: drain → extract_facts → reconcile → write
```

### Layered retrieval (why it's fast)

| Layer | Latency | Triggers | Example |
|-------|---------|----------|---------|
| Slot-direct | 0ms | Keyword match in query | "where do i live?" |
| LRU embed cache | 0ms | Same text seen before | Repeated "how are you?" |
| Word-count gate | 0ms | `< N` word messages | "ok", "thanks" |
| Local semantic | ~5ms | New substantive msg | "I started learning piano" |
| API semantic | 50-450ms | New msg with API provider | Same as above, remote embed |

## Reconcile flow (write path)

After extracting candidate facts from an exchange:

1. **Secret guard**: regex drops cards (13-19 digits), API keys (`sk-...`), AWS keys (`AKIA...`), long hex/base64 — code-enforced double layer after prompt guard
2. **Near-duplicate search**: cosine sim ≥ 0.65 against existing memories
3. **Skip or reconcile**:
   - No similar memories → ADD all directly (no LLM call — saves 50%+ cost)
   - Similar memories found → ONE reconcile LLM call decides per-candidate: ADD / UPDATE / SUPERSEDE / DELETE / SKIP
4. **Apply ops**: write to `memories` table with importance + slot + optional TTL, audit to `memory_events`
5. **Eviction**: `enforce_cap(500)` archives lowest-scored active memories (age × access_count)

### Importance extraction

LLM scores each fact 0-1 during extraction. Used in retrieval scoring:

```
score = cosine_sim × importance × (1 + log(access_count)) × recency_decay
```

High-importance facts (identity: name, location) survive longer than low-importance (hobby, preference).

## Database schema

```sql
memories (
  id UUID PK, org_id TEXT, user_id TEXT, slot TEXT,
  content TEXT, embedding vector, importance REAL DEFAULT 0.5,
  expires_at TIMESTAMPTZ,
  status TEXT DEFAULT 'active' -- active/archived/consolidated/superseded
  superseded_by TEXT, access_count INT, ...
)

memory_events (id, org_id, user_id, memory_id, action, detail JSONB, ...)

pending_extractions (id, org_id, user_id, turn JSONB, claimed_by, claimed_at, ...)

-- Indexes
HNSW on embedding WHERE status='active' (≤2000 dim models only)
B-tree on (org_id, user_id)
B-tree on (org_id, user_id, slot) WHERE status='active'
B-tree on expires_at WHERE expires_at IS NOT NULL
```

**Why soft-delete?** `status = 'archived'` instead of DELETE. Audit trail preserved. Eviction + reset use soft-delete.

## Extraction queue (crash-safe)

Every turn is `INSERT`ed into `pending_extractions` immediately after reply.
Background worker calls `drain()` which uses `UPDATE ... WHERE (claimed_by IS NULL OR claimed_at < 5 min) RETURNING *` — atomic row-level claims. Two workers never double-process. If worker crashes, stale claims re-claimed after 5 min.

## Embedding providers

| Provider | Config | Dims | Install |
|----------|--------|------|---------|
| `api` (default) | `MEMHERO_EMBED_MODEL`, `MEMHERO_EMBED_API_KEY`, `MEMHERO_EMBED_BASE_URL` | varies | none |
| `local` | `MEMHERO_LOCAL_MODEL` (default `all-MiniLM-L6-v2`) | 384 | `uv sync --extra local-embeddings` |

Vector dimensions auto-detected. Override with `MEMHERO_VECTOR_DIMS`. LRU cache (4096 entries) avoids recompute.

## Interfaces

| Entry point | Port | Type | Use case |
|-------------|------|------|----------|
| `make ui` | 8765 | HTML (stdlib server) | Demo, manual testing |
| `uv run memhero-ui` | 7860 | Gradio | Feature-rich UI |
| `uv run memhero-api` | 8000 | FastAPI | Production API |
| `uv run memhero-cli <user>` | — | REPL | Debug, scripting |
| `uv run memhero-mcp` | — | MCP stdio | LLM client memory |

## Evals

9 Langfuse scenarios × memory ON/OFF:
- `growth-stress` — 30 facts over 15 turns, recall last fact
- `cross-convo-person` — recall a fact from earlier conversation
- `precision-unrelated` — don't hallucinate on unrelated queries
- `secret-guard` — assert no secrets stored in DB
- `explicit-forget` — forget removes a specific memory
- `preference-diet` — dietary preference remembered
- `temporal-facts` — time-qualified move history
- `conflict-move` — city change → only latest remembered
- `recall-city` — basic recall from prior turn

Memory-OFF is baseline: fresh session each turn, no cross-turn knowledge.

## Per-turn cost

| Step | Mode | Latency |
|------|------|---------|
| Slot-direct | Always | 0ms (DB index scan) |
| Semantic embed | API provider | 50-450ms |
| Semantic embed | Local provider | ~5ms |
| HNSW search | Always | ~1ms |
| Chat reply | Always | 1000-3000ms (dominant) |
| Write path | Async queue | Off critical path |

Net memory overhead: 0ms (slot-direct) to ~5ms (local embed) to ~450ms (API embed).
LLM reply is the bottleneck, not memory.