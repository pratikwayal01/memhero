# memhero architecture

## Design goal

Long-term memory for LLM agents. Extract facts from conversations, store in
Postgres+pgvector, retrieve semantically on later turns. Single Python package
with multiple interfaces (CLI, HTTP API, HTML UI, Gradio UI, MCP).

## Design rationale — why this architecture

### Problem: LLMs have no long-term memory

Every conversation starts from zero. Chatbots don't remember your name,
preferences, or past decisions. Solutions fall into three categories:

**1. Full context dump** — stuff entire history into system prompt.
- Breaks at scale: 100 conversations = 100k tokens → slow, expensive, LLM attention degrades.

**2. Embedding everything** (mem0, Chroma-based) — embed every message, retrieve on similarity.
- Every turn pays embedding cost. Chitchat ("ok", "thanks") wastes compute.
- No dedup: "I live in Delhi" × 3 times = 3 near-identical embeddings stored.
- No conflict resolution: "I moved to Bangalore" doesn't update old fact.

**3. Slot-based recall** — store facts in named slots, retrieve by slot name.
- Fast, zero embed. But rigid: new topics get no slot. Scaling means adding slots forever.

**Our approach**: hybrid — slot-direct for known topics + semantic for everything
else + LLM-based reconcile for conflict resolution. Best of all three.

### Design decisions (why each choice)

| Decision | Alternatives considered | Why this |
|----------|------------------------|----------|
| **Postgres+pgvector** not Pinecone/Weaviate | Separate vector DB adds ops burden | Already have Postgres. pgvector is mature, HNSW is fast. One less service to manage. |
| **Extract facts, not embed raw messages** | Embed every turn (mem0 approach) | Facts are 10-50× smaller. 1 fact ≈ 1 sentence. 100 turns ≈ 30 facts, not 100 embeddings. |
| **LLM reconcile, not vector dedup** | Cosine similarity threshold alone | "I live in Delhi" and "I live in Bangalore" are 0.7 cosine similar but CONFLICT. Cosine can't distinguish "same topic, updated" from "same fact, duplicate". LLM understands nuance. |
| **Slot-direct, not only semantic** | Pure vector search for everything | "where do i live?" is predictable. 15 keyword patterns cover 80% of user queries. Zero embed cost for the common case. |
| **Soft-delete, not hard DELETE** | DELETE rows | Audit trail. Superseded facts are still queryable for past-tense. Archiving preserves history for compliance/debugging. |
| **Queue, not inline extraction** | `learn()` inline on reply path | If process crashes mid-extraction, facts lost forever. Queue persists first, processes later. Atomic claims prevent double-processing. |
| **No ORM** | SQLAlchemy | Raw psycopg is 3× faster, zero abstraction. Schema is 3 tables. ORM adds nothing but latency and dependency weight. |
| **stdlib HTTP server, not Flask** | Flask/FastAPI for UI | HTML UI is a single page. stdlib `ThreadingHTTPServer` is 0 deps, starts in 50ms. FastAPI is for the API path where it adds value (OpenAPI, typing). |

## Comparison: memhero vs mem0 vs agent-memory

| Feature | mem0 | agent-memory | memhero |
|---------|------|-------------|----------|
| Embedding per message | Every turn | Every turn (LRU cached) | Slot-direct skips 80%+. LRU cache for rest |
| Conflict handling | Cosine threshold | Slot-based SUPERSEDE | LLM reconcile: ADD/UPDATE/SUPERSEDE/DELETE per fact |
| Extraction | Structured json per turn | Slot-classified extraction | Fact extraction with importance + slot + TTL |
| Memory dedup | Vector threshold | Exact content match | Cosine ≥ 0.65 + LLM reconcile for near-duplicates |
| Queue/durability | None | Postgres queue with claims | Postgres queue with atomic claims, same pattern |
| Multi-tenancy | User-level | User-level | org_id + user_id compound isolation |
| Soft-delete | No | No | status: active/archived/consolidated/superseded |
| Embedding options | API only | Local only (fastembed) | API + local (sentence-transformers) |
| TTL/expiry | No | No | expires_at per fact, filtered in search |
| Interfaces | API + SDK | HTTP server + HTML | CLI, HTTP API, HTML UI, Gradio, FastAPI, MCP |

## How we handle memory — the full lifecycle

### 1. Creation (extraction)

From every user-assistant exchange, the LLM extracts discrete facts. Not raw messages — distilled facts. The prompt instructs: "return a JSON array of facts, each with slot name and importance 0-1." Facts are third-person declarative sentences.

```json
[{"content": "The user lives in Delhi", "slot": "location", "importance": 0.9},
 {"content": "The user enjoys cooking pasta", "slot": "hobby", "importance": 0.3}]
```

Secrets are stripped twice: once by prompt ("never extract secrets"), once by regex guard (cards, keys, tokens).

### 2. Deduplication

Each candidate fact is embedded, then cosine-checked against existing memories (≥ 0.65 similarity). If nothing similar exists, ADD directly — no LLM call needed, saves cost.

If similar memories exist, ONE reconcile LLM call decides per-candidate: ADD (genuinely new), UPDATE (refine in place), SUPERSEDE (replace outdated/conflicting), DELETE (user said it's false), SKIP (exact duplicate).

**Why SUPERSEDE instead of DELETE?** "I live in Delhi" → "I moved to Bangalore" — the Delhi fact is marked `status='superseded'` with `superseded_by` pointing to the Bangalore fact. Past-tense queries ("where did I live before?") can still find it. Future compaction jobs can consolidate old superseded facts.

### 3. Storage

Facts are stored in Postgres with:
- `embedding` (vector) — for semantic search
- `slot` — for direct keyword lookup (zero-embed path)
- `importance` (0-1) — LLM-scored, used in retrieval ranking
- `expires_at` — optional TTL for transient facts ("visiting Paris next week" → auto-archive after 7 days)
- `status` — active/archived/consolidated/superseded. Soft-delete for audit trail.

### 4. Retrieval

Three layers, tried in order:

1. **Slot-direct** — if user query contains known slot keywords ("live", "job", "allergy"), fetch directly by slot column. Zero embedding cost, pure DB index scan.
2. **LRU embed cache** — same query text = same embedding vector cached in-process (4096 entries).
3. **Semantic search** — embed query → pgvector HNSW index → return top-k=50 → score = `cosine_sim × importance × (1 + log(access_count)) × recency_decay` → select top k=8.

Retrieved facts are injected into the system prompt as `[WHAT YOU REMEMBER ABOUT THIS USER]`. LLM uses them contextually but trusts the user's current statement over stored memories.

### 5. Growth management

Three limits prevent unbounded growth:

1. **Slot dedup** — same slot gets SUPERSEDE'd, not duplicated. User can't accumulate 50 "location: X" facts — only latest is active.
2. **Scoring decay** — facts older than 90 days get ×0.3 penalty. They drop out of top-k retrieval even if still stored. Effectively invisible for 99% of queries.
3. **Hard cap (500)** — `enforce_cap` archives lowest-scoring active facts beyond 500 per user. Scoring formula: `age_penalty − access_bonus`, so frequently-accessed old facts survive while stale facts go first.

Future: background compaction job (periodic cosine clustering → LLM consolidation → mark old as `consolidated`). Reduces storage while preserving information density.

### 6. Forgetting

Two paths:
- **User-initiated**: LLM-based forget via `/api/forget` (natural language query → retrieve candidates → LLM selects which to delete → hard DELETE). UI also supports click-to-delete on individual memory cards.
- **Automatic**: TTL expiry (`expires_at`), cap eviction (archive), slot supersession (old marked inactive).

### 7. Consistency (read-after-write)

After a turn, facts are INSERTed into `pending_extractions`. The next turn calls `drain_pending()` BEFORE retrieval — any queued facts from the previous turn are processed first. This closes the async consistency gap: user says "I moved to Bangalore", next turn "where do I live?" → sees the new fact.

In demo mode (HTML UI), extraction runs synchronously (`background=False`) — facts stored before reply returns. Zero gap.

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