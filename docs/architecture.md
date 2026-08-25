# memhero architecture

## Design goal

Long-term memory for LLM agents. Extract facts from conversations, store in
vector DB, retrieve semantically on later turns. Single Python package with
multiple interfaces (CLI, HTTP, Gradio UI, MCP).

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Chat LLM | OpenAI-compatible (Gemini/OpenRouter) | Swap with env vars |
| Embeddings | Gemini `gemini-embedding-001` (3072d) or `text-embedding-3-small` (1536d) | Separate base_url/key support |
| Vector DB | Postgres + pgvector | No new infra — already have Postgres |
| Tracing | Langfuse (local docker) | Every LLM call + embedding traced |
| DB migration | Raw SQL in `schema.sql` | No Alembic overhead for 3 tables |

## Code layout

```
memhero/
  __init__.py
  config.py        # Config dataclass from env vars
  core.py          # MemoryStore (Postgres+pgvector), search/near_dupes/add/delete
  llm.py           # chat(), embed(), extract_facts(), reconcile(), guard_secret()
  service.py       # ChatService — turn flow, background fact extraction
  api.py           # FastAPI server (:8000)
  http_server.py   # Stdlib HTTP server + HTML UI (:8765)
  ui.py            # Gradio UI (:7860)
  cli.py           # Interactive REPL
  mcp_server.py    # MCP stdio server (remember/recall/forget/memories tools)
  templates/
    index.html     # Dark-themed HTML UI
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
  ▼
embed(msg) ──► pgvector HNSW search (k=8, min_sim=0.25)
  │
  ▼
[WHAT YOU REMEMBER ABOUT THIS USER] block injected into system prompt
  │
  ▼
chat(messages) ──► reply returned to user
  │
  ▼ (background thread)
extract_facts(user_msg, reply) ──► guard_secret() ──► near_duplicates()
  │
  ├─ no matches → ADD all candidates (no LLM call needed)
  └─ matches → reconcile() LLM call → ADD / UPDATE / SUPERSEDE / DELETE / SKIP
```

## Reconcile flow (write path)

After extracting candidate facts from an exchange:

1. **Secret guard**: regex drops secrets (cards, keys, auth tokens)
2. **Near-duplicate search**: cosine sim ≥ 0.65 against existing memories
3. **Skip or reconcile**:
   - No similar memories → ADD all candidates directly (no LLM call — saves cost)
   - Similar memories found → ONE reconcile LLM call decides per-candidate: ADD / UPDATE / SUPERSEDE / DELETE / SKIP
4. **Apply ops**: write to `memories` table, audit to `memory_events`

Supersede example: "I live in Delhi" then later "moved to Bangalore" → old
row marked SUPERSEDE, new row ADDed. Query-time only returns active rows.

## Database schema

```sql
-- extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS citext;  -- case-insensitive user_id

-- tables
memories (id uuid PK, user_id citext, content text, embedding vector(3072),
          deleted_reason text DEFAULT NULL, superseded_by text DEFAULT NULL,
          access_count int, last_accessed timestamptz, created_at, updated_at)

memory_events (id uuid PK, user_id citext, memory_id uuid FK, op text,
               payload jsonb, created_at timestamptz)

-- index: HNSW on embedding for similarity search
```

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

## Budget per turn

| Step | Cost |
|------|------|
| Embed user message | 1 embedding call (~50ms) |
| HNSW search | ~1ms |
| Chat reply | 1 LLM call |
| Background extract | 1 embed + ≤2 small LLM calls (off critical path) |

Memory adds ~50ms to the reply path (embed + search). Everything else is async.