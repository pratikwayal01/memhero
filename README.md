# memhero

Conversational memory layer for LLM agents. Extract facts from conversations,
store in Postgres+pgvector, retrieve semantically on later turns. Exposed as
CLI, HTTP API, HTML UI, Gradio UI, and MCP stdio server.

## Features

- **Slot-direct retrieval** — keyword-based lookup skips embedding entirely for known queries ("where do i live?" → `location` slot)
- **LRU embed cache** — 4096 entries, same text = zero recompute
- **Crash-safe extraction queue** — `pending_extractions` with `claimed_by` atomic claims, multi-worker safe
- **Multi-tenancy** — `org_id` isolation on all tables
- **Importance ranking** — LLM scores facts 0-1, used in retrieval: sim × importance × recency × access
- **Soft-delete + TTL** — `status` column: active/archived/consolidated, `expires_at` for auto-expiry
- **Dual LLM models** — separate chat (quality) and aux (extraction/reconcile) models for cost control
- **Layered retrieval** — slot-direct (0ms) → LRU cache (0ms) → semantic search (~5-50ms). Degrades gracefully

## Run

```sh
docker compose up -d          # db :5434, langfuse :3001 (dev@memhero.local / memhero-dev)
cp .env.example .env          # fill keys
uv sync

make ui                       # HTML UI on :8765 (dark theme, memory toggle, click-to-delete)
uv run memhero-ui             # Gradio UI on :7860 (full-featured)
uv run memhero-api            # FastAPI on :8000 — POST /chat, /forget, GET /users/{id}/memories
uv run memhero-cli <user>     # REPL with /memories /forget <text>
uv run memhero-mcp            # MCP stdio server: remember/recall/forget/memories tools
```

MCP registration (opencode): already in `~/.config/opencode/opencode.json`
as `memhero` (local stdio). For other clients:

```json
{"command": "uv", "args": ["run", "--directory", "/path/to/memhero", "memhero-mcp"]}
```

## LLM setup

Two models — split for cost control:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
MEMHERO_CHAT_MODEL=gpt-4o-mini          # user-facing replies (quality matters)
MEMHERO_AUX_MODEL=gpt-4o-mini           # extraction/reconcile (cheaper model works here)
```

`MEMHERO_AUX_MODEL` falls back to `MEMHERO_CHAT_MODEL` if not set. Separate API clients prevent rate-limit collisions.

## Embedding providers

Set `MEMHERO_EMBED_PROVIDER`:

| Provider | Config | Dims | Install |
|----------|--------|------|---------|
| `api` (default) | `MEMHERO_EMBED_MODEL`, `MEMHERO_EMBED_API_KEY`, `MEMHERO_EMBED_BASE_URL` | varies | none |
| `local` | `MEMHERO_LOCAL_MODEL` (default `all-MiniLM-L6-v2`) | 384 | `uv sync --extra local-embeddings` |

Vector dimensions auto-detected on first embed call. Override with `MEMHERO_VECTOR_DIMS`. Local embeddings have zero API cost (~5ms CPU).

## Architecture

See [docs/architecture.md](docs/architecture.md) — covers turn flow, reconcile
pipeline, DB schema, code layout, interfaces table, eval scenarios, and budget.

## Evals

9 scenarios × memory ON/OFF (fresh session baseline for OFF), scored with
substring checks + LLM rubric judge in Langfuse:

```sh
uv run python -m eval.run_eval
```

## Per-turn latency

| Path | Cost | When |
|------|------|------|
| Slot-direct | 0ms | Known-slot queries ("where do i live?") |
| LRU cache hit | 0ms | Repeated messages |
| Word-count gate | 0ms | `MEMHERO_SKIP_RETRIEVAL_UNDER_WORDS=n` |
| Local embed + HNSW | ~5ms | New substantive messages |
| API embed + HNSW | 50-450ms | New messages with API provider |

LLM reply: 1000-3000ms (dominant cost). Extraction: async, off critical path.