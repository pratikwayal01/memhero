# memhero

Conversational memory layer for LLM agents. Extract facts from conversations,
store in Postgres+pgvector, retrieve semantically on later turns. Exposed as
CLI, HTTP API, HTML UI, Gradio UI, and MCP stdio server.

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

## Architecture

See [docs/architecture.md](docs/architecture.md) — covers turn flow, reconcile
pipeline, DB schema, code layout, interfaces table, eval scenarios, and budget.

TL;DR: embed user msg → HNSW search → inject `[MEMORY]` block → chat reply.
Background: extract facts → guard secrets → dedup → reconcile LLM → apply ops.

## Embedding providers

Set `MEMHERO_EMBED_PROVIDER`:

| Provider | Config | Dims | Install |
|----------|--------|------|---------|
| `api` (default) | `MEMHERO_EMBED_MODEL`, `MEMHERO_EMBED_API_KEY`, `MEMHERO_EMBED_BASE_URL` | varies | none |
| `local` | `MEMHERO_LOCAL_MODEL` (default `all-MiniLM-L6-v2`) | 384 | `uv sync --group local-embeddings` |

Vector dimensions auto-detected on first embed call. Override with `MEMHERO_VECTOR_DIMS`.

## Evals

9 scenarios × memory ON/OFF (fresh session baseline for OFF), scored with
substring checks + LLM rubric judge in Langfuse:

```sh
uv run python -m eval.run_eval
```

## Budget per turn

- Reply path: 1 embed call (~50ms) + 1 HNSW query (~1ms) added
- Background: 1 embed + ≤2 small LLM calls (off critical path)