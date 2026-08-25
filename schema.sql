CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           TEXT NOT NULL,
    slot              TEXT,
    content           TEXT NOT NULL,
    embedding         vector NOT NULL,
    access_count      INT NOT NULL DEFAULT 0,
    last_accessed_at  TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ponytail: HNSW caps at 2000 dims; local MiniLM=384 works, Gemini=3072 doesn't.
-- Index is optional — add when using ≤2000 dim embedding models.
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS memories_user_idx ON memories (user_id);
CREATE INDEX IF NOT EXISTS memories_slot_idx ON memories (user_id, slot);

CREATE TABLE IF NOT EXISTS memory_events (
    id          BIGSERIAL PRIMARY KEY,
    memory_id   UUID,
    user_id     TEXT NOT NULL,
    action      TEXT NOT NULL,          -- ADD / UPDATE / SUPERSEDE / DELETE / SKIP / FORGET / GUARD
    detail      JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memory_events_user_idx ON memory_events (user_id, created_at DESC);

-- crash-safe extraction queue: turns are enqueued immediately, processed async.
-- claimed_by/claimed_at prevents double-processing across workers/restarts.
CREATE TABLE IF NOT EXISTS pending_extractions (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    conversation TEXT NOT NULL DEFAULT '',
    turn        JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_by  TEXT,
    claimed_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS pending_extractions_user_idx ON pending_extractions (user_id, created_at);
