CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id            TEXT NOT NULL DEFAULT '',
    user_id           TEXT NOT NULL,
    slot              TEXT,
    content           TEXT NOT NULL,
    embedding         vector NOT NULL,
    importance        REAL NOT NULL DEFAULT 0.5,
    expires_at        TIMESTAMPTZ,
    status            TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','superseded','archived','consolidated')),
    superseded_by     TEXT,
    access_count      INT NOT NULL DEFAULT 0,
    last_accessed_at  TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS memories_org_user_idx ON memories (org_id, user_id);
CREATE INDEX IF NOT EXISTS memories_slot_idx ON memories (org_id, user_id, slot) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS memories_expires_idx ON memories (expires_at) WHERE expires_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS memory_events (
    id          BIGSERIAL PRIMARY KEY,
    memory_id   UUID,
    org_id      TEXT NOT NULL DEFAULT '',
    user_id     TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memory_events_org_user_idx ON memory_events (org_id, user_id, created_at DESC);

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
