CREATE TABLE nodes (
    id          TEXT PRIMARY KEY,
    root_id     TEXT,
    hierarchy   TEXT,
    file_name   TEXT,
    kind        TEXT,
    content     TEXT,
    linked_node_ids TEXT[],
    embedding halfvec(2560),

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
