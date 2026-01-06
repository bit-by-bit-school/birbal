CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE nodes (
    id          TEXT PRIMARY KEY,
    root_id     TEXT,
    hierarchy   TEXT,
    file_name   TEXT,
    kind        TEXT,
    content     TEXT,

    content_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED,

    embedding halfvec(%(vector_dims)s),

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX nodes_tsv_idx   ON nodes USING GIN(content_tsv);
CREATE INDEX nodes_vec_idx   ON nodes USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX nodes_root_idx  ON nodes(root_id);
