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

    embedding halfvec(4000),

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
