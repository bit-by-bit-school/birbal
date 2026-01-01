#

import psycopg
from psycopg.rows import dict_row
from birbal.stores.base import VectorStore
from birbal.config import config


class PostgresStore(VectorStore):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.conn = psycopg.connect(config["postgres_dsn"])
        self._migrate()

    def _migrate(self):
        migrations_dir = config["migrations_dir"]

        with self.conn.cursor() as cur:
            cur.execute(
                """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                migrated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
            )
        self.conn.commit()

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT filename FROM schema_migrations")
            applied_migrations = {row["filename"] for row in cur.fetchall()}

            for file in sorted(migrations_dir.glob("*.sql")):
                if file.name in applied_migrations:
                    continue

                sql = file.read_text()
                print(f"Applying migration {file.name}", flush=True)
                cur.execute(sql, config)
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (file.name,),
                )
                self.conn.commit()

    def add_texts(self, texts, metadatas, ids):
        with self.conn.cursor() as cur:
            for text, meta, node_id in zip(texts, metadatas, ids):
                embedding = self.embeddings.embed_query(text)

                cur.execute(
                    """
                INSERT INTO nodes (id, root_id, hierarchy, file_name, kind, content, embedding)
                VALUES (%(node_id)s, %(root_id)s, %(hierarchy)s, %(file_name)s, %(kind)s, %(content)s, %(embedding)s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW();
                """,
                    {
                        "node_id": node_id,
                        "root_id": meta.get("root_id"),
                        "hierarchy": meta.get("hierarchy"),
                        "file_name": meta.get("file_name"),
                        "kind": meta.get("kind"),
                        "content": text,
                        "embedding": embedding,
                    },
                )

        self.conn.commit()

    def _hybrid_query(self, query_text, query_embedding, k):
        reciprocal_rank_fusion_smoothing = 50
        lexical_weight = 1
        # same as supabase https://supabase.com/docs/guides/ai/hybrid-search

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH full_text AS (
                    SELECT
                        id,
                        content,
                        row_number() OVER (
                            ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('english', %(q)s)) DESC
                        ) AS rank_ix
                    FROM nodes
                    WHERE content_tsv @@ websearch_to_tsquery('english', %(q)s)
                    ORDER BY rank_ix
                    LIMIT %(k)s * 2
                ),
                semantic AS (
                    SELECT
                        id,
                        content,
                        row_number() OVER (
                            ORDER BY embedding <=> (%(emb)s)::halfvec
                        ) AS rank_ix
                    FROM nodes
                    ORDER BY rank_ix
                    LIMIT %(k)s * 2
                )
                SELECT COALESCE(ft.content, sem.content) AS content
                FROM full_text ft
                FULL OUTER JOIN semantic sem 
                USING (id)
                ORDER BY
                    COALESCE(1.0 / (%(rrf)s + ft.rank_ix), 0) * %(lw)s +
                    COALESCE(1.0 / (%(rrf)s + sem.rank_ix), 0) DESC
                LIMIT %(k)s;
                """,
                {
                    "q": query_text,
                    "emb": query_embedding,
                    "k": k,
                    "lw": lexical_weight,
                    "rrf": reciprocal_rank_fusion_smoothing,
                },
            )

            return [row["content"] for row in cur.fetchall()]

    def similarity_search(self, query_str):
        query_embedding = self.embeddings.embed_query(query_str)
        k = config["k_nearest_neighbors_to_retrieve"]
        return self._hybrid_query(query_str, query_embedding, k)

    def filter_by_metadata(self, metadata_field, metadata_value):
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
            SELECT content
            FROM nodes
            WHERE {metadata_field} = %(val)s;
            """,
                {"val": metadata_value},
            )
            return [row["content"] for row in cur.fetchall()]
