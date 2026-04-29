#
from datetime import timezone
import psycopg
from psycopg.rows import dict_row
from birbal.stores.base import VectorStore, FileStat
from birbal.config import config


class PostgresStore(VectorStore):
    def __init__(self, embedder):
        self.embedder = embedder
        self.conn = psycopg.connect(config["postgres_dsn"])
        self._migrate()

    def _migrate(self):
        print("Running database migrations...")

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
                print(f"Applying migration {file.name}")
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (file.name,),
                )
                self.conn.commit()

        print("Migrations complete.")

    def upsert_nodes(self, nodes: list[dict]):
        query = """
        INSERT INTO nodes (id, root_id, content, embedding, file_name, hierarchy, kind, linked_node_ids, updated_at)
        VALUES (%(id)s, %(root_id)s, %(content)s, %(embedding)s, %(file_name)s, %(hierarchy)s, %(kind)s, %(linked_node_ids)s, NOW())
        ON CONFLICT (id) DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            updated_at = NOW();
        """
        with self.conn.cursor() as cur:
            cur.executemany(query, nodes)
        self.conn.commit()

    def delete_by_filenames(self, filenames):
        with self.conn.cursor() as cur:
            cur.execute(
                """
               DELETE FROM nodes WHERE file_name = ANY(%s)
                """,
                (list(filenames),),
            )

        self.conn.commit()

    def get_file_stats(self):
        with self.conn.cursor() as cur:
            cur.execute(
                """
               SELECT file_name, MIN(updated_at) FROM nodes GROUP BY file_name
                """,
            )
            return [
                FileStat(file_name=fname, last_indexed_at=ts.astimezone(timezone.utc))
                for fname, ts in cur.fetchall()
            ]

    def _hybrid_query(self, query_text, query_embedding, k):
        reciprocal_rank_fusion_smoothing = 50
        lexical_weight = 1
        documents_per_search = k * 2
        # same as supabase https://supabase.com/docs/guides/ai/hybrid-search

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH full_text AS (
                    SELECT
                        id,
                        file_name,
                        content,
                        row_number() OVER (
                            ORDER BY content <@> to_bm25query(%(q)s, 'nodes_content_bm25_idx')
                        ) AS rank_ix
                    FROM nodes
                    ORDER BY content <@> to_bm25query(%(q)s, 'nodes_content_bm25_idx')
                    LIMIT %(d)s
                ),
                semantic AS (
                    SELECT
                        id,
                        file_name,
                        content,
                        row_number() OVER (
                            ORDER BY embedding <=> (%(emb)s)::halfvec
                        ) AS rank_ix
                    FROM nodes
                    ORDER BY rank_ix
                    LIMIT %(d)s
                )
                SELECT COALESCE(ft.file_name, sem.file_name) AS filename, 
                       COALESCE(ft.content, sem.content) AS content
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
                    "d": documents_per_search,
                    "lw": lexical_weight,
                    "rrf": reciprocal_rank_fusion_smoothing,
                },
            )

            return cur.fetchall()

    def _semantic_unlinked_neighbors(self, note_id, limit):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH target_note AS (
                    SELECT id, embedding, linked_node_ids
                    FROM nodes
                    WHERE root_id = %(id)s
                ),
                excluded_ids AS (
                    SELECT %(id)s AS exclude_id
                    UNION
    
                    SELECT unnest(linked_node_ids)
                    FROM target_note
                    WHERE linked_node_ids IS NOT NULL
    
                    UNION
    
                    SELECT root_id
                    FROM nodes
                    WHERE %(id)s = ANY(linked_node_ids)
                ),
                candidate_matches AS (
                    SELECT 
                        t.id AS source_chunk_id,
                        c.root_id AS matched_root_id,
                        c.file_name,
                        (c.embedding <=> t.embedding) AS distance
                    FROM target_note t
                    CROSS JOIN LATERAL (
                        SELECT root_id, file_name, embedding
                        FROM nodes n
                        WHERE NOT EXISTS (
                            SELECT 1 FROM excluded_ids e WHERE e.exclude_id = n.root_id
                        )
                        ORDER BY n.embedding <=> t.embedding
                        LIMIT 20
                        ) AS c
                )
                SELECT 
                    matched_root_id AS root_id,
                    MIN(distance) AS best_distance
                FROM candidate_matches
                GROUP BY matched_root_id
                ORDER BY best_distance ASC
                LIMIT %(k)s;
                """,
                {
                    "id": note_id,
                    "k": limit
                }
            )
            return cur.fetchall()

    def similarity_search(self, query_str):
        query_embedding = self.embedder.embed_query(query_str)
        k = config["k_nearest_neighbors_to_retrieve"]
        results = self._hybrid_query(query_str, query_embedding, k)
        return [row["content"] for row in results]

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

    def find_similar_unlinked(self, note_id, limit):
        results = self._semantic_unlinked_neighbors(note_id, limit)
        return [row[0] for row in results]