CREATE INDEX nodes_root_idx  ON nodes(root_id);
CREATE INDEX nodes_vec_idx   ON nodes USING hnsw(embedding halfvec_cosine_ops);
CREATE INDEX nodes_content_bm25_idx ON nodes USING bm25(content) WITH (text_config='english');