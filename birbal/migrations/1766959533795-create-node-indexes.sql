CREATE INDEX nodes_tsv_idx   ON nodes USING GIN(content_tsv);
CREATE INDEX nodes_vec_idx   ON nodes USING hnsw(embedding halfvec_cosine_ops);
CREATE INDEX nodes_root_idx  ON nodes(root_id);
