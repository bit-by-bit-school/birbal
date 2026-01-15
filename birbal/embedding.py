# This module converts a provided data frame into a vector embedding and stores it
import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from birbal.ai import get_embedder
from birbal.store import get_store
from birbal.config import config


def _apply_hierarchy_context(splits: list[str], hierarchy: str) -> list[str]:
    """Prefixes all splits after the first with the hierarchy context."""
    if not splits:
        return []
    if len(splits) == 1:
        return splits
    return [splits[0]] + [f"[{hierarchy}] {t}" for t in splits[1:]]


def _create_chunk_id(base_id, index, num_chunks):
    return base_id if num_chunks == 1 else f"{base_id}.{index}"


def _create_metadata(row: dict) -> dict:
    """Constructs metadata from row data."""
    return {
        "root_id": row.get("root_id") or row.get("id"),
        "file_name": row["file_name"],
        "hierarchy": row.get("hierarchy") or row.get("title", ""),
        "kind": row.get("kind"),
    }


def _create_chunk(text, index, num_chunks, row):
    metadata = _create_metadata(row)
    return {
        "id": _create_chunk_id(row.get("id"), index, num_chunks),
        "content": text,
        **metadata,
    }


def _split_row(row, splitter):
    """Splits a single row into a list of chunks."""
    hierarchy = row.get("hierarchy") or row.get("title", "")
    raw_splits = splitter.split_text(row.get("text"))
    processed_texts = _apply_hierarchy_context(raw_splits, hierarchy)

    return [
        _create_chunk(text, i, len(processed_texts), row)
        for i, text in enumerate(processed_texts)
    ]


def _prepare_chunks(df):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["text_split_chunk_size"],
        chunk_overlap=config["text_split_chunk_overlap"],
    )

    return [
        chunk
        for _, row in df.iterrows()
        for chunk in _split_row(row.to_dict(), text_splitter)
    ]


def _batch_embed_chunks(chunks, embedder):
    """Chunks is a list of dicts. We add an 'embedding' key to each."""
    batch_size = config["embedding_batch_size"]
    texts = [c["content"] for c in chunks]
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    print(f"Embedding {len(texts)} chunks (Batch size: {batch_size})...", flush=True)

    all_embeddings = [
        emb for batch in batches for emb in embedder.embed_documents(batch)
    ]

    return [{**chunk, "embedding": emb} for chunk, emb in zip(chunks, all_embeddings)]


def ingest_dataframe(df):
    """
    Embed a Pandas DataFrame into a vector store.

    The input DataFrame MUST contain the following columns:

        - id (str)
            A unique, stable identifier for each row.
            This value is used as the document ID and MUST be unique across all rows.

        - text (str)
            The full text content that will be embedded and indexed.

        - title (str)
            A short human-readable title for the document or note.

        - file_name (str)
            The source filename associated with this document.

    It can additionally contain the following columns:

        - root_id (str)
            An identifier shared across subrows.
            This can be used to query for an entire subtree.

        - hierarchy (str)
            A series of node titles going from the current node through all its ancestors (if they exist) separated by >.
    """
    chunks = _prepare_chunks(df)
    embedder = get_embedder()
    embedded_chunks = _batch_embed_chunks(chunks, embedder)
    vectordb = get_store()
    vectordb.upsert_nodes(embedded_chunks)
