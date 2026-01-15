from birbal.stores import PostgresStore
from birbal.ai import get_embedder
from birbal.config import config

_store = None


def get_store():
    global _store
    if _store is None:
        embedder = get_embedder()
        _store = PostgresStore(embedder)
    return _store


def query_vector(query_str):
    vectordb = get_store()
    return vectordb.similarity_search(query_str)


def query_by_id(root_id: str):
    """Return all documents who lie in the subtree rooted at `root_id`."""
    vectordb = get_store()
    return vectordb.filter_by_metadata(metadata_field="root_id", metadata_value=root_id)
