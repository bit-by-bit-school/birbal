from birbal.stores import PostgresStore, ChromaStore
from birbal.ai import get_ai_provider
from birbal.config import config

_store = None


def get_store():
    global _store
    if _store is None:
        ai = get_ai_provider()
        embeddings = ai.get_embeddings()
        if config["vector_backend"] == "chroma":
            _store = ChromaStore(embeddings)
        else:
            _store = PostgresStore(embeddings)
    return _store
