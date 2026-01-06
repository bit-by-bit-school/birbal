from birbal.stores import PostgresStore
from birbal.ai import get_ai_provider
from birbal.config import config

_store = None


def get_store():
    global _store
    if _store is None:
        ai = get_ai_provider()
        embedder = ai.get_embedder()
        _store = PostgresStore(embedder)
    return _store
