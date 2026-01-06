from birbal.stores import PostgresStore
from birbal.ai import get_ai_provider
from birbal.config import config

_store = None


def get_store():
    global _store
    if _store is None:
        ai = get_ai_provider()
        embeddings = ai.get_embeddings()
        _store = PostgresStore(embeddings)
    return _store
