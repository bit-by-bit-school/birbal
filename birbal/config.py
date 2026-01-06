import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def _pg_dsn():
    return (
        f"dbname={os.getenv('POSTGRES_DB')} "
        f"user={os.getenv('POSTGRES_USER')} "
        f"password={os.getenv('POSTGRES_PASSWORD')} "
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', 5432)}"
    )


config = {
    "port": int(os.getenv("PORT", 8080)),
    "file_dir": os.getenv("FILE_DIR"),
    "ollama_host": os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"),
    "embedding_model": os.getenv("EMBEDDING_MODEL"),
    "large_language_model": os.getenv("LARGE_LANGUAGE_MODEL"),
    "vector_backend": os.getenv("VECTOR_BACKEND", "pg"),
    "vector_dims": int(os.getenv("VECTOR_DIMENSIONS", 4000)),
    "text_split_chunk_size": int(os.getenv("TEXT_SPLIT_CHUNK_SIZE", 300)),
    "text_split_chunk_overlap": int(os.getenv("TEXT_SPLIT_CHUNK_OVERLAP", 0)),
    "embedding_batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", 100)),
    "context_window_size": int(os.getenv("CONTEXT_WINDOW_SIZE", 2048)),
    "k_nearest_neighbors_to_retrieve": int(
        os.getenv("K_NEAREST_NEIGHBORS_TO_RETRIEVE", 7)
    ),
    "postgres_dsn": _pg_dsn(),
    "migrations_dir": Path(os.getenv("MIGRATIONS_DIR", "/birbal/migrations")).resolve()
}
