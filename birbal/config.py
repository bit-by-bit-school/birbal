import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "port": int(os.getenv("PORT", 8080)),
    "file_dir": os.getenv("FILE_DIR"),
    "embedding_model": os.getenv("EMBEDDING_MODEL"),
    "large_language_model": os.getenv("LARGE_LANGUAGE_MODEL"),
    "vector_backend": os.getenv("VECTOR_BACKEND", "pg"),
    "text_split_chunk_size": int(os.getenv("TEXT_SPLIT_CHUNK_SIZE", 300)),
    "text_split_chunk_overlap": int(os.getenv("TEXT_SPLIT_CHUNK_OVERLAP", 0)),
    "k_nearest_neighbors_to_retrieve": int(
        os.getenv("K_NEAREST_NEIGHBORS_TO_RETRIEVE", 7)
    ),
    "chroma_dir": os.getenv("CHROMA_DIR", "./chroma/"),
    "chroma_collection_name": os.getenv("CHROMA_COLLECTION_NAME", "notes"),
}
