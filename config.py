import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "port": int(os.getenv("PORT")) or 8080,
    "roam_dir": os.getenv("ROAM_DIR"),
    "persist_dir": os.getenv("PERSIST_DIR") or "./chroma/",
    "collection_name": os.getenv("COLLECTION_NAME"),
    "embedding_model": os.getenv("EMBEDDING_MODEL"),
    "large_language_model": os.getenv("LARGE_LANGUAGE_MODEL"),
    "text_split_chunk_size": int(os.getenv("TEXT_SPLIT_CHUNK_SIZE")) or 300,
    "text_split_chunk_overlap": int(os.getenv("TEXT_SPLIT_CHUNK_OVERLAP")) or 0,
    "k_nearest_neighbors_to_retrieve": int(os.getenv("K_NEAREST_NEIGHBORS_TO_RETRIEVE")) or 7,
}
