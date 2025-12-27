from langchain_chroma import Chroma
from config import config

class ChromaStore:
    def __init__(self, embeddings):
        self.conn = Chroma(
            collection_name=config["collection_name"],
            embedding_function=embeddings,
            persist_directory=config["persist_dir"],
        )

    def add_texts(self, texts, metadatas, ids):
        self.conn.add_texts(texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query_str):
        retriever = self.conn.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config["k_nearest_neighbors_to_retrieve"]},
        )
        return retriever.invoke(query_str)

    def filter_by_metadata(self, metadata_field, metadata_value):
        results = self.conn.get(where={metadata_field: metadata_value})
        return results["documents"]
