# This module takes a user query and fetches the relevant context and provides it to the LLM

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import config


def query_vector(query_str):
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    vectordb = Chroma(
        collection_name=config["collection_name"],
        embedding_function=embeddings,
        persist_directory=config["persist_dir"],
    )
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config["k_nearest_neighbors_to_retrieve"]},
    )
    retrieved_docs = retriever.invoke(query_str)

    return retrieved_docs

def query_by_id(root_id: str):
    """Return all documents who lie in the subtree rooted at `root_id`."""
    lower = id_prefix
    upper = id_prefix + ".\uffff"

    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    vectordb = Chroma(
        collection_name=config["collection_name"],
        embedding_function=embeddings,
        persist_directory=config["persist_dir"],
    )

    raw = vectordb.get(where={"root_id": root_id})
    return raw["documents"]

# # Retrieve docs
# retrieved_docs = vectordb.similarity_search_with_score(query_str, k=15)
# org_link_format = "[%.2f]: [[id:%s][%s]] \n %s"
# docs = [org_link_format % (score, doc.metadata["ID"],
#                            doc.metadata["title"].strip(),
#                                doc.metadata["hierarchy"].strip())
#                          for doc, score in retrieved_docs]
