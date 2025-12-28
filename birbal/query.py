# This module queries the vector db and provides the desired document contents
from birbal.store import get_store


def query_vector(query_str):
    vectordb = get_store()
    return vectordb.similarity_search(query_str)


def query_by_id(root_id: str):
    """Return all documents who lie in the subtree rooted at `root_id`."""
    vectordb = get_store()
    return vectordb.filter_by_metadata(metadata_field="root_id", metadata_value=root_id)


# # Retrieve docs
# retrieved_docs = vectordb.similarity_search_with_score(query_str, k=15)
# org_link_format = "[%.2f]: [[id:%s][%s]] \n %s"
# docs = [org_link_format % (score, doc.metadata["ID"],
#                            doc.metadata["title"].strip(),
#                                doc.metadata["hierarchy"].strip())
#                          for doc, score in retrieved_docs]
