# This module takes a user query and fetches the relevant context and provides it to the LLM

import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def query_vector(query_str):
    persist_directory = os.getenv('PERSIST_DIR')
    embedding_model = os.getenv('EMBEDDING_MODEL')
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    vectordb = Chroma(collection_name="org_roam", embedding_function=embeddings, persist_directory=persist_directory)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 7}) # parametrize
    retrieved_docs = retriever.invoke(query_str)

    return retrieved_docs


# # Retrieve docs
# retrieved_docs = vectordb.similarity_search_with_score(query_str, k=15)
# org_link_format = "[%.2f]: [[id:%s][%s]] \n %s"
# docs = [org_link_format % (score, doc.metadata["ID"],
#                            doc.metadata["title"].strip(),
#                                doc.metadata["hierarchy"].strip())
#                          for doc, score in retrieved_docs]



