# This module converts a provided data frame into a vector embedding and stores it

import os
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def embed_df(df):
    persist_directory = os.getenv('PERSIST_DIR')
    embedding_model = os.getenv('EMBEDDING_MODEL')
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectordb = Chroma(collection_name="org_roam", embedding_function=embeddings, persist_directory=persist_directory)

    vectordb.add_texts(list(df["text_to_encode"].values),
                       metadatas=[{"ID": df.iloc[i]["node_id"],
                                   "title": df.iloc[i]["node_title"],
                                   "hierarchy": df.iloc[i]["node_hierarchy"],
                                   "file_name": df.iloc[i]["file_name"]} for i in range(len(df))],
                       ids=list(df["node_id"]))

def split_nodes():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0) # parametrize these in env vars also

    for index, row in data.iterrows():
        org_id = row["node_id"]
        title = row["node_title"]
        file_name = row["file_name"]
        node_hierarchy = row["node_hierarchy"]
        texts = text_splitter.split_text(row["node_text_nested_exclusive"])
        texts = ["[" + node_hierarchy + "] " + text for text in texts]
        metadatas = [
            {
                "source": f"{index}-{i}",
                "ID": org_id,
                "title": title,
                "hierarchy": node_hierarchy,
                "file_name": file_name,
            }
            for i in range(len(texts))
        ]
        ids = [f"{index}-{i}" for i in range(len(texts))]
        vectordb.add_texts(texts, metadatas=metadatas, ids=ids)