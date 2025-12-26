# This module converts a provided data frame into a vector embedding and stores it

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import config

def embed_df(df):
    embeddings = OllamaEmbeddings(model=config['embedding_model'])
    vectordb = Chroma(collection_name="org_roam", embedding_function=embeddings, persist_directory=config['persist_dir'])

    vectordb.add_texts(list(df["text_to_encode"].values),
                       metadatas=[{"ID": df.iloc[i]["node_id"],
                                   "title": df.iloc[i]["node_title"],
                                   "hierarchy": df.iloc[i]["node_hierarchy"],
                                   "file_name": df.iloc[i]["file_name"]} for i in range(len(df))],
                       ids=list(df["node_id"]))

def split_nodes():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['text_split_chunk_size'], 
        chunk_overlap=config['text_split_chunk_overlap'])

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