# This module converts a provided data frame into a vector embedding and stores it

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import config


def embed_df(df):
    """
    Embed a Pandas DataFrame into a Chroma vector store.

    The input DataFrame MUST contain the following columns:

        - id (str)
            A unique, stable identifier for each row.
            This value is used as the Chroma document ID and MUST be unique across all rows.

        - text (str)
            The full text content that will be embedded and indexed.

        - title (str)
            A short human-readable title for the document or note.

        - file_name (str)
            The source filename associated with this document.

    It can additionally contain the following columns:

        - root_id (str)
            An identifier shared across subrows.
            This can be used to query for an entire subtree.

        - hierarchy (str)
            A series of node titles going from the current node through all its ancestors (if they exist) separated by >.
    """
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    vectordb = Chroma(
        collection_name=config["collection_name"],
        embedding_function=embeddings,
        persist_directory=config["persist_dir"],
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["text_split_chunk_size"],
        chunk_overlap=config["text_split_chunk_overlap"],
    )

    for index, row in df.iterrows():
        id = row["id"]
        root_id = row["root_id"] or id
        title = row["title"]
        file_name = row["file_name"]
        hierarchy = row["hierarchy"] or title
        texts = text_splitter.split_text(row["text"])
        texts = [texts[0]] + ["[" + hierarchy + "] " + t for t in texts[1:]]
        metadatas = [
            {
                "ID": id,
                "root_id": root_id,
                "title": title,
                "hierarchy": hierarchy,
                "file_name": file_name,
            }
            for i in range(len(texts))
        ]
        if len(texts) == 1:
            ids = [id]
        else:
            ids = [f"{id}.{i}" for i in range(len(texts))]

        vectordb.add_texts(texts, metadatas=metadatas, ids=ids)
