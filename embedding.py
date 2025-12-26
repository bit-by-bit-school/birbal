# This module converts a provided data frame into a vector embedding and stores it

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import config


def embed_df(df):
    """
    Embed a Pandas DataFrame into a Chroma vector store.

    Required DataFrame schema:

    The input DataFrame MUST contain the following columns:

        - id (str)
            A unique, stable identifier for each row.
            This value is used as the Chroma document ID and MUST be unique across all rows.

        - text_to_encode (str)
            The full text content that will be embedded and indexed.

        - title (str)
            A short human-readable title for the document or note.

        - file_name (str)
            The source filename associated with this document.

    It can additionally contain the following columns:

        - hierarchy (str)
            A list of node titles going from the current node to its ancestor (if they exist).
    """
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    vectordb = Chroma(
        collection_name="notes",
        embedding_function=embeddings,
        persist_directory=config["persist_dir"],
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["text_split_chunk_size"],
        chunk_overlap=config["text_split_chunk_overlap"],
    )

    for index, row in df.iterrows():
        id = row["id"]
        title = row["title"]
        file_name = row["file_name"]
        hierarchy = row["hierarchy"] or row["title"]
        texts = text_splitter.split_text(row["text_to_encode"])
        texts = ["[" + hierarchy + "] " + text for text in texts]
        metadatas = [
            {
                "ID": id,
                "title": title,
                "hierarchy": hierarchy,
                "file_name": file_name,
            }
            for i in range(len(texts))
        ]
        ids = [f"{id}-{i}" for i in range(len(texts))]
        vectordb.add_texts(texts, metadatas=metadatas, ids=ids)
