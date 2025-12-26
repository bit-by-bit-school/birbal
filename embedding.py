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

        - hierarchy (str)
            A hierarchical path or outline location for the document
            (e.g. "Projects/AI/Embeddings").

        - file_name (str)
            The source filename associated with this document.
    """
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    vectordb = Chroma(
        collection_name="org_roam",
        embedding_function=embeddings,
        persist_directory=config["persist_dir"],
    )

    vectordb.add_texts(
        list(df["text_to_encode"].values),
        metadatas=[
            {
                "ID": df.iloc[i]["id"],
                "title": df.iloc[i]["title"],
                "hierarchy": df.iloc[i]["hierarchy"],
                "file_name": df.iloc[i]["file_name"],
            }
            for i in range(len(df))
        ],
        ids=list(df["id"]),
    )


def split_nodes(nodes):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["text_split_chunk_size"],
        chunk_overlap=config["text_split_chunk_overlap"],
    )

    for index, row in data.iterrows():
        org_id = row["id"]
        title = row["title"]
        file_name = row["file_name"]
        node_hierarchy = row["hierarchy"]
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
