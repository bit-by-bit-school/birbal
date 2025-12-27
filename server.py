# This module runs an http server for querying the llm or vector db directly

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from llm import query_llm
from query import query_vector, query_by_id


app = FastAPI()


@app.get("/query")
def query(q: str = Query(..., min_length=1)):
    return StreamingResponse(query_llm(q), media_type="text/plain")


@app.get("/search", response_class=PlainTextResponse)
def query(id: str = Query(..., min_length=1)):
    retrieved_docs = query_by_id(id)
    docs_content = "\n\n".join(retrieved_docs)
    return docs_content
