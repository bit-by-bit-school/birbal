from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from llm import query_llm

app = FastAPI()

@app.get("/query")
def query(q: str = Query(..., min_length=1)):
    return StreamingResponse(
        query_llm(q),
        media_type="text/plain"
    )
