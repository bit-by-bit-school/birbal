from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from llm import query_llm

app = FastAPI()

@app.get("/query", response_class=PlainTextResponse)
def query(q: str = Query(..., min_length=1)):
    result = query_llm(q)
    return result