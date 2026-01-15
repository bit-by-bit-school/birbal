# This module runs an http server for querying the llm or vector db directly
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
import asyncio
from contextlib import asynccontextmanager
from birbal.sources import *
from birbal.ai import query_llm
from birbal.store import query_vector, query_by_id
from birbal.sync import sync_store, sync_file, delete_file_from_store
from birbal.config import config


async def _safety_net_poller():
    while True:
        print("Running periodic sync...", flush=True)
        await asyncio.to_thread(sync_store)
        print("Periodic sync complete.", flush=True)
        await asyncio.sleep(config("sync_interval"))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    fs = FileSystemSource("org")
    watcher_task = asyncio.create_task(fs.watch(sync_file, delete_file_from_store))
    poller_task = asyncio.create_task(_safety_net_poller())

    yield

    watcher_task.cancel()
    poller_task.cancel()


app = FastAPI(lifespan=_lifespan)


def run_query(query):
    retrieved_docs = query_vector(query)
    docs_content = "\n\n".join(retrieved_docs)
    print(docs_content, flush=True)
    query_llm(query, docs_content)


@app.get("/query")
def query(q: str = Query(..., min_length=1)):
    return StreamingResponse(
        run_query(q),
        media_type="text/plain",
    )


@app.get("/search", response_class=PlainTextResponse)
def query(id: str = Query(..., min_length=1)):
    retrieved_docs = query_by_id(id)
    docs_content = "\n\n".join(retrieved_docs)
    return docs_content
