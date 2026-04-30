# This module runs an http server for querying the llm or vector db directly
import asyncio
import json
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from contextlib import asynccontextmanager
from birbal.sources import *
from birbal.ai import query_llm
from birbal.store import get_store, query_vector, query_by_id, query_similar_unlinked_by_id
from birbal.sync import sync_store, sync_file, delete_file_from_store
from birbal.config import config


async def _safety_net_poller():
    while True:
        print("Running periodic sync...")
        await asyncio.to_thread(sync_store)
        print("Periodic sync complete.")
        await asyncio.sleep(config["sync_interval"])


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await asyncio.to_thread(get_store)
    
    watcher_tasks = []
    for source in config["sources"]:
        fs = FileSystemSource(path=source["path"], formats=source["formats"])
        task = asyncio.create_task(fs.watch(sync_file, delete_file_from_store))
        watcher_tasks.append(task)

    poller_task = asyncio.create_task(_safety_net_poller())

    yield

    # Cleanup tasks
    for task in watcher_tasks:
        task.cancel()
    poller_task.cancel()


app = FastAPI(lifespan=_lifespan)


def run_query(query):
    retrieved_docs = query_vector(query)

    context_blocks = []
    id_lookup_map = {}
    
    for doc in retrieved_docs:
        root_title = doc["hierarchy"].split(" > ")[-1].strip()
        id_lookup_map[root_title] = doc["root_id"]
        content = doc["content"][doc["content"].find('\n') + 1:]
        
        formatted_block = f"""
        <document>
            <note_title>{root_title}</note_title>
            <content>
                {content}
            </content>
        </document>
        """

        context_blocks.append(formatted_block)
        
    docs_content = "\n".join(context_blocks)
    print(docs_content)
    
    full_response = ""
    for chunk in query_llm(query, docs_content):
        full_response += chunk
        yield chunk

    try:
        if "SOURCES" in full_response:
            sources_text = full_response.split("SOURCES")[-1].strip()
            source_titles = [line.strip("- *").strip() for line in sources_text.splitlines() if line.strip()]            
            valid_ids = [id_lookup_map[title] for title in source_titles if title in id_lookup_map]
            metadata_payload = f"\n\n===BIRBAL_METADATA===\n{json.dumps({'source_ids': valid_ids})}"
            yield metadata_payload

    except Exception as e:
        print(f"Error mapping sources: {e}")


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


@app.get("/similar")
def query(id: str = Query(..., min_length=1)):
    return query_similar_unlinked_by_id(id, 10)
