from birbal.ingest import ingest
from birbal.server import app
from birbal.config import config
import uvicorn


def run_ingest():
    print("Starting ingestion...")
    ingest()
    print("Ingestion complete.")


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=config["port"])
