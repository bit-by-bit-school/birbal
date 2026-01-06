# This module syncs store state to file_dir state
import os
import glob
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from birbal.parsers import *
from birbal.sources import *
from birbal.embedding import embed_df
from birbal.store import get_store
from birbal.config import config


def ingest_files(paths, store):
    parser = OrgParser()
    accumulated_df = pd.concat([parser.parse(path) for path in paths])
    embed_df(accumulated_df)


def delete_orphaned_nodes(db_stats, local_stats, store):
    db_filenames = {s.file_name for s in db_stats}
    local_filenames = {s.location for s in local_stats}
    orphaned = db_filenames - local_filenames

    if orphaned:
        print(f"Deleting {len(orphaned)} nodes for missing files", flush=True)
        store.delete_by_filenames(orphaned)


def update_stale_nodes(db_stats, local_stats, store):
    db_map = {s.file_name: s for s in db_stats}
    local_map = {s.location: s for s in local_stats}

    stale = {
        fname
        for fname in (set(local_map.keys()) & set(db_map.keys()))
        if local_map[fname].last_modified_at > db_map[fname].last_indexed_at
    }

    if stale:
        print(f"Re-indexing {len(stale)} modified files", flush=True)
        store.delete_by_filenames(stale)
        ingest_files(stale, store)
        # could wrap in transaction


def ingest_new_files(db_stats, local_stats, store):
    db_filenames = {s.file_name for s in db_stats}
    local_filenames = {s.location for s in local_stats}
    new = local_filenames - db_filenames

    if new:
        print(f"Ingesting {len(new)} files", flush=True)
        ingest_files(new, store)


def sync_store():
    db = get_store()
    fs = FileSystemSource()
    db_stats = db.get_file_stats()
    local_stats = fs.get_source_stats("org")

    print("Syncing...", flush=True)
    delete_orphaned_nodes(db_stats, local_stats, db)
    update_stale_nodes(db_stats, local_stats, db)
    ingest_new_files(db_stats, local_stats, db)
    print("Sync complete.", flush=True)
