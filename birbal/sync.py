# This module syncs store state to source state
import pandas as pd
from typing import List, Set
import os
from pathlib import Path
from birbal.config import config
from birbal.store import get_store
from birbal.sources import FileSystemSource
from birbal.parsers.base import get_parser_for_extension
from birbal.embedding import ingest_dataframe


def _ingest_files(paths: List[str], store):
    try:
        dataframes = []
        for path in paths:
            extension = Path(path).suffix.lstrip(".")
            parser = get_parser_for_extension(extension)

            if parser is None:
                print(f"Skipping {path} due to unsupported extension {extension}")
                continue

            df = parser.parse_from_path(str(path))
            
            if df is not None and not df.empty:
                dataframes.append(df)

        if dataframes:
            accumulated_df = pd.concat(dataframes, ignore_index=True)
            ingest_dataframe(accumulated_df)

    except Exception as e:
        print(f"Error ingesting files: {e}")


def _delete_orphaned_nodes(db_stats, local_stats, store):
    db_filenames = {s.file_name for s in db_stats}
    local_filenames = {s.location for s in local_stats}
    orphaned = db_filenames - local_filenames

    if orphaned:
        print(f"Deleting {len(orphaned)} nodes for missing files")
        store.delete_by_filenames(orphaned)


def _update_stale_nodes(db_stats, local_stats, store):
    db_map = {s.file_name: s for s in db_stats}
    local_map = {s.location: s for s in local_stats}

    stale = {
        fname
        for fname in (set(local_map.keys()) & set(db_map.keys()))
        if local_map[fname].last_modified_at > db_map[fname].last_indexed_at
    }

    if stale:
        print(f"Re-indexing {len(stale)} modified files")
        store.delete_by_filenames(stale)
        _ingest_files(stale, store)


def _ingest_new_files(db_stats, local_stats, store):
    db_filenames = {s.file_name for s in db_stats}
    local_filenames = {s.location for s in local_stats}
    new = local_filenames - db_filenames

    if new:
        print(f"Ingesting {len(new)} files")
        _ingest_files(new, store)


def sync_file(path):
    store = get_store()
    store.delete_by_filenames([path])
    _ingest_files([path], store)


def delete_file_from_store(path):
    store = get_store()
    store.delete_by_filenames([path])


def sync_store():
    db = get_store()
    db_stats = db.get_file_stats()
    
    all_local_stats = []
    
    for source in config["sources"]:
        fs = FileSystemSource(path=source["path"], formats=source["formats"])
        all_local_stats.extend(fs.get_source_stats())

    print("Syncing...")
    _delete_orphaned_nodes(db_stats, all_local_stats, db)
    _update_stale_nodes(db_stats, all_local_stats, db)
    _ingest_new_files(db_stats, all_local_stats, db)
    print("Sync complete.")