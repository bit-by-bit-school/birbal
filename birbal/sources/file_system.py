from pathlib import Path
from datetime import datetime, timezone
import asyncio
from watchfiles import awatch, Change
from birbal.sources.base import Source, SourceStat
from birbal.config import config


class FileSystemSource(Source):
    def __init__(self, extension):
        self.source_dir = config["file_dir"]
        self.extension = extension

    def get_source_stats(self):
        root = Path(self.source_dir)
        files = root.rglob(f"*.{self.extension}")

        return [
            SourceStat(
                location=str(path),
                last_modified_at=datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ),
            )
            for path in files
        ]

    async def watch(self, upsert_fn, delete_fn):
        print(f"Watcher started on {self.source_dir}", flush=True)

        async for changes in awatch(self.source_dir):
            for change_type, path in changes:
                if not path.endswith(f".{self.extension}"):
                    continue

                if change_type in (Change.added, Change.modified):
                    print(f"Watcher detected update: {path}", flush=True)
                    upsert_fn(path)

                elif change_type == Change.deleted:
                    print(f"Watcher detected deletion: {path}", flush=True)
                    delete_fn(path)
