from pathlib import Path
import logging
from typing import List
from watchfiles import awatch, Change
from birbal.sources.base import Source, SourceStat
from birbal.config import config
from datetime import datetime, timezone

class FileSystemSource(Source):
    def __init__(self, path: str, formats: List[str]):
        self.source_dir = Path(path).resolve()
        self.formats = [fmt.lower() for fmt in formats]
        
    def get_source_stats(self) -> List[SourceStat]:
        root = self.source_dir
        files = [file for ext in self.formats for file in root.rglob(f"*.{ext}")]
        
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
        try:
            print(f"Watcher started on {self.source_dir}")
            
            allowed_extensions = set(f".{ext}" for ext in self.formats)
            
            async for changes in awatch(str(self.source_dir)):
                for change_type, path_str in changes:
                    path = Path(path_str)
                    
                    if not path.is_file():
                        continue
                        
                    # Skip files that don't match allowed formats
                    ext = f".{path.name.split('.')[-1].lower()}"
                    if ext not in allowed_extensions:
                        continue

                    # Handle events
                    if change_type in (Change.added, Change.modified):
                        upsert_fn(str(path))
                    elif change_type == Change.deleted:
                        delete_fn(str(path))
        except Exception as e:
            logging.error(f"Error in file watcher: {str(e)}")
