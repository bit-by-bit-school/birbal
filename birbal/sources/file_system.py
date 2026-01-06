from pathlib import Path
from datetime import datetime, timezone
from birbal.sources.base import Source, SourceStat
from birbal.config import config

class FileSystemSource(Source):
    def __init__(self):
        self.source_dir = config["file_dir"]

    def get_source_stats(self, extension):
        root = Path(config["file_dir"])
        files = root.rglob(f"*.{extension}")

        return [
            SourceStat(
                location = str(path),
                last_modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
            )
            for path in files
        ]
