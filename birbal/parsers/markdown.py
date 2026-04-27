import pandas as pd
import uuid
from pathlib import Path
from typing import Union
from langchain_text_splitters import MarkdownHeaderTextSplitter


class MarkdownParser(DocumentParser):
    def __init__(self):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on, strip_headers=False
        )

    def _build_doc_hierarchy(self, metadata: dict) -> str:
        """Flattens Header metadata into 'H1 > H2 > H3'"""
        levels = ["Header 1", "Header 2", "Header 3"]
        parts = [metadata.get(lvl) for lvl in levels if metadata.get(lvl)]
        return " > ".join(parts)

    def parse(
        self, content: str, source_name: str, root_id: str, base_hierarchy: str = ""
    ) -> pd.DataFrame:
        """
        Parses markdown string into a DataFrame.
        'base_hierarchy' allows us to prepend the URL path or File path.
        """
        chunks = self.splitter.split_text(content)

        records = []
        for chunk in chunks:
            # Get the internal document structure (H1 > H2)
            doc_hierarchy = self._build_doc_hierarchy(chunk.metadata)

            # Combine external (URL/File path) with internal (Headings)
            full_hierarchy = f"{base_hierarchy} > {doc_hierarchy}".strip(" > ")

            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "root_id": root_id,
                    "content": f"[{full_hierarchy}] {chunk.page_content}",
                    "file_name": source_name,
                    "hierarchy": full_hierarchy,
                    "kind": "markdown",
                }
            )

        return pd.DataFrame(records)
