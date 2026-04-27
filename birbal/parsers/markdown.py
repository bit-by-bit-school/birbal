import os
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

    def _parse(self, content: str, source_name: str, root_id: str) -> pd.DataFrame:
        """
        Parses markdown string into a DataFrame.
        """
        chunks = self.splitter.split_text(content)

        records = []
        for chunk in chunks:
            # Get the internal document structure (H1 > H2)
            doc_hierarchy = self._build_doc_hierarchy(chunk.metadata)

            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "root_id": root_id,
                    "content": f"[{doc_hierarchy}] {chunk.page_content}" if doc_hierarchy else chunk.page_content,
                    "file_name": source_name,
                    "hierarchy": doc_hierarchy,
                    "kind": "markdown",
                }
            )

        return pd.DataFrame(records)

    def parse_from_path(self, path: str) -> pd.DataFrame:
        # Read the file content using standard python
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the file name without the extension to use as the source_name
        file_name_with_ext = os.path.basename(path)
        file_name_no_ext, _ = os.path.splitext(file_name_with_ext)
        
        # Extract the file name with the extension to use as the root_id
        root_id = file_name_with_ext
        
        # Call the private _parse helper
        return self._parse(content=content, source_name=file_name_no_ext, root_id=root_id)

    def parse_from_data(self, data: str) -> pd.DataFrame:
        # This method is called when raw string data is provided without a file path
        # Use default fallback IDs
        return self._parse(content=data, source_name="markdown_data", root_id="markdown_data.md")
