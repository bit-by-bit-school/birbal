import pandas as pd
import httpx
import uuid
from pathlib import Path
from google_labs_html_chunker.html_chunker import HtmlChunker
from urllib.parse import urlparse
import trafilatura


class HtmlParser(DocumentParser):
    def __init__(self):
        self.chunker = HtmlChunker(
            max_words_per_aggregate_passage=200,
            greedily_aggregate_sibling_nodes=True,
            html_tags_to_exclude={"noscript", "script", "style"},
        )

    def _get_url_hierarchy(self, url):
        """Converts /category/subfolder/page.html -> 'category > subfolder > page'"""
        path_parts = urlparse(url).path.strip("/").split("/")
        segments = [p.split(".")[0] for p in path_parts if p]
        return " > ".join(segments) if segments else "root"

    def _parse(self, html, source_url):
        """Private helper that contains the actual parsing logic and requires context."""
        # Clean and Extract Metadata
        metadata = trafilatura.extract_metadata(html)
        page_title = metadata.title if metadata and metadata.title else source_url

        # Extract clean main content
        clean_content = (
            trafilatura.extract(html, output_format="xml", include_comments=False)
            or html
        )

        # Semantic Chunking
        passages = self.chunker.chunk(clean_content)

        # Build the Hierarchy
        url_hierarchy = self._get_url_hierarchy(source_url)
        hierarchy = f"{url_hierarchy} > {page_title}" if url_hierarchy else page_title

        records = []
        for passage in passages:
            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "root_id": source_url,
                    "content": f"[{hierarchy}] {passage}",
                    "file_name": source_url,
                    "title": page_title,
                    "hierarchy": hierarchy,
                    "kind": "web_page",
                }
            )

        return pd.DataFrame(records)

    def parse_from_data(self, data):
        """
        Parses raw HTML string. Since no URL is provided via the interface,
        we use a placeholder URL to prevent the hierarchy builder from crashing.
        """
        return self._parse(html=data, source_url="raw_html_data")

    def parse_from_path(self, path):
        """
        Fetches a URL and chunks the HTML into a DataFrame.
        Here, `path` is treated as the URL.
        """
        try:
            with httpx.Client(follow_redirects=True, timeout=15.0) as client:
                response = client.get(path)
                response.raise_for_status()

            # Pass the URL into our private parser so we get proper metadata
            return self._parse(html=response.text, source_url=path)

        except Exception as e:
            print(f"Failed to parse html document for {path}: {e}")
            return pd.DataFrame()
