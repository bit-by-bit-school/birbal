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
        self.md_parser = MarkdownParser()

    def _get_url_hierarchy(self, url):
        """Converts /category/subfolder/page.html -> 'category > subfolder > page'"""
        path_parts = urlparse(url).path.strip("/").split("/")
        # Filter out empty segments and strip file extensions
        segments = [p.split(".")[0] for p in path_parts if p]
        return " > ".join(segments) if segments else "root"

    def parse_from_data(self, html):
        # 2. Clean and Extract Metadata
        metadata = trafilatura.extract_metadata(html)
        page_title = metadata.title if metadata and metadata.title else url

        # Extract clean main content and metadata using trafilatura, TODO choose one of md or xml
        md_content = (
            trafilatura.extract(html, output_format="markdown", include_comments=False)
            or html
        )

        clean_content = (
            trafilatura.extract(html, output_format="xml", include_comments=False)
            or html
        )

        # 3. Semantic Chunking
        passages = self.chunker.chunk(clean_content)

        # 3. Build the Hybrid Hierarchy
        url_hierarchy = self._get_url_hierarchy(url)
        hierarchy = f"{url_hierarchy} > {page_title}" if url_hierarchy else page_title

        # TODO choose one of the below returns
        return self.md_parser.parse(
            content=md_content,
            source_name=url_str,
            root_id=url_str,
            base_hierarchy=combined_base,
        )

        return pd.DataFrame(
            [
                {
                    "id": str(uuid.uuid4()),
                    "root_id": url,
                    "content": f"[{hierarchy}] {passage}",
                    "file_name": url,
                    "title": page_title,
                    "hierarchy": hierarchy,
                    "kind": "web_page",
                }
                for passage in passages
            ]
        )

    def parse_from_path(self, url):
        """
        Fetches a URL and chunks the HTML into a DataFrame.
        """
        try:
            with httpx.Client(follow_redirects=True, timeout=15.0) as client:
                response = client.get(url)
                response.raise_for_status()

            return parse_from_data(response.text)

        except Exception as e:
            print(f"Failed to parse html document for {url}: {e}")
            return pd.DataFrame()
