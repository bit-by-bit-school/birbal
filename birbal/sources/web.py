from usp.tree import sitemap_tree_for_homepage
from urllib.parse import urlparse
from datetime import datetime
from birbal.sources.base import Source, SourceStat
from birbal.config import config


class WebSource(Source):
    def __init__(self):
        self.domain = config["web_domain"].rstrip("/")  # TODO update config
        self.target_netloc = urlparse(self.domain_url).netloc

    def _is_same_domain(self, url):
        """Checks if the URL netloc matches the target domain."""
        try:
            return urlparse(url).netloc == self.target_netloc
        except Exception:
            return False

    def _determine_last_mod(self, page):
        # If last modified unavailable assume page needs updating.
        return page.last_modified if page.last_modified else datetime.now()

    def get_source_stats(self):
        """Fetches sitemap for domain and parses it to get SourceStats."""

        print(f"Crawling sitemap tree for: {self.domain_url}" )

        try:
            tree = sitemap_tree_for_homepage(self.domain_url)
            return [
                SourceStat(
                    location=page.url, last_modified_at=self._determine_last_mod(page)
                )
                for page in tree.all_pages()
                if self._is_same_domain(page.url)
            ]

        except Exception as e:
            print(f"Error parsing sitemaps for {self.domain_url}: {e}")

        return []

    async def watch(self, extension, upsert_fn, delete_fn):
        # Web does not support watching
        return None
