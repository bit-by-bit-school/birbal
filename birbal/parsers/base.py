from abc import ABC, abstractmethod
import pandas as pd

class DocumentParser(ABC):
    """
    Parses a single document into a normalized dataframe.
    """

    @abstractmethod
    def parse_from_path(self, path: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def parse_from_data(self, data: str) -> pd.DataFrame:
        ...

class DocumentParserRegistry:
    @staticmethod
    def get_parser_for_extension(extension: str) -> 'DocumentParser':
        extension = extension.lower()
        if extension in ("org", ".org"):
            from .org import OrgParser
            return OrgParser()
        elif extension in ("md", "markdown", ".md"):
            from .markdown import MarkdownParser
            return MarkdownParser()
        elif extension in ("html", ".html"):
            from .html import HtmlParser
            return HtmlParser()
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

__all__ = ["DocumentParserRegistry"]
