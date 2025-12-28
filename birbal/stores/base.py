from abc import ABC, abstractmethod
from typing import List


class VectorStore(ABC):
    """
    Abstract interface for vector-backed knowledge stores.
    """

    @abstractmethod
    def add_texts(
        self, texts: List[str], metadatas: List[dict], ids: List[str]
    ) -> None: ...

    @abstractmethod
    def similarity_search(self, query_str: str) -> List[str]: ...

    @abstractmethod
    def filter_by_metadata(
        self, metadata_field: str, metadata_value: str
    ) -> List[str]: ...
