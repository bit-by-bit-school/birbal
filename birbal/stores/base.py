from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Set, Dict, Any
from datetime import datetime


@dataclass(frozen=True)
class FileStat:
    file_name: str
    last_indexed_at: datetime


class VectorStore(ABC):
    """
    Abstract interface for vector-backed knowledge stores.
    """

    @abstractmethod
    def upsert_nodes(
        self, nodes: List[Dict[str, Any]]
    ) -> None: ...

    @abstractmethod
    def delete_by_filenames(self, filenames: Set[str]) -> None: ...

    @abstractmethod
    def get_file_stats(self) -> List[FileStat]: ...

    @abstractmethod
    def similarity_search(self, query_str: str) -> List[str]: ...

    @abstractmethod
    def filter_by_metadata(
        self, metadata_field: str, metadata_value: str
    ) -> List[str]: ...
