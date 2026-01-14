import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Callable
from datetime import datetime


@dataclass(frozen=True)
class SourceStat:
    location: str
    last_modified_at: datetime


class Source(ABC):
    """
    Abstract interface for knowledge sources.
    """

    @abstractmethod
    def get_source_stats(self) -> List[SourceStat]: ...

    @abstractmethod
    async def watch(
        self,
        upsert_fn: Callable[[str], None],
        delete_fn: Callable[[str], None],
    ) -> None: ...
