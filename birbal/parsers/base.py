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
