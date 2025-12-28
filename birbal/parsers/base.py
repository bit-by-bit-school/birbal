from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path

class DocumentParser(ABC):
    """
    Parses a single document into a normalized dataframe.
    """

    @abstractmethod
    def parse(self, path: Path) -> pd.DataFrame:
        ...
