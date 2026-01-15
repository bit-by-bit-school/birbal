# Defines the API for AI Providers
from abc import ABC, abstractmethod
from typing import List, Union, Generator, Any


class Embedder(ABC):
    """Interface for all embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        pass


class LLM(ABC):
    """Interface for all Large Language Models."""

    @abstractmethod
    def invoke(self, messages: List[dict]) -> Union[str, Generator[Any, None, None]]:
        """Execute a chat completion. Returns string if sync, generator if streaming."""
        pass


class AIProvider(ABC):
    """
    Abstract interface for LLM / embedding providers.
    """

    @abstractmethod
    def get_embedder(self) -> Embedder:
        pass

    @abstractmethod
    def get_llm(self) -> LLM:
        pass
