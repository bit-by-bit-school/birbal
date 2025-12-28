from langchain_ollama import OllamaEmbeddings, ChatOllama
from birbal.models.base import AIProvider
from birbal.config import config

class OllamaProvider(AIProvider):
    def get_embeddings(self):
        return OllamaEmbeddings(model=config["embedding_model"])

    def get_llm(self, *, stream=True):
        return ChatOllama(
            model=config["large_language_model"],
            temperature=0,
            stream=stream,
        )