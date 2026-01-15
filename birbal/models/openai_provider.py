from openai import OpenAI
from birbal.models.base import AIProvider, Embedder, LLM
from birbal.config import config


class OpenAiEmbedder(Embedder):
    def __init__(self, model, dimensions, **kwargs):
        self.client = OpenAI(base_url=config["embed_host"], api_key=config["embed_api_key"])
        self.model = model
        self.dimensions = dimensions
        self.settings = kwargs

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=self.model, input=texts, encoding_format="float", **self.settings
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAiLLM(LLM):
    def __init__(self, model, stream, **kwargs):
        self.client = OpenAI(base_url=config["llm_host"], api_key=config["llm_api_key"])
        self.model = model
        self.stream = stream
        self.settings = kwargs

    def invoke(self, messages):
        return self.client.chat.completions.create(
            model=self.model, messages=messages, stream=self.stream, **self.settings
        )


class OpenAiProvider(AIProvider):
    def get_embedder(self):
        return OpenAiEmbedder(
            model=config["embed_model"], dimensions=config["vector_dims"]
        )

    def get_llm(self):
        return OpenAiLLM(model=config["large_language_model"], stream=True)
