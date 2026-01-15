from mlx_lm import load, generate, stream_generate
from birbal.models.base import AIProvider
from birbal.config import config


class MlxEmbedder:
    def __init__(self, model, dimensions, **kwargs):
        self.model, self.tokenizer = load(model)
        self.dimensions = dimensions
        self.options = kwargs

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            tokens = self.tokenizer.encode(text, return_tensors="mx")
            output = self.model.embed_text(tokens)
            vector = output.mean(axis=1).tolist()[0]
            embeddings.append(vector[: self.dimensions])
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class MlxLLM:
    def __init__(self, model, stream, **kwargs):
        self.model, self.tokenizer = load(model)
        self.stream = stream
        self.options = kwargs

    def invoke(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        if self.stream:
            return stream_generate(self.model, self.tokenizer, prompt, **self.options)

        return generate(self.model, self.tokenizer, prompt, **self.options)


# cannot run via Docker, only locally
class MlxProvider(AIProvider):
    def get_embedder(self):
        return MlxEmbedder(
            model=config["embed_model"], dimensions=config["vector_dims"]
        )

    def get_llm(self):
        return MlxLLM(
            model=config["large_language_model"],
            stream=True,
            max_tokens=config["context_window_size"],
            temp=0.0,
        )
