from birbal.models import *
from birbal.config import config

_embedder = None
_llm = None


def _get_provider(provider):
    match provider:
        case "ollama":
            return OllamaProvider()
        case "openai":
            return OpenAiProvider()
        case _:
            raise ValueError(f"Unsupported AI provider {provider}")


def get_embedder():
    embed_backend = config["embed_provider"]
    global _embedder
    if _embedder is None:
        _embedder = _get_provider(embed_backend).get_embedder()
    return _embedder


def get_llm():
    llm_backend = config["llm_provider"]
    global _llm
    if _llm is None:
        _llm = _get_provider(llm_backend).get_llm()
    return _llm


def query_llm(query, context):
    system_prompt = f"""
        You are a research assistant. Answer the user's question using ONLY the provided context. 
        If the answer is not in the context, state that you do not know.

        <context>
        {context}
        </context>

        At the end of your response, provide a section titled "SOURCES" listing only the sources used to answer the question.
        """

    llm = get_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    for chunk in llm.invoke(messages):
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]
