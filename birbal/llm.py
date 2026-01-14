# This module takes a user query and generates a RAG context-fuelled response
from birbal.ai import get_ai_provider
from birbal.query import query_vector


def query_llm(query):
    retrieved_docs = query_vector(query)
    docs_content = "\n\n".join(retrieved_docs)
    print(docs_content, flush=True)

    system_prompt = f"""
        You are a research assistant. Answer the user's question using ONLY the provided context. 
        If the answer is not in the context, state that you do not know.

        <context>
        {docs_content}
        </context>

        At the end of your response, provide a section titled "SOURCES" listing only the sources used to answer the question.
        """
    
    llm = get_ai_provider().get_llm(stream=True)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    for chunk in llm.invoke(messages):
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]
