# This module takes a user query and generates a RAG context-fuelled response
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from birbal.ai import get_ai_provider
from birbal.query import query_vector


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = query_vector(last_query)

    docs_content = "\n\n".join(retrieved_docs)

    print(docs_content, flush=True)
    system_message = (
        "You are a retrieval-augmented assistant. Answer based ONLY on the context below, and do NOT hallucinate."
        f"\n\n{docs_content}"
    )

    return system_message


def query_llm(query):
    llm = get_ai_provider().get_llm()
    agent = create_agent(llm, tools=[], middleware=[prompt_with_context])
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        msg = step["messages"][-1]
        if msg.type == "ai":
            yield msg.content
