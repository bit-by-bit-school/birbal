import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from query import query_vector

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = query_vector(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    print(docs_content)
    system_message = (
        "You are a retrieval-augmented assistant. Answer based ONLY on the context below, and do NOT hallucinate."
        f"\n\n{docs_content}"
    )

    return system_message


def query_llm(query):
    llm_model = os.getenv('LARGE_LANGUAGE_MODEL')
    llm = ChatOllama(
        model=llm_model,
        temperature=0,
        # other params...
    )
    agent = create_agent(llm, tools=[], middleware=[prompt_with_context])
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()