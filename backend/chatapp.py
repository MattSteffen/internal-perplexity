import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
import requests
from typing import List, Optional
import json

TOOL_SERVER_URL = "http://localhost:8000"


@tool
def milvus_search(queries: List[str], filters: Optional[List[str]] = None) -> str:
    """Performs a semantic search using the given queries and optional filters."""
    try:
        response = requests.post(
            f"{TOOL_SERVER_URL}/search",
            json={"queries": queries, "filters": filters or []},
        )
        response.raise_for_status()
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"


@tool
def milvus_query(filters: List[str]) -> str:
    """Runs a filtered query without semantic search. Only filters are used."""
    try:
        response = requests.post(
            f"{TOOL_SERVER_URL}/query",
            json={"filters": filters},
        )
        response.raise_for_status()
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"


@tool
def comb(perspective: str, filters: Optional[List[str]] = None) -> str:
    """Iteratively reads through documents in the database, collecting tidbits that might be related to the user's desires from a given perspective."""
    try:
        response = requests.post(
            f"{TOOL_SERVER_URL}/comb",
            json={"perspective": perspective, "filters": filters or []},
        )
        response.raise_for_status()
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"


def get_agent():
    llm = ChatOllama(model="qwen3:1.7b", temperature=0)
    tools = [milvus_search, milvus_query, comb]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    st.title("LLM Tool Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            agent = get_agent()
            response = agent.invoke({"input": prompt})
            st.markdown(response["output"])
        st.session_state.messages.append(
            {"role": "assistant", "content": response["output"]}
        )


if __name__ == "__main__":
    main()