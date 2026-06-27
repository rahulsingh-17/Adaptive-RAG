"""
ReAct agent setup for document retrieval and question answering.
"""

import os

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import Config
from src.llms.openai import llm
from src.rag.retriever_setup import get_retriever

config = Config()


def build_agent_executor() -> AgentExecutor:
    """
    Build a ReAct agent executor wired to the CURRENT retriever tool.

    This is rebuilt on every call rather than cached once at import
    time. Caching it at import time was a bug: get_retriever() returns
    a tool bound to whatever vectorstore exists *at that moment*, and
    document_upload.py later swaps in a brand new vectorstore. A
    module-level agent_executor built before any upload would keep
    answering from the empty/dummy vectorstore forever, even after
    real documents were uploaded (see tests/test_retriever_freshness.py).

    Returns:
        A freshly constructed AgentExecutor using the latest retriever.
    """
    tools = [get_retriever()]

    prompt = ChatPromptTemplate.from_messages([
        ("system", config.prompt("system_prompt")),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])

    react_agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=2,
        verbose=True,
        return_intermediate_steps=True
    )
