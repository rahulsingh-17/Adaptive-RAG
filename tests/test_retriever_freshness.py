"""
Regression test for the "stale retriever tool" bug.

Bug: src/rag/reAct_agent.py used to build its AgentExecutor's tools
list ONCE at import time, by calling get_retriever() before any
documents existed. document_upload.py later swaps in a brand new FAISS
vectorstore on upload, but the already-built AgentExecutor kept
pointing at the original (empty/dummy) one — so uploaded documents
were invisible to the agent that actually answers questions, even
though the query classifier (which calls get_retriever() fresh every
time) correctly thought retrieval should be used.

Confirmed before the fix: querying the agent's tool after uploading a
document still returned the dummy placeholder text
"No documents have been uploaded yet. Please upload a document first."
instead of the uploaded content.

This test builds the agent the way production code does (build_agent_executor)
and checks it sees newly uploaded content immediately, with zero real
API calls (embeddings are faked).
"""

from langchain_core.documents import Document

import src.rag.retriever_setup as retriever_setup
from src.rag.reAct_agent import build_agent_executor


def test_freshly_built_agent_sees_documents_uploaded_after_startup(fake_embeddings):
    # Step 1: "app starts up" — nothing uploaded yet, agent built once.
    first_agent = build_agent_executor()
    first_tool = first_agent.tools[0]
    first_answer = first_tool.func("anything")
    assert "No documents have been uploaded yet" in first_answer

    # Step 2: a document is uploaded — retriever_chain swaps in a new
    # vectorstore, exactly like document_upload.py does.
    real_doc = [Document(
        page_content="The secret launch code is ALPHA-NOVEMBER-7.",
        metadata={},
    )]
    assert retriever_setup.retriever_chain(real_doc) is True

    # Step 3: build the agent again (this happens fresh on every
    # /rag/query call via retriever_node) — it must now see the upload.
    second_agent = build_agent_executor()
    second_tool = second_agent.tools[0]
    second_answer = second_tool.func("launch code")

    assert "ALPHA-NOVEMBER-7" in second_answer
    assert "No documents have been uploaded yet" not in second_answer


def test_two_calls_to_build_agent_executor_without_upload_both_use_dummy_store(fake_embeddings):
    """Sanity check: without any upload, repeated calls stay consistent."""
    agent_a = build_agent_executor()
    agent_b = build_agent_executor()

    answer_a = agent_a.tools[0].func("anything")
    answer_b = agent_b.tools[0].func("anything")

    assert "No documents have been uploaded yet" in answer_a
    assert "No documents have been uploaded yet" in answer_b
