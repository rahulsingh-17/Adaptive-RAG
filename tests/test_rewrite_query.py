"""
Tests for src/rag/graph_builder.py's rewrite_query node.

The LLM call inside rewrite_query is mocked so this makes zero real
API calls — we're testing the retry_count bookkeeping, not the LLM's
rewriting ability.
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.rag.graph_builder import rewrite_query


def _fake_chain_invoke(self, *args, **kwargs):
    return AIMessage(content="rewritten query text")


class TestRewriteQueryRetryCount:
    def test_first_rewrite_sets_retry_count_to_one(self):
        with patch("langchain_core.runnables.base.RunnableSequence.invoke", _fake_chain_invoke):
            result = rewrite_query({"latest_query": "original query", "retry_count": 0})

        assert result["retry_count"] == 1
        assert result["latest_query"] == "rewritten query text"

    def test_retry_count_increments_from_existing_value(self):
        with patch("langchain_core.runnables.base.RunnableSequence.invoke", _fake_chain_invoke):
            result = rewrite_query({"latest_query": "original query", "retry_count": 1})

        assert result["retry_count"] == 2

    def test_missing_retry_count_defaults_to_zero_before_incrementing(self):
        with patch("langchain_core.runnables.base.RunnableSequence.invoke", _fake_chain_invoke):
            result = rewrite_query({"latest_query": "original query"})

        assert result["retry_count"] == 1
