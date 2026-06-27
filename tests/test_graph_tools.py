"""
Tests for src/tools/graph_tools.py routing logic.

routing_tool and doc_tool are pure functions of State, so these tests
need no mocking and make zero API calls.
"""

from src.tools.graph_tools import routing_tool, doc_tool, MAX_RETRIES


class TestRoutingTool:
    def test_routes_to_retriever_when_index(self):
        assert routing_tool({"route": "index"}) == "retriever"

    def test_routes_to_general_llm_when_general(self):
        assert routing_tool({"route": "general"}) == "general_llm"

    def test_routes_to_web_search_for_anything_else(self):
        assert routing_tool({"route": "search"}) == "web_search"
        assert routing_tool({"route": "anything_unrecognized"}) == "web_search"


class TestDocTool:
    def test_generates_when_score_is_yes(self):
        assert doc_tool({"binary_score": "yes", "retry_count": 0}) == "generate"

    def test_generates_when_score_is_yes_regardless_of_retry_count(self):
        # A late "yes" should still short-circuit straight to generate.
        assert doc_tool({"binary_score": "yes", "retry_count": MAX_RETRIES}) == "generate"

    def test_rewrites_when_score_is_no_and_retries_remain(self):
        assert doc_tool({"binary_score": "no", "retry_count": 0}) == "rewrite"

    def test_missing_retry_count_defaults_to_zero_and_rewrites(self):
        # state.get("retry_count") may be None on the very first pass.
        assert doc_tool({"binary_score": "no"}) == "rewrite"

    def test_gives_up_gracefully_once_max_retries_reached(self):
        """
        Regression test for the infinite-loop bug: previously doc_tool
        had no retry awareness at all, so a persistently irrelevant
        document would cause grade -> rewrite -> retriever -> grade to
        loop forever (confirmed by running it with score='no' fixed
        and watching it never terminate).
        """
        assert doc_tool({"binary_score": "no", "retry_count": MAX_RETRIES}) == "no_relevant_info"

    def test_never_loops_past_max_retries_across_many_iterations(self):
        decisions = []
        state = {"binary_score": "no", "retry_count": 0}
        for _ in range(20):
            decision = doc_tool(state)
            decisions.append(decision)
            if decision != "rewrite":
                break
            state["retry_count"] += 1

        assert decisions[-1] == "no_relevant_info"
        # Must terminate well before 20 iterations.
        assert len(decisions) <= MAX_RETRIES + 1
