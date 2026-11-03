import os
import unittest
from unittest.mock import Mock, patch

import arxiv
import requests
from langchain_core.tools import Tool

from qa_agent import tools


class ToolsTests(unittest.TestCase):
    def test_arxiv_search_supports_langchain_wrapper(self):
        self.assertTrue(callable(getattr(arxiv.Search, "results", None)))

    def test_tavily_replaces_duckduckgo_without_network_call(self):
        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "verification-only"}),
            patch.object(tools, "load_dotenv"),
        ):
            loaded = tools.get_tools()

        self.assertEqual([tool.name for tool in loaded], ["tavily_search", "arxiv", "wikipedia"])
        self.assertTrue(all(tool.handle_tool_error for tool in loaded))
        self.assertEqual(loaded[0].search_depth, "basic")
        self.assertEqual(loaded[0].max_results, 3)
        self.assertFalse(loaded[0].include_raw_content)

    def test_tavily_sets_http_timeout(self):
        wrapper = tools.BoundedTavilyAPIWrapper(tavily_api_key="verification-only")
        with patch.object(tools.requests, "post") as post:
            post.return_value.json.return_value = {"results": [{"title": "Evidence"}]}
            result = wrapper.raw_results(query="geography", max_results=3)
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 10))
        self.assertEqual(post.call_args.kwargs["json"], {"query": "geography", "max_results": 3})
        post.return_value.raise_for_status.assert_called_once()
        self.assertTrue(result["results"])

    def test_timeout_does_not_retry(self):
        search = Mock(side_effect=requests.exceptions.Timeout("slow search"))
        tool = tools._make_resilient(Tool(name="search", description="Search", func=search))
        with patch.object(tools.time, "sleep") as sleep:
            result = tool.invoke("query")
        search.assert_called_once()
        sleep.assert_not_called()
        self.assertIn("暂时不可用", result)

    def test_tavily_error_response_is_not_treated_as_success(self):
        search = Mock(return_value={"error": requests.exceptions.Timeout("slow search")})
        tool = tools._make_resilient(Tool(name="search", description="Search", func=search))
        self.assertIn("暂时不可用", tool.invoke("query"))
        search.assert_called_once()

    def test_transient_error_still_retries(self):
        search = Mock(side_effect=[ConnectionError("temporary"), "Evidence"])
        tool = tools._make_resilient(Tool(name="search", description="Search", func=search))
        with patch.object(tools.time, "sleep") as sleep:
            self.assertEqual(tool.invoke("query"), "Evidence")
        self.assertEqual(search.call_count, 2)
        sleep.assert_called_once_with(3)

    def test_missing_tavily_key_has_clear_error(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(tools, "load_dotenv"),
        ):
            with self.assertRaisesRegex(ValueError, "TAVILY_API_KEY"):
                tools.get_tools()
