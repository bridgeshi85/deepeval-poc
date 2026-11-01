import os
import unittest
from unittest.mock import patch

import arxiv

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

    def test_missing_tavily_key_has_clear_error(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(tools, "load_dotenv"),
        ):
            with self.assertRaisesRegex(ValueError, "TAVILY_API_KEY"):
                tools.get_tools()
