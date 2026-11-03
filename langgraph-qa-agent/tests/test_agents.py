import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import Tool
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError

from qa_agent import agents


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class AgentBudgetTests(unittest.TestCase):
    def test_researchers_limit_searches_and_can_still_answer(self):
        for factory in (agents.create_research_agent, agents.create_critique_researcher):
            with self.subTest(factory=factory.__name__):
                search = Mock(return_value="Evidence")
                tool = Tool(name="search", description="Search", func=lambda query: search(query))
                model = ToolCallingFakeModel(responses=[
                    AIMessage(content="", tool_calls=[
                        {"name": "search", "args": {"tool_input": "query"}, "id": str(i)}
                        for i in range(5)
                    ]),
                    AIMessage(content="C. Final answer"),
                ])
                with patch.object(agents, "get_llm", return_value=model), patch.object(
                    agents, "get_tools", return_value=[tool]
                ):
                    agent = factory()
                self.assertEqual(agents._ask(agent, "Question"), "C. Final answer")
                self.assertEqual(search.call_count, 3)

    def test_model_loop_fails_explicitly_instead_of_returning_budget_text_as_answer(self):
        search = Mock(return_value="Evidence")
        model = ToolCallingFakeModel(responses=[AIMessage(content="", tool_calls=[
            {"name": "search", "args": {"tool_input": "query"}, "id": str(i)}
        ]) for i in range(4)])
        with patch.object(agents, "get_llm", return_value=model), patch.object(
            agents, "get_tools", return_value=[Tool(name="search", description="Search", func=lambda query: search(query))]
        ):
            agent = agents.create_research_agent()
        with self.assertRaises(ModelCallLimitExceededError):
            agents._ask(agent, "Question")
        self.assertEqual(search.call_count, 3)
