import importlib
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from qa_agent import agents
from qa_agent.trace import log_event


class RunTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Compile the real graph without constructing network-backed models.
        with (
            patch.object(agents, "create_research_agent"),
            patch.object(agents, "create_critique_researcher"),
            patch.object(agents, "create_reflection_chain"),
        ):
            cls.runner = importlib.import_module("qa_agent.run")
            cls.graph_module = importlib.import_module("qa_agent.graph")

    async def run_sample(self, reflections):
        sample = {
            "question": "Example geography question?",
            "choices": ["North", "South", "East", "West"],
            "answer": 2,
        }
        researcher = Mock()
        researcher.invoke.return_value = {"messages": [AIMessage(content="A. Draft")]}
        reviser = Mock()
        reviser.invoke.return_value = {"messages": [AIMessage(content="C. Revised")]}
        reflector = Mock()
        reflector.invoke.side_effect = reflections
        output = io.StringIO()
        with (
            patch.object(self.runner, "load_dataset", return_value=[sample]) as load,
            patch.object(self.graph_module, "initial_researcher", researcher),
            patch.object(self.graph_module, "critique_researcher", reviser),
            patch.object(self.graph_module, "reflection_chain", reflector),
            patch.object(
                self.runner.graph, "astream", wraps=self.runner.graph.astream
            ) as stream,
            redirect_stdout(output),
        ):
            await self.runner.run()

        load.assert_called_once_with(
            "cais/mmlu", "high_school_geography", split="test"
        )
        stream.assert_called_once_with(
            {
                "question": sample["question"],
                "options": "A. North\nB. South\nC. East\nD. West",
            },
            stream_mode=["updates"],
        )
        self.assertIn("MMLU 标准答案：\nC. East", output.getvalue())
        self.assertIn("仅供人工核对", output.getvalue())
        self.assertIn("[更新: research_start]", output.getvalue())
        self.assertIn("[更新: reflect]", output.getvalue())
        return output.getvalue(), researcher, reviser, reflector

    async def test_approved_answer_ends_after_first_research(self):
        output, researcher, reviser, reflector = await self.run_sample(
            [agents.ReflectionResponse(answer="C. Confirmed")]
        )
        self.assertIn("Agent 答案：\nC. Confirmed", output)
        researcher.invoke.assert_called_once_with(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "题目：\nExample geography question?\n\n选项：\n"
                            "A. North\nB. South\nC. East\nD. West"
                        ),
                    }
                ]
            }
        )
        reviser.invoke.assert_not_called()
        self.assertEqual(reflector.invoke.call_count, 1)

    async def test_critique_reaches_researcher_and_final_answer_is_updated(self):
        output, _, reviser, reflector = await self.run_sample(
            [
                agents.ReflectionResponse(critique="Check the map"),
                agents.ReflectionResponse(answer="C. Confirmed"),
            ]
        )
        self.assertIn("Agent 答案：\nC. Confirmed", output)
        reviser.invoke.assert_called_once()
        prompt = reviser.invoke.call_args.args[0]["messages"][0]["content"]
        self.assertIn("A. Draft", prompt)
        self.assertIn("Check the map", prompt)
        self.assertEqual(reflector.invoke.call_count, 2)

    async def test_step_limit_keeps_latest_draft(self):
        output, _, reviser, reflector = await self.run_sample(
            [agents.ReflectionResponse(critique="More evidence")] * 3
        )
        self.assertIn("Agent 答案：\nC. Revised", output)
        self.assertEqual(reviser.invoke.call_count, 2)
        self.assertEqual(reflector.invoke.call_count, 3)

    async def test_missing_answer_is_an_explicit_failure(self):
        async def empty_stream(*args, **kwargs):
            yield "updates", {"reflect": {"feedback": "No answer"}}

        with (
            patch.object(
                self.runner,
                "load_dataset",
                return_value=[{"question": "Q", "choices": ["X", "Y"], "answer": 0}],
            ),
            patch.object(self.runner.graph, "astream", side_effect=empty_stream),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "未产生 Agent 答案"):
                await self.runner.run()


class TraceTests(unittest.TestCase):
    def test_tool_calls_and_feedback_are_logged(self):
        output = io.StringIO()
        with redirect_stdout(output):
            log_event(
                "research",
                {
                    "feedback": "Check the map",
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "search",
                                    "args": {"query": "geography"},
                                    "id": "call_1",
                                }
                            ],
                        )
                    ],
                },
            )
        self.assertIn("Check the map", output.getvalue())
        self.assertIn("工具调用: search 参数: {'query': 'geography'}", output.getvalue())


if __name__ == "__main__":
    unittest.main()
