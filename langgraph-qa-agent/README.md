# langgraph-qa-agent（选题 4，待开发）

**选题 4 —— 跟着 Packt 官方书，30 分钟搭一个能跑的 LangGraph Agent**

以 `references/generative_ai_with_langchain/chapter6/question_answering.ipynb` 为蓝本
（Research Agent + Reflection Agent + LangGraph 状态机），做一个 MMLU 多选题问答 Agent。
模型驱动换成 Qwen / DashScope（复用 `../deepeval-local-llm-poc/custom_llm.py` 的
`ChatOpenAI` + DashScope base_url 写法）。

详细的需求 / 设计 / 开发步骤见 [PLAN.md](PLAN.md)。
选题背景见 [`../docs/content-plan-agent-eval-series.md`](../docs/content-plan-agent-eval-series.md)
的「附录：选题 4 的最终方案」。

状态：目录占位，尚未开始编码。
