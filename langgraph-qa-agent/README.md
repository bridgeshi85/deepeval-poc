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

## Agent API 兼容性

本项目使用 LangChain 1.x。现有研究员使用的 `create_tool_calling_agent` 和
`AgentExecutor` 已迁移到 `langchain_classic.agents`，因此依赖中显式包含
`langchain-classic`。无需降级 `langchain`。

在项目目录安装依赖后，可运行研究员和反思链的本地示例（需要配置模型环境变量，
且会调用外部模型和检索服务）：

```powershell
python -m pip install -r .\requirements.txt
python .\qa_agent\agents.py
```

## 写作参考：LangChain v1 的 Agent API 迁移

官方来源、迁移对照及文章切入点已单独整理到
[Agent API 迁移备忘](../docs/langchain-agent-migration-memo.md)。
