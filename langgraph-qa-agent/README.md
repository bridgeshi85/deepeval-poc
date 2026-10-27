# langgraph-qa-agent（选题 4）

**选题 4 —— 跟着 Packt 官方书，30 分钟搭一个能跑的 LangGraph Agent**

以 `references/generative_ai_with_langchain/chapter6/question_answering.ipynb` 为蓝本
（Research Agent + Reflection Agent + LangGraph 状态机），做一个 MMLU 多选题问答 Agent。
模型驱动换成 Qwen / DashScope（复用 `../deepeval-local-llm-poc/custom_llm.py` 的
`ChatOpenAI` + DashScope base_url 写法）。

详细的需求 / 设计 / 开发步骤见 [PLAN.md](PLAN.md)。
选题背景见 [`../docs/content-plan-agent-eval-series.md`](../docs/content-plan-agent-eval-series.md)
的「附录：选题 4 的最终方案」。

已实现单题 MMLU 加载、研究与反思循环、流式节点日志和人工答案核对入口。

## 运行一道 MMLU 题目

在本目录的 `.env` 或环境变量中配置 `DASHSCOPE_API_KEY` 和
`DASHSCOPE_MODEL_NAME`，然后在本目录执行：

```powershell
.\.venv\Scripts\python.exe -m qa_agent.run
```

首次加载数据集需要网络连接；模型调用和检索工具也需要联网。入口固定读取
`cais/mmlu` 的 `high_school_geography` 配置、`test` split 的第 0 条样本，
一次只运行一道题。

MMLU 原始数据包含 `question`（题干字符串）、`choices`（选项字符串列表）、
`answer`（从 0 开始的正确选项索引），以及 `subject`。入口把 `choices` 转成
`A.` 至 `D.` 的多行选项；**标准答案不传入图或模型**，仅在图结束后转成
“字母 + 选项内容”打印，供人工核对，不计算正确率或其他自动评测指标。

运行顺序为 `research_start → reflect → (research → reflect)*`：
教授确认答案时结束，否则最多研究三轮。每个流式更新交给 `log_event` 打印；
结束时展示最新 Agent 答案和 MMLU 标准答案。达到轮数上限并不表示教授已确认，
也不保证答案正确。**Demo 只演示流程，不承诺答题准确率。**

| LangGraph 概念 | 本 Demo 的对应实现 |
|---|---|
| 节点 | `research_start` 首次研究、`research` 根据反馈重研究、`reflect` 教授审查 |
| 边 | 固定边连接研究和审查；`reflect` 使用 `Command(goto=...)` 结束或继续 |
| 状态 | `question`、`options`、`current_answer`、`feedback`、`steps` |

`graph.py` 只定义和编译图，不再带手写问题运行入口。`run.py` 使用
`asyncio.run` 启动异步函数，并消费 `graph.astream(..., stream_mode=["updates"])`。
当前外层节点仅返回答案、反馈和轮数，因此日志不展示内部 Agent 的每次工具调用；
`log_event` 可以展示传入事件中 `messages` 携带的工具调用，但这不是完整工具 trace。

## 真实运行记录（2026-10-27）

首次联网已成功下载该配置，`test` split 有 198 条记录。第 0 条题目询问阻碍
自给自足经济发展的主要缺失因素。实际运行日志节选：

```text
MMLU: high_school_geography / test / sample 0
--- [Node: research_start] 开始首次研究 ---
--- [更新: research_start] ---
steps: 1
--- [Node: reflect] 教授审查中 ---
--- [更新: reflect] ---
h2 connection driver error: connection aborted
```

研究员初稿选择 **B. a well-connected transportation infrastructure.**；
教授没有确认答案，而是要求核对原始题源及权威依据，流程因此进入继续研究分支。
后续阶段长时间没有新输出，并出现上述连接中断日志，本次检查已手动停止，
**尚未验证真实联网运行完整结束**。不能把这段记录描述为成功跑完。

随后从已缓存的数据中读取第 0 条标准答案，`answer=1`，对应 **B**。
人工核对结果是“初稿选项与标准答案一致”，不是最终答案验收，也不代表准确率。
配置联网环境后重新运行入口，才能补齐最终 Agent 答案与标准答案的完整输出。

## Agent API 兼容性

本项目使用 LangChain 1.x，研究员通过 `langchain.agents.create_agent` 构建。
输入使用 `messages`，输出从最后一条 AI 消息中读取文本；图节点复用
`agents.py` 的调用封装，避免继续使用旧版 `AgentExecutor` 的 `output` 字段。

在项目目录安装依赖后，可运行研究员和反思链的本地示例（需要配置模型环境变量，
且会调用外部模型和检索服务）：

```powershell
python -m pip install -r .\requirements.txt
python .\qa_agent\agents.py
```

## 写作参考：LangChain v1 的 Agent API 迁移

官方来源、迁移对照及文章切入点已单独整理到
[Agent API 迁移备忘](../docs/langchain-agent-migration-memo.md)。
