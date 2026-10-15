# LangGraph QA Agent 开发步骤

本文按照“先验证单个组件，再连接状态图”的顺序实现 Demo，方便边开发边记录 guide 文章素材。

## 开发前准备

在项目目录启用独立虚拟环境并安装依赖：

```powershell
cd langgraph-qa-agent
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

在仓库根目录 `.env` 中配置 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_MODEL_NAME`。不要把密钥写入代码或提交到仓库。

> DuckDuckGo 工具在当前 `langchain-community` 版本中依赖 `ddgs`，而不是旧包 `duckduckgo-search`；依赖已记录在 `requirements.txt`。

## 1. 封装 Qwen 模型

**文件：** `qa_agent/llm.py`

参考 `deepeval-local-llm-poc/custom_llm.py`，实现 `get_llm()`：

- 使用 `langchain_openai.ChatOpenAI`。
- `base_url` 设置为 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
- 从环境变量读取 API Key 和模型名称。
- 配置缺失时给出清楚的错误信息。

**检查：**先确认 `get_llm()` 能创建模型对象，再用 `.invoke("ping")` 做一次真实 API 调用。真实调用需要有效的 DashScope Key。

**文章素材：**记录 OpenAI 兼容接口如何让 LangChain 复用 `ChatOpenAI` 接入 Qwen。

## 2. 加载检索工具

**文件：** `qa_agent/tools.py`

通过 `langchain_community.agent_toolkits.load_tools` 加载 `ddg-search`、`arxiv` 和 `wikipedia`，并封装成一个工具加载函数。

**检查：**打印加载出的工具名称，预期包含 DuckDuckGo、Arxiv 和 Wikipedia。只检查工具构建时不需要发起联网搜索。

**文章素材：**说明工具提供外部检索能力，但检索结果仍需要模型筛选和解释。

## 3. 实现 Research Agent 和 Reflection Chain

**文件：** `qa_agent/agents.py`

先实现并分开验证三个组件：

1. 初始 Research Agent：读取题干和选项，调用检索工具，给出答案和论证。
2. Research-with-critique Agent：除题目外，还接收已有 `answer` 和教授的 `feedback`，据此继续研究。
3. Reflection Chain：用教授角色检查答案，通过 `llm.with_structured_output(Response)` 返回结构化结果。

定义 Pydantic `Response`，表达“确认答案”或“提出 critique”。要考虑并处理模型同时输出 `answer` 和 `critique`、或两者都为空的情况。

**检查：**先用一道手写问题单独调用初始 Research Agent，再单独调用 Reflection Chain；确认输入输出字段后再接状态图。

**文章素材：**展示普通文本输出与结构化输出的差别，以及 critique 如何反馈到下一轮研究。

## 4. 定义状态和节点

**文件：** `qa_agent/graph.py`

定义 `ReflectionAgentState`，统一保存题目、选项、答案、反馈、消息和当前步数等运行数据。然后实现三个节点：

- `research_start`：执行首次研究并初始化状态。
- `research`：读取已有答案和反馈，执行后续研究。
- `reflect`：评审答案，更新答案、critique 和步数。

每个节点只返回本次需要更新的状态字段。字段命名要前后一致，确保 critique 能传递给下一轮 Research Agent。

**检查：**逐一对照状态定义和节点的读取、返回字段；在接入真实 LLM 前先确认数据流完整。

**文章素材：**解释 LangGraph 的 state 是节点之间共享和更新的数据。

## 5. 配置条件路由并编译图

**文件：** `qa_agent/graph.py`

实现 `_should_end`，并按以下流程组装 `StateGraph`：

```text
START -> research_start -> reflect
reflect -> END 或 research
research -> reflect
```

终止逻辑至少覆盖：教授确认答案时结束；达到最大步数时结束；仍有 critique 且未超限时继续研究。确认步数在哪个节点更新，避免循环无法终止。

**检查：**用手工构造的状态分别验证“确认答案”“需要重研究”“达到步数上限”三个路由分支，然后编译图。

**文章素材：**用流程图解释节点、边和条件边如何组成 Agent 的控制流。

## 6. 增加过程日志

**文件：** `qa_agent/trace.py`

实现 `log_event(node_name, event)`，格式化输出节点名称、状态增量、工具调用名称和参数，以及每轮 critique。

**检查：**先用手工构造的 update 事件测试日志格式，再接入 `graph.astream(..., stream_mode=["updates"])`。以当前安装版本实际返回的事件结构为准。

**文章素材：**保留一段真实运行日志，并标注其中的节点切换、工具调用和 critique。

## 7. 实现 MMLU 加载与运行入口

**文件：** `qa_agent/run.py`

实现以下流程：

1. 用 `datasets.load_dataset` 加载 `cais/mmlu` 的 `high_school_geography` 配置。
2. 从样本中选一道题，整理题干、选项和标准答案。
3. 用 `asyncio.run` 启动异步运行函数。
4. 遍历 `graph.astream(..., stream_mode=["updates"])`，将事件交给 `log_event`。
5. 结束后打印 Agent 答案和 MMLU 标准答案，供人工核对。

**检查：**先只跑一道题，确认整个流程能正常结束。首次下载数据集需要网络连接。不要把标准答案比较当作自动评测指标。

**文章素材：**记录数据格式、真实运行过程和人工核对结果，同时说明 Demo 不承诺答题准确率。

## 8. 完整验收与文章整理

在 `langgraph-qa-agent/` 目录执行：

```powershell
python -m qa_agent.run
python -m pip check
```

再手动尝试至少三道题，观察答案确认、critique 后重新研究和步数上限等路径。若真实运行没有触发某条路径，用手工状态验证路由，不要把未观察到的结果描述成真实运行结果。

在 `README.md` 补充运行方法、一次真实日志，以及“节点 / 边 / 状态”概念映射表。文章可以沿用本文件顺序：模型与工具、两个 Agent、状态和节点、条件路由、日志、真实运行与边界。

## 本 Demo 的范围

- 目标是跑通一个带检索和反思循环的 LangGraph Agent，并让过程日志清晰可读。
- 不接入 DeepEval，不做批量评测或指标计算。
- 不导出 OpenTelemetry trace。
- MMLU 标准答案只用于人工核对，不以答对率作为验收目标。
