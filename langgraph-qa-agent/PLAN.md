# 选题 4 Demo Agent —— 需求 / 设计 / 开发 步骤

## Context

选题 4《30 分钟搭一个能跑的 LangGraph Agent》是 Agent 评测三部曲的入门铺垫篇，
目标是"跟着官方书代码走一遍，理解 Agent 的基本组成"。素材是
`../references/generative_ai_with_langchain/chapter6/question_answering.ipynb`
（MMLU 多选题问答 Agent：Research Agent + Reflection Agent + LangGraph 状态机）。

按 需求 → 设计 → 开发 三段推进，产出一个能跑通的 demo agent，后续文章围绕它写。

关键约束：

- 模型驱动改用 **Qwen / DashScope**（复用 `../deepeval-local-llm-poc/custom_llm.py`
  里的 `ChatOpenAI` + `base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`
  写法，读环境变量 `DASHSCOPE_API_KEY` / `DASHSCOPE_MODEL_NAME`，`.env` 复用仓库根一份）。
  原书用 `ChatGoogleGenerativeAI`。
- 本项目自带 `langgraph-qa-agent/requirements.txt`、各自 `.venv`。需要的包：
  `langgraph`、`langchain` / `langchain-core` / `langchain-openai` / `langchain-community`、
  `pydantic`、检索工具包（`duckduckgo-search`、`arxiv`、`wikipedia`）、`datasets`、
  `python-dotenv`。版本在开发第 2 步锁定。
- 代码放本目录下的包 `qa_agent/`。

---

## 第一阶段：需求（明确这个 Agent 要做成什么）

1. **一句话定义**：输入一道 MMLU 多选题（题干 + 选项），Agent 通过联网检索收集证据，
   给出带论证的答案；再由"教授"角色反思，要么确认、要么打回重研究，直到答案被确认或
   达到最大推理步数。

2. **功能清单（要在文章里逐条对上 LangGraph 概念）**：
   - F1 数据加载：从 MMLU（`cais/mmlu`, `high_school_geography`）取样，暴露
     `question` / `options` / 标准答案，用于人工核对（入门篇不做自动评测）。
   - F2 Research Agent：`create_react_agent`，挂 3 个检索工具（DuckDuckGo / Arxiv /
     Wikipedia），system prompt 设定"逐步思考、必须给论证、不许臆测、用工具找证据"。
   - F3 Research-with-critique Agent：同上，但 prompt 额外吃 `answer` + `feedback`，
     做二次改进。
   - F4 Reflection 步骤：教授 prompt + `llm.with_structured_output(Response)`，
     `Response` 为 Pydantic 模型，二选一填 `answer` 或 `critique`。
   - F5 状态机：`StateGraph(ReflectionAgentState)`，节点 `research_start` /
     `research` / `reflect`，边 START→research_start→reflect，research→reflect，
     `reflect` 走条件边 `_should_end`（有终答且无 critique → END；步数超限 → END；
     否则 → research）。
   - F6 可观测（本篇重点、为选题 3 铺垫）：用**最原始的 print/logging** 打印每个节点
     的输入输出、工具调用、每轮 critique，让读者肉眼看到"决策过程"。
   - F7 运行入口：`graph.astream(..., stream_mode=["updates"])` 跑一道真实题目，
     打印全过程 + 最终答案 vs 标准答案。

3. **非目标（明确排除，避免 scope 蔓延）**：
   - 不接 DeepEval / 不写指标、不做批量评测（那是选题 1/2）。
   - 不做 OTel / trace 导出（选题 3）。
   - 不追求答对率，只要求跑通、日志清晰。

4. **验收标准**：`python -m qa_agent.run` 能完整跑完一道题，控制台按节点顺序
   打印中间过程，最后给出结构化最终答案；README 记录 3 个概念映射（节点/边/状态）。

---

## 第二阶段：设计（模块拆分与关键改造点）

1. **目录结构**（建议）：

   ```text
   langgraph-qa-agent/
     requirements.txt
     README.md          # 概念映射表（写文章时直接引用）
     PLAN.md            # 本文件
     qa_agent/
       __init__.py
       llm.py           # Qwen ChatOpenAI 封装
       tools.py         # load_tools 三件套
       agents.py        # research_agent / research_agent_with_critique / reflection_chain
       graph.py         # State 定义 + StateGraph 组装 + _should_end
       trace.py         # F6：统一的节点/工具日志打印
       run.py           # F1 数据加载 + F7 运行入口
   ```

2. **对照原 notebook 的改造点**：

   | notebook 位置 | 原实现 | 改造 |
   |---|---|---|
   | cell 8 | `ChatGoogleGenerativeAI("gemini-2.5-flash")` | `qa_agent/llm.py` 用 DashScope `ChatOpenAI`，`temperature=1.0` 保留 |
   | cell 9 | `load_tools(["ddg-search","arxiv","wikipedia"], llm)` | 不变，但确认当前 langchain 里 `load_tools` 的 import 路径（`langchain_community.agent_toolkits.load_tools` 或 `langchain.agents.load_tools`） |
   | cell 10/12 | `create_react_agent` from `langgraph.prebuilt` | 不变，需装 `langgraph` |
   | cell 13-15 | `Response` / `ReflectionAgentState` / 三个节点函数 | 不变 |
   | cell 16 | `draw_mermaid_png()` | 改成保存 png 到文件，或直接在 README 里贴静态 mermaid（去掉对 graphviz/浏览器的依赖） |
   | cell 18 | `async for ... graph.astream` | 移到 `run.py`，包一层 `asyncio.run`；日志走 `trace.py` |

3. **`trace.py` 设计（F6 的核心）**：一个 `log_event(node_name, event)` 函数，
   在 `astream` 循环里对每个 update 事件格式化打印——节点名、该节点返回的 state 增量、
   若含 `messages` 则打印工具调用名和参数。入门篇不引任何 tracing 库，纯 `print` +
   分隔线，读者能直接抄。

4. **配置**：复用仓库根 `.env` 的 `DASHSCOPE_API_KEY` / `DASHSCOPE_MODEL_NAME`（如
   `qwen-plus`）；`llm.py` 用 `python-dotenv` 加载。检索工具走公网，无需 key。

---

## 第三阶段：开发（落地顺序）

1. 建 `langgraph-qa-agent/qa_agent/` 骨架 + 空文件，建本项目 `.venv`。
2. 写 `requirements.txt`，`pip install`，锁版本；确认 `langgraph` /
   `create_react_agent` / `load_tools` 的可用 import 路径（跑一个 import 冒烟测试）。
3. `llm.py`：参考 `../deepeval-local-llm-poc/custom_llm.py` 的 DashScope `ChatOpenAI`
   构造逻辑，暴露 `get_llm()`；单测一次 `.invoke("ping")`。
4. `tools.py`：`load_tools` 三件套；单测 `ddg-search` 能返回结果。
5. `agents.py`：搬 cell 9-15 的 prompt / `Response` / 两个 react agent /
   `reflection_chain`。单独跑 `research_agent.invoke({...})` 验证工具调用链。
6. `graph.py`：搬 `ReflectionAgentState`、三个节点函数、`_should_end`、
   `StateGraph` 组装、`compile()`。
7. `trace.py` + `run.py`：F1 数据加载（`datasets.load_dataset`，取样 100 条），
   选一道题，`graph.astream` + `log_event`，打印最终 `Response` 与标准答案对比。
8. `README.md`：填「节点 / 边 / 状态」概念映射表、运行命令、一次真实运行的日志片段。

---

## 验证

- `python -m qa_agent.run`（在 `langgraph-qa-agent/` 下）：完整跑完一道 MMLU 题目，
  控制台按 `research_start → reflect → (research → reflect)* → END` 顺序打印中间过程，
  末尾输出结构化最终答案 + 标准答案。
- 手动核对：至少跑 3 道不同题目，确认状态机的循环与终止条件都被触发过一次。
- `pip check` 无依赖冲突；本项目独立于 `deepeval-local-llm-poc/`，互不影响。
