# 内容系列规划：Agent 评测三部曲 + Agent 开发入门

> 聚焦 **Agent 评测三部曲 + Agent 开发入门** 的组合，把选题 3（自建 Collector）和 RAGAS vs DeepEval 整合进体系，形成有阅读顺序的系列。
>
> 创建日期：2026-08-31

---

## Agent 评测三部曲

### 🎯 选题 1（概念篇）：Agent 评测到底测什么——从"能不能用"到"哪里出错"

**💡 素材关联**：已学的 Agent 评测三层框架（最终答案层/轨迹层/生产层）+ Confident AI、Morphllm 的资料

**文章价值**：市面上讲 Agent 评测的文章大多直接跳到"用 XX 框架打分"，但很少讲清楚"为什么端到端评测测不出 Agent 的问题"这个根本逻辑。这篇打好概念地基，也是后面两篇实战文章的理论支撑，读者读完能自己判断"我的场景该测哪一层"。

**目标用户**：刚接触 Agent 评测、被"trajectory""tool correctness"这些术语绕晕的测试/开发工程师。

**关键问题**：

1. 单次 LLM 调用评测和 Agent 评测的本质区别是什么（为什么同一任务两次运行路径不同但都算对）？
2. 三层框架（答案层/轨迹层/生产层）分别对应什么指标、什么工具？
3. 什么场景下只测最终答案就够，什么场景必须拆到轨迹层？

**参考资料**：LLM Agent Evaluation Metrics in 2026 - Confident AI、AI Agent Evaluation 2026 - Morphllm

---

### 🎯 选题 2（实战篇）：给客服 Agent 搭评测 Harness——一个 Task Completion 过了但 Tool Correctness 没过的真实案例

**💡 素材关联**：已跑通的客服 Agent Demo（`agent_plain.py` → `agent_instrumented.py` → 三个测试文件），尤其是那个"故意设计成边界情况"的第三个 golden

**文章价值**：这是选题 1 的落地版——不讲抽象指标定义，直接复盘一个真实跑出来的分歧案例：为什么任务完成度过了，工具调用正确性却没过，这种分歧到底暴露了 Agent 的什么问题。有真实分数、真实 trace 截图，比纯教程更有说服力。

**目标用户**：已经理解概念、想看"评测到底怎么落地成代码"的读者；正在把自己的 Agent 接入 DeepEval 的工程师。

**关键问题**：

1. 4 行埋点代码（`CallbackHandler` + `@observe`）具体做了什么，为什么改造成本这么低？
2. `test_agent.py`、`test_agent_extended.py`、`test_agent_correctness.py` 三个文件分别验证了什么，为什么要拆开写？
3. 那个边界案例里，Task Completion 和 Tool Correctness 具体在哪个环节出现分歧？这说明了什么设计问题？

**参考资料**：自己的 Agent 评测 Demo repo（GitHub 可直接放代码链接引流）

---

### 🎯 选题 3（可观测篇）：DeepEval Trace 数据不出内网——自建 OTel Collector 追踪 Agent 调用链

**💡 素材关联**：自建 Collector 方案 + Chapter 9《Production Deployment and Observability》

**文章价值**：三部曲的收尾——评测跑通之后，下一个问题是"生产环境的 Agent trace 数据能不能不出公司内网"。这篇直接给出可落地的自建方案，是中文互联网目前几乎空白的方向，也最贴合 K8s/Helm 的技术背景，写出来差异化最强。

**目标用户**：数据合规要求高、想把 Agent 可观测性纳入现有基础设施（ELK/Grafana）的团队。

**关键问题**：

1. DeepEval 默认的 `ConfidentSpanExporter` 把数据发到哪，怎么替换成自定义 `OTLPSpanExporter`？
2. 用 Helm 部署 OTel Collector 的具体步骤，Collector 后面接 ELK 要配什么 exporter？
3. 用选题 2 那个客服 Agent 案例做验证——自建链路能不能同样定位到 Task Completion / Tool Correctness 分歧的那次调用？

**参考资料**：OTel Collector Helm Chart、Confident AI OpenTelemetry 文档

---

## Agent 开发入门（1篇，为三部曲做铺垫）

### 🎯 选题 4：跟着 Packt 官方书，30 分钟搭一个能跑的 LangGraph Agent

**💡 素材关联**：`generative_ai_with_langchain`（chapter6，question_answering.ipynb；原计划 chapter5 Building Intelligent Agents，最终选定 chapter6 QA Agent，见下方附录）

**文章价值**：给还没有 Agent 开发经验的读者铺垫——不涉及评测，纯粹是"跟着官方书代码走一遍，理解 Agent 的基本组成（工具定义、LangGraph 节点、状态流转）"。写完这篇，读者才有能力理解后面三部曲里"埋点""trace"这些概念在操作什么对象。

**目标用户**：有 Python 基础但没搭过 Agent 的测试/开发工程师；准备学 Agent 评测但缺基础的读者（给三部曲导流）。

**关键问题**：

1. LangGraph 的节点、边、状态在书里的例子中分别对应什么？
2. 一个最简单的工具调用型 Agent 需要哪几个组成部分？
3. 跑起来之后，怎么用最原始的方式（打印日志）看到它内部的决策过程，为下一篇"埋点评测"做铺垫？

**参考资料**：generative_ai_with_langchain - chapter6（代码已存于本仓库 `references/generative_ai_with_langchain/chapter6/`）

---

## 附加篇：RAGAS vs DeepEval

### 🎯 选题 5：RAGAS vs DeepEval，在同一个 RAG Bot 上打分差多少

**💡 素材关联**：`deepeval-local-llm-poc/` 子项目（已有 5 个 DeepEval 指标的真实分数）

**文章价值**：不属于 Agent 系列，但成本最低、最容易出活——直接用已经跑出来的 golden dataset 再跑一遍 RAGAS，做真实数据对比，而非转述别人的框架对比文章。

**目标用户**：纠结选 RAGAS 还是 DeepEval 的团队。

**关键问题**：

1. 同一批 golden dataset，两个框架的 Faithfulness/Relevancy 分数差多少？
2. 分数差异是 judge 模型不同导致，还是指标定义细节不同？

**参考资料**：`deepeval-local-llm-poc/` 子项目

---

## 建议的写作/发布顺序

1. **选题 4**（Agent 开发入门）→
2. **选题 1**（评测概念）→
3. **选题 2**（评测实战）→
4. **选题 3**（Trace 可观测）→
5. **选题 5**（RAGAS vs DeepEval，可穿插在任意位置发布，不依赖前面的阅读顺序）

这个顺序刚好也是学习节奏——写文章和学习进度同步推进，不需要额外倒回去补素材。

---

## 附录：选题 4 的最终方案（基于 chapter6 question_answering.ipynb）

### 📋 question_answering.ipynb 核心架构分析

**🎯 Agent 组成（简洁清晰）**

1. **研究代理（Research Agent）**
   - 工具：DuckDuckGo 搜索、Arxiv、Wikipedia
   - 角色：学生角色，逐步思考，收集证据
2. **反思代理（Reflection Agent）**
   - 模型：作为教授进行反思和评价
   - 输出：答案 OR 批评反馈
3. **状态机流程**

```
START
 ↓
[research_start] → 初始调查
 ↓
[reflect] → 教授评价（✓答案 / ✗反馈）
 ↓
[research] → 基于反馈改进
 ↓
循环直到：答案正确 OR 超过 max_reasoning_steps
 ↓
END
```

**🎓 为什么适合选题 4（入门篇）**

| 维度 | 评分 | 原因 |
|------|------|------|
| 代码复杂度 | ⭐⭐ | 只需要 LangGraph 基础概念 |
| 工具数量 | ⭐⭐⭐ | 3 个工具（DuckDuckGo、Arxiv、Wikipedia） |
| 状态管理 | ⭐⭐⭐ | 相对简洁的 TypedDict 状态 |
| 输入/输出 | ⭐⭐⭐⭐ | 非常清晰（问题→选项→答案） |
| 可评测性 | ⭐⭐⭐⭐⭐ | MMLU 数据集，有标准答案 |

**✏️ 选题 4 的实现方案**

核心内容：
1. **LangGraph 三要素**：节点、边、状态
2. **工具集成**：load_tools 快速接入搜索工具
3. **状态流转**：从 research_start → reflect → 条件判断
4. **结构化输出**：Pydantic Response 模型

实战步骤：
- Step 1: 加载 MMLU 数据集，看清问题结构
- Step 2: 搭建 Research Agent（创意生成，多轮搜索）
- Step 3: 搭建 Reflection Agent（质量评估）
- Step 4: 用 StateGraph 连接两个 agent，加入 should_end 条件
- Step 5: 跑一个真实例子，理解内部决策

为什么这个例子好用：
- ✅ 数据集有标准答案（可验证正确性）
- ✅ 展示了 agent 的典型工作模式（搜索→反思→改进）
- ✅ 代码量少但概念全（状态、工具、条件判断）
- ✅ 输出清晰（students' reasoning + final answer）

**🎯 三部曲最终方案**

| 序号 | 标题 | Demo Agent | 难度 | 评测维度 |
|------|------|-----------|------|---------|
| 选题 4 | 30分钟搭一个 QA Agent | question_answering.ipynb | ⭐⭐ | 无评测（入门铺垫） |
| 选题 1 | Agent 评测测什么 | 可用 ToT 作高阶案例 | ⭐⭐⭐ | 概念：最终答案层 vs 轨迹层 |
| 选题 2 | 客服 Agent 评测实战 | agent_plain.py | ⭐⭐⭐ | 实战：Task Completion vs Tool Correctness |
| 选题 3 | 自建 OTel Collector | 同选题 2 | ⭐⭐⭐⭐ | 生产：链路追踪 + ELK 存储 |

question_answering.ipynb 正是入门篇需要的：简洁、完整、有数据集、有标准答案。直接以它为基础写选题 4。
