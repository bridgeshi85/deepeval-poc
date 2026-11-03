# LangSmith 接入与基础评估步骤

按“先打通 Trace，再建立数据集，最后跑实验”的顺序进行。LangSmith Trace 用于记录每次运行；Dataset 和 Evaluation 用于比较不同版本的答案质量。

## 1. 创建 LangSmith 项目并配置环境变量

在 LangSmith 创建 API Key 和项目，例如 `langgraph-qa-agent-dev`。在项目使用的 `.env` 中配置：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的_langsmith_api_key
LANGSMITH_PROJECT=langgraph-qa-agent-dev
```

不要将真实 API Key 提交到 Git。LangSmith Trace 会记录输入、输出以及部分运行元数据；如果题目或上下文包含敏感内容，先确认数据可以发送到托管平台。

## 2. 验证自动 Trace

LangGraph 和 LangChain 的调用可以通过环境变量启用 LangSmith tracing，通常不必修改 `qa_agent/graph.py`。先运行现有程序：

```bash
cd langgraph-qa-agent
python -m qa_agent.run
```

然后在 LangSmith 项目中检查是否出现一次运行，并确认能看到 `research_start`、`reflect`，以及可能出现的 `research` 和工具调用。如果没有 Trace，检查 API Key、项目名、环境变量是否在启动进程前加载，以及当前依赖环境中是否有 `langsmith`。建议将 `langsmith` 作为明确依赖加入 `requirements.txt`，避免依赖其他包间接安装。

## 3. 准备一小份固定评估数据

先从 MMLU `high_school_geography` 准备 10～20 条固定样本，不建议一开始就跑完整数据集：Agent 会联网搜索，并可能经过多轮研究，整批运行会较慢，也会产生 API 成本。

每条样本至少包含：

- `inputs`：`question` 和 `options`
- `outputs`：标准答案，建议保存选项字母，如 `C`

可以先在 LangSmith 创建一个版本化数据集，例如 `mmlu-geography-smoke-v1`，再导入这些样本。标准答案保存为字母会比保存完整选项文本更方便评估。

## 4. 写一个适配当前 Graph 的评估入口

当前 Graph 的输入是 `question`、`options`，运行结果中最终答案在 `current_answer`。评估 target 可以把两者映射成 LangSmith 统一的输入输出格式：

```python
def target(inputs: dict) -> dict:
    result = graph.invoke(inputs)
    return {"answer": result["current_answer"]}
```

如果 evaluator 能拿到模型的完整解释文本，也可以暂时返回完整答案；不过首个评估建议只比较选项字母，减少格式差异造成的误判。

## 5. 先做准确率评估

实现一个简单 evaluator，从 Agent 输出和标准答案中提取 `A`～`E`，比较是否一致，并返回一个分数。然后使用 LangSmith 的 `evaluate(...)` 针对刚才的数据集运行实验。

首轮结果重点观察：

- 总体准确率，以及逐题错误情况
- Trace 中失败的工具调用、异常和过长的研究循环
- Agent 是否输出了无法解析的选项
- 调整 prompt 或模型后，新旧实验的分数变化

## 6. 再增加质量指标

当前 Agent 最终输出主要是自由文本，因此更适合先做选项准确率。之后如果需要评估论证质量，可以增加 LLM-as-judge evaluator，例如判断论证是否支持答案、是否使用了可靠证据。

Agent 使用检索工具，但若没有稳定的引用或证据字段，groundedness 一类指标较难可靠计算；可先改为结构化输出“答案、理由、来源”，再补充对应评估器。

## 首轮验收目标

先完成最小闭环：一个 LangSmith 项目、10～20 条固定样本、一个选项准确率 evaluator、一次可复现的实验。验证 tracing 和评估流程后，再扩展更多数据和指标。
