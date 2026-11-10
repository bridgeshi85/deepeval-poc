# 备忘：LangChain v1 的 Agent API 迁移

记录日期：2026-10-20。用途：后续文章写作参考。

## 官方来源

- [LangChain v1 migration guide — Migrate to create_agent](https://docs.langchain.com/oss/python/migrate/langchain-v1#migrate-to-create_agent)
- [create_agent API reference](https://reference.langchain.com/python/langchain/agents/factory/create_agent)

官方迁移指南的核心观点（中文转述，非原文引用）：
LangChain 在 v1.0 之前推荐使用 `langgraph.prebuilt.create_react_agent` 构建 Agent；
现在推荐使用 `langchain.agents.create_agent`。

## 文章中需要区分的两条路径

1. **旧代码兼容修复**：`create_tool_calling_agent` 和 `AgentExecutor` 的旧版用法
   可以从 `langchain_classic.agents` 导入。此前遇到的 `ImportError` 属于这一类，
   改导入路径不等于完成新版 Agent API 迁移。
2. **新版 API 迁移**：新开发优先使用 `langchain.agents.create_agent`。
   官方上述段落主要对比的是 `langgraph.prebuilt.create_react_agent` 与
   `create_agent`，并非 `create_tool_calling_agent` 的逐项替换说明。

## 与本项目相关的迁移要点

| 维度 | 旧版研究员实现 | 使用 `create_agent` 后 |
| --- | --- | --- |
| 构建方式 | `create_tool_calling_agent` + `AgentExecutor` | `create_agent`，返回基于 LangGraph 的 Agent 图 |
| 系统提示词 | `ChatPromptTemplate` 中的 system 消息 | `system_prompt` 参数 |
| 用户输入 | 模板变量，如 `question`、`options` | 将题目、选项等组织为 `messages` |
| 工具调用历史 | 提示词中声明 `agent_scratchpad` | 框架维护消息和工具调用循环 |
| 答案读取 | `result["output"]` | 从 `result["messages"]` 的最终 AI 消息读取；内容可能是文本或内容块 |
| 教授反思链 | `prompt \| llm.with_structured_output(...)` | 可以保留，不必因研究员迁移而改写 |

注意：表格是针对本项目旧版实现的迁移整理，不是官方对照表的逐字翻译。
真正迁移时需要同步检查初始研究员、接收批评的研究员及其调用方，
不能只替换 import。

## 可用于文章的切入点

“复现书中的 Agent 示例时，首先遇到的可能不是模型问题，而是框架 API 代际差异。”
可以从本次导入报错展开，说明“使用 classic 包保留旧接口”与
“迁移到当前推荐的 `create_agent`”是两个不同决策，再引出
LangChain 的高层 Agent API 与 LangGraph 底层执行图的关系。

发表前应重新核对官方文档及示例依赖版本；以上是写作素材，不代表项目已完成迁移。
