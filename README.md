# AI 开发练习 Monorepo

集中练习 AI 应用开发与评测的仓库。每个练习是一个独立的平级子目录，自带依赖与说明；
参考素材和跨项目文档放在仓库根共享。

## 布局约定

- **每个练习一个平级目录**，目录内自带 `requirements.txt`，各自建 `.venv`。
- **共享 `.env`**：`DASHSCOPE_API_KEY` / `DASHSCOPE_MODEL_NAME` 等密钥放仓库根的一份
  `.env`（见 `.env.example`），各子项目复用。
- **`references/`**：课程 / 官方书的参考代码素材，只读，多个项目共用。
- **`docs/`**：跨项目的内容规划、选题等文档。

## 练习项目

| 目录 | 主题 | 状态 |
|---|---|---|
| [`deepeval-local-llm-poc/`](deepeval-local-llm-poc/README.md) | 用 DeepEval + 本地/云端 LLM 对 RAG 应用做系统化评测 | ✅ 可运行 |
| [`langgraph-qa-agent/`](langgraph-qa-agent/README.md) | 选题 4：跟 Packt 官方书搭一个 LangGraph QA Agent | 🚧 待开发 |

## 共享目录

- [`docs/content-plan-agent-eval-series.md`](docs/content-plan-agent-eval-series.md) — Agent 评测三部曲内容规划
- `references/generative_ai_with_langchain/` — Packt《Generative AI with LangChain》第 2 版章节代码
- `references/agent-demo/` — 客服 Agent 评测 demo 素材
