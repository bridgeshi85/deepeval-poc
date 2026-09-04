# 基于 DeepEval 的本地化 LLM 评估实战指南 (POC)

本项目以一个真实的 RAG 应用为被测对象，演示如何用 **DeepEval** 框架配合**本地 / 云端 LLM** 对其进行系统化评估。

> 本项目是仓库 monorepo 中的一个练习项目。以下所有命令均在 `deepeval-local-llm-poc/` 目录下执行；
> `.env` 复用仓库根目录的一份共享配置。

---

## 目录结构

```
deepeval-local-llm-poc/            # 本项目目录
├── custom_llm.py                  # DeepEval 自定义模型适配层（QwenModel + OllamaEmbeddingModel）
├── requirements.txt
├── .env.example                   # 环境变量模板
│
├── rag/
│   ├── build_vectordb.py          # 一次性脚本：抓取网页 → 切片 → 写入 Chroma 向量库
│   └── rag_app.py                 # 被测对象：MCPRagBot（加载向量库 + 本地 LLM 作答）
│
├── data/
│   ├── mcp_knowledge.txt          # 原始知识库文档
│   └── goldens/                   # synthesizer 自动生成的 goldens 输出目录
│
├── evaluation/
│   ├── goldens_dataset.py         # Part 1：用 DeepEval Synthesizer 自动生成黄金测试集
│   └── test_deepeval_metrics_poc.py  # Part 2：用 5 个 RAG 指标对 RAG Bot 进行完整评估
│
└── chroma_db/                     # 本地向量数据库（由 build_vectordb.py 生成）
```

---

## 技术栈

| 层次 | 技术 |
|------|------|
| RAG 框架 | LangChain（`langchain`, `langchain-community`, `langchain-core`） |
| 向量数据库 | ChromaDB（本地持久化） |
| 本地 LLM / Embedding | Ollama（`qwen2.5:7b` 作答，`qwen3-embedding:4b` 向量化） |
| 云端裁判 LLM | 阿里云 DashScope（`qwen-max` 或同系列模型，兼容 OpenAI API） |
| 评估框架 | DeepEval 3.x |

---

## 快速开始

### 1. 安装依赖

```bash
# 在本项目目录 deepeval-local-llm-poc/ 下
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

本项目复用**仓库根目录**的共享 `.env`：

```bash
cp ../.env.example ../.env
```

编辑 `../.env`，填入你的 DashScope API Key 和模型名：

```dotenv
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
DASHSCOPE_MODEL_NAME=qwen-max
```

### 3. 启动本地 Ollama 模型

确保已安装 [Ollama](https://ollama.com/)，然后拉取所需模型：

```bash
ollama pull qwen2.5:7b
ollama pull qwen3-embedding:4b
```

### 4. 构建向量知识库（仅需执行一次）

脚本会抓取 MCP 主题网页，切片后嵌入写入本地 `chroma_db/`：

```bash
python rag/build_vectordb.py
```

主要配置项（位于文件顶部常量区）：

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `TARGET_URL` | `https://www.descope.com/learn/post/mcp` | 知识库来源网页 |
| `EMBEDDING_MODEL` | `qwen3-embedding:4b` | Ollama 嵌入模型 |
| `PERSIST_DIRECTORY` | `../chroma_db` | 向量库存储目录 |
| `CHUNK_SIZE` | `500` | 文本切片大小 |
| `CHUNK_OVERLAP` | `50` | 切片重叠字符数 |

### 5. 验证 RAG Bot 可以正常作答

```bash
python rag/rag_app.py
```

主要配置项（位于文件顶部常量区）：

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL` | `qwen3-embedding:4b` | 必须与构建时一致 |
| `LLM_MODEL` | `qwen2.5:7b` | Ollama 对话模型 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `RETRIEVER_K` | `3` | 每次检索返回的片段数 |

---

## 评估流程

### Part 1 — 自动生成黄金测试集

使用 DeepEval `Synthesizer` 读取 `data/mcp_knowledge.txt`，自动出题并保存到 `data/goldens/`：

```bash
python evaluation/goldens_dataset.py
```

- 裁判模型：`QwenModel`（DashScope 云端）
- 嵌入模型：`OllamaEmbeddingModel`（本地 Ollama）
- 输出路径固定为 `data/goldens/`（使用 `__file__` 绝对路径，从任意目录运行均可）

> 你也可以直接使用 `data/goldens.csv`（手工整理版），跳过此步骤。

---

### Part 2 — 5 指标 RAG 评估

从 `data/goldens.csv` 加载黄金测试集，驱动 RAG Bot 作答，再由 `QwenModel` 担任裁判，评估以下 5 个维度：

| 指标 | 评估维度 | 是否需要 expected_output |
|------|----------|--------------------------|
| `AnswerRelevancyMetric` | 回答是否真正回答了问题 | 否 |
| `FaithfulnessMetric` | 回答是否忠实于检索内容，没有幻觉 | 否 |
| `ContextualRelevancyMetric` | 检索到的文档与问题是否相关 | 否 |
| `ContextualPrecisionMetric` | 最相关的文档是否排在最前面 | 是 |
| `ContextualRecallMetric` | 回答所需的信息是否都被检索到 | 是 |

**运行方式（从项目根目录）：**

```bash
deepeval test run evaluation/test_deepeval_metrics_poc.py
```

评估结束后会在终端打印汇总表格，包含每个指标的平均分、通过率和阈值：

```
============================================================
📊  评估结果汇总
============================================================
  AnswerRelevancy                  avg=0.85  passed=4/5  threshold=0.7
  Faithfulness                     avg=0.91  passed=5/5  threshold=0.8
  ...
============================================================
```

**指标调优参考：**

- `ContextualRelevancy / Precision / Recall` 低 → 调整 `chunk_size` 或更换 embedding 模型
- `Faithfulness` 低 → 收紧 system prompt 或降低 `temperature`
- `AnswerRelevancy` 低 → 检查 prompt 模板或检索质量

---

## 自定义 LLM 适配层

`custom_llm.py` 提供两个可复用的适配类：

### `QwenModel`（裁判 LLM）

通过 DashScope OpenAI 兼容接口接入千问系列模型，读取 `.env` 中的 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_MODEL_NAME`。

```python
from custom_llm import QwenModel
judge = QwenModel()
```

### `OllamaEmbeddingModel`（本地嵌入）

封装本地 Ollama 嵌入模型，供 `Synthesizer` 使用：

```python
from custom_llm import OllamaEmbeddingModel
embedder = OllamaEmbeddingModel(model_name="qwen3-embedding:4b")
```

两个类均继承 DeepEval 的 `DeepEvalBaseLLM` / `DeepEvalBaseEmbeddingModel`，可直接传入任意 DeepEval 指标或 `Synthesizer`。

---

## 常见问题

**Q: `No test cases found`**  
A: `deepeval test run` 只识别包含 `def test_xxx():` 函数的文件。确保测试入口函数以 `test_` 开头。

**Q: `AttributeError: 'QwenModel' object has no attribute 'model'`**  
A: `DeepEvalBaseLLM.__init__()` 会立即调用 `load_model()`。正确做法是在 `load_model()` 里创建并返回模型对象，或在 `super().__init__()` 之前给 `self.model` 赋值。本项目已采用后者。

**Q: `TypeError: evaluate() got an unexpected keyword argument 'run_async'`**  
A: DeepEval 3.x 已移除 `run_async` 参数，默认自动并发执行。直接删除该参数即可。
