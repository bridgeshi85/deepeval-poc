# POC Guide: 从玄学到科学 - 基于 DeepEval 与 RAGAS 的本地化 LLM 评估实战

一个简易的本地 RAG 应用评估 POC，展示了如何使用 DeepEval 和 RAGAS 框架对基于 LangChain + Chroma 构建的 RAG 应用进行系统性评估。

## 项目结构

```
├── config.py                  # 配置管理
├── main.py                    # 主入口
├── part1_baseline_rag.py      # Part 1: Baseline RAG 应用
├── part2_goldens_dataset.py    # Part 2: 测试数据集
├── part3_deepeval_metrics.py   # Part 3: DeepEval 指标
├── part4_ragas_evaluation.py   # Part 4: RAGAS 评估
├── requirements.txt
└── .env.example
```

## Part 1: 环境准备与被测对象 (The Baseline)

基于 LangChain 0.1.46+ 和 Chroma 构建的简易本地 RAG 应用。

**Pro Tip**: 使用 `retriever.invoke(question)` 替代 `retriever.get_relevant_documents(question)`，`qa_chain.invoke()` 替代 `qa_chain.run()`。

## Part 2: 构建"考题库" (Datasets & Goldens)

测试用例结构:
- `input`: 用户问题
- `expected_output`: 参考答案
- `retrieved_contexts`: 检索到的上下文
- `actual_output`: 模型实际输出

## Part 3: 实战 DeepEval

- **业务指标**: GEval 捏出的 Concise (简洁度) 和 Completeness (完整度)
- **系统指标**: FaithfulnessMetric (忠实度) 和 ContextualPrecisionMetric (上下文精确度)
- **高阶玩法**: 使用本地 Ollama 模型做裁判，连接 Confident AI 查看可视化报告

## Part 4: 实战 RAGAS

RAGAS 框架特点:
- 将数据打包为 `EvaluationDataset`
- 输出 DataFrame 格式，适合 CI/CD 流水线历史版本对比

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入必要的 API keys

# 3. 运行评估
python main.py
```

## 评估指标一览

| 指标类型 | 指标名称 | 说明 |
|---------|---------|------|
| 业务指标 | Conciseness | 响应简洁度 |
| 业务指标 | Completeness | 响应完整度 |
| 系统指标 | Faithfulness | 响应与检索上下文的一致性 |
| 系统指标 | ContextualPrecision | 检索质量 |
| RAGAS | AnswerCorrectness | 答案正确性 |
| RAGAS | ContextualRecall | 上下文召回率 |
