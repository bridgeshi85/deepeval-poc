"""
RAG Evaluation with 5 DeepEval Metrics

评估维度分两组：
  🔍 检索器 (Retriever)
      - ContextualRelevancyMetric  : 检索到的文档和问题相关吗？       [无需 expected_output]
      - ContextualPrecisionMetric  : 最相关的文档排在最前面吗？        [需要 expected_output]
      - ContextualRecallMetric     : 回答所需的信息都找到了吗？        [需要 expected_output]

  ⚙️  生成器 (Generator)
      - AnswerRelevancyMetric      : 答案真正回答了用户的问题吗？      [无需 expected_output]
      - FaithfulnessMetric         : 答案有没有编造 context 里没有的内容？[无需 expected_output]

运行方式（从项目根目录）：
    deepeval test run evaluation/test_deepeval_metrics_poc.py
"""

import csv
import sys
from pathlib import Path
from typing import List, Tuple

from deepeval.dataset import EvaluationDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from custom_llm import QwenModel
from rag.rag_app import ask_rag_bot

# ── 常量配置 ────────────────────────────────────────────────────────────────
GOLDENS_CSV = PROJECT_ROOT / "data" / "goldens.csv"

# 各指标的通过阈值（0-1），可按实际结果调整
THRESHOLDS = {
    "answer_relevancy": 0.7,
    "faithfulness": 0.8,  # 幻觉风险最高，阈值设严一点
    "contextual_relevancy": 0.7,
    "contextual_precision": 0.7,
    "contextual_recall": 0.7,
}


# ── Step 1: 从 CSV 加载 Goldens 并调用 RAG Bot 构建 TestCase ────────────────
def load_test_cases() -> List[LLMTestCase]:
    """
    通过goldens.csv加载DataSet → 调用 RAG Bot 获取 actual_output 和 retrieval_context
    → 组装成 DeepEval LLMTestCase 列表。

    CSV 列说明：
        question      → input（用户提问）
        ideal_answer  → expected_output（标准答案，供 Precision / Recall 使用）
        reference_docs → context（黄金参考文档，供 Precision / Recall 使用）
    """
    if not GOLDENS_CSV.exists():
        raise FileNotFoundError(
            f"找不到 Goldens 文件: {GOLDENS_CSV}\n"
            "请先运行生成数据集的脚本。"
        )

    # 1. 初始化空数据集
    dataset = EvaluationDataset()

    # 2. 从 CSV 一键导入黄金样本 (Goldens)
    dataset.add_goldens_from_csv_file(
        file_path=str(GOLDENS_CSV),
        input_col_name="input",  # 对应 CSV 里的问题列
        expected_output_col_name="expected_output",  # 对应 CSV 里的标准答案列
        context_col_name="context",  # 对应 CSV 里的参考文档列
        context_col_delimiter="|"  # 如果多段 context 是用竖线拼接的
    )

    total = len(dataset.goldens)
    print(f"📂 已通过原生 API 加载 {total} 条 Golden，开始调用 RAG Bot...\n")

    # 3. 遍历 Goldens，让 RAG 系统作答并组装 TestCase
    for i, golden in enumerate(dataset.goldens, start=1):
        print(f"  [{i}/{total}] 正在作答: {golden.input[:60]}{'...' if len(golden.input) > 60 else ''}")

        # 调用业务 RAG 系统，获取系统的实际回答和检索到的文档
        actual_output, retrieval_context = ask_rag_bot(golden.input)

        # 组装单条测试用例
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=actual_output,
            expected_output=golden.expected_output,
            context=golden.context,  # 期望的参考上下文
            retrieval_context=retrieval_context  # 实际检索到的文档片段
        )

        # 将作答完毕的 test_case 添加回 dataset
        dataset.add_test_case(test_case)

    print(f"\n✅ TestCase 构建完成，共 {len(dataset.test_cases)} 条。\n")

    return dataset.test_cases


# ── Step 2: 创建 5 个 RAG 评估指标 ──────────────────────────────────────────
def create_rag_metrics(judge: QwenModel) -> Tuple[list, list, list]:
    """
    返回三组指标：
        generator_metrics  : 生成器相关（不需要 expected_output）
        retriever_ref_free : 检索器 · 无参考（不需要 expected_output）
        retriever_ref_based: 检索器 · 有参考（需要 expected_output + context）
    """
    # ── 生成器指标 ───────────────────────────────────────────────────────────
    answer_relevancy = AnswerRelevancyMetric(
        threshold=THRESHOLDS["answer_relevancy"],
        model=judge,
        include_reason=True,
    )
    faithfulness = FaithfulnessMetric(
        threshold=THRESHOLDS["faithfulness"],
        model=judge,
        include_reason=True,
    )

    # ── 检索器指标（无需参考答案）───────────────────────────────────────────
    contextual_relevancy = ContextualRelevancyMetric(
        threshold=THRESHOLDS["contextual_relevancy"],
        model=judge,
        include_reason=True,
    )

    # ── 检索器指标（需要 expected_output + context）──────────────────────────
    contextual_precision = ContextualPrecisionMetric(
        threshold=THRESHOLDS["contextual_precision"],
        model=judge,
        include_reason=True,
    )
    contextual_recall = ContextualRecallMetric(
        threshold=THRESHOLDS["contextual_recall"],
        model=judge,
        include_reason=True,
    )

    generator_metrics = [answer_relevancy, faithfulness]
    retriever_ref_free = [contextual_relevancy]
    retriever_ref_based = [contextual_precision, contextual_recall]

    return generator_metrics, retriever_ref_free, retriever_ref_based


# ── Step 3: 打印结果汇总 ────────────────────────────────────────────────────
def print_summary(test_cases: List[LLMTestCase], all_metrics: list) -> None:
    """在 evaluate() 完成后打印每个指标的平均分和通过率。"""
    print("\n" + "=" * 60)
    print("📊  评估结果汇总")
    print("=" * 60)

    for metric in all_metrics:
        scores = []
        passed = 0
        for tc in test_cases:
            # DeepEval 在 evaluate() 后会把结果写回 test_case 的 metrics_data
            for md in getattr(tc, "metrics_data", []):
                if md.name == metric.__name__:
                    scores.append(md.score)
                    if md.success:
                        passed += 1
                    break

        if scores:
            avg = sum(scores) / len(scores)
            total = len(scores)
            print(
                f"  {metric.__name__:<32} "
                f"avg={avg:.3f}  "
                f"passed={passed}/{total}  "
                f"threshold={metric.threshold}"
            )
        else:
            print(f"  {metric.__name__:<32} (无评分数据)")

    print("=" * 60 + "\n")

    print("💡 结果解读：")
    print("  - ContextualRelevancy / Precision / Recall 低  → 检索环节有问题，调 chunk_size 或 embedding 模型")
    print("  - Faithfulness 低                              → 生成器在编造，收紧 system prompt 或降低 temperature")
    print("  - AnswerRelevancy 低                           → 答非所问，检查 prompt 模板或检索质量\n")


def test_run_evaluation() -> None:
    print("🚀 RAG 评估启动\n")
    print(f"   法官模型  : QwenModel (DashScope)")
    print(f"   数据集    : {GOLDENS_CSV.name}")
    print(f"   指标数量  : 5 个（2 Generator + 3 Retriever）\n")

    # 1. 初始化评估法官
    judge = QwenModel()

    # 2. 加载 Golden 并构建 TestCase
    test_cases = load_test_cases()

    # 3. 创建指标
    gen_metrics, ret_free, ret_ref = create_rag_metrics(judge)
    all_metrics = gen_metrics + ret_free + ret_ref

    # 4. 运行评估
    # evaluate() 会并发调用 LLM Judge，结果同时上传 Confident AI Dashboard（如已登录）
    evaluate(
        test_cases=test_cases,
        metrics=all_metrics,
    )

    # 5. 打印本地汇总
    print_summary(test_cases, all_metrics)
