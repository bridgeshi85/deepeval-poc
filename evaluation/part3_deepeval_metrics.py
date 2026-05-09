"""
Part 3: RAG Evaluation with 5 DeepEval Metrics

评估维度分两组：
  🔍 检索器 (Retriever)
      - ContextualRelevancyMetric  : 检索到的文档和问题相关吗？       [无需 expected_output]
      - ContextualPrecisionMetric  : 最相关的文档排在最前面吗？        [需要 expected_output]
      - ContextualRecallMetric     : 回答所需的信息都找到了吗？        [需要 expected_output]

  ⚙️  生成器 (Generator)
      - AnswerRelevancyMetric      : 答案真正回答了用户的问题吗？      [无需 expected_output]
      - FaithfulnessMetric         : 答案有没有编造 context 里没有的内容？[无需 expected_output]

运行方式（从项目根目录）：
    python -m evaluation.part3_deepeval_metrics
    或通过 pytest：
    deepeval test run evaluation/part3_deepeval_metrics.py
"""

import csv
import sys
from pathlib import Path
from typing import List, Tuple

# ── 路径修正：确保从项目根目录运行时，包内模块可正常导入 ──────────────────
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
    "answer_relevancy":     0.7,
    "faithfulness":         0.8,   # 幻觉风险最高，阈值设严一点
    "contextual_relevancy": 0.7,
    "contextual_precision": 0.7,
    "contextual_recall":    0.7,
}


# ── Step 1: 从 CSV 加载 Goldens 并调用 RAG Bot 构建 TestCase ────────────────
def load_test_cases() -> List[LLMTestCase]:
    """
    读取 goldens.csv → 调用 RAG Bot 获取 actual_output 和 retrieval_context
    → 组装成 DeepEval LLMTestCase 列表。

    CSV 列说明：
        question      → input（用户提问）
        ideal_answer  → expected_output（标准答案，供 Precision / Recall 使用）
        reference_docs → context（黄金参考文档，供 Precision / Recall 使用）
    """
    if not GOLDENS_CSV.exists():
        raise FileNotFoundError(
            f"找不到 Goldens 文件: {GOLDENS_CSV}\n"
            "请先运行 part2_goldens_dataset.py 生成数据集。"
        )

    test_cases: List[LLMTestCase] = []

    with open(GOLDENS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    print(f"📂 已加载 {total} 条 Golden，开始调用 RAG Bot 生成实际输出...\n")

    for i, row in enumerate(rows, start=1):
        question = row["question"].strip()
        expected = row.get("ideal_answer", "").strip()
        ref_docs  = row.get("reference_docs", "").strip()

        print(f"  [{i}/{total}] {question[:60]}{'...' if len(question) > 60 else ''}")

        # 调用 RAG Bot，获取实际回答和检索到的文档片段
        actual_output, retrieval_context = ask_rag_bot(question)

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=actual_output,          # RAG Bot 的真实输出
                expected_output=expected,             # 标准答案（Precision / Recall 依赖）
                context=[ref_docs] if ref_docs else None,  # 黄金参考上下文（同上）
                retrieval_context=retrieval_context,  # 实际检索到的文档片段（5 个指标都用）
            )
        )

    print(f"\n✅ TestCase 构建完成，共 {len(test_cases)} 条。\n")
    return test_cases


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

    generator_metrics   = [answer_relevancy, faithfulness]
    retriever_ref_free  = [contextual_relevancy]
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
            avg   = sum(scores) / len(scores)
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


# ── 主流程 ───────────────────────────────────────────────────────────────────
def run_evaluation() -> None:
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

    print("📋 本次评估指标：")
    print("   🔍 检索器")
    for m in ret_free + ret_ref:
        needs_ref = "需要 expected_output" if m in ret_ref else "无需参考答案"
        print(f"      · {m.__name__} ({needs_ref})")
    print("   ⚙️  生成器")
    for m in gen_metrics:
        print(f"      · {m.__name__} (无需参考答案)")
    print()

    # 4. 运行评估
    #    evaluate() 会并发调用 LLM Judge，结果同时上传 Confident AI Dashboard（如已登录）
    evaluate(
        test_cases=test_cases,
        metrics=all_metrics,
        run_async=True,          # 并发打分，节省时间
        show_indicator=True,     # 显示进度条
    )

    # 5. 打印本地汇总
    print_summary(test_cases, all_metrics)


# ── pytest 入口（支持 deepeval test run）───────────────────────────────────
import pytest


def _get_test_cases_for_pytest() -> List[LLMTestCase]:
    """延迟加载，避免 import 时就触发 RAG Bot 初始化。"""
    try:
        return load_test_cases()
    except Exception:
        return []


@pytest.mark.parametrize("test_case", _get_test_cases_for_pytest())
def test_rag_pipeline(test_case: LLMTestCase):
    """pytest 风格入口：每条 TestCase 单独断言，方便 CI 集成。"""
    judge = QwenModel()
    gen_metrics, ret_free, ret_ref = create_rag_metrics(judge)
    from deepeval import assert_test
    assert_test(test_case, gen_metrics + ret_free + ret_ref)


# ── 直接运行入口 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation()
