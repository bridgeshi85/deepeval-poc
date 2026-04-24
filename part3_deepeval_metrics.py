"""
Part 3: DeepEval Metrics
Business metrics (GEval) and System metrics for RAG evaluation.
"""
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from typing import List

# Business Metrics using GEval
def create_conciseness_metric():
    """Conciseness metric - measures if response is concise."""
    return GEval(
        name="Conciseness",
        criteria="The response should be concise and to the point, avoiding unnecessary verbosity.",
        evaluation_steps=[
            "Check if the response addresses the question directly",
            "Check if the response contains excessive irrelevant details",
            "Rate the conciseness from 0 to 1"
        ],
        evaluation_params=[LLMTestCase.EXPECTED_OUTPUT, LLMTestCase.ACTUAL_OUTPUT]
    )


def create_completeness_metric():
    """Completeness metric - measures if response covers all aspects."""
    return GEval(
        name="Completeness",
        criteria="The response should be complete and cover all key aspects of the question.",
        evaluation_steps=[
            "Check if all parts of the question are addressed",
            "Check if the response provides sufficient detail",
            "Rate the completeness from 0 to 1"
        ],
        evaluation_params=[LLMTestCase.INPUT, LLMTestCase.ACTUAL_OUTPUT]
    )


# System Metrics
def create_faithfulness_metric():
    """Faithfulness metric - measures if response matches retrieved context."""
    from deepeval.metrics import FaithfulnessMetric
    return FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4",
        include_reason=True
    )


def create_contextual_precision_metric():
    """Contextual Precision metric - measures retrieval quality."""
    from deepeval.metrics import ContextualPrecisionMetric
    return ContextualPrecisionMetric(
        threshold=0.7,
        model="gpt-4",
        include_reason=True
    )


def evaluate_with_deepeval(test_case: LLMTestCase, metrics: List) -> dict:
    """Run DeepEval metrics on a test case."""
    results = {}
    for metric in metrics:
        metric.measure(test_case)
        results[metric.name] = {
            "score": metric.score,
            "reason": metric.reason if hasattr(metric, 'reason') else None
        }
    return results


if __name__ == "__main__":
    # Example usage
    # test_case = LLMTestCase(
    #     input="What is LangChain?",
    #     actual_output="LangChain is a framework for LLM applications.",
    #     expected_output="LangChain is a framework for developing applications powered by language models.",
    #     retrieval_context=["LangChain is a framework...", "It provides tools..."]
    # )
    #
    # metrics = [create_conciseness_metric(), create_faithfulness_metric()]
    # results = evaluate_with_deepeval(test_case, metrics)
    pass
