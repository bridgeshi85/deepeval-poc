"""
Part 4: RAGAS Evaluation
RAGAS framework for data analysis perspective evaluation.
"""
import pandas as pd
from ragas import EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    contextual_precision,
    contextual_recall
)


def create_ragas_dataset(test_cases) -> EvaluationDataset:
    """Convert test cases to RAGAS EvaluationDataset format."""
    data = {
        "user_input": [tc.input for tc in test_cases],
        "retrieved_contexts": [tc.retrieved_contexts for tc in test_cases],
        "response": [tc.actual_output for tc in test_cases],
        "reference": [tc.expected_output for tc in test_cases]
    }
    return EvaluationDataset.from_dict(data)


def evaluate_with_ragas(dataset: EvaluationDataset, llm_model: str = "gpt-4") -> pd.DataFrame:
    """Run RAGAS evaluation and return results as DataFrame."""
    metrics = [
        faithfulness,
        answer_correctness,
        contextual_precision,
        contextual_recall
    ]

    result = evaluate(dataset, metrics=metrics)

    # Convert to DataFrame for analysis
    scores = {}
    for metric_name in result.keys():
        scores[metric_name] = [s[0] for s in result[metric_name].scores]

    return pd.DataFrame(scores)


def run_evaluation(dataset: EvaluationDataset) -> pd.DataFrame:
    """Run full RAGAS evaluation pipeline."""
    from ragas.evaluation import evaluate

    metrics = [
        faithfulness,
        answer_correctness,
        contextual_precision,
        contextual_recall
    ]

    results = evaluate(dataset, metrics=metrics)

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "question": row.user_input,
            "faithfulness": row.faithfulness,
            "answer_correctness": row.answer_correctness,
            "contextual_precision": row.contextual_precision,
            "contextual_recall": row.contextual_recall
        }
        for row in results
    ])

    return df


if __name__ == "__main__":
    # Example usage
    # from part2_goldens_dataset import create_test_dataset
    # dataset = create_ragas_dataset(create_test_dataset())
    # results_df = run_evaluation(dataset)
    # print(results_df)
    pass
