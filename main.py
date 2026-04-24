"""
Main entry point for POC evaluation pipeline.
Demonstrates the complete flow from baseline RAG to evaluation.
"""
from part1_baseline_rag import create_rag_chain, query
from part2_goldens_dataset import create_test_dataset, EvaluationDataset
from part3_deepeval_metrics import (
    create_conciseness_metric,
    create_completeness_metric,
    create_faithfulness_metric,
    create_contextual_precision_metric,
    evaluate_with_deepeval
)
from part4_ragas_evaluation import create_ragas_dataset, run_evaluation
from deepeval.test_case import LLMTestCase


def run_deepeval_evaluation(dataset: EvaluationDataset, rag_chain):
    """Run DeepEval evaluation on dataset."""
    print("=" * 60)
    print("Running DeepEval Evaluation")
    print("=" * 60)

    metrics = [
        create_conciseness_metric(),
        create_completeness_metric(),
        create_faithfulness_metric(),
        create_contextual_precision_metric()
    ]

    for tc in dataset:
        # Query RAG chain
        result = query(rag_chain, tc.input)
        tc.actual_output = result["answer"]
        tc.retrieved_contexts = [doc.page_content for doc in result["source_documents"]]

        # Create DeepEval test case
        test_case = LLMTestCase(
            input=tc.input,
            actual_output=tc.actual_output,
            expected_output=tc.expected_output,
            retrieval_context=tc.retrieved_contexts
        )

        # Evaluate
        results = evaluate_with_deepeval(test_case, metrics)
        print(f"\nQ: {tc.input}")
        for metric_name, result in results.items():
            print(f"  {metric_name}: {result['score']:.2f}")


def run_ragas_evaluation(dataset: EvaluationDataset, rag_chain):
    """Run RAGAS evaluation on dataset."""
    print("\n" + "=" * 60)
    print("Running RAGAS Evaluation")
    print("=" * 60)

    # Populate dataset with actual responses
    for tc in dataset:
        result = query(rag_chain, tc.input)
        tc.actual_output = result["answer"]
        tc.retrieved_contexts = [doc.page_content for doc in result["source_documents"]]

    # Convert to RAGAS format
    ragas_dataset = create_ragas_dataset(dataset)

    # Run evaluation
    results_df = run_evaluation(ragas_dataset)
    print("\n", results_df)

    return results_df


def main():
    """Main evaluation pipeline."""
    print("POC: Local LLM RAG Evaluation with DeepEval & RAGAS")
    print("=" * 60)

    # 1. Create test dataset
    dataset = create_test_dataset()
    print(f"Loaded {len(dataset)} test cases")

    # 2. Create RAG chain (requires vector store to be populated)
    # rag_chain = create_rag_chain(vector_store)

    # 3. Run evaluations
    # run_deepeval_evaluation(dataset, rag_chain)
    # run_ragas_evaluation(dataset, rag_chain)

    print("\nPipeline ready. Uncomment the RAG chain setup to run evaluations.")


if __name__ == "__main__":
    main()
