"""
Part 2: Goldens Dataset - Test Data Structure
Defines the structure for test cases with input, expected_output, and captured contexts.
"""
from dataclasses import dataclass, field
from typing import List
from langchain.schema import Document


@dataclass
class GoldenTestCase:
    """A single test case for RAG evaluation."""
    input: str                          # User question
    expected_output: str                # Reference answer
    retrieved_contexts: List[str] = field(default_factory=list)  # Retrieved docs
    actual_output: str = ""             # Model response (filled during test)

    @classmethod
    def from_question(cls, question: str, expected: str):
        return cls(input=question, expected_output=expected)


@dataclass
class EvaluationDataset:
    """Collection of test cases for evaluation."""
    test_cases: List[GoldenTestCase] = field(default_factory=list)

    def add(self, question: str, expected: str) -> GoldenTestCase:
        """Add a test case to the dataset."""
        tc = GoldenTestCase.from_question(question, expected)
        self.test_cases.append(tc)
        return tc

    def __len__(self):
        return len(self.test_cases)

    def __iter__(self):
        return iter(self.test_cases)


# Sample test data for POC
SAMPLE_TEST_DATA = [
    {
        "input": "What is LangChain?",
        "expected_output": "LangChain is a framework for developing applications powered by language models."
    },
    {
        "input": "What are the main components of LangChain?",
        "expected_output": "LangChain consists of Models, Prompts, Chains, Agents, and Memory components."
    },
    {
        "input": "How does RAG improve LLM responses?",
        "expected_output": "RAG (Retrieval-Augmented Generation) improves LLM responses by retrieving relevant context from a knowledge base."
    },
]


def create_test_dataset() -> EvaluationDataset:
    """Create a test dataset from sample data."""
    dataset = EvaluationDataset()
    for item in SAMPLE_TEST_DATA:
        dataset.add(item["input"], item["expected_output"])
    return dataset


if __name__ == "__main__":
    dataset = create_test_dataset()
    for tc in dataset:
        print(f"Q: {tc.input}")
        print(f"A: {tc.expected_output}")
        print("---")
