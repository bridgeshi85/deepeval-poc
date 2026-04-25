"""
Part 2: Goldens Dataset - Test Data Structure
Defines the structure for test cases with input, expected_output, and captured contexts.
"""
from dataclasses import dataclass, field
from typing import List


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
        "input": "What is MCP?",
        "expected_output": "MCP is an open protocol for connecting language-model applications to external tools and data sources."
    },
    {
        "input": "What are the core roles in MCP?",
        "expected_output": "The core roles in MCP are the host application, the client, and the server."
    },
    {
        "input": "Why would a team use MCP in an LLM app?",
        "expected_output": "A team would use MCP to standardize how an LLM app connects to tools, data sources, and internal systems instead of building one-off integrations."
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
