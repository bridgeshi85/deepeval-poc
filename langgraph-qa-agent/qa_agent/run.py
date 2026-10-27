import asyncio
import sys
from pathlib import Path

from datasets import load_dataset

# 支持 `python qa_agent/run.py` 直接运行：把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_agent.graph import app as graph  # noqa: E402
from qa_agent.trace import log_event  # noqa: E402


async def run() -> None:
    dataset = load_dataset("cais/mmlu", "high_school_geography", split="test")
    sample = dataset[0]
    choices = sample["choices"]
    options = "\n".join(
        f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)
    )
    answer_index = sample["answer"]
    reference_answer = f"{chr(ord('A') + answer_index)}. {choices[answer_index]}"

    print("MMLU: high_school_geography / test / sample 0")
    print(f"\n题目：\n{sample['question']}\n\n选项：\n{options}")
    inputs = {"question": sample["question"], "options": options}
    agent_answer = None
    async for _, updates in graph.astream(inputs, stream_mode=["updates"]):
        for node_name, event in updates.items():
            log_event(node_name, event)
            if "current_answer" in event:
                agent_answer = event["current_answer"]

    if not agent_answer:
        raise RuntimeError("图运行结束，但未产生 Agent 答案。")

    print("\n=== 最终结果（仅供人工核对） ===")
    print(f"Agent 答案：\n{agent_answer}")
    print(f"\nMMLU 标准答案：\n{reference_answer}")
    print("\nDemo 不承诺答题准确率；标准答案仅供人工核对，不作为自动评测指标。")


if __name__ == "__main__":
    asyncio.run(run())