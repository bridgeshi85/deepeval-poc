import csv
import sys
from pathlib import Path

# 确保能 import 到项目根的 custom_llm（不依赖运行时 cwd）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer.config import ContextConstructionConfig
from custom_llm import QwenModel, OllamaEmbeddingModel


def generate_auto_goldens():
    print("🚀 准备启动 AI 自动出题机 ...")
    # 获取当前文件的绝对路径，并计算项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # evaluation/ 的上一级是项目根目录
    data_dir = project_root / "data"
    knowledge_file = data_dir / "mcp_knowledge.txt"
    output_csv = data_dir / "goldens.csv"

    # 确保输出目录存在
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. 初始化模型实例
    qwen_model = QwenModel()
    local_embedder = OllamaEmbeddingModel()

    # 2. 设置文档分块配置（用于上下文构建）
    context_config = ContextConstructionConfig(
        critic_model=qwen_model,  # 评判模型，用于生成问题和答案
        embedder=local_embedder,  # 嵌入模型，用于语义分块
        chunk_size=500,  # 每个文档块的最大字符数
        chunk_overlap=50  # 相邻块之间的重叠字符数，保持上下文连贯
    )

    # 3. 创建合成器并生成 Goldens
    synthesizer = Synthesizer(
        model=qwen_model,
    )
    print(f"📖 AI 正在阅读文档，并使用进化算法生成高难度考题 ...")

    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[str(knowledge_file)],  # 使用绝对路径
        max_goldens_per_context=2,  # 每个context生成2个测试用例
        context_construction_config=context_config
    )

    dataset = EvaluationDataset(goldens=goldens)

    # 4 保存到 CSV
    print(f"💾 正在将考题写入 {output_csv} ...")
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "expected_output", "context"])
        for golden in dataset.goldens:
            context_str = " | ".join(golden.context) if golden.context else ""
            writer.writerow([
                golden.input,
                golden.expected_output,
                context_str
            ])

    print(f"\n✅ 太棒了！成功生成数据集并保存至 {output_csv}")
    print(f"一共生成了 {len(dataset.goldens)} 道 Golden 考题。")


if __name__ == "__main__":
    generate_auto_goldens()
