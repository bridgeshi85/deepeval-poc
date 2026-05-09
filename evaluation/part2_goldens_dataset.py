from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer.config import ContextConstructionConfig
from custom_llm import QwenModel, OllamaEmbeddingModel  # 假设自定义类保存在 custom_llm.py


def generate_auto_goldens():
    print("🚀 准备启动 AI 自动出题机 ...")

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
    synthesizer = Synthesizer()
    print(f"📖 AI 正在阅读文档，并使用进化算法生成高难度考题 ...")

    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=["../data/mcp_knowledge.txt"],  # 支持多文档列表
        max_goldens_per_context=5,  # 每个文档块最多生成 5 个测试用例
        context_construction_config=context_config
    )

    # 4. 组装成 EvaluationDataset 对象
    dataset = EvaluationDataset(goldens=goldens)

    # 5. 导出为 CSV 文件，实现“测试代码”与“测试数据”的物理隔离
    output_dir = "./"
    dataset.save_as(file_type="csv", directory=output_dir)

    # 6. 打印统计信息
    print(f"\n✅ 太棒了！成功生成数据集并保存至 {output_dir}")
    print(f"    一共生成了 {len(dataset.goldens)} 道 Golden 考题。")


if __name__ == "__main__":
    generate_auto_goldens()
