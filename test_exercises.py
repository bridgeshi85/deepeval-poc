from deepeval.evaluate import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI  # 借用 LangChain 的标准接口调用兼容 API


# 如果使用本地 ollama，可以换成: from langchain_ollama import ChatOllama

# 1. 创建自定义的 LLM 裁判类
class QwenJudgeModel(DeepEvalBaseLLM):
    def __init__(self, *args, **kwargs):
        # 方案 A：接入阿里云通义千问 API (使用 OpenAI 兼容格式)

        self.model = ChatOpenAI(
            api_key="",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3.6-max-preview"  # 或 qwen-max, qwen-turbo
        )
        super().__init__(*args, **kwargs)

        # 方案 B：如果你的千问是跑在本地的 Ollama 上，用下面这行替换上面：
        # self.model = ChatOllama(model="qwen2.5:latest", base_url="http://localhost:11434")

    def load_model(self):
        return self.model

    # DeepEval 需要同步生成方法
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        return response.content

    # DeepEval 需要异步生成方法 (为了并发测试加速)
    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Qwen-Judge"


# 2. 实例化你的自定义裁判模型
custom_qwen_judge = QwenJudgeModel()

# 3. 在指标中显式指定使用该模型
answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=custom_qwen_judge  # <-- 关键点：覆盖默认的 OpenAI 模型
)


# answer_relevancy_metric.measure(test_case)
# print(f"得分: {answer_relevancy_metric.score}")
# print(f"评价理由: {answer_relevancy_metric.reason}")

def test_answer_relevancy():
    custom_qwen_judge = QwenJudgeModel()

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=custom_qwen_judge,
    )

    test_case = LLMTestCase(
        input="Who is the current president of the United States of America?",
        actual_output="Joe Biden",
        retrieval_context=[
            "Joe Biden serves as the current president of America."
        ],
    )

    assert_test(test_case, [metric])
