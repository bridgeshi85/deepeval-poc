import os

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseEmbeddingModel, DeepEvalBaseLLM
from dotenv import load_dotenv

load_dotenv()


class QwenModel(DeepEvalBaseLLM):
    def __init__(self, *args, **kwargs):
        # 接入阿里云通义千问 API (使用 OpenAI 兼容格式)
        self.model = ChatOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),  # 生产环境中建议使用 os.getenv("DASHSCOPE_API_KEY")
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=os.getenv("DASHSCOPE_MODEL_NAME")
        )
        super().__init__(*args, **kwargs)

    def load_model(self) -> ChatOpenAI:
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Qwen-Judge"


class OllamaEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name="qwen3-embedding:4b", *args, **kwargs):
        # 接入本地 Ollama 的 Embedding 模型
        self.embedder = OllamaEmbeddings(model=model_name)
        super().__init__(*args, **kwargs)

    def load_model(self) -> OllamaEmbeddings:
        return self.embedder

    def embed_text(self, text: str) -> list[float]:
        return self.load_model().embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.load_model().embed_documents(texts)

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.load_model().aembed_query(text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.load_model().aembed_documents(texts)

    def get_model_name(self):
        return "Ollama-Qwen3-Embedding"
