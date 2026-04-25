import os
from typing import List, Tuple
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ==========================================
# 配置参数
# ==========================================
PERSIST_DIRECTORY = "../chroma_db"
EMBEDDING_MODEL = "qwen3-embedding:4b"

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 300

# 检索配置：召回几个最相关的文档片段
RETRIEVER_K = 3


# ==========================================

class MCPRagBot:
    def __init__(self):
        # 1. 直接加载本地已有的向量库，不再爬取网页
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})

        # 2. 初始化本地 LLM
        self.llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )

        # 3. 定义 Prompt
        template = """基于以下检索到的上下文来回答问题：

        {context}

        请提供简明扼要的回答。如果上下文中没有答案，请直接说不知道。

        问题: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(template)

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join([d.page_content for d in docs])

    def ask(self, question: str) -> Tuple[str, List[str]]:
        """返回: (生成的答案, 检索到的文档片段列表)"""
        # 检索
        retrieved_docs = self.retriever.invoke(question)
        retrieval_context_list = [doc.page_content for doc in retrieved_docs]
        formatted_context = self._format_docs(retrieved_docs)

        # 生成
        chain = self.prompt | self.llm | StrOutputParser()
        actual_output = chain.invoke({
            "context": formatted_context,
            "question": question
        })

        return actual_output, retrieval_context_list


# 提供一个单例供外部调用
rag_bot = MCPRagBot()


def ask_rag_bot(question: str) -> Tuple[str, List[str]]:
    return rag_bot.ask(question)


# --- 简单的单测代码，方便读者拷贝后直接运行验证 ---
if __name__ == "__main__":
    test_question = "What is MCP?"
    print(f"🧐 正在提问: {test_question}")
    print("⏳ 机器人正在检索并生成回答...\n")

    ans, contexts = ask_rag_bot(test_question)

    print(f"🤖 回答:\n{ans}\n")
    print("-" * 40)
    print(f"📚 一共检索到了 {len(contexts)} 个文档片段。")