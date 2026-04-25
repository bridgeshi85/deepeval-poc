"""Part 1: Baseline RAG application for the DeepEval POC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    EMBEDDING_MODEL,
    KNOWLEDGE_BASE_PATH,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PERSIST_DIRECTORY,
    RETRIEVER_TOP_K,
)


DEFAULT_PROMPT = """你是一个用于评估的简洁 RAG 助手。请仅基于提供的上下文回答问题。

上下文:
{context}

要求:
- 优先直接回答问题
- 如果上下文无法支持答案，明确回答“不知道”
- 回答尽量简洁，方便后续评估

问题: {question}
"""


def load_documents(file_path: str = KNOWLEDGE_BASE_PATH) -> List[Document]:
    """Load knowledge base documents from a local text file."""
    source_path = Path(file_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {source_path}")
    loader = TextLoader(str(source_path), encoding="utf-8")
    return loader.load()


def create_vector_store(
    documents: List[Document],
    persist_directory: Optional[str] = PERSIST_DIRECTORY,
) -> Chroma:
    """Create a Chroma vector store from local documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def _build_generation_chain(llm: ChatOllama):
    prompt = ChatPromptTemplate.from_template(DEFAULT_PROMPT)
    return prompt | llm | StrOutputParser()


@dataclass
class RAGChainBundle:
    retriever: Any
    generation_chain: Any


def create_rag_chain(vector_store: Chroma) -> RAGChainBundle:
    """Create a lightweight RAG chain bundle."""
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=0.3,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    return RAGChainBundle(
        retriever=retriever,
        generation_chain=_build_generation_chain(llm),
    )


def format_docs(docs: List[Document]) -> str:
    """Convert retrieved documents into a prompt-ready context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def query(chain: RAGChainBundle, question: str) -> dict:
    """Query the RAG chain and return answer plus retrieved contexts."""
    retriever = chain.retriever
    generation_chain = chain.generation_chain

    retrieved_docs = retriever.invoke(question)
    answer = generation_chain.invoke(
        {
            "context": format_docs(retrieved_docs),
            "question": question,
        }
    )

    return {
        "question": question,
        "answer": answer,
        "source_documents": retrieved_docs,
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
    }


class SimpleRAGChatbot:
    """Small wrapper class that DeepEval can treat as the system under test."""

    def __init__(
        self,
        knowledge_base_path: str = KNOWLEDGE_BASE_PATH,
        persist_directory: Optional[str] = PERSIST_DIRECTORY,
    ):
        self.knowledge_base_path = knowledge_base_path
        self.documents = load_documents(knowledge_base_path)
        self.vector_store = create_vector_store(self.documents, persist_directory)
        self.chain = create_rag_chain(self.vector_store)

    def ask(self, question: str) -> dict:
        return query(self.chain, question)


def build_demo_chatbot(
    knowledge_base_path: str = KNOWLEDGE_BASE_PATH,
    persist_directory: Optional[str] = PERSIST_DIRECTORY,
) -> SimpleRAGChatbot:
    """Factory used by demos and evaluation scripts."""
    return SimpleRAGChatbot(
        knowledge_base_path=knowledge_base_path,
        persist_directory=persist_directory,
    )


if __name__ == "__main__":
    bot = build_demo_chatbot()
    result = bot.ask("What is MCP?")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Retrieved Chunks: {len(result['retrieved_contexts'])}")
