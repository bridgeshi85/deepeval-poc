"""
Part 1: Baseline RAG Application
A simple LangChain + Chroma RAG application for POC evaluation.
"""
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from config import LLM_MODEL, OLLAMA_BASE_URL, PERSIST_DIRECTORY


def load_documents(file_path: str):
    """Load documents from a text file."""
    loader = TextLoader(file_path)
    return loader.load()


def create_vector_store(documents, persist_directory: str = PERSIST_DIRECTORY):
    """Create Chroma vector store from documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_BASE_URL
    )

    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_directory
    )


def create_rag_chain(vector_store):
    """Create a RAG QA chain."""
    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )

    retriever = vector_store.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain


def query(chain, question: str) -> dict:
    """Query the RAG chain and return result with metadata."""
    result = chain.invoke(question)
    return {
        "question": question,
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }


if __name__ == "__main__":
    # Example usage
    # 1. Load documents
    docs = load_documents("your_data.txt")

    # 2. Create vector store
    # vector_store = create_vector_store(docs)

    # 3. Create RAG chain
    # chain = create_rag_chain(vector_store)

    # 4. Query
    # result = query(chain, "What is the main topic?")
    # print(result["answer"])
    pass
