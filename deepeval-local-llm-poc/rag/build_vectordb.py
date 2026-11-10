import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ==========================================
# 配置参数
# ==========================================
TARGET_URL = "https://www.descope.com/learn/post/mcp"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "qwen3-embedding:4b"
# 相对本文件定位，从任意 cwd 运行结果一致（与 rag_app.py 保持一致）
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../chroma_db")


# ==========================================

def build_vector_db():
    print("🚀 开始构建本地向量知识库...")

    # 1. 抓取网页数据
    print(f"1. 正在抓取数据 ({TARGET_URL})...")
    loader = WebBaseLoader(TARGET_URL)
    data = loader.load()

    # 2. 文本切片
    print(f"2. 正在切分文本 (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(data)

    # 3. 嵌入并持久化保存到本地目录
    print(f"3. 正在调用本地模型 [{EMBEDDING_MODEL}] 生成向量...")
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print(f"4. 正在将数据持久化保存到硬盘目录: {PERSIST_DIRECTORY} ...")
    # persist_directory 是关键，它会将数据写到硬盘
    Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=PERSIST_DIRECTORY
    )

    print("✅ 向量库构建完成！")


if __name__ == "__main__":
    build_vector_db()
