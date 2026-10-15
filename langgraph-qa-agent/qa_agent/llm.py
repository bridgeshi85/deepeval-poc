import os
from dotenv import load_dotenv
# 换用最新的通用初始化函数
from langchain.chat_models import init_chat_model

load_dotenv()

def get_llm():
    """
    使用 init_chat_model 初始化 DashScope (通义千问) 模型
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    model_name = os.getenv("DASHSCOPE_MODEL_NAME")

    if not api_key:
        raise ValueError("环境变量缺失: 请在 .env 文件中配置 DASHSCOPE_API_KEY")
    if not model_name:
        raise ValueError("环境变量缺失: 请在 .env 文件中配置 DASHSCOPE_MODEL_NAME")

    # 使用统一工厂函数
    llm = init_chat_model(
        model=model_name,
        model_provider="openai",  # 告诉工厂函数使用 OpenAI 兼容规范
        api_key=api_key,
        base_url="https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
        max_retries=3
    )
    
    return llm

def test_llm_connection():
    print("正在初始化 LLM...")
    try:
        llm = get_llm()
        # 注意：init_chat_model 返回的对象其实就是 ChatOpenAI 的实例
        print(f"✅ 模型对象创建成功!")
    except Exception as e:
        print(f"❌ 模型对象创建失败: {e}")
        return

    print("正在测试 API 连通性 (发送 'ping')...")
    try:
        response = llm.invoke("ping")
        print("✅ API 调用成功！")
        print(f"模型回复: {response.content}")
    except Exception as e:
        print(f"❌ API 调用失败: {e}")

if __name__ == "__main__":
    test_llm_connection()