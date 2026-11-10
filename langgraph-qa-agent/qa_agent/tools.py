import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import timedelta
from functools import wraps

import wikipedia
import requests
from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import ToolException
from langchain_tavily import TavilySearch
from langchain_tavily._utilities import TavilySearchAPIWrapper, TAVILY_API_URL

# Wikimedia 要求使用可识别的 User-Agent；通用 UA 更容易被限流
WIKIPEDIA_USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "deepeval-poc-qa-agent/0.1 (https://github.com/bridgeshi85/deepeval-poc)",
)
RETRY_DELAYS = (3,)
TOOL_TIMEOUT = 20  # 单次工具调用的最长等待秒数


class BoundedTavilyAPIWrapper(TavilySearchAPIWrapper):
    def raw_results(self, **kwargs):
        # 此版本的 Tavily 包未给 requests.post 设置超时。
        response = requests.post(
            f"{self.api_base_url or TAVILY_API_URL}/search",
            json={key: value for key, value in kwargs.items() if value is not None},
            headers={
                "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Client-Source": "langchain-tavily",
            },
            timeout=(5, 10),
        )
        response.raise_for_status()
        return response.json()


def _make_resilient(tool):
    """外部检索偶发限流/网络错误时重试，最终转成工具错误交给模型，而不是让整个 Agent 崩溃。"""
    original_run = tool._run

    def run_with_timeout(*args, **kwargs):
        # 底层搜索库没有可靠的超时；超时后放弃等待，线程在后台自行结束
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(copy_context().run, original_run, *args, **kwargs).result(timeout=TOOL_TIMEOUT)
        finally:
            pool.shutdown(wait=False)

    @wraps(original_run)
    def run_with_retry(*args, **kwargs):
        started = time.monotonic()
        print(f"[tool:{tool.name}] 开始检索", flush=True)
        last_error = None
        for delay in (*RETRY_DELAYS, None):
            try:
                result = run_with_timeout(*args, **kwargs)
                # Tavily 将网络异常作为 error 字段返回，需要转回异常才能统一处理。
                if isinstance(result, dict) and result.get("error"):
                    error = result["error"]
                    if isinstance(error, Exception):
                        raise error
                    raise ToolException(str(error))
                print(
                    f"[tool:{tool.name}] 完成，耗时 {time.monotonic() - started:.1f}s",
                    flush=True,
                )
                return result
            except Exception as e:
                print(f"[tool:{tool.name}] 调用失败: {type(e).__name__} {e}")
                last_error = e
                if isinstance(e, (TimeoutError, requests.exceptions.Timeout, ToolException)):
                    break
                if delay is not None:
                    time.sleep(delay)
        raise ToolException(
            f"工具 {tool.name} 暂时不可用（{type(last_error).__name__}），"
            "请换用其他工具或调整查询后重试。"
        )

    # 工具是 pydantic 模型，直接赋值会被拒绝，这里绕过字段校验替换 _run
    object.__setattr__(tool, "_run", run_with_retry)
    tool.handle_tool_error = True
    return tool


def get_tools():
    """
    加载并返回所需的检索工具列表。
    这里仅仅是实例化工具对象，不会触发真实的外部 API 搜索，但需要 Tavily Key。
    """
    load_dotenv()
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("环境变量缺失: 请在 .env 文件中配置 TAVILY_API_KEY")

    wikipedia.wikipedia.USER_AGENT = WIKIPEDIA_USER_AGENT
    wikipedia.set_rate_limiting(True, min_wait=timedelta(seconds=1))

    tools = [
        TavilySearch(
            api_wrapper=BoundedTavilyAPIWrapper(),
            max_results=3,
            search_depth="basic",
            auto_parameters=False,
            include_answer=False,
            include_raw_content=False,
        ),
        *load_tools(["arxiv", "wikipedia"]),
    ]
    return [_make_resilient(tool) for tool in tools]

if __name__ == "__main__":
    tools = get_tools()
    print("✅ 成功加载工具:")
    for tool in tools:
        print(f" - {tool.name}: {tool.description[:60]}...")
