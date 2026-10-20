import os
import time
from datetime import timedelta

import wikipedia
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import ToolException

# Wikimedia 要求使用可识别的 User-Agent；通用 UA 更容易被限流
WIKIPEDIA_USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "deepeval-poc-qa-agent/0.1 (https://github.com/bridgeshi85/deepeval-poc)",
)
RETRY_DELAYS = (3, 10)


def _make_resilient(tool):
    """外部检索偶发限流/网络错误时重试，最终转成工具错误交给模型，而不是让整个 Agent 崩溃。"""
    original_run = tool._run

    def run_with_retry(*args, **kwargs):
        last_error = None
        for delay in (*RETRY_DELAYS, None):
            try:
                return original_run(*args, **kwargs)
            except Exception as e:
                last_error = e
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
    这里仅仅是实例化工具对象，不会触发真实的外部 API 搜索。
    """
    wikipedia.wikipedia.USER_AGENT = WIKIPEDIA_USER_AGENT
    wikipedia.set_rate_limiting(True, min_wait=timedelta(seconds=1))

    # LangChain 内置了对这些常见搜索工具的封装
    tools = load_tools(["ddg-search", "arxiv", "wikipedia"])
    return [_make_resilient(tool) for tool in tools]

if __name__ == "__main__":
    tools = get_tools()
    print("✅ 成功加载工具:")
    for tool in tools:
        print(f" - {tool.name}: {tool.description[:60]}...")