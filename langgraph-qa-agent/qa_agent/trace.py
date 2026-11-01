from collections.abc import Mapping
from pprint import pformat
from typing import Any


def log_event(node_name: str, event: Mapping[str, Any]) -> None:
    """只打印文章需要的轮次、反馈和工具调用，不展开内部状态。"""
    if node_name in ("research_start", "research") and "steps" in event:
        print(f"\n第 {event['steps']} 轮研究完成")
    elif node_name == "reflect":
        if event.get("feedback"):
            print(f"教授反馈：{event['feedback']}")
        else:
            print("教授审查完成")

    for message in event.get("messages", []):
        calls = (
            message.get("tool_calls", [])
            if isinstance(message, Mapping)
            else getattr(message, "tool_calls", [])
        )
        for call in calls:
            print(f"工具调用: {call['name']} 参数: {pformat(call['args'])}")
