from collections.abc import Mapping
from pprint import pformat
from typing import Any


def log_event(node_name: str, event: Mapping[str, Any]) -> None:
    """打印节点状态增量；事件含消息时同时展示工具调用。"""
    print(f"\n--- [更新: {node_name}] ---")
    for key, value in event.items():
        print(f"{key}: {pformat(value, sort_dicts=False)}")
    for message in event.get("messages", []):
        calls = (
            message.get("tool_calls", [])
            if isinstance(message, Mapping)
            else getattr(message, "tool_calls", [])
        )
        for call in calls:
            print(f"工具调用: {call['name']} 参数: {pformat(call['args'])}")