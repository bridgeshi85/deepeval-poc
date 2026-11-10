"""
agent_plain.py
==============
A simple customer-support agent built with LangChain + Claude.
No evaluation code anywhere. This is what a developer would write
if they'd never heard of DeepEval.

The agent has two tools:
  - get_order_status(order_id)  -> shipping info
  - get_refund_policy(category) -> refund rules

It uses Claude to decide which tool to call and to compose the answer.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()
from langchain_core.tools import tool
from langchain.agents import create_agent


# ---------------------------------------------------------------------------
# Fake in-memory data — keeps the demo offline and deterministic.
# ---------------------------------------------------------------------------
ORDERS = {
    "ORD-1042": {"status": "Shipped",    "eta": "2026-05-13"},
    "ORD-2099": {"status": "Delivered",  "eta": "2026-05-08"},
    "ORD-7777": {"status": "Processing", "eta": "2026-05-15"},
}

REFUND_POLICIES = {
    "electronics": "Electronics can be returned within 15 days, unopened.",
    "clothing":    "Clothing can be returned within 30 days with tags attached.",
    "food":        "Food items are non-returnable for safety reasons.",
}


# ---------------------------------------------------------------------------
# Tools — plain Python functions exposed to the agent via @tool.
# ---------------------------------------------------------------------------
@tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order by its ID."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"No order found with ID {order_id}."
    return f"Order {order_id} is {order['status']}. ETA: {order['eta']}."


@tool
def get_refund_policy(category: str) -> str:
    """Return the refund policy for a product category."""
    policy = REFUND_POLICIES.get(category.lower())
    if not policy:
        return f"No refund policy on file for '{category}'."
    return policy


# ---------------------------------------------------------------------------
# The agent — a LangGraph ReAct agent powered by Claude.
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

agent = create_agent(
    model=llm,
    tools=[get_order_status, get_refund_policy],
    system_prompt=(
        "You are a friendly customer-support agent. "
        "Use the available tools to answer order and refund questions. "
        "Keep replies short and helpful."
    ),
)


# ---------------------------------------------------------------------------
# Single entry point — this is what test code will import.
# ---------------------------------------------------------------------------
def support_agent(user_input: str) -> str:
    """Run the agent on a user message and return the final reply."""
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print(support_agent("Where is my order ORD-1099?"))
