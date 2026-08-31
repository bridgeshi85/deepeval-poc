"""
agent_instrumented.py
=====================
SAME agent as agent_plain.py, with DeepEval observability added.

What changed vs agent_plain.py? Only 4 things — all marked with "NEW":

  (1) Import CallbackHandler from deepeval.integrations.langchain
  (2) Build the callback handler once (no metrics baked in — metrics are
      attached by the test file at run time)
  (3) Pass it via config={"callbacks": [...]} on .invoke()
  (4) (Optional) Add @observe on the outer Python wrapper so any
      non-LangChain code we add later (guardrails, post-processing)
      also shows up as spans

That's it. Four lines, and now QA can run component-level evals
without ever touching this file again.
"""

import asyncio
import os

import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()
from langchain_core.tools import tool
from langchain.agents import create_agent

# --- (1) NEW: import DeepEval tracing primitives --------------------------
from deepeval.integrations.langchain import CallbackHandler
from deepeval.tracing.context import update_current_trace


# ---------------------------------------------------------------------------
# Same fake data as agent_plain.py.
# ---------------------------------------------------------------------------
ORDERS = {
    "ORD-1042": {"status": "Shipped",    "eta": "2026-05-13", "category": "Electronics"},
    "ORD-2099": {"status": "Delivered",  "eta": "2026-05-08", "category": "clothing"},
    "ORD-7777": {"status": "Processing", "eta": "2026-05-15", "category": "food"},
}

REFUND_POLICIES = {
    "electronics": "Electronics can be returned within 15 days, unopened.",
    "clothing":    "Clothing can be returned within 30 days with tags attached.",
    "food":        "Food items are non-returnable for safety reasons.",
}


# ---------------------------------------------------------------------------
# Same tools as agent_plain.py.
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
# Same agent as agent_plain.py.
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# Load one extra tool from the MCP server at startup and merge it in.
_mcp_tools = asyncio.run(
    MultiServerMCPClient({
        "shopease": {
            "command": "python",
            "args": [os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")],
            "transport": "stdio",
        }
    }).get_tools()
)

agent = create_agent(
    model=llm,
    tools=[get_order_status, get_refund_policy] + _mcp_tools,
    system_prompt=(
        "You are a friendly customer-support agent. "
        "Use the available tools to answer order and refund questions. "
        "Keep replies short and helpful."
    ),
)


# --- (2) NEW: a single shared callback handler ----------------------------
# This handler captures every LangChain LLM call, tool call, and chain step
# as a DeepEval span automatically. No per-function decoration needed for
# anything LangChain owns.
deepeval_callback = CallbackHandler()


def support_agent(user_input: str) -> str:
    """Run the agent on a user message and return the final reply."""
    # --- (3) NEW: pass the callback handler in the .invoke() config ------
    result = asyncio.run(agent.ainvoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": [deepeval_callback]},
    ))
    reply = result["messages"][-1].content
    update_current_trace(output=reply)
    return reply


if __name__ == "__main__":
    print(support_agent("Where is my order ORD-1042?"))
