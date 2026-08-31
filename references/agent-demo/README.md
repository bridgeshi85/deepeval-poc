# DeepEval Agentic Testing Demo

A minimal teaching repo that shows how to evaluate a LangChain agent with
DeepEval's component-level evaluation: **Task Completion** and **Tool
Correctness** metrics, applied to a cross-vendor setup (Claude as the
agent, GPT-4o as the judge).

---

## Files in this repo

| File | Who writes it (in real life) | What it does |
|---|---|---|
| `agent_plain.py` | Developer (before evals exist) | A normal LangChain + Claude agent. No DeepEval code anywhere. |
| `agent_instrumented.py` | Developer (after QA asks) | The **same** agent, with **4 lines added** to make it observable to DeepEval. |
| `test_agent.py` | QA / Tester (you) | Imports the agent, defines goldens, runs metrics. |

---

## The teaching point

The diff between `agent_plain.py` and `agent_instrumented.py` is **4 lines**.
That's the entire ask QA makes of dev:

1. `from deepeval.integrations.langchain import CallbackHandler`
2. `deepeval_callback = CallbackHandler()`
3. `config={"callbacks": [deepeval_callback]}` on `.invoke()`
4. `@observe(name="support_agent")` on the outer wrapper function

Everything else stays exactly the same.

Show students both files side by side. Once they see the diff is tiny,
they stop being intimidated by component-level evaluation.

---

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# 3. (Optional but recommended) log in to Confident AI to see traces in a UI
deepeval login

# 4. Sanity check: run the agent on its own
python agent_instrumented.py

# 5. Run the evaluation
python test_agent.py
```

---

## What you'll see when you run `test_agent.py`

DeepEval will:
1. Invoke `support_agent(...)` for each golden input
2. Capture the full trace (LangChain LLM call + tool call + final message)
3. Apply **Task Completion** to the whole trace
4. Apply **Tool Correctness** to the tool-call span
5. Print scores + reasons per golden

The third golden is intentionally borderline — watch for the interesting
case where Task Completion may pass but Tool Correctness fails (or vice
versa). That's the moment that makes component-level evaluation click.

---

## Suggested lesson flow

| Minute | Activity |
|---|---|
| 0–5 | Show `agent_plain.py`. Ask: "How would you test this?" |
| 5–10 | Show `agent_instrumented.py` side-by-side. Walk through the 4-line diff. |
| 10–25 | Walk through `test_agent.py` line by line. |
| 25–35 | Run it live. Discuss the borderline result. |
| 35+ | Open discussion: what other metrics? what other goldens? |

---

## Reality checks before teaching

DeepEval's integration surface evolves. Before your first session, run the
demo end-to-end once and confirm:

- `ChatAnthropic(model="claude-sonnet-4-5")` — if this errors, swap to the
  latest Claude model from the Anthropic console.
- `from deepeval.integrations.langchain import CallbackHandler` — works as
  written in recent DeepEval versions; older versions may use a different
  import path.
- `Golden(expected_tools=[ToolCall(name="...")])` — accepted in current
  DeepEval; in older versions you may need to pass via `additional_metadata`
  and unpack via `update_current_span`.

Fix any version issue once, then it's solid for the whole course.
