# Introduction — Why AI Needs a Completely Different Kind of Testing

> **Read this before opening any test file.** This document sets the foundation for everything that follows. It answers four questions:
> 1. What exactly are we testing when we say "test an AI system"?
> 2. Why can't we test it the way we test traditional software?
> 3. What are Evals, and how does DeepEval implement them?
> 4. What is a Judge LLM and why do we need one?
>
> Once you're comfortable with these concepts, `test_agent.py` (the first test file) will make immediate sense.

---

## 1. What Exactly Is AI Testing?

When someone says "test a web application," you know exactly what that means — click a button, check the database, verify the response code. The system is well-defined: inputs go in, predictable outputs come out.

But AI systems are different. There is no single "AI application" shape. In this course, we deal with four distinct types of AI systems, each with its own testing challenges:

### 1.1 AI Agents

An AI Agent is an LLM that can **take actions** — it decides which tools to call, in what order, with what arguments, and then interprets the results to form a final answer.

**Example from our codebase:** `agent_plain.py` is a customer-support agent with two tools — `get_order_status()` and `get_refund_policy()`. When a user asks "Where is my order ORD-1042?", the agent must:
1. **Decide** that this is an order status question (not a refund question).
2. **Choose** the `get_order_status` tool.
3. **Extract** the order ID "ORD-1042" from the user's message and pass it as the argument.
4. **Interpret** the tool's response and formulate a helpful reply.

**What can go wrong:**
- The agent calls the wrong tool (asks for refund policy when the user wants order status).
- The agent calls the right tool with wrong arguments (passes "1042" instead of "ORD-1042").
- The agent calls the right tool, gets the right data, but misinterprets or garbles it in the final reply.
- The agent calls unnecessary extra tools, wasting time and money.
- The agent ignores its system prompt instructions (goes off-brand, gives long answers when told to be brief).

Every one of these failure modes needs a different evaluation strategy. That's why `test_agent.py` uses **TaskCompletionMetric** (did the whole job get done?) and **ToolCorrectnessMetric** (did it pick the right tool?).

### 1.2 MCP (Model Context Protocol) Tools

MCP is a protocol that lets AI agents connect to **external tool servers**. Instead of defining tools directly in code, the agent discovers tools dynamically from an MCP server at runtime.

**Example from our codebase:** `mcp_server.py` exposes a `get_shipping_options` tool via MCP. The instrumented agent in `agent_instrumented.py` connects to this MCP server at startup and adds the shipping tool to its toolkit alongside the regular Python tools.

**What's different about testing MCP:**
- The agent doesn't just call a local function — it calls a tool discovered at runtime from an external server.
- Tool names and descriptions come from the MCP server, not from the agent's own code.
- The agent must correctly identify that a user question about shipping should route to the MCP tool rather than the locally defined tools.

In `test_agent.py`, the fourth golden tests exactly this — it asks "What shipping options do you offer?" and expects the agent to call `get_shipping_options`, which is the MCP-served tool. The testing mechanics are the same (ToolCorrectnessMetric doesn't care whether a tool is local or MCP), but the *integration surface* is wider.

### 1.3 Chatbots (Multi-Turn Conversations)

A Chatbot maintains conversation history across multiple user messages. Unlike a single-turn agent that processes one question in isolation, a chatbot must remember what was said earlier and stay in character throughout.

**Example from our codebase:** `chatbot.py` is a multi-turn support bot. A user might say:
- Turn 1: "My order ID is ORD-1042."
- Turn 2: "Is it going to arrive on time?"
- Turn 3: "What was the ETA you just mentioned?"

The chatbot must carry context forward — it needs to remember the order ID from Turn 1 to answer Turn 2, and remember the ETA it provided in Turn 2 to answer Turn 3.

**What can go wrong:**
- The chatbot forgets information from earlier turns (memory failure).
- The chatbot breaks character — e.g., a customer support bot starts giving cooking advice.
- Individual turns are off-topic even if the conversation as a whole seems reasonable.
- The conversation technically progresses but never actually resolves the customer's issue.

These failures require **conversational metrics**: KnowledgeRetention, RoleAdherence, TurnRelevancy, and ConversationalGEval — all covered in `test_chatbot.py`.

### 1.4 RAG (Retrieval-Augmented Generation)

A RAG Agent doesn't rely on the LLM's training data to answer questions. Instead, it first **retrieves** relevant documents from a knowledge base, then uses those documents as context to generate an answer.

**Example from our codebase:** `rag_agent.py` has a vector store loaded with 9 policy documents (shipping, returns, cancellation, etc.). When a user asks "What is the return policy for electronics?", the agent:
1. **Retrieves** the top 3 most relevant policy documents from the vector store.
2. **Reads** those documents.
3. **Generates** an answer based on the retrieved content.

**What can go wrong — the RAG-specific failures:**
- **Hallucination:** The agent says something that isn't in the retrieved documents. The retrieval says "15 days" but the agent says "30 days."
- **Bad retrieval ranking:** The most relevant document is ranked 3rd while irrelevant ones are ranked 1st and 2nd. The LLM might miss the important one.
- **Incomplete retrieval:** The knowledge base has the answer, but the retriever failed to surface the right document.
- **Irrelevant answer:** The retrieved documents are fine but the agent goes off-topic in its response anyway.

RAG testing requires a specialized set of metrics (Faithfulness, ContextualPrecision, ContextualRecall, AnswerRelevancy) covered in `test_rag_agent.py`.

---

## 2. Why Traditional Testing Doesn't Work

### The Determinism Problem

Traditional software testing relies on one fundamental assumption: **the same input produces the same output every time.**

```
# Traditional test — this works because software is deterministic
assert add(2, 3) == 5          # Always true. Every time. Forever.
assert login("admin", "pass123") == "Welcome, admin!"  # Predictable.
```

You write an assertion. You run it 1000 times. It passes 1000 times. If it fails, something is broken.

Now try this with an AI agent:

```
# AI agent test — this DOES NOT work
response = support_agent("Where is my order ORD-1042?")
assert response == "Order ORD-1042 is Shipped. ETA: 2026-05-13."
```

**Why this fails:** Run this 10 times and you might get 10 slightly different responses:
- "Order ORD-1042 has been shipped and should arrive by May 13, 2026."
- "Your order ORD-1042 is currently shipped! Expected delivery: 2026-05-13."
- "I checked your order — ORD-1042 is on its way! It's expected to arrive by May 13th."
- "Great news! Order ORD-1042 has shipped. ETA is May 13, 2026."

All four responses are **correct**. All four convey the same information. But none of them match the exact string in the assertion. The test fails 100% of the time even though the agent works perfectly.

### Why AI Systems Are Non-Deterministic

LLMs generate text by **sampling from probability distributions**.

...It can produce different outputs for the same input. And with temperature > 0, the variation is intentional and by design.

### The String-Match Trap

The traditional testing instinct is to match strings:

```
# All of these are bad ideas for AI testing:
assert "shipped" in response.lower()        # What if it says "on its way"?
assert response.startswith("Order ORD")     # What if it says "Your order"?
assert len(response) < 100                  # Arbitrary, ignores content quality
```

You can play whack-a-mole with regex patterns and substring checks, but you'll never cover the full space of valid responses. And you'll spend more time maintaining your test assertions than actually testing the agent.

### What We Actually Want to Test

The question isn't "did the agent produce this exact string?" — it's:

- **Did the agent give the right information?** (semantic correctness, not string matching)
- **Did it use the right tool?** (behavioral, not output-based)
- **Was the response relevant to the question?** (semantic judgment)
- **Did it follow its instructions?** (compliance checking)
- **Was it efficient?** (step counting, not output checking)
- **Did it hallucinate?** (factual grounding against sources)
- **Was it safe, unbiased, and privacy-preserving?** (guardrails)

None of these can be answered with `assert response == expected`. Every one of them requires **judgment** — the ability to read the response, understand its meaning, and evaluate its quality.

This is where **Evals** come in.

---

## 3. What Are Evals? How Does DeepEval Work?

### Evals — The Concept

"Eval" is short for **evaluation**. In the AI world, an eval is a structured, repeatable process for measuring how well an AI system performs on a specific quality dimension.

Think of it this way:

| Traditional Testing | AI Evals |
|---|---|
| Binary: pass or fail | Scored: 0.0 to 1.0 |
| Exact match | Semantic judgment |
| Deterministic | Probabilistic |
| "Is the output correct?" | "How good is the output?" |
| Written by QA engineers | Judged by another LLM |

An eval doesn't ask "is this exactly right?" — it asks "how good is this, on a scale?" and then you set a **threshold** to decide what's good enough.

### DeepEval — The Framework

DeepEval is an open-source Python framework for evaluating LLM applications. It provides:

1. **Metrics** — Pre-built evaluation criteria (AnswerRelevancy, Faithfulness, ToolCorrectness, etc.) plus the ability to define custom criteria via GEval.
2. **Test cases** — Structured containers for inputs, outputs, and expected values (Goldens).
3. **A judge LLM** — An LLM that scores the outputs (explained in Section 4 below).
4. **Tracing/instrumentation** — Hooks into LangChain, OpenAI, etc. to automatically capture what the agent did (tool calls, intermediate steps, retrieved documents).
5. **A test runner** — Integration with pytest, or standalone `evaluate()` / `evals_iterator()` methods.

### How a DeepEval Evaluation Works — Step by Step

Here's what happens when you run a test file like `test_agent.py`:

```
Step 1: Define your test data
    ┌──────────────────────────────────────────────────────┐
    │  Golden(                                             │
    │      input="Where is my order ORD-1042?",            │
    │      expected_tools=[ToolCall(name="get_order_status")]│
    │  )                                                   │
    └──────────────────────────────────────────────────────┘
    A "Golden" is one test scenario — the input you'll send
    to the agent, plus any expected values to compare against.

Step 2: Choose your metrics
    ┌──────────────────────────────────────────────────────┐
    │  task_completion  = TaskCompletionMetric(threshold=0.7)│
    │  tool_correctness = ToolCorrectnessMetric()           │
    └──────────────────────────────────────────────────────┘
    Each metric knows how to evaluate one quality dimension.
    The threshold is YOUR definition of "good enough."

Step 3: Run the agent
    ┌──────────────────────────────────────────────────────┐
    │  support_agent("Where is my order ORD-1042?")        │
    └──────────────────────────────────────────────────────┘
    The agent runs for real — it calls tools, generates a
    response. DeepEval's instrumentation captures everything
    as a "trace" (the sequence of steps the agent took).

Step 4: The Judge LLM scores the trace
    ┌──────────────────────────────────────────────────────┐
    │  Judge (GPT-4o) examines:                            │
    │    - The user input                                  │
    │    - The agent's full execution trace                 │
    │    - The final response                              │
    │    - The expected values from the Golden              │
    │                                                      │
    │  It produces a score: 0.85                           │
    │  Threshold was 0.7 → PASS ✓                          │
    └──────────────────────────────────────────────────────┘
```

### How Scoring Works

Every DeepEval metric produces a score between **0.0 and 1.0**:

| Score | Meaning |
|-------|---------|
| 1.0 | Perfect — the output fully satisfies the metric |
| 0.7–0.9 | Good — minor issues but overall solid |
| 0.4–0.6 | Mediocre — significant room for improvement |
| 0.1–0.3 | Poor — major failures detected |
| 0.0 | Complete failure on this metric |

You set a **threshold** for each metric — this is your pass/fail line. If the score is at or above the threshold, the test passes. If it's below, the test fails.

**Threshold is a design decision, not a rule.** Different metrics and different use cases warrant different thresholds:

- **Quality metrics** (AnswerRelevancy, TaskCompletion, GEval): typically `0.7` — you want high quality but allow some flexibility.
- **Safety metrics** (Bias, Toxicity, PII): set at `0.5` in our files — these scores can be noisy, but any production system should push these higher.
- **Efficiency metrics** (StepEfficiency): typically `0.5` — agents rarely take the theoretically optimal path, so you allow more slack.

### Two Types of Metrics — Deterministic vs. LLM-Judged

Not all metrics use a judge LLM:

**Deterministic metrics** calculate the score using exact logic — no LLM involved:
- `ToolCorrectnessMetric` — compares actual tool calls against expected tool calls. It's a set comparison. Either the agent called `get_order_status` or it didn't. No interpretation needed.

**LLM-judged metrics** ask another LLM to evaluate the quality:
- `TaskCompletionMetric` — there's no formula for "did the agent complete the task." It requires understanding the user's intent, the agent's actions, and whether the outcome makes sense. Only an LLM can judge this.
- `AnswerRelevancyMetric`, `FaithfulnessMetric`, `GEval`, all safety metrics — all require semantic understanding that only an LLM can provide.

---

## 4. The Judge LLM — Why We Use a Second LLM to Grade the First

### The Core Idea

If we can't use string matching to evaluate AI outputs, and we need semantic understanding to judge quality, who does the judging? The answer: **another LLM**.

This is the **Judge LLM** (also called the "evaluator model"). It's a separate LLM whose job is to read the agent's work and score it.

```
┌─────────────┐       ┌─────────────────┐       ┌─────────────┐
│   User       │       │   Agent LLM     │       │  Judge LLM  │
│   Question   │──────▶│   (under test)  │──────▶│  (scorer)   │
│              │       │                 │       │             │
│ "Where is my │       │  Calls tools,   │       │ Reviews the │
│  order?"     │       │  generates      │       │ trace and   │
│              │       │  response       │       │ output,     │
│              │       │                 │       │ gives score │
└─────────────┘       └─────────────────┘       └─────────────┘
                                                       │
                                                  Score: 0.85
                                                  PASS ✓
```

### Why Not Just Have the Agent Grade Itself?

Using the same LLM to both generate and evaluate creates a conflict of interest — like grading your own exam. Research has shown that LLMs tend to rate their own outputs more favorably than outputs from other models. This is called **self-evaluation bias**.

### Cross-Vendor Setup — Our Approach

In our test files, we intentionally use **different providers** for the agent and the judge:

| Role | Model | Provider | Purpose |
|------|-------|----------|---------|
| Agent LLM (system under test) | Claude / GPT-4o | Anthropic / OpenAI | Runs the agent, generates responses |
| Judge LLM (evaluator) | GPT-4o | OpenAI | Scores the agent's performance |

This cross-vendor setup means:
- The agent (the student) and the judge (the teacher) are from different "families."
- The judge has no bias toward the agent's outputs.
- The evaluation is more objective and trustworthy.

### What the Judge Actually Does

When a metric like `TaskCompletionMetric` runs, here's what happens internally:

1. DeepEval constructs an **evaluation prompt** — a carefully designed prompt that tells the judge LLM exactly what to evaluate and how.
2. The prompt includes the user's input, the agent's full trace (tool calls, intermediate steps, final response), and the evaluation criteria.
3. The judge LLM reads everything and produces:
   - A **score** (0.0 to 1.0)
   - A **reason** (textual explanation of why it gave that score)
4. DeepEval compares the score to the threshold and reports pass/fail.

The evaluation prompts are designed by DeepEval's authors using prompt engineering best practices — you don't write them yourself. Each metric has its own evaluation prompt template optimized for that specific quality dimension.

### The Judge Is Not Perfect — And That's OK

The judge LLM is itself an AI, so it's also non-deterministic. The same agent output might receive a 0.78 on one run and a 0.82 on the next. This is why:

- **Thresholds should have margin.** If your true minimum acceptable quality is 0.8, set the threshold to 0.7 to absorb scoring variance.
- **Run evals multiple times** when the results are borderline.
- **Look at the reasons**, not just the scores. DeepEval's metrics often provide a textual explanation of the score — this is more informative than the number alone.

---

## 5. Putting It All Together — The Bridge to `test_agent.py`

Now you have the full mental model:

1. **AI systems are non-deterministic** — same input, different output every time. Traditional `assert` testing breaks down.

2. **Evals replace assertions** — instead of "is the output exactly X?", we ask "how good is the output on dimension Y?" and get a score from 0 to 1.

3. **DeepEval provides the framework** — metrics, test cases (Goldens), instrumentation (tracing), and the evaluation loop.

4. **A Judge LLM does the scoring** — a separate, unbiased model reads the agent's work and grades it.

With this foundation, open `test_agent.py`. You'll see:
- **Two metrics:** `TaskCompletionMetric` (did the agent do the job?) and `ToolCorrectnessMetric` (did it pick the right tool?).
- **Four Goldens:** Four test scenarios, each with an input and expected tools.
- **The evaluation loop:** `evals_iterator` walks through each golden, runs the agent, captures the trace, and lets the judge score it.
- **Cross-vendor setup:** Claude runs the agent, GPT-4o runs the judge.

Everything in `test_agent.py` is a direct application of the concepts from this document. Let's go.

---

## Quick Glossary

| Term | Definition |
|------|-----------|
| **Eval** | A structured process for measuring AI output quality on a specific dimension. Produces a score (0–1), not a pass/fail. |
| **Metric** | A specific evaluation criterion (e.g., AnswerRelevancy, Faithfulness, ToolCorrectness). Each metric measures one quality dimension. |
| **Golden** | A test case — contains the input to send to the agent, plus any expected values (expected output, expected tools) for comparison. |
| **EvaluationDataset** | A collection of Goldens that form one evaluation suite. |
| **Judge LLM** | A separate LLM used to score the agent's outputs. Not the same model being tested. |
| **Threshold** | The minimum score (0–1) for a metric to pass. Set by the test author based on quality requirements. |
| **Trace** | The full record of an agent's execution — every tool call, intermediate step, and final response. Captured by DeepEval's instrumentation. |
| **Non-deterministic** | Producing different outputs for the same input across runs. This is inherent to LLMs and is the reason traditional testing doesn't work. |
| **Cross-vendor** | Using different LLM providers for the agent (e.g., Anthropic's Claude) and the judge (e.g., OpenAI's GPT-4o) to avoid self-evaluation bias. |
| **Instrumentation** | Code added to the agent (like DeepEval's `CallbackHandler` and `@observe`) that captures execution traces without changing the agent's behavior. |
| **DeepEval** | An open-source Python framework for evaluating LLM applications. Provides metrics, tracing, and test infrastructure. |

  
  Tracing = visibility inside the agent's execution

  When your agent runs, it does multiple things internally — an LLM call, then a tool call, then another LLM call. Tracing captures each of those internal
  steps as spans so DeepEval can see what happened at each level, not just the final output.

  ---
  Why safety tests don't need tracing
  
  BiasMetric, ToxicityMetric, PIILeakageMetric only look at the final text output. They don't care how the agent got there — no tools, no intermediate steps.
  Just: "read this response, is it biased?" So no tracing needed.

  ---
  Why the other tests do need tracing
  
  - ToolCorrectnessMetric — needs to know which tools were called during the run. That information lives inside the agent's execution, not in the final output.
   Tracing captures the tool span so DeepEval can read tools_called.
  - TaskCompletionMetric, StepEfficiencyMetric — the judge needs to see the full sequence of steps the agent took, not just the answer.
  - ContextualPrecision/Recall — needs to see which chunks were retrieved inside the RAG pipeline. That's captured as a retrieval span.

  ---
  The one-line version:
  
  ▎ If the metric only reads the final reply → no tracing needed.
  ▎ If the metric needs to look inside the agent's steps → tracing required.

  @observe creates the outer trace, CallbackHandler captures every LangChain span inside it, and update_current_trace is how you explicitly attach
  tools_called, retrieval_context, and expected_tools so the metrics can find them.

- tools_called is hardcoded — you're telling DeepEval "these tools were called", you're not actually running the agent
  - actual_output is hardcoded — you're telling DeepEval "this is what the agent said"
  - No evals_iterator, no @observe, no CallbackHandler — no tracing at all

  DeepEval just takes your word for it and scores against expected_tools.

  ---
  In our test_agent.py the same metric works differently:

  # us — agent runs live, tools captured via tracing
  for golden in dataset.evals_iterator(metrics=[tool_correctness]):
      support_agent(golden.input)   # @observe captures tools_called automatically

  The @observe + CallbackHandler watches the agent run in real time and populates tools_called on the trace. evals_iterator then reads that trace to build the
  LLMTestCase internally — you never write tools_called yourself.

  ---
  So the docs example is essentially the static pattern applied to tool testing — useful if you already have a captured conversation log and want to check if
  the right tools fired. But if you want to test the live agent, tracing is what makes it work without you having to hardcode anything.
To capture live tools_called and actual_output from a real agent, you should instrument your agent with the @observe decorator and use update_current_span / update_current_trace to record the data as it flows through your app. Then drive evaluation with EvaluationDataset and evals_iterator().