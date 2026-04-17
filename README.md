# LLM Learning (From Scratch)

This repo is a hands-on, framework-light curriculum to understand LLM systems by building them piece by piece in Python.

**Goal 0 — Foundations**
Build the minimal core of an LLM “runtime” you can control end-to-end.

What we are building:
1. Raw HTTP client (no SDKs)
2. Prompt → API → response parsing
3. Token budgeting (simple length estimates + truncation)

Planned outputs:
1. `llm_client.py` - a minimal client with a working raw HTTP request path
2. Basic parsing + error handling
3. Token budgeting helpers


**Goal 1 — Prompts as Data**
Treat prompts as structured data instead of plain strings.

What we are building:
1. Prompt template system (string interpolation + validation)
2. System/user/assistant message assembly
3. Prompt registry stored in JSON or YAML

Planned outputs:
1. `prompts/` folder
2. `prompt_engine.py`


**Goal 2 — Tool Calling (Manual)**
Define tools and route calls yourself.

What we are building:
1. Tool schema registry
2. Simple parser: model outputs “CALL_TOOL(name, args)”
3. Execute tool and return tool result to model

Planned outputs:
1. `tools.py`
2. `tool_router.py`


**Goal 3 — Agent Loop (Single Agent)**
Implement a simple planning loop without autonomy yet.

What we are building:
1. “Think → Act → Observe” loop
2. Max‑steps guard
3. Stop conditions

Planned outputs:
1. agent.py with run(task) that uses tools