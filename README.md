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

How to run it:
1. Set your LLM env vars:
   - `export LLM_API_URL="https://<your-llm-endpoint>"`
   - `export LLM_API_KEY="..."` (optional if your API doesn’t require it)
   - `export LLM_MODEL="..."` (optional)
2. Run the minimal client demo:
   - `python llm_client.py`


**Goal 1 — Prompts as Data**
Treat prompts as structured data instead of plain strings.

What we are building:
1. Prompt template system (string interpolation + validation)
2. System/user/assistant message assembly
3. Prompt registry stored in JSON or YAML

Planned outputs:
1. `prompts/` folder
2. `prompt_engine.py`

How to run it:
- `python prompt_engine.py`


**Goal 2 — Tool Calling (Manual)**
Define tools and route calls yourself.

What we are building:
1. Tool schema registry
2. Simple parser: model outputs “CALL_TOOL(name, args)”
3. Execute tool and return tool result to model

Planned outputs:
1. `tools.py`
2. `tool_router.py`

How to run it:
- `python tool_router.py`


**Goal 3 — Agent Loop (Single Agent)**
Implement a simple planning loop without autonomy yet.

What we are building:
1. “Think → Act → Observe” loop
2. Max‑steps guard
3. Stop conditions

Planned outputs:
1. agent.py with run(task) that uses tools

How to run it:
1. Set your LLM env vars (same as Goal 0).
2. Run a task through the agent loop:
   - `python agent.py --task "Compute 12*(3+4) using the calculator tool."`

**Goal 4 — Multi-agent Orchestrator**
Build a small, inspectable multi-agent system that runs multiple roles per task.

What we are building:
1. Agent roles:
- `researcher`: gathers relevant facts/assumptions and open questions
- `planner`: produces an execution plan and success criteria
- `critic`: ranks candidates and highlights risks/fixes
- `executor`: produces candidate solutions and the final merged answer

2. Orchestrator:
- runs researcher → planner → multiple executor candidates
- ranks candidates with critic
- optional voting (critic/planner/researcher) to pick a winner
- final merge/refinement pass produces a single `FINAL: ...`

Planned outputs:
1. orchestrator.py (CLI entrypoint)

How to run it:
1. `python orchestrator.py --task "Explain the difference between rate limiting and backpressure."` 
2. Multiple candidates + verbose trace: `python orchestrator.py --task "..." --candidates 3 --verbose`
3. Disable voting (critic-only ranking): `python orchestrator.py --task "..." --no-vote`


## Running everything together

From the repo root:

1. Set your LLM env vars:
   - `export LLM_API_URL="https://<your-llm-endpoint>"`
   - `export LLM_API_KEY="..."` (optional if your API doesn’t require it)
   - `export LLM_MODEL="..."` (optional)

2. Run a task:
   - `python agent.py --task "Compute 12*(3+4) using the calculator tool."`

3. Run using a prompt from `prompts/registry.json`:
   - `python agent.py --prompt summarize --var topic="rate limiting" --var length="3 bullets"`

4. Use memory (file-based, keyword retrieval):
   - Remember runs: `python agent.py --task "..." --remember`
   - Retrieve context: `python agent.py --task "..." --memory-search-k 3`