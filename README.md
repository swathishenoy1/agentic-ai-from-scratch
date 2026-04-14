# LLM Learning (From Scratch)

This repo is a hands-on, framework-light curriculum to understand LLM systems by building them piece by piece in Python.

**Goal 0 — Foundations**
Build the minimal core of an LLM “runtime” you can control end-to-end.

What we’re building:
1. Raw HTTP client (no SDKs)
2. Prompt → API → response parsing
3. Token budgeting (simple length estimates + truncation)

Current files:
1. `llm_client.py` — a minimal client with a `complete(prompt, **params)` function

Planned outputs for Goal 0:
1. `llm_client.py` with a working raw HTTP request path
2. Basic parsing + error handling
3. Token budgeting helpers

**Goal 1 — Prompts as Data**
Treat prompts as structured data instead of plain strings.

What we’re building:
1. Prompt template system (string interpolation + validation)
2. System/user/assistant message assembly
3. Prompt registry stored in JSON or YAML

Planned outputs for Goal 1:
1. `prompts/` folder
2. `prompt_engine.py`
