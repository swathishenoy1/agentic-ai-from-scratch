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

Next: we will expand into prompt templates, tool calling, and agent loops once Goal 0 is solid.
