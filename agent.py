import json
from typing import Any, Dict, List, Optional

from llm_client import complete
from tool_router import ToolCallError, execute_tool_call
from tools import ToolRegistry, ToolError, default_registry, tool_spec


class AgentError(Exception):
    pass


def _format_messages(messages: List[Dict[str, str]]) -> str:
    # Keep this plain and inspectable; no chat SDK needed.
    parts: List[str] = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"{role}:\n{msg['content']}\n")
    return "\n".join(parts).strip() + "\n"


def _tool_catalog(registry: ToolRegistry) -> str:
    specs = [tool_spec(t) for t in registry.list().values()]
    return json.dumps(specs, ensure_ascii=True, indent=2)


def _is_final(text: str) -> Optional[str]:
    marker = "FINAL:"
    idx = text.find(marker)
    if idx == -1:
        return None
    return text[idx + len(marker) :].strip()


def run(
    task: str,
    *,
    registry: Optional[ToolRegistry] = None,
    max_steps: int = 8,
    model: Optional[str] = None,
    temperature: float = 0.0,
    llm_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Simple Think -> Act -> Observe loop (not autonomous).

    Stop conditions:
      - model emits "FINAL: ..."
      - model emits no tool call AND no FINAL (return its text)
      - max_steps reached (raise)
    """
    if max_steps <= 0:
        raise AgentError("max_steps must be > 0")

    if registry is None:
        registry = default_registry()

    tools_json = _tool_catalog(registry)
    system = (
        "You are a tool-using assistant.\n"
        "You must follow a Think -> Act -> Observe loop.\n\n"
        "Available tools (JSON):\n"
        f"{tools_json}\n\n"
        "When you want to use a tool, output exactly one call like:\n"
        'CALL_TOOL(tool_name, {"arg":"value"})\n\n'
        "When you are done, output:\n"
        "FINAL: <answer>\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Task:\n{task}"},
    ]

    extra = dict(llm_params or {})
    if model:
        extra["model"] = model
    extra.setdefault("temperature", temperature)

    for step in range(1, max_steps + 1):
        prompt = _format_messages(messages)
        model_text = complete(prompt, **extra).strip()
        messages.append({"role": "assistant", "content": model_text})

        final = _is_final(model_text)
        if final is not None:
            return final

        try:
            tool_out = execute_tool_call(model_text, registry)
        except (ToolCallError, ToolError) as e:
            # Observation includes the error so the model can correct itself.
            messages.append({"role": "user", "content": f"OBSERVATION (tool error): {e}"})
            continue

        if tool_out is None:
            # No tool call; treat this as a stop.
            return model_text

        messages.append({"role": "user", "content": f"OBSERVATION: {tool_out}"})

    raise AgentError(f"Max steps reached ({max_steps}) without FINAL.")


if __name__ == "__main__":
    # Requires your LLM env vars to be set (LLM_API_URL etc).
    print(run("Compute 12*(3+4) using the calculator tool."))
