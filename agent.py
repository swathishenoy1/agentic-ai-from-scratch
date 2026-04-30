import json
import argparse
from typing import Any, Dict, List, Optional

from llm_client import complete
from tool_router import ToolCallError, execute_tool_call
from tools import ToolRegistry, ToolError, default_registry, tool_spec
from memory import write as memory_write, search as memory_search


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


def _tool_system_prompt(registry: ToolRegistry) -> str:
    tools_json = _tool_catalog(registry)
    return (
        "You are a tool-using assistant.\n"
        "You must follow a Think -> Act -> Observe loop.\n\n"
        "Available tools (JSON):\n"
        f"{tools_json}\n\n"
        "When you want to use a tool, output exactly one call like:\n"
        'CALL_TOOL(tool_name, {"arg":"value"})\n\n'
        "When you are done, output:\n"
        "FINAL: <answer>\n"
    )


def run_messages(
    messages: List[Dict[str, str]],
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

    # Ensure we always include the tool system instructions. If the caller already
    # provided a system message, append tool instructions to it; otherwise add one.
    tool_system = _tool_system_prompt(registry)
    if messages and messages[0].get("role") == "system":
        messages = list(messages)
        messages[0] = {
            "role": "system",
            "content": tool_system + "\n\n" + messages[0].get("content", ""),
        }
    else:
        messages = [{"role": "system", "content": tool_system}] + list(messages)

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


def run(
    task: str,
    *,
    registry: Optional[ToolRegistry] = None,
    max_steps: int = 8,
    model: Optional[str] = None,
    temperature: float = 0.0,
    llm_params: Optional[Dict[str, Any]] = None,
) -> str:
    return run_messages(
        [{"role": "user", "content": f"Task:\n{task}"}],
        registry=registry,
        max_steps=max_steps,
        model=model,
        temperature=temperature,
        llm_params=llm_params,
    )


def _parse_kv(pairs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not pairs:
        return out
    for item in pairs:
        if "=" not in item:
            raise AgentError(f"Invalid --var value (expected key=value): {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise AgentError(f"Invalid --var key in: {item}")
        out[k] = v
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the toy agent loop.")
    parser.add_argument("--task", type=str, help="Task string to execute.")
    parser.add_argument("--prompt", type=str, help="Prompt name from prompts/registry.json.")
    parser.add_argument(
        "--var",
        action="append",
        help="Variables for --prompt in key=value form. May be repeated.",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument(
        "--memory-search-k",
        type=int,
        default=0,
        help="If >0, retrieve up to K long-term memory hits and attach as context.",
    )
    parser.add_argument("--memory-namespace", type=str, default="default")
    parser.add_argument("--memory-dir", type=str, default=None)
    parser.add_argument(
        "--remember",
        action="store_true",
        help="Store the task and final answer to long-term memory.",
    )

    args = parser.parse_args(argv)

    if bool(args.task) == bool(args.prompt):
        raise AgentError("Provide exactly one of --task or --prompt.")

    base_messages: List[Dict[str, str]]
    task_text: str
    if args.task:
        task_text = args.task
        base_messages = [{"role": "user", "content": f"Task:\n{task_text}"}]
    else:
        from prompt_engine import render_messages

        variables = _parse_kv(args.var)
        rendered = render_messages(args.prompt, variables)
        task_text = " ".join(m["content"] for m in rendered if m.get("role") == "user")
        base_messages = rendered

    if args.memory_search_k and args.memory_search_k > 0:
        hits = memory_search(
            task_text,
            k=args.memory_search_k,
            namespace=args.memory_namespace,
            store_dir=args.memory_dir,
        )
        if hits:
            mem_lines = "\n".join(f"- {h.get('text','')}" for h in hits)
            base_messages = [
                {
                    "role": "system",
                    "content": "Relevant long-term memory (may be incomplete):\n" + mem_lines,
                }
            ] + base_messages

    answer = run_messages(
        base_messages,
        max_steps=args.max_steps,
        model=args.model,
        temperature=args.temperature,
    )

    if args.remember:
        memory_write(
            f"Task: {task_text}\nAnswer: {answer}",
            metadata={"kind": "run"},
            namespace=args.memory_namespace,
            store_dir=args.memory_dir,
        )

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
