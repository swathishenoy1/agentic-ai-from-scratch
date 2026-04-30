import json
from typing import Any, Dict, Optional, Tuple

from tools import ToolRegistry, ToolError, serialize_result


class ToolCallError(Exception):
    pass


def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    # Intentionally strict: the agent system prompt asks the model to output
    # exactly one tool call (and nothing else) when invoking tools.
    s = text.strip()
    if not s.startswith("CALL_TOOL("):
        return None

    i = len("CALL_TOOL(")
    n = len(s)

    def _skip_ws(idx: int) -> int:
        while idx < n and s[idx].isspace():
            idx += 1
        return idx

    i = _skip_ws(i)

    # Parse tool name (quoted or unquoted)
    if i >= n:
        raise ToolCallError("Incomplete tool call (missing name)")

    if s[i] in {"'", '"'}:
        quote = s[i]
        i += 1
        start = i
        while i < n and s[i] != quote:
            i += 1
        if i >= n:
            raise ToolCallError("Unterminated quoted tool name")
        name = s[start:i]
        i += 1
    else:
        start = i
        while i < n and s[i] not in {",", " ", "\t", "\n", "\r"}:
            i += 1
        name = s[start:i].strip()
        if not name:
            raise ToolCallError("Tool name is empty")

    i = _skip_ws(i)
    if i >= n or s[i] != ",":
        raise ToolCallError("Expected ',' after tool name")
    i += 1
    i = _skip_ws(i)

    # Parse JSON args object using the JSON decoder (avoids greedy regex issues).
    if i >= n or s[i] != "{":
        raise ToolCallError("Tool args must be a JSON object starting with '{'")

    decoder = json.JSONDecoder()
    try:
        args_obj, end = decoder.raw_decode(s, idx=i)
    except json.JSONDecodeError as e:
        raise ToolCallError(f"Invalid JSON args for tool call: {e}") from e

    i = _skip_ws(end)
    if i >= n or s[i] != ")":
        raise ToolCallError("Expected ')' after tool args")
    i += 1
    if s[i:].strip():
        raise ToolCallError("Unexpected trailing text after tool call")

    if not isinstance(args_obj, dict):
        raise ToolCallError("Tool args must be a JSON object")

    return name, args_obj


def execute_tool_call(text: str, registry: ToolRegistry) -> Optional[str]:
    parsed = parse_tool_call(text)
    if not parsed:
        return None

    name, args = parsed
    tool = registry.get(name)
    try:
        result = tool.handler(args)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Unhandled tool error: {e}") from e

    return serialize_result(result)


if __name__ == "__main__":
    reg = ToolRegistry()
    from tools import default_registry

    reg = default_registry()
    output = execute_tool_call('CALL_TOOL(calculator, {"expression":"2+2*3"})', reg)
    print(output)
