import json
import re
from typing import Any, Dict, Optional, Tuple

from tools import ToolRegistry, ToolError, serialize_result


CALL_RE = re.compile(
    r"CALL_TOOL\(\s*(?P<name>(\"[^\"]+\"|'[^']+'|[A-Za-z0-9_\-]+))\s*,\s*(?P<args>\{.*\})\s*\)",
    re.DOTALL,
)


class ToolCallError(Exception):
    pass


def parse_tool_call(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    match = CALL_RE.search(text)
    if not match:
        return None

    raw_name = match.group("name").strip()
    if (raw_name.startswith('"') and raw_name.endswith('"')) or (
        raw_name.startswith("'") and raw_name.endswith("'")
    ):
        name = raw_name[1:-1]
    else:
        name = raw_name

    raw_args = match.group("args")
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as e:
        raise ToolCallError(f"Invalid JSON args for tool call: {e}") from e

    if not isinstance(args, dict):
        raise ToolCallError("Tool args must be a JSON object")

    return name, args


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
