import json
from dataclasses import dataclass
from typing import Any, Callable, Dict


class ToolError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if not tool:
            raise ToolError(f"Tool not found: {name}")
        return tool

    def list(self) -> Dict[str, Tool]:
        return dict(self._tools)


def tool_spec(tool: Tool) -> Dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


# Example tools (keep them tiny and inspectable)
def _calculator(args: Dict[str, Any]) -> Dict[str, Any]:
    expr = args.get("expression", "")
    if not isinstance(expr, str) or not expr.strip():
        raise ToolError("calculator requires a non-empty 'expression' string")
    # Safe-ish eval: only allow digits and basic operators
    allowed = set("0123456789+-*/(). ")
    if any(ch not in allowed for ch in expr):
        raise ToolError("calculator expression contains invalid characters")
    try:
        value = eval(expr, {"__builtins__": {}}, {})
    except Exception as e:
        raise ToolError(f"calculator error: {e}") from e
    return {"result": value}


def _echo(args: Dict[str, Any]) -> Dict[str, Any]:
    return {"echo": args}


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a basic math expression.",
            input_schema={"expression": "string"},
            handler=_calculator,
        )
    )
    registry.register(
        Tool(
            name="echo",
            description="Echo back provided arguments.",
            input_schema={"any": "object"},
            handler=_echo,
        )
    )
    return registry


def serialize_result(result: Any) -> str:
    try:
        return json.dumps(result, ensure_ascii=True)
    except TypeError:
        return json.dumps({"result": str(result)}, ensure_ascii=True)
