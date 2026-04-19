import json
import ast
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

    expr = expr.strip()
    if len(expr) > 200:
        raise ToolError("calculator expression is too long")

    allowed = set("0123456789+-*/(). %\t\n\r")
    if any(ch not in allowed for ch in expr):
        raise ToolError("calculator expression contains invalid characters")

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            # Limit literal size to avoid huge ints chewing CPU/memory.
            if isinstance(node.value, int) and len(str(abs(node.value))) > 30:
                raise ToolError("calculator number literal is too large")
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = _eval_node(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)
        ):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left % right
        raise ToolError("calculator expression contains unsupported operations")

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"calculator syntax error: {e.msg}") from e

    try:
        value = _eval_node(parsed)
    except ZeroDivisionError as e:
        raise ToolError("calculator error: division by zero") from e

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
