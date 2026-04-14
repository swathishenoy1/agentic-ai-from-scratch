import json
import os
import string
from typing import Any, Dict, List, Set


class PromptEngineError(Exception):
    pass


DEFAULT_REGISTRY_PATH = os.environ.get(
    "PROMPT_REGISTRY_PATH",
    os.path.join(os.path.dirname(__file__), "prompts", "registry.json"),
)


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise PromptEngineError(f"Prompt registry not found: {path}") from e
    except json.JSONDecodeError as e:
        raise PromptEngineError(f"Invalid JSON in registry: {path} ({e})") from e


def _extract_fields(template: str) -> Set[str]:
    fields: Set[str] = set()
    for literal, field, format_spec, conv in string.Formatter().parse(template):
        if field:
            # Strip any attribute/index access to keep it simple
            field = field.split(".")[0].split("[")[0]
            fields.add(field)
    return fields


def _collect_required_vars(messages: List[Dict[str, Any]]) -> Set[str]:
    required: Set[str] = set()
    for msg in messages:
        template = msg.get("template", "")
        if not isinstance(template, str):
            raise PromptEngineError("Each message must have a string 'template'.")
        required |= _extract_fields(template)
    return required


def load_registry(path: str = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    data = _load_json(path)
    if "prompts" not in data or not isinstance(data["prompts"], dict):
        raise PromptEngineError("Registry must contain a top-level 'prompts' object.")
    return data


def get_prompt(name: str, registry_path: str = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    prompt = registry["prompts"].get(name)
    if not prompt:
        raise PromptEngineError(f"Prompt not found: {name}")
    return prompt


def validate_prompt(prompt: Dict[str, Any]) -> None:
    messages = prompt.get("messages")
    if not isinstance(messages, list) or not messages:
        raise PromptEngineError("Prompt must include a non-empty 'messages' list.")

    for msg in messages:
        role = msg.get("role")
        if role not in {"system", "user", "assistant"}:
            raise PromptEngineError(f"Invalid role: {role}")

    required = _collect_required_vars(messages)
    declared = set(prompt.get("variables", []))

    if declared and required - declared:
        missing_decl = ", ".join(sorted(required - declared))
        raise PromptEngineError(f"Registry missing variable declarations: {missing_decl}")


def render_messages(
    name: str,
    variables: Dict[str, Any],
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> List[Dict[str, str]]:
    prompt = get_prompt(name, registry_path)
    validate_prompt(prompt)

    messages = prompt["messages"]
    required = _collect_required_vars(messages)

    missing = required - set(variables.keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise PromptEngineError(f"Missing variables: {missing_list}")

    rendered: List[Dict[str, str]] = []
    for msg in messages:
        rendered.append(
            {
                "role": msg["role"],
                "content": msg["template"].format(**variables),
            }
        )
    return rendered


if __name__ == "__main__":
    # Example: python prompt_engine.py
    msgs = render_messages(
        "summarize",
        {"topic": "rate limiting", "length": "3 bullets"},
    )
    for m in msgs:
        print(f"[{m['role']}] {m['content']}")
