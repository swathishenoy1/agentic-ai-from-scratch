import json
import os
import time
from typing import Any, Dict

import requests


# Environment configuration (provider-agnostic)
DEFAULT_API_URL = os.environ.get("LLM_API_URL", "").strip()
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "").strip()
DEFAULT_TIMEOUT = float(os.environ.get("LLM_TIMEOUT_SEC", "30"))

# Token budgeting (very rough heuristic: ~4 chars per token)
DEFAULT_MAX_INPUT_TOKENS = int(os.environ.get("LLM_MAX_INPUT_TOKENS", "2000"))
CHARS_PER_TOKEN = int(os.environ.get("LLM_CHARS_PER_TOKEN", "4"))


class LLMClientError(Exception):
    pass


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    # Keep both the beginning (often system/tool instructions) and the end
    # (most recent turns). This reduces the chance of truncating critical
    # agent/tool formatting rules.
    head_chars = max(256, max_chars // 4)
    head = text[: min(len(text), head_chars)]
    tail_chars = max_chars - len(head)
    if tail_chars <= 0:
        return head
    tail = text[-tail_chars:]
    return head + "\n...\n" + tail


def _build_payload(prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # Basic schema: prompt + model + optional params
    payload = {"prompt": prompt}
    if params.get("model"):
        payload["model"] = params["model"]
    if params.get("temperature") is not None:
        payload["temperature"] = params["temperature"]
    if params.get("max_tokens") is not None:
        payload["max_tokens"] = params["max_tokens"]
    # Pass through any extra params
    for k, v in params.items():
        if k not in payload:
            payload[k] = v
    return payload


def _parse_response(data: Dict[str, Any]) -> str:
    """
    Provider-agnostic extraction with common fallbacks.
    You can override this logic to match your API.
    """
    # Common OpenAI-like shape: {"choices":[{"text": "..."}]}
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                if "text" in first and isinstance(first["text"], str):
                    return first["text"]
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
        # Other common shapes
        if isinstance(data.get("output_text"), str):
            return data["output_text"]
        if isinstance(data.get("response"), str):
            return data["response"]
        if isinstance(data.get("text"), str):
            return data["text"]
    raise LLMClientError("Unable to parse response text. Update _parse_response for your API shape.")


def complete(prompt: str, **params: Any) -> str:
    """
    Send a raw HTTP request to an LLM API without any SDK.

    Required env vars:
      - LLM_API_URL (full URL to the completion endpoint)
      - LLM_API_KEY (optional if your API doesn't require it)
    Optional env vars:
      - LLM_MODEL, LLM_TIMEOUT_SEC, LLM_MAX_INPUT_TOKENS, LLM_CHARS_PER_TOKEN
    """
    api_url = params.pop("api_url", DEFAULT_API_URL)
    api_key = params.pop("api_key", DEFAULT_API_KEY)
    model = params.pop("model", DEFAULT_MODEL)
    if not api_url:
        raise LLMClientError("LLM_API_URL is not set (and no api_url override was provided).")

    max_input_tokens = int(params.pop("max_input_tokens", DEFAULT_MAX_INPUT_TOKENS))
    token_estimate = _estimate_tokens(prompt)
    if token_estimate > max_input_tokens:
        prompt = _truncate_to_token_budget(prompt, max_input_tokens)

    # Assemble request
    payload = _build_payload(prompt, {"model": model, **params})
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.time()
    resp = requests.post(
        api_url,
        headers=headers,
        data=body,
        timeout=DEFAULT_TIMEOUT,
    )
    resp_body = resp.text

    if resp.status_code < 200 or resp.status_code >= 300:
        raise LLMClientError(f"HTTP {resp.status_code}: {resp_body}")

    try:
        data = json.loads(resp_body)
    except json.JSONDecodeError as e:
        raise LLMClientError(f"Failed to parse JSON response: {e}. Raw: {resp_body}") from e

    _ = time.time() - start  # available for logging if needed
    return _parse_response(data)


if __name__ == "__main__":
    # Minimal self-test (requires LLM_API_URL)
    print(complete("Hello there!"))
