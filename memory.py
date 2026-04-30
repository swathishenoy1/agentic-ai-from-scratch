import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class MemoryError(Exception):
    pass


def _now_ts() -> float:
    return time.time()


def _default_store_dir() -> str:
    # Keep store local and inspectable by default.
    return os.environ.get(
        "MEMORY_DIR",
        os.path.join(os.path.dirname(__file__), ".memory"),
    )


def _namespace_path(store_dir: str, namespace: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-./]", "_", namespace).strip("/")
    if not safe:
        raise MemoryError("namespace must be a non-empty string")
    return os.path.join(store_dir, safe)


def _jsonl_path(store_dir: str, namespace: str) -> str:
    return os.path.join(_namespace_path(store_dir, namespace), "memory.jsonl")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _score(text: str, query_tokens: Sequence[str]) -> float:
    if not text:
        return 0.0
    hay = text.lower()
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    token_counts: Dict[str, int] = {}
    for t in tokens:
        token_counts[t] = token_counts.get(t, 0) + 1

    score = 0.0
    for qt in query_tokens:
        score += float(token_counts.get(qt, 0))
        if qt and qt in hay:
            score += 0.25
    return score


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    ts: float
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "text": self.text,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MemoryRecord":
        if not isinstance(data, dict):
            raise MemoryError("Memory record must be an object")
        rid = data.get("id")
        ts = data.get("ts")
        text = data.get("text")
        metadata = data.get("metadata", {})
        if not isinstance(rid, str) or not rid:
            raise MemoryError("Memory record missing 'id'")
        if not isinstance(ts, (int, float)):
            raise MemoryError("Memory record missing/invalid 'ts'")
        if not isinstance(text, str):
            raise MemoryError("Memory record missing/invalid 'text'")
        if not isinstance(metadata, dict):
            raise MemoryError("Memory record missing/invalid 'metadata'")
        return MemoryRecord(id=rid, ts=float(ts), text=text, metadata=metadata)


class ConversationBuffer:
    """
    Short-term memory: keep last N messages in memory.
    """

    def __init__(self, max_items: int = 20) -> None:
        if max_items <= 0:
            raise MemoryError("max_items must be > 0")
        self._max_items = max_items
        self._items: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        if role not in {"system", "user", "assistant"}:
            raise MemoryError(f"Invalid role: {role}")
        if not isinstance(content, str):
            raise MemoryError("content must be a string")
        self._items.append({"role": role, "content": content})
        if len(self._items) > self._max_items:
            self._items = self._items[-self._max_items :]

    def get(self) -> List[Dict[str, str]]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()


def write(
    text: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    namespace: str = "default",
    store_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Long-term memory: append a record to a JSONL file.
    Returns the stored record as a dict.
    """
    if not isinstance(text, str) or not text.strip():
        raise MemoryError("text must be a non-empty string")

    store_dir = store_dir or _default_store_dir()
    path = _jsonl_path(store_dir, namespace)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    record = MemoryRecord(
        id=str(uuid.uuid4()),
        ts=_now_ts(),
        text=text,
        metadata=dict(metadata or {}),
    )

    line = json.dumps(record.to_dict(), ensure_ascii=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    return record.to_dict()


def _iter_records(path: str) -> Iterable[MemoryRecord]:
    if not os.path.exists(path):
        return
        yield  # pragma: no cover
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                yield MemoryRecord.from_dict(data)
            except Exception:
                # Skip corrupted lines; keep store append-only and resilient.
                continue


def search(
    query: str,
    *,
    k: int = 5,
    namespace: str = "default",
    store_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Simple keyword retrieval over long-term store.

    Returns up to k records sorted by (score desc, ts desc).
    """
    if not isinstance(query, str) or not query.strip():
        raise MemoryError("query must be a non-empty string")
    if k <= 0:
        return []

    store_dir = store_dir or _default_store_dir()
    path = _jsonl_path(store_dir, namespace)

    qtokens = _tokenize(query)
    if not qtokens:
        raise MemoryError("query must contain at least one alphanumeric token")

    scored: List[Tuple[float, MemoryRecord]] = []
    for rec in _iter_records(path):
        s = _score(rec.text, qtokens)
        if s <= 0:
            continue
        scored.append((s, rec))

    scored.sort(key=lambda pair: (pair[0], pair[1].ts), reverse=True)
    return [rec.to_dict() for _, rec in scored[:k]]


if __name__ == "__main__":
    write("Remember: Swathi prefers concise outputs.", metadata={"kind": "preference"})
    hits = search("prefers concise", k=3)
    print(json.dumps(hits, indent=2))
