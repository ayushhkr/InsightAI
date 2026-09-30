"""Small process-local cache for deterministic repeated LLM requests."""
from __future__ import annotations
import hashlib, json, re

class LLMResponseCache:
    def __init__(self, max_entries: int = 128): self.max_entries, self._items = max_entries, {}
    def key(self, dataset_metadata: dict, question: str, history: list, model_name: str) -> str | None:
        dataset_fingerprint = dataset_metadata.get("dataset_fingerprint") or hashlib.sha256(json.dumps(dataset_metadata, sort_keys=True, default=str).encode()).hexdigest()
        payload = {"dataset_fingerprint": dataset_fingerprint, "question": " ".join(question.lower().split()), "history": history or [], "model": model_name}
        serialized = json.dumps(payload, sort_keys=True, default=str)
        if re.search(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]", serialized): return None
        return hashlib.sha256(serialized.encode()).hexdigest()
    def get(self, key: str | None): return self._items.get(key) if key else None
    def set(self, key: str | None, value: dict) -> None:
        if not key: return
        if len(self._items) >= self.max_entries: self._items.pop(next(iter(self._items)))
        self._items[key] = value
