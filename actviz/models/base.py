"""Interface every model adapter implements, plus tokenizer helpers they share."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..trace import Trace


class ModelAdapter(ABC):
    @classmethod
    @abstractmethod
    def load(cls, repo: str, revision: str | None = None) -> "ModelAdapter":
        """Download (or reuse cached) weights and tokenizer."""

    @abstractmethod
    def tokenize(self, prompt: str, bos: bool = True) -> list[int]:
        ...

    @abstractmethod
    def trace(self, ids: list[int], prompt: str) -> Trace:
        """Run the forward pass on `ids`, recording every intermediate and the page layout."""


def token_text(tokenizer, i: int) -> str:
    """Human-readable text of one token; falls back to the raw BPE piece for
    tokens that are only part of a multi-byte character."""
    s = tokenizer.decode([i], skip_special_tokens=False)
    if "�" in s:
        s = tokenizer.id_to_token(i) or s
    return s


def token_infos(tokenizer, ids: list[int]) -> list[dict]:
    return [{"id": int(i), "text": token_text(tokenizer, i)} for i in ids]


def vocab_labels(tokenizer, n: int) -> list[str]:
    return [token_text(tokenizer, i) for i in range(n)]
