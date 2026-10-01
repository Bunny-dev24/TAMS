"""Shared token counting utility (tiktoken, with a rough fallback)."""
from __future__ import annotations

_encoding = None
_tried = False


def _get_encoding():
    global _encoding, _tried
    if not _tried:
        _tried = True
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoding = None
    return _encoding


def count_tokens(text: str) -> int:
    enc = _get_encoding()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))
