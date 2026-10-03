"""One-off smoke test for the real (non-fake) OpenAI LLM backend. Exercises
chat(), chat_json(), and embed() independently so a failure in one does not
hide results from the others.

Note: chat()/chat_json() degrade gracefully instead of raising (by design —
see tams/llm/client.py), so this script detects failure by checking for the
known fallback sentinel / empty-dict result rather than relying on
exceptions. embed() currently has NO graceful degradation (a known gap), so
its failures surface as real exceptions.

Run with: python3 scripts/smoke_test_provider.py
"""
from __future__ import annotations

import sys

from tams.config import load_settings
from tams.llm.client import LLMClient

CHAT_FALLBACK_SENTINEL = "having trouble reaching the language model"


def main() -> int:
    settings = load_settings()
    print(f"openai_api_key set: {bool(settings.openai_api_key)}")
    print(f"orchestrator_model: {settings.orchestrator_model}")
    print(f"extractor_model: {settings.extractor_model}")
    print(f"embedding_model: {settings.embedding_model}")
    print("-" * 60)

    if not settings.openai_api_key:
        print("FAIL: no API key loaded — fake mode would be used. Aborting.")
        return 1

    client = LLMClient(settings)
    results: dict[str, bool] = {}

    # 1. Plain chat
    try:
        out = client.chat(
            messages=[
                {"role": "system", "content": "You are a terse test assistant."},
                {"role": "user", "content": "Reply with exactly the word: PONG"},
            ],
            model=settings.orchestrator_model,
        )
        print(f"[chat] text={out.text!r} prompt_tokens={out.prompt_tokens} completion_tokens={out.completion_tokens}")
        # Note: a response can legitimately have 0 reported tokens in some
        # edge cases, so we don't treat that alone as a failure signal here.
        ok = CHAT_FALLBACK_SENTINEL not in out.text
        results["chat"] = ok
        if not ok:
            print("[chat] -> looks like a degraded fallback, not a real response")
    except Exception as e:
        print(f"[chat] FAILED (unexpected raise): {type(e).__name__}: {e}")
        results["chat"] = False

    print("-" * 60)

    # 2. JSON-mode chat (used by extractors)
    try:
        schema = {"value": "int"}
        out = client.chat_json(
            messages=[
                {
                    "role": "user",
                    "content": (
                        'Extract the number as JSON: {"value": <int>}. '
                        'Text: "The timeout is set to 30 seconds."'
                    ),
                }
            ],
            model=settings.extractor_model,
            schema=schema,
        )
        print(f"[chat_json] -> {out!r}")
        results["chat_json"] = bool(out)
        if not out:
            print("[chat_json] -> empty dict = degraded fallback, not a real response")
    except Exception as e:
        print(f"[chat_json] FAILED (unexpected raise): {type(e).__name__}: {e}")
        results["chat_json"] = False

    print("-" * 60)

    # 3. Embeddings. No graceful degradation exists yet for this call, so a
    # failure here raises directly.
    try:
        vec = client.embed(["hello world"])
        print(f"[embed] -> vector of length {len(vec[0])}, first 5 dims: {vec[0][:5]}")
        results["embed"] = True
    except Exception as e:
        print(f"[embed] FAILED: {type(e).__name__}: {e}")
        results["embed"] = False

    print("=" * 60)
    for k, v in results.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    return 0 if all(results.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
