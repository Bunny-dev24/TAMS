"""Validates the retry-then-degrade-gracefully behavior added to chat(),
chat_json(), and embed() for the real (non-fake) OpenAI code path."""
from __future__ import annotations

from types import SimpleNamespace

from tams.llm.client import LLMClient


class _FlakyOpenAI:
    """Fails `fail_times` calls, then succeeds. Mimics transient API errors."""

    def __init__(self, fail_times: int, success_payload):
        self.fail_times = fail_times
        self.calls = 0
        self.success_payload = success_payload
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._embed_create)

    def _create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient API error")
        return self.success_payload

    def _embed_create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient API error")
        return self.success_payload


def _make_real_client(settings, openai_stub) -> LLMClient:
    settings = settings.model_copy(update={"openai_api_key": "sk-test-fake-key"})
    client = LLMClient(settings)
    client._openai = openai_stub  # bypass real network entirely
    return client


def test_chat_retries_then_succeeds(settings):
    payload = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello from retry"))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
    )
    stub = _FlakyOpenAI(fail_times=2, success_payload=payload)
    client = _make_real_client(settings, stub)

    result = client.chat([{"role": "user", "content": "hi"}], model="m")
    assert result.text == "hello from retry"
    assert stub.calls == 3  # failed twice, succeeded on the 3rd (max retries)


def test_chat_degrades_gracefully_after_exhausting_retries(settings):
    stub = _FlakyOpenAI(fail_times=99, success_payload=None)  # always fails
    client = _make_real_client(settings, stub)

    result = client.chat([{"role": "user", "content": "hi"}], model="m")
    assert "trouble reaching" in result.text.lower()
    assert stub.calls == 3  # stopped after max attempts, never raised


def test_chat_json_retries_on_malformed_json_then_succeeds(settings):
    bad = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not json!"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    good = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"items": [1, 2]}'))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2),
    )

    class _JsonFlaky(_FlakyOpenAI):
        def _create(self, **kwargs):
            self.calls += 1
            return bad if self.calls == 1 else good

    stub = _JsonFlaky(fail_times=0, success_payload=good)
    client = _make_real_client(settings, stub)

    result = client.chat_json([{"role": "user", "content": "hi"}], model="m", schema={"items": "int[]"})
    assert result == {"items": [1, 2]}
    assert stub.calls == 2


def test_chat_json_degrades_to_empty_dict_after_exhausting_retries(settings):
    stub = _FlakyOpenAI(fail_times=99, success_payload=None)
    client = _make_real_client(settings, stub)

    result = client.chat_json([{"role": "user", "content": "hi"}], model="m", schema={})
    assert result == {}
    assert stub.calls == 3


def test_llm_client_passes_custom_base_url_to_openai_sdk(settings, monkeypatch):
    """Verifies the Abacus-AI-or-any-OpenAI-compatible-provider wiring: when
    llm_base_url is set, it must reach the OpenAI SDK constructor so chat/
    embeddings calls route to that provider instead of api.openai.com."""
    captured = {}

    class _FakeOpenAICtor:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAICtor)

    custom_settings = settings.model_copy(
        update={"openai_api_key": "abacus-test-key", "llm_base_url": "https://routellm.abacus.ai/v1"}
    )
    LLMClient(custom_settings)

    assert captured["api_key"] == "abacus-test-key"
    assert captured["base_url"] == "https://routellm.abacus.ai/v1"


def test_embed_retries_then_succeeds(settings):
    payload = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])],
        usage=SimpleNamespace(prompt_tokens=4),
    )
    stub = _FlakyOpenAI(fail_times=1, success_payload=payload)
    client = _make_real_client(settings, stub)

    vectors = client.embed(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert stub.calls == 2


def test_chat_handles_usage_object_with_none_token_fields(settings):
    """Regression test: some OpenAI-SDK-compatible gateways (observed with
    Abacus AI RouteLLM in real-mode testing) return a `usage` object that is
    present but has None fields instead of omitting it or zero-filling it.
    This must not crash chat()/chat_json()/embed() or their cost tracking."""
    payload = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="PONG"))],
        usage=SimpleNamespace(prompt_tokens=None, completion_tokens=None),
    )
    stub = _FlakyOpenAI(fail_times=0, success_payload=payload)
    client = _make_real_client(settings, stub)

    result = client.chat([{"role": "user", "content": "hi"}], model="m")
    assert result.text == "PONG"
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


def test_embed_handles_usage_object_with_none_prompt_tokens(settings):
    payload = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2])],
        usage=SimpleNamespace(prompt_tokens=None),
    )
    stub = _FlakyOpenAI(fail_times=0, success_payload=payload)
    client = _make_real_client(settings, stub)

    vectors = client.embed(["a"])
    assert vectors == [[0.1, 0.2]]
