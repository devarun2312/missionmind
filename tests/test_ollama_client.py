"""
Tests for OllamaClient and the provider-selection factory.

All tests are fully offline — no running Ollama server is required.
Network calls are mocked via unittest.mock / httpx mocking.

Test coverage
-------------
1.  Happy path — valid 200 response returns expected LLMResponse
2.  message.content is used (not choices[])
3.  thinking field is ignored
4.  Missing 'message' key → LLMClientError
5.  Missing 'content' key inside message → LLMClientError
6.  Empty content string → LLMClientError
7.  Non-2xx HTTP response → LLMClientError
8.  httpx.TimeoutException → LLMClientError
9.  httpx.RequestError (connection refused) → LLMClientError
10. provider=ollama selects OllamaClient
11. provider=watsonx selects WatsonxClient
12. Unknown provider falls back to WatsonxClient
13. OllamaClient satisfies the LLMClient protocol
14. Request payload includes stream=False and format="json"
15. Token counts read from Ollama fields (prompt_eval_count, eval_count)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from missionmind.agents.client import (
    LLMClient,
    LLMClientError,
    LLMResponse,
    OllamaClient,
    WatsonxClient,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_body(
    content: str = '{"ok": true}',
    model: str = "granite4.2:3b",
    thinking: str = "internal reasoning text",
    prompt_eval_count: int = 42,
    eval_count: int = 88,
) -> dict:
    """Build a realistic Ollama /api/chat response body."""
    return {
        "model": model,
        "created_at": "2025-01-01T00:00:00.000Z",
        "message": {
            "role": "assistant",
            "content": content,
        },
        "thinking": thinking,      # must be ignored
        "done": True,
        "done_reason": "stop",
        "total_duration": 5_000_000_000,
        "load_duration": 100_000,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


def _make_fake_http_resp(body: dict, status_code: int = 200) -> MagicMock:
    fake = MagicMock()
    fake.status_code = status_code
    fake.json.return_value = body
    fake.text = json.dumps(body)
    return fake


# ---------------------------------------------------------------------------
# OllamaClient._parse_response (static, no network)
# ---------------------------------------------------------------------------

class TestOllamaClientParseResponse:
    def test_happy_path_extracts_content(self):
        data = _ollama_body(content='{"scored_targets": []}')
        result = OllamaClient._parse_response(data, "granite4.2:3b")
        assert result.content == '{"scored_targets": []}'

    def test_model_field_used(self):
        data = _ollama_body(model="granite4.2:3b")
        result = OllamaClient._parse_response(data)
        assert result.model == "granite4.2:3b"

    def test_model_hint_used_as_fallback(self):
        data = _ollama_body()
        del data["model"]
        result = OllamaClient._parse_response(data, model_hint="fallback-model")
        assert result.model == "fallback-model"

    def test_thinking_field_is_ignored(self):
        """The 'thinking' field must NOT appear anywhere in the returned LLMResponse."""
        data = _ollama_body(content='{"answer": 1}', thinking="secret chain-of-thought")
        result = OllamaClient._parse_response(data)
        assert "thinking" not in result.content
        assert "secret" not in result.content

    def test_token_counts_from_ollama_fields(self):
        data = _ollama_body(prompt_eval_count=15, eval_count=30)
        result = OllamaClient._parse_response(data)
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 30

    def test_missing_message_key_raises(self):
        data = {"model": "m", "done": True}   # no 'message'
        with pytest.raises(LLMClientError, match="message.content"):
            OllamaClient._parse_response(data)

    def test_missing_content_key_raises(self):
        data = {"model": "m", "message": {"role": "assistant"}}  # no 'content'
        with pytest.raises(LLMClientError, match="message.content"):
            OllamaClient._parse_response(data)

    def test_empty_content_raises(self):
        data = _ollama_body(content="")
        with pytest.raises(LLMClientError, match="empty"):
            OllamaClient._parse_response(data)

    def test_message_is_none_raises(self):
        data = {"model": "m", "message": None}
        with pytest.raises(LLMClientError, match="message.content"):
            OllamaClient._parse_response(data)


# ---------------------------------------------------------------------------
# OllamaClient.complete() — mocked httpx transport
# ---------------------------------------------------------------------------

class TestOllamaClientComplete:
    async def test_complete_happy_path(self):
        fake_resp = _make_fake_http_resp(_ollama_body(content='{"result": "ok"}'))
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_resp)):
            client = OllamaClient(base_url="http://localhost:11434")
            result = await client.complete(
                system_prompt="You are a scientist.",
                user_message='{"waypoints": []}',
            )
        assert result.content == '{"result": "ok"}'
        assert isinstance(result, LLMResponse)

    async def test_complete_uses_message_content_not_choices(self):
        """Ensures OllamaClient reads response["message"]["content"],
        not the OpenAI-style response["choices"][0]["message"]["content"]."""
        body = _ollama_body(content='{"check": "ollama_path"}')
        # Add a misleading 'choices' key that should be ignored
        body["choices"] = [{"message": {"content": '{"check": "wrong_path"}'}}]
        fake_resp = _make_fake_http_resp(body)
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_resp)):
            client = OllamaClient(base_url="http://localhost:11434")
            result = await client.complete(system_prompt="s", user_message="u")
        assert result.content == '{"check": "ollama_path"}'

    async def test_non_2xx_response_raises(self):
        fake_resp = _make_fake_http_resp({}, status_code=500)
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_resp)):
            client = OllamaClient(base_url="http://localhost:11434")
            with pytest.raises(LLMClientError, match="HTTP 500"):
                await client.complete(system_prompt="s", user_message="u")

    async def test_404_raises(self):
        fake_resp = _make_fake_http_resp({}, status_code=404)
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_resp)):
            client = OllamaClient(base_url="http://localhost:11434")
            with pytest.raises(LLMClientError, match="HTTP 404"):
                await client.complete(system_prompt="s", user_message="u")

    async def test_timeout_raises_llm_client_error(self):
        with patch(
            "httpx.AsyncClient.post",
            new=AsyncMock(side_effect=httpx.TimeoutException("timed out")),
        ):
            client = OllamaClient(base_url="http://localhost:11434", timeout=5.0)
            with pytest.raises(LLMClientError, match="timed out"):
                await client.complete(system_prompt="s", user_message="u")

    async def test_connection_refused_raises_llm_client_error(self):
        with patch(
            "httpx.AsyncClient.post",
            new=AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            ),
        ):
            client = OllamaClient(base_url="http://localhost:11434")
            with pytest.raises(LLMClientError, match="Ollama request failed"):
                await client.complete(system_prompt="s", user_message="u")

    async def test_request_payload_has_stream_false_and_format_json(self):
        """Verify the request body sent to Ollama has stream=False, format='json'."""
        fake_resp = _make_fake_http_resp(_ollama_body())
        captured: list[dict] = []

        async def capture_post(self, url, *, json=None, **kwargs):  # noqa: A002
            captured.append(json or {})
            return fake_resp

        with patch("httpx.AsyncClient.post", new=capture_post):
            client = OllamaClient(base_url="http://localhost:11434")
            await client.complete(system_prompt="sys", user_message="usr")

        assert len(captured) == 1
        payload = captured[0]
        assert payload["stream"] is False
        assert payload["format"] == "json"
        assert payload["messages"][0] == {"role": "system", "content": "sys"}
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"].startswith("usr")
        assert "Return ONLY one complete valid JSON object" in payload["messages"][1]["content"]

    async def test_model_override_forwarded(self):
        fake_resp = _make_fake_http_resp(_ollama_body(model="custom-model"))
        captured: list[dict] = []

        async def capture_post(self, url, *, json=None, **kwargs):  # noqa: A002
            captured.append(json or {})
            return fake_resp

        with patch("httpx.AsyncClient.post", new=capture_post):
            client = OllamaClient(base_url="http://localhost:11434")
            await client.complete(
                system_prompt="s", user_message="u", model="custom-model"
            )

        assert captured[0]["model"] == "custom-model"


# ---------------------------------------------------------------------------
# OllamaClient construction
# ---------------------------------------------------------------------------

class TestOllamaClientConstruction:
    def test_default_url_from_config(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.1:11434")
        # Re-import config so the env var is picked up
        import importlib
        from missionmind import config as cfg
        importlib.reload(cfg)
        client = OllamaClient()
        assert "10.0.0.1" in client._base_url
        importlib.reload(cfg)  # restore

    def test_explicit_base_url_wins(self):
        client = OllamaClient(base_url="http://custom:9999")
        assert client._base_url == "http://custom:9999"

    def test_trailing_slash_stripped(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert not client._base_url.endswith("/")

    def test_default_model_from_config(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "llama3")
        import importlib
        from missionmind import config as cfg
        importlib.reload(cfg)
        client = OllamaClient()
        assert client._default_model == "llama3"
        importlib.reload(cfg)

    def test_explicit_model_wins(self):
        client = OllamaClient(default_model="my-model")
        assert client._default_model == "my-model"

    def test_satisfies_llm_client_protocol(self):
        client = OllamaClient(base_url="http://localhost:11434")
        assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# Provider selection via _build_llm_client()
# ---------------------------------------------------------------------------

class TestProviderSelection:
    def test_ollama_provider_returns_ollama_client(self, monkeypatch):
        monkeypatch.setenv("MISSIONMIND_LLM_PROVIDER", "ollama")
        import importlib
        from missionmind import config as cfg
        importlib.reload(cfg)
        from missionmind.planning import planner
        importlib.reload(planner)
        client = planner._build_llm_client()
        assert isinstance(client, OllamaClient)
        importlib.reload(cfg)
        importlib.reload(planner)

    def test_watsonx_provider_returns_watsonx_client(self, monkeypatch):
        monkeypatch.setenv("MISSIONMIND_LLM_PROVIDER", "watsonx")
        import importlib
        from missionmind import config as cfg
        importlib.reload(cfg)
        from missionmind.planning import planner
        importlib.reload(planner)
        client = planner._build_llm_client()
        assert isinstance(client, WatsonxClient)
        importlib.reload(cfg)
        importlib.reload(planner)

    def test_unknown_provider_falls_back_to_watsonx(self, monkeypatch):
        monkeypatch.setenv("MISSIONMIND_LLM_PROVIDER", "openai")
        import importlib
        from missionmind import config as cfg
        importlib.reload(cfg)
        from missionmind.planning import planner
        importlib.reload(planner)
        client = planner._build_llm_client()
        assert isinstance(client, WatsonxClient)
        importlib.reload(cfg)
        importlib.reload(planner)

    def test_default_is_watsonx(self, monkeypatch):
        monkeypatch.delenv("MISSIONMIND_LLM_PROVIDER", raising=False)
        import importlib
        from missionmind import config as cfg
        importlib.reload(cfg)
        from missionmind.planning import planner
        importlib.reload(planner)
        client = planner._build_llm_client()
        assert isinstance(client, WatsonxClient)
        importlib.reload(cfg)
        importlib.reload(planner)
