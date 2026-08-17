"""
Tests for Sub-Task 2: BaseAgent & AI Client Abstraction.

Covers:
- LLMResponse dataclass.
- LLMClient protocol structural conformance.
- WatsonxClient URL/header construction (no live network calls).
- WatsonxClient._parse_response() happy path and error path.
- WatsonxClient.complete() with a mocked httpx transport.
- AgentResponseError attributes and __str__.
- BaseAgent.run() happy path via a minimal concrete subclass.
- BaseAgent.run() raises AgentResponseError on bad JSON.
- BaseAgent.run() raises AgentResponseError on schema mismatch.
- BaseAgent.run() retries on LLMClientError and eventually raises.
- BaseAgent.run() does NOT retry on AgentResponseError (structural failure).
- BaseAgent._serialise_context() handles non-serialisable values gracefully.
- agents package re-exports.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missionmind.agents.client import (
    LLMClient,
    LLMClientError,
    LLMResponse,
    WatsonxClient,
)
from missionmind.agents.base_agent import AgentResponseError, BaseAgent
from missionmind.schemas.outputs import ScienceAnalysis, ScoredTarget


# ---------------------------------------------------------------------------
# Helpers — minimal concrete agent for testing
# ---------------------------------------------------------------------------

VALID_SCIENCE_JSON = json.dumps({
    "scored_targets": [
        {"waypoint_id": "wp-1", "scientific_value": 0.9, "justification": "iron-rich basalt"},
        {"waypoint_id": "wp-2", "scientific_value": 0.4, "justification": "sandy plain"},
    ],
    "priority_order": ["wp-1", "wp-2"],
    "reasoning": "Prioritised by mineralogical interest.",
})


def _make_mock_client(content: str = VALID_SCIENCE_JSON) -> MagicMock:
    """Return a mock that behaves like a valid LLMClient.

    ``complete`` is an ``AsyncMock`` so ``await client.complete(...)`` works.
    """
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=LLMResponse(
        content=content,
        model="test-model",
        prompt_tokens=50,
        completion_tokens=120,
    ))
    return mock


class ScienceAgentStub(BaseAgent[ScienceAnalysis]):
    """Minimal concrete subclass used exclusively in tests."""

    @property
    def name(self) -> str:
        return "science_stub"

    @property
    def system_prompt(self) -> str:
        return "You are a planetary scientist. Respond in JSON."

    @property
    def response_schema(self) -> type[ScienceAnalysis]:
        return ScienceAnalysis


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_construction(self):
        r = LLMResponse(content='{"ok": true}', model="gpt-4o")
        assert r.content == '{"ok": true}'
        assert r.model == "gpt-4o"
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0

    def test_full_construction(self):
        r = LLMResponse(
            content="hello",
            model="granite",
            prompt_tokens=10,
            completion_tokens=20,
        )
        assert r.prompt_tokens == 10
        assert r.completion_tokens == 20

    def test_frozen(self):
        r = LLMResponse(content="x", model="m")
        with pytest.raises((AttributeError, TypeError)):
            r.content = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LLMClient protocol
# ---------------------------------------------------------------------------

class TestLLMClientProtocol:
    def test_watsonx_client_satisfies_protocol(self):
        """WatsonxClient must be accepted as a valid LLMClient."""
        client = WatsonxClient(api_key="test", base_url="http://localhost")
        assert isinstance(client, LLMClient)

    def test_mock_with_async_complete_is_usable_as_llm_client(self):
        """Any object with an async complete() coroutine can be used as LLMClient.

        Note: runtime_checkable Protocol only checks attribute *existence*, not
        whether it's a coroutine function.  We verify usability behaviourally
        instead of via isinstance().
        """
        mock = _make_mock_client()
        # The attribute exists and is callable — that's what BaseAgent needs.
        assert callable(mock.complete)
        # It is an AsyncMock — awaiting it returns the expected LLMResponse.
        import asyncio
        result = asyncio.run(mock.complete(system_prompt="s", user_message="u"))
        assert isinstance(result, LLMResponse)


# ---------------------------------------------------------------------------
# WatsonxClient construction
# ---------------------------------------------------------------------------

class TestWatsonxClientConstruction:
    def test_default_url_uses_watsonx(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.setenv("IBM_WATSONX_URL", "https://eu-de.ml.cloud.ibm.com")
        client = WatsonxClient()
        assert "eu-de.ml.cloud.ibm.com" in client._base_url
        assert "/ml/v1/text/chat" in client._base_url

    def test_explicit_base_url_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1/chat/completions")
        client = WatsonxClient()
        assert client._base_url == "http://localhost:11434/v1/chat/completions"

    def test_api_key_from_env_watsonx(self, monkeypatch):
        monkeypatch.setenv("IBM_WATSONX_API_KEY", "wx-secret-123")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = WatsonxClient()
        assert client._api_key == "wx-secret-123"

    def test_api_key_falls_back_to_openai(self, monkeypatch):
        monkeypatch.delenv("IBM_WATSONX_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
        client = WatsonxClient()
        assert client._api_key == "sk-openai-fallback"

    def test_explicit_constructor_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("IBM_WATSONX_API_KEY", "env-key")
        client = WatsonxClient(api_key="constructor-key")
        assert client._api_key == "constructor-key"

    def test_project_id_from_env(self, monkeypatch):
        monkeypatch.setenv("IBM_WATSONX_PROJECT_ID", "proj-abc")
        client = WatsonxClient()
        assert client._project_id == "proj-abc"

    def test_default_model_from_config(self):
        from missionmind import config
        client = WatsonxClient()
        assert client._default_model == config.LLM_MODEL_NAME


# ---------------------------------------------------------------------------
# WatsonxClient._parse_response
# ---------------------------------------------------------------------------

class TestWatsonxClientParseResponse:
    def test_happy_path(self):
        data = {
            "choices": [{"message": {"content": '{"answer": 42}'}}],
            "model": "granite-3",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = WatsonxClient._parse_response(data)
        assert result.content == '{"answer": 42}'
        assert result.model == "granite-3"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5

    def test_missing_choices_raises(self):
        with pytest.raises(LLMClientError):
            WatsonxClient._parse_response({})

    def test_empty_choices_raises(self):
        with pytest.raises(LLMClientError):
            WatsonxClient._parse_response({"choices": []})

    def test_missing_usage_defaults_to_zero(self):
        data = {"choices": [{"message": {"content": "hi"}}], "model": "m"}
        result = WatsonxClient._parse_response(data)
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


# ---------------------------------------------------------------------------
# WatsonxClient._build_payload
# ---------------------------------------------------------------------------

class TestWatsonxClientBuildPayload:
    def test_payload_structure(self):
        client = WatsonxClient(api_key="k", base_url="http://x", project_id="p")
        payload = client._build_payload(
            system_prompt="sys",
            user_message="usr",
            model="model-x",
            temperature=0.1,
            max_tokens=512,
        )
        assert payload["model"] == "model-x"
        assert payload["messages"][0] == {"role": "system", "content": "sys"}
        assert payload["messages"][1] == {"role": "user", "content": "usr"}
        assert payload["temperature"] == pytest.approx(0.1)
        assert payload["max_tokens"] == 512
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["project_id"] == "p"

    def test_no_project_id_omitted(self):
        client = WatsonxClient(api_key="k", base_url="http://x", project_id="")
        payload = client._build_payload(
            system_prompt="s",
            user_message="u",
            model="m",
            temperature=0.2,
            max_tokens=100,
        )
        assert "project_id" not in payload

    def test_authorization_header(self):
        client = WatsonxClient(api_key="my-api-key", base_url="http://x")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer my-api-key"

    def test_no_api_key_no_auth_header(self):
        client = WatsonxClient(api_key="", base_url="http://x")
        headers = client._build_headers()
        assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# WatsonxClient.complete() with mocked httpx transport
# ---------------------------------------------------------------------------

class TestWatsonxClientComplete:
    """These tests mock ``httpx.AsyncClient.post`` to avoid real network calls."""

    def _make_fake_response(self, content: str, status_code: int = 200):
        """Return a MagicMock that looks like an httpx.Response."""
        fake = MagicMock()
        fake.status_code = status_code
        fake.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        fake.text = content
        return fake

    async def test_complete_returns_llm_response(self):
        fake_resp = self._make_fake_response('{"ok": true}')
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_resp)):
            client = WatsonxClient(api_key="k", base_url="http://test/chat")
            result = await client.complete(
                system_prompt="sys", user_message="usr"
            )
        assert result.content == '{"ok": true}'
        assert result.model == "test-model"

    async def test_complete_raises_on_http_error(self):
        fake_resp = self._make_fake_response("Internal Server Error", status_code=500)
        fake_resp.text = "Internal Server Error"
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=fake_resp)):
            client = WatsonxClient(api_key="k", base_url="http://test/chat")
            with pytest.raises(LLMClientError, match="HTTP 500"):
                await client.complete(system_prompt="s", user_message="u")

    async def test_complete_raises_on_timeout(self):
        import httpx as _httpx
        with patch(
            "httpx.AsyncClient.post",
            new=AsyncMock(side_effect=_httpx.TimeoutException("timed out")),
        ):
            client = WatsonxClient(api_key="k", base_url="http://test/chat")
            with pytest.raises(LLMClientError, match="timed out"):
                await client.complete(system_prompt="s", user_message="u")

    async def test_complete_raises_on_connection_error(self):
        import httpx as _httpx
        with patch(
            "httpx.AsyncClient.post",
            new=AsyncMock(side_effect=_httpx.ConnectError("refused")),
        ):
            client = WatsonxClient(api_key="k", base_url="http://test/chat")
            with pytest.raises(LLMClientError, match="refused"):
                await client.complete(system_prompt="s", user_message="u")


# ---------------------------------------------------------------------------
# AgentResponseError
# ---------------------------------------------------------------------------

class TestAgentResponseError:
    def test_basic_message(self):
        err = AgentResponseError("bad JSON")
        assert str(err) == "bad JSON"

    def test_with_agent_name(self):
        err = AgentResponseError("bad JSON", agent_name="science")
        assert "science" in str(err)

    def test_with_raw_response(self):
        err = AgentResponseError("bad JSON", raw_response='{"x"')
        assert '{"x"' in str(err)

    def test_raw_response_truncated(self):
        long_raw = "x" * 2_000
        err = AgentResponseError("msg", raw_response=long_raw)
        assert len(err.raw_response) == 1_000

    def test_cause_stored(self):
        cause = ValueError("root cause")
        err = AgentResponseError("outer", cause=cause)
        assert err.cause is cause

    def test_is_value_error(self):
        err = AgentResponseError("msg")
        assert isinstance(err, ValueError)


# ---------------------------------------------------------------------------
# BaseAgent.run() — happy path
# ---------------------------------------------------------------------------

class TestBaseAgentRunHappyPath:
    async def test_run_returns_validated_model(self):
        mock_client = _make_mock_client(VALID_SCIENCE_JSON)
        agent = ScienceAgentStub(llm_client=mock_client)
        result = await agent.run({"candidate_waypoints": []})
        assert isinstance(result, ScienceAnalysis)
        assert len(result.scored_targets) == 2
        assert result.scored_targets[0].waypoint_id == "wp-1"
        assert result.priority_order == ["wp-1", "wp-2"]

    async def test_run_calls_llm_with_system_prompt(self):
        mock_client = _make_mock_client()
        agent = ScienceAgentStub(llm_client=mock_client)
        await agent.run({})
        mock_client.complete.assert_called_once()
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["system_prompt"] == agent.system_prompt

    async def test_run_serialises_context_as_user_message(self):
        mock_client = _make_mock_client()
        agent = ScienceAgentStub(llm_client=mock_client)
        context = {"waypoints": [{"id": "wp-1", "x": 100}]}
        await agent.run(context)
        call_kwargs = mock_client.complete.call_args.kwargs
        user_msg = call_kwargs["user_message"]
        parsed = json.loads(user_msg)
        assert parsed["waypoints"][0]["id"] == "wp-1"

    async def test_run_accepts_empty_context(self):
        mock_client = _make_mock_client()
        agent = ScienceAgentStub(llm_client=mock_client)
        result = await agent.run({})
        assert isinstance(result, ScienceAnalysis)


# ---------------------------------------------------------------------------
# BaseAgent.run() — structural failures (no retry)
# ---------------------------------------------------------------------------

class TestBaseAgentStructuralFailures:
    async def test_run_raises_on_invalid_json(self):
        mock_client = _make_mock_client("this is not json {{{")
        agent = ScienceAgentStub(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run({})
        assert exc_info.value.agent_name == "science_stub"
        assert "this is not json" in exc_info.value.raw_response
        # LLM was called exactly once — no retries on structural failure
        mock_client.complete.assert_called_once()

    async def test_run_raises_on_schema_mismatch(self):
        """Valid JSON but wrong shape for ScienceAnalysis."""
        bad_json = json.dumps({"totally": "wrong", "shape": 123})
        mock_client = _make_mock_client(bad_json)
        agent = ScienceAgentStub(llm_client=mock_client)
        # ScienceAnalysis has optional fields with defaults, so a dict with
        # extra unknown keys passes validation (Pydantic ignores extras by
        # default).  Let's test an actual schema violation instead —
        # scientific_value > 1.0 in a scored target.
        bad_value_json = json.dumps({
            "scored_targets": [
                {"waypoint_id": "wp-1", "scientific_value": 5.0}
            ],
            "priority_order": [],
            "reasoning": "",
        })
        mock_client2 = _make_mock_client(bad_value_json)
        agent2 = ScienceAgentStub(llm_client=mock_client2)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent2.run({})
        assert exc_info.value.agent_name == "science_stub"
        mock_client2.complete.assert_called_once()

    async def test_agent_response_error_has_cause(self):
        mock_client = _make_mock_client("not json")
        agent = ScienceAgentStub(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run({})
        import json as _json
        assert isinstance(exc_info.value.cause, _json.JSONDecodeError)


# ---------------------------------------------------------------------------
# BaseAgent.run() — transient failures and retry
# ---------------------------------------------------------------------------

class TestBaseAgentRetry:
    async def test_run_retries_on_llm_client_error(self):
        """Fails twice then succeeds — should resolve after retries."""
        fail = LLMClientError("temporary network error")
        success = LLMResponse(content=VALID_SCIENCE_JSON, model="m")
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=[fail, fail, success])

        agent = ScienceAgentStub(
            llm_client=mock_client,
            max_retries=2,
            retry_delay=0.0,   # no real sleep in tests
        )
        result = await agent.run({})
        assert isinstance(result, ScienceAnalysis)
        assert mock_client.complete.call_count == 3

    async def test_run_raises_after_exhausting_retries(self):
        """Fails on every attempt — should raise LLMClientError."""
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(side_effect=LLMClientError("always failing"))

        agent = ScienceAgentStub(
            llm_client=mock_client,
            max_retries=2,
            retry_delay=0.0,
        )
        with pytest.raises(LLMClientError):
            await agent.run({})
        # 1 initial attempt + 2 retries = 3 total calls
        assert mock_client.complete.call_count == 3

    async def test_no_retry_on_agent_response_error(self):
        """Structural parse failure must NOT be retried."""
        mock_client = _make_mock_client("bad json {{")
        agent = ScienceAgentStub(
            llm_client=mock_client,
            max_retries=2,
            retry_delay=0.0,
        )
        with pytest.raises(AgentResponseError):
            await agent.run({})
        # Called only once — no retries
        mock_client.complete.assert_called_once()


# ---------------------------------------------------------------------------
# BaseAgent._serialise_context
# ---------------------------------------------------------------------------

class TestBaseAgentSerialiseContext:
    def test_serialises_simple_dict(self):
        result = BaseAgent._serialise_context({"x": 1, "y": "hello"})
        assert json.loads(result) == {"x": 1, "y": "hello"}

    def test_serialises_nested_dict(self):
        context = {"waypoints": [{"id": "wp-1", "x": 10.5}]}
        result = BaseAgent._serialise_context(context)
        parsed = json.loads(result)
        assert parsed["waypoints"][0]["x"] == pytest.approx(10.5)

    def test_non_serialisable_falls_back_to_str(self):
        """Objects that can't be JSON-serialised should be converted to str."""
        from datetime import datetime
        context = {"ts": datetime(2025, 1, 1)}
        result = BaseAgent._serialise_context(context)
        parsed = json.loads(result)
        # datetime is stringified
        assert isinstance(parsed["ts"], str)

    def test_empty_context_serialises_to_empty_object(self):
        result = BaseAgent._serialise_context({})
        assert result == "{}"


# ---------------------------------------------------------------------------
# agents package re-exports
# ---------------------------------------------------------------------------

class TestAgentsPackageExports:
    def test_exports(self):
        from missionmind.agents import (
            AgentResponseError,
            BaseAgent,
            LLMClient,
            LLMResponse,
            WatsonxClient,
        )
        assert issubclass(AgentResponseError, ValueError)
        assert issubclass(WatsonxClient, object)
        r = LLMResponse(content="x", model="m")
        assert r.content == "x"
