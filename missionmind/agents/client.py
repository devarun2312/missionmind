"""
LLM client abstraction for MissionMind.

Architecture
------------
Every AI agent in MissionMind calls exactly ONE thing: an ``LLMClient``.
``LLMClient`` is a *protocol* (a structural interface) — any object that
implements the ``complete()`` coroutine is a valid client.  This means:

* In production, ``WatsonxClient`` calls the IBM watsonx AI REST endpoint.
* In tests, a ``MockLLMClient`` can return canned JSON with zero network calls.
* The same code path works with any OpenAI-compatible endpoint (e.g. local
  Ollama, GPT-4o during development) just by changing env vars.

IBM watsonx AI compatibility
-----------------------------
IBM watsonx AI exposes an OpenAI-compatible ``/ml/v1/text/chat`` endpoint.
We construct standard ``ChatCompletionRequest`` payloads and POST them via
``httpx.AsyncClient``.  The response is standard OpenAI JSON.

Required environment variables (production)
--------------------------------------------
IBM_WATSONX_API_KEY   Your IBM Cloud API key.
IBM_WATSONX_URL       Watsonx AI endpoint, e.g.
                      https://us-south.ml.cloud.ibm.com
IBM_WATSONX_PROJECT_ID  Your watsonx project ID.
LLM_MODEL_NAME        Model ID, e.g. "ibm/granite-3-8b-instruct".

Optional overrides (local development / other providers)
---------------------------------------------------------
LLM_BASE_URL          Full chat-completions URL.  Overrides watsonx defaults.
                      E.g. http://localhost:11434/v1/chat/completions for Ollama
                      or https://api.openai.com/v1/chat/completions for OpenAI.
OPENAI_API_KEY        Used as the Bearer token when LLM_BASE_URL is set and
                      IBM_WATSONX_API_KEY is absent.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from missionmind import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMResponse:
    """The normalised result returned by any ``LLMClient.complete()`` call.

    Attributes
    ----------
    content:
        The raw text content of the first choice returned by the model.
        For JSON-mode requests this will be a JSON string.
    model:
        The model identifier reported by the server.
    prompt_tokens:
        Number of tokens consumed by the prompt (0 if not reported).
    completion_tokens:
        Number of tokens in the completion (0 if not reported).
    """

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---------------------------------------------------------------------------
# Protocol (interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """Structural interface for any LLM backend.

    Any object that provides an async ``complete()`` method matching this
    signature is a valid ``LLMClient`` — no subclassing required.

    This makes it trivial to swap the backend (watsonx, OpenAI, Ollama, mock)
    without touching agent code.
    """

    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a chat-completion request and return a normalised response.

        Parameters
        ----------
        system_prompt:
            The role/instruction text placed in the ``system`` message slot.
        user_message:
            The mission context serialised as a user-turn message.
        model:
            Override the default model for this call.
        temperature:
            Sampling temperature (lower = more deterministic).
        max_tokens:
            Maximum tokens to generate.

        Returns
        -------
        LLMResponse
            Normalised response with ``content`` guaranteed to be non-empty.

        Raises
        ------
        LLMClientError
            On network failure, timeout, or a non-2xx HTTP response.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMClientError(RuntimeError):
    """Raised when the LLM backend returns an error or is unreachable."""


# ---------------------------------------------------------------------------
# Production implementation — IBM watsonx AI
# ---------------------------------------------------------------------------

class WatsonxClient:
    """Async LLM client targeting IBM watsonx AI's chat-completions endpoint.

    The client is compatible with *any* OpenAI-style ``/chat/completions``
    endpoint; point ``LLM_BASE_URL`` at a different URL to redirect calls.

    Parameters
    ----------
    api_key:
        IBM Cloud / watsonx API key.  Defaults to the ``IBM_WATSONX_API_KEY``
        env var, then falls back to ``OPENAI_API_KEY`` for local dev.
    base_url:
        Full URL of the chat completions endpoint.  Defaults to the watsonx
        construction from ``IBM_WATSONX_URL``, or ``LLM_BASE_URL`` if set.
    project_id:
        Watsonx project ID required for IBM's endpoint.  Ignored when using a
        non-IBM ``base_url``.
    default_model:
        Model ID used when callers don't supply one.  Defaults to
        ``config.LLM_MODEL_NAME``.
    timeout:
        Per-request timeout in seconds.
    """

    #: IBM watsonx AI chat completions path template.
    _WATSONX_PATH = "/ml/v1/text/chat?version=2024-05-01"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        project_id: str | None = None,
        default_model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("IBM_WATSONX_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self._project_id = project_id or os.environ.get(
            "IBM_WATSONX_PROJECT_ID", ""
        )
        self._base_url = base_url or self._resolve_base_url()
        self._default_model = default_model or config.LLM_MODEL_NAME
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """POST a chat-completion request to the configured endpoint."""
        chosen_model = model or self._default_model
        payload = self._build_payload(
            system_prompt=system_prompt,
            user_message=user_message,
            model=chosen_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        headers = self._build_headers()

        logger.debug(
            "LLM request | model=%s | url=%s | prompt_chars=%d",
            chosen_model,
            self._base_url,
            len(system_prompt) + len(user_message),
        )

        async with httpx.AsyncClient(timeout=self._timeout) as http:
            try:
                resp = await http.post(
                    self._base_url,
                    json=payload,
                    headers=headers,
                )
            except httpx.TimeoutException as exc:
                raise LLMClientError(
                    f"LLM request timed out after {self._timeout}s"
                ) from exc
            except httpx.RequestError as exc:
                raise LLMClientError(
                    f"LLM request failed: {exc}"
                ) from exc

        if resp.status_code >= 400:
            raise LLMClientError(
                f"LLM endpoint returned HTTP {resp.status_code}: {resp.text[:400]}"
            )

        return self._parse_response(resp.json())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_base_url() -> str:
        """Construct the completions URL from environment variables."""
        explicit = os.environ.get("LLM_BASE_URL")
        if explicit:
            return explicit
        watsonx_url = os.environ.get(
            "IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
        )
        return watsonx_url.rstrip("/") + WatsonxClient._WATSONX_PATH

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_payload(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        # Watsonx requires project_id; harmless for other providers.
        if self._project_id:
            payload["project_id"] = self._project_id
        return payload

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> LLMResponse:
        """Extract content and token usage from a chat-completions JSON body."""
        try:
            content: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMClientError(
                f"Unexpected LLM response shape: {json.dumps(data)[:400]}"
            ) from exc

        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            model=data.get("model", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
