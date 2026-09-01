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
import re
from decimal import Decimal, InvalidOperation
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


# ---------------------------------------------------------------------------
# Local development implementation — Ollama (IBM Granite)
# ---------------------------------------------------------------------------

class OllamaClient:
    """Async LLM client targeting a local Ollama server.

    Calls ``POST {base_url}/api/chat`` with ``stream=false`` and
    ``format="json"`` so the model is strongly encouraged to return valid JSON.

    Only ``response["message"]["content"]`` is forwarded to ``BaseAgent``
    for Pydantic parsing.  All other fields — ``thinking``, timing metadata,
    token counts — are silently discarded.

    Parameters
    ----------
    base_url:
        Base URL of the Ollama server (no trailing slash).
        Defaults to ``config.OLLAMA_BASE_URL`` (``http://127.0.0.1:11434``).
    default_model:
        Model tag used when callers don't override it.
        Defaults to ``config.OLLAMA_MODEL`` (``granite4.2:3b``).
    timeout:
        Per-request timeout in seconds.  Granite on CPU can be slow — the
        default is intentionally generous.
    """

    _CHAT_PATH = "/api/chat"
    _NUMERIC_ADD_EXPR = re.compile(
        r'(?P<prefix>"(?:[^"\\]|\\.)+"\s*:\s*)'
        r'(?P<expr>-?\d+(?:\.\d+)?(?:\s*\+\s*-?\d+(?:\.\d+)?)+)'
        r'(?P<suffix>\s*[,}])'
    )

    def __init__(
        self,
        *,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self._default_model = default_model or config.OLLAMA_MODEL
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public interface — satisfies the LLMClient protocol
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
        """POST a chat request to Ollama and return a normalised LLMResponse.

        Parameters match the ``LLMClient`` protocol exactly so
        ``OllamaClient`` is a drop-in replacement for ``WatsonxClient``.
        """
        chosen_model = model or self._default_model
        url = self._base_url + self._CHAT_PATH
        structured_user_message = (
            user_message
            + "\n\nIMPORTANT OUTPUT RULES:"
            + "\n- Return ONLY one complete valid JSON object."
            + "\n- Keep every justification/reasoning string concise: maximum one sentence."
            + "\n- Do not show calculations, internal reasoning, or chain-of-thought."
            + "\n- Every numeric JSON value must be a literal number."
            + "\n- Calculate arithmetic before producing JSON; never put arithmetic expressions in JSON values."
            + "\n- Do not include markdown or text outside the JSON object."
        )
        payload: dict[str, Any] = {
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": structured_user_message},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_predict": max(max_tokens, 4096),
                "num_ctx": 8192,
            },
        }

        logger.debug(
            "OllamaClient request | model=%s | url=%s | prompt_chars=%d",
            chosen_model,
            url,
            len(system_prompt) + len(user_message),
        )

        async with httpx.AsyncClient(timeout=self._timeout) as http:
            try:
                resp = await http.post(url, json=payload)
            except httpx.TimeoutException as exc:
                raise LLMClientError(
                    f"Ollama request timed out after {self._timeout}s"
                ) from exc
            except httpx.RequestError as exc:
                raise LLMClientError(
                    f"Ollama request failed (is Ollama running?): {exc}"
                ) from exc

        if resp.status_code >= 400:
            raise LLMClientError(
                f"Ollama endpoint returned HTTP {resp.status_code}: {resp.text[:400]}"
            )

        return self._parse_response(resp.json(), chosen_model)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
        _NUMERIC_ADD_EXPR = re.compile(
        r'(?P<prefix>"(?:[^"\\]|\\.)+"\s*:\s*)'
        r'(?P<expr>-?\d+(?:\.\d+)?(?:\s*\+\s*-?\d+(?:\.\d+)?)+)'
        r'(?P<suffix>\s*[,}])'
    )

    @staticmethod
    def _repair_numeric_expressions(content: str) -> str:
        """Convert simple numeric addition expressions into JSON numbers.

        Granite occasionally emits values such as:
            "total_energy": 22.5 + 14.0 + 19.0

        That is mathematically valid but not valid JSON. Only simple
        addition of numeric literals is repaired; arbitrary expressions
        are never evaluated.
        """

        def replace(match: re.Match[str]) -> str:
            expression = match.group("expr")

            try:
                total = sum(
                    (Decimal(part.strip()) for part in expression.split("+")),
                    Decimal("0"),
                )
            except InvalidOperation:
                return match.group(0)

            return (
                f'{match.group("prefix")}'
                f'{format(total, "f")}'
                f'{match.group("suffix")}'
            )

        return OllamaClient._NUMERIC_ADD_EXPR.sub(replace, content)
    @staticmethod
    def _parse_response(data: dict[str, Any], model_hint: str = "") -> LLMResponse:
        """Extract ``message.content`` from the Ollama /api/chat response.

        The ``thinking`` field and all timing/metadata fields are
        intentionally discarded.

        Raises
        ------
        LLMClientError
            If ``message`` or ``content`` are absent or if content is empty.
        """
        try:
            content: str = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMClientError(
                f"Ollama response missing 'message.content': {json.dumps(data)[:400]}"
            ) from exc

        if not content:
            raise LLMClientError(
                "Ollama response contained an empty 'message.content'"
            )
        content = OllamaClient._repair_numeric_expressions(content)
        return LLMResponse(
            content=content,
            model=data.get("model", model_hint),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
