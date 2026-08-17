"""
BaseAgent — the shared abstract base class for all MissionMind AI agents.

Design
------
Every specialised agent (Science, Resource, Safety, Commander) inherits from
``BaseAgent`` and only needs to supply two things:

1. ``system_prompt`` — a string that tells the model its role and output format.
2. ``response_schema`` — the Pydantic model class the response must conform to.

``BaseAgent.run(context)`` does the rest:

    context dict
        │
        ▼  (serialised to JSON user message)
    LLMClient.complete()
        │
        ▼  (raw JSON string from model)
    JSON parse
        │
        ▼  (Pydantic validation)
    TypedResponseModel  ──or──  AgentResponseError


Retry policy
------------
Transient network failures (``LLMClientError``) are retried up to
``max_retries`` times with an exponential back-off.  Structural failures
(JSON parse error, Pydantic validation error) are *not* retried — a bad
response from the model on the first attempt is unlikely to improve on retry.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from missionmind.agents.client import LLMClient, LLMClientError, WatsonxClient

logger = logging.getLogger(__name__)

# Generic type variable so subclasses can express their concrete response type.
ResponseT = TypeVar("ResponseT", bound=BaseModel)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class AgentResponseError(ValueError):
    """Raised when an AI agent's response cannot be parsed or validated.

    This is a ``ValueError`` subclass so callers can catch it alongside other
    input-validation errors.  It always contains a human-readable message that
    explains *what* failed and *which agent* produced the bad response.

    Attributes
    ----------
    agent_name:
        The name of the agent that produced the bad response.
    raw_response:
        The raw string returned by the LLM, truncated to 1 000 characters.
    cause:
        The original exception (``json.JSONDecodeError`` or
        ``pydantic.ValidationError``) that caused the failure.
    """

    def __init__(
        self,
        message: str,
        *,
        agent_name: str = "",
        raw_response: str = "",
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.agent_name = agent_name
        self.raw_response = raw_response[:1_000]
        self.cause = cause

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.agent_name:
            parts.append(f"agent={self.agent_name!r}")
        if self.raw_response:
            parts.append(f"raw={self.raw_response!r}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent(ABC, Generic[ResponseT]):
    """Abstract base class for all MissionMind AI agents.

    Subclasses must implement:

    * ``system_prompt`` — the full system-role instruction sent to the model.
    * ``response_schema`` — the Pydantic ``BaseModel`` subclass the JSON
      response must conform to.
    * ``name`` — a short identifier used in log messages and error reports.

    Constructor parameters
    ----------------------
    llm_client:
        The LLM backend to use.  Defaults to a ``WatsonxClient`` built from
        environment variables.  Inject a mock in tests to avoid network calls.
    max_retries:
        Number of additional attempts on transient ``LLMClientError``.
        Does not apply to structural failures (bad JSON / schema mismatch).
    retry_delay:
        Initial back-off delay in seconds.  Doubles on each subsequent retry.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self._client: LLMClient = llm_client or WatsonxClient()
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this agent, e.g. ``"science"``."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Full system-role instruction text sent to the LLM."""

    @property
    @abstractmethod
    def response_schema(self) -> type[ResponseT]:
        """Pydantic model class the LLM response must conform to."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self, context: dict[str, Any]) -> ResponseT:
        """Execute the agent: call the LLM and return a validated response.

        Parameters
        ----------
        context:
            A dict of mission-relevant data (waypoints, rover state, etc.).
            It is serialised to JSON and sent as the user turn.

        Returns
        -------
        ResponseT
            A fully validated Pydantic model instance.

        Raises
        ------
        AgentResponseError
            If the LLM returns invalid JSON or output that doesn't match
            ``response_schema``.  This is raised immediately without retrying.
        LLMClientError
            If all retry attempts are exhausted due to network / server errors.
        """
        user_message = self._serialise_context(context)
        attempt = 0
        delay = self._retry_delay

        while True:
            try:
                llm_response = await self._client.complete(
                    system_prompt=self.system_prompt,
                    user_message=user_message,
                )
                break  # success — exit retry loop
            except LLMClientError as exc:
                attempt += 1
                if attempt > self._max_retries:
                    logger.error(
                        "[%s] LLM call failed after %d attempt(s): %s",
                        self.name,
                        attempt,
                        exc,
                    )
                    raise
                logger.warning(
                    "[%s] LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                    self.name,
                    attempt,
                    self._max_retries + 1,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
                delay *= 2.0

        return self._parse_and_validate(llm_response.content)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_context(context: dict[str, Any]) -> str:
        """Convert the context dict to a compact JSON string user message."""
        try:
            return json.dumps(context, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            # Fallback: stringify each value individually.
            safe = {k: str(v) for k, v in context.items()}
            logger.warning("Context serialisation fallback triggered: %s", exc)
            return json.dumps(safe)

    def _parse_and_validate(self, raw: str) -> ResponseT:
        """Parse raw LLM output as JSON and validate against response_schema.

        Parameters
        ----------
        raw:
            The raw string content from the LLM response.

        Returns
        -------
        ResponseT
            Validated Pydantic model instance.

        Raises
        ------
        AgentResponseError
            On JSON parse failure or Pydantic validation failure.
        """
        # Step 1 — JSON parse
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AgentResponseError(
                f"[{self.name}] Response is not valid JSON",
                agent_name=self.name,
                raw_response=raw,
                cause=exc,
            ) from exc

        # Step 2 — Pydantic validation
        try:
            return self.response_schema.model_validate(data)
        except ValidationError as exc:
            raise AgentResponseError(
                f"[{self.name}] Response does not match {self.response_schema.__name__}",
                agent_name=self.name,
                raw_response=raw,
                cause=exc,
            ) from exc
