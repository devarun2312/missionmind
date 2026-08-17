"""
Science Agent — planetary geologist / astrobiologist.

Responsibility
--------------
Analyse a set of candidate rover waypoints and score each one by scientific
value (0.0 – 1.0).  The agent considers geology, mineralogy, water/ice
evidence, atmospheric relevance, biosignature potential, and alignment with
the stated mission objectives.

The Science Agent intentionally knows nothing about:
- battery / energy budgets      → ResourceAgent's responsibility
- terrain safety rules          → SafetyAgent + SafetyValidator
- final route selection         → MissionCommander
- whether the rover can reach a waypoint

Context keys expected by ``run()``
------------------------------------
candidate_waypoints : list[dict]
    Each entry is a serialised ``Waypoint`` dict (id, x, y, scientific_value,
    terrain_risk, label, …).
rover_position : dict
    ``{"x": float, "y": float}`` — current rover location.
mission_objectives : list[str]
    Plain-text mission goals, e.g. ["search for biosignatures",
    "characterise subsurface water-ice"].

Returns
-------
ScienceAnalysis
    Validated Pydantic model containing per-waypoint scores, a priority
    ordering, and an overall reasoning narrative.
"""

from __future__ import annotations

import pathlib

from missionmind.agents.base_agent import BaseAgent
from missionmind.agents.client import LLMClient
from missionmind.schemas.outputs import ScienceAnalysis

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "science_prompt.md"


def _load_prompt() -> str:
    """Read the science system prompt from disk once at import time."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


# Cache the prompt text so repeated instantiations don't re-read the file.
_SCIENCE_SYSTEM_PROMPT: str = _load_prompt()


# ---------------------------------------------------------------------------
# ScienceAgent
# ---------------------------------------------------------------------------

class ScienceAgent(BaseAgent[ScienceAnalysis]):
    """Concrete agent that scores candidate waypoints by scientific value.

    Inherits all LLM call logic, retry handling, JSON parsing, and Pydantic
    validation from :class:`~missionmind.agents.base_agent.BaseAgent`.
    The only additions here are:

    * The planetary-science system prompt.
    * The ``ScienceAnalysis`` response schema.
    * A convenience ``name`` for logging / error messages.

    Parameters
    ----------
    llm_client:
        Injected LLM backend.  Defaults to ``WatsonxClient()`` (reads IBM
        credentials from environment variables).  Pass a mock in tests.
    max_retries:
        Forwarded to ``BaseAgent``.
    retry_delay:
        Forwarded to ``BaseAgent``.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    # ------------------------------------------------------------------
    # BaseAgent abstract properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "science"

    @property
    def system_prompt(self) -> str:
        return _SCIENCE_SYSTEM_PROMPT

    @property
    def response_schema(self) -> type[ScienceAnalysis]:
        return ScienceAnalysis
