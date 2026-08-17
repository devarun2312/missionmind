"""
Safety Agent — Mars mission safety officer (AI soft-constraint assessment).

Responsibility
--------------
Assess per-waypoint terrain and environmental risk, classify overall mission
risk level, and recommend exclusions for waypoints that appear too hazardous.

This is SOFT-CONSTRAINT AI reasoning only.  Hard safety rules (energy budget
limits, terrain risk ceilings, mandatory return waypoint, etc.) are enforced
by the deterministic ``SafetyValidator`` implemented in Sub-Task 6.  The
Safety Agent's recommendations influence the MissionCommander's plan but are
not the final safety gate.

The Safety Agent intentionally knows nothing about:
- scientific value of waypoints         → ScienceAgent's responsibility
- energy / time budgets                 → ResourceAgent's responsibility
- final route construction              → MissionCommander
- hard constraint enforcement           → SafetyValidator (Sub-Task 6)
- mission replanning                    → Replanner (Sub-Task 9)

Context keys expected by ``run()``
------------------------------------
candidate_waypoints : list[dict]
    Serialised ``Waypoint`` dicts (id, x, y, terrain_risk, label, …).
weather_forecast : dict
    Keys: dust_storm_probability (float), temperature_min_c (float),
    temperature_max_c (float), wind_speed_mps (float),
    forecast_hours (int).
comm_windows : list[dict]
    Each entry: {"start_utc": str, "duration_minutes": int}.
terrain_map : dict[str, dict]
    Keyed by waypoint ID.  Per-waypoint: slope_degrees (float),
    surface_type (str), surveyed (bool).

The agent also injects ``MAX_TERRAIN_RISK_SCORE`` from ``config.py`` into the
context so the LLM always uses the authoritative threshold.

Returns
-------
RiskAssessment
    Validated Pydantic model containing per-waypoint risks, an overall risk
    level, recommended exclusions, and a reasoning narrative.
"""

from __future__ import annotations

import pathlib

from missionmind import config
from missionmind.agents.base_agent import BaseAgent
from missionmind.agents.client import LLMClient
from missionmind.schemas.outputs import RiskAssessment

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_PATH = (
    pathlib.Path(__file__).parent.parent / "prompts" / "safety_prompt.md"
)


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


_SAFETY_SYSTEM_PROMPT: str = _load_prompt()


# ---------------------------------------------------------------------------
# SafetyAgent
# ---------------------------------------------------------------------------

class SafetyAgent(BaseAgent[RiskAssessment]):
    """Concrete agent that performs AI-based soft-constraint risk assessment.

    Inherits all LLM call logic, retry handling, JSON parsing, and Pydantic
    validation from :class:`~missionmind.agents.base_agent.BaseAgent`.
    The only additions here are:

    * The safety-officer system prompt.
    * The ``RiskAssessment`` response schema.
    * Automatic injection of ``MAX_TERRAIN_RISK_SCORE`` into every context
      dict so the LLM always reasons against the authoritative threshold.

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
        return "safety"

    @property
    def system_prompt(self) -> str:
        return _SAFETY_SYSTEM_PROMPT

    @property
    def response_schema(self) -> type[RiskAssessment]:
        return RiskAssessment

    # ------------------------------------------------------------------
    # Context enrichment
    # ------------------------------------------------------------------

    async def run(self, context: dict) -> RiskAssessment:
        """Run the safety agent, injecting the configured terrain risk threshold.

        Adds ``max_terrain_risk_score`` from ``config.py`` into the context
        before delegating to ``BaseAgent.run()``.  This ensures the LLM
        always flags waypoints against the same threshold used by the
        deterministic validator, and that callers cannot accidentally omit
        or override this critical value.

        Parameters
        ----------
        context:
            Must contain ``candidate_waypoints``, ``weather_forecast``,
            ``comm_windows``, ``terrain_map``.  Any existing
            ``max_terrain_risk_score`` key is silently overwritten with the
            value from ``config.py``.

        Returns
        -------
        RiskAssessment
            Validated Pydantic model from the LLM response.
        """
        enriched = {
            **context,
            "max_terrain_risk_score": config.MAX_TERRAIN_RISK_SCORE,
        }
        return await super().run(enriched)
