"""
Resource Agent — rover power systems engineer / mission controller.

Responsibility
--------------
Evaluate the rover's current battery state, estimate the energy cost of
reaching each candidate waypoint, and recommend a feasible subset that fits
within the usable energy budget while preserving the mandatory safe-return
reserve.

The Resource Agent intentionally knows nothing about:
- scientific value of waypoints         → ScienceAgent's responsibility
- terrain hazard classification         → SafetyAgent + SafetyValidator
- final route selection                 → MissionCommander
- mission replanning                    → Replanner

Context keys expected by ``run()``
------------------------------------
battery_pct : float
    Current battery state as a fraction of full capacity (0.0 – 1.0).
battery_capacity_wh : float
    Total battery capacity in watt-hours.
candidate_waypoints : list[dict]
    Each entry is a serialised ``Waypoint`` dict (id, x, y,
    estimated_travel_time_minutes, estimated_energy_wh, label, …).
rover_speed_mps : float
    Nominal rover driving speed in metres per second.
power_consumption_w : float
    Rover power draw in watts during nominal driving.

The agent also injects the configured ``MIN_RETURN_BATTERY_PCT`` from
``config.py`` into the context so the LLM can reason about the reserve.

Returns
-------
ResourceBudget
    Validated Pydantic model containing available energy, available time,
    the recommended feasible waypoint IDs, per-waypoint energy estimates,
    and a reasoning narrative.
"""

from __future__ import annotations

import pathlib

from missionmind import config
from missionmind.agents.base_agent import BaseAgent
from missionmind.agents.client import LLMClient
from missionmind.schemas.outputs import ResourceBudget

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPT_PATH = (
    pathlib.Path(__file__).parent.parent / "prompts" / "resource_prompt.md"
)


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


_RESOURCE_SYSTEM_PROMPT: str = _load_prompt()


# ---------------------------------------------------------------------------
# ResourceAgent
# ---------------------------------------------------------------------------

class ResourceAgent(BaseAgent[ResourceBudget]):
    """Concrete agent that evaluates the rover's energy/time budget.

    Inherits all LLM call logic, retry handling, JSON parsing, and Pydantic
    validation from :class:`~missionmind.agents.base_agent.BaseAgent`.
    The only additions here are:

    * The power-systems-engineer system prompt.
    * The ``ResourceBudget`` response schema.
    * Automatic injection of ``MIN_RETURN_BATTERY_PCT`` into every context
      dict so the LLM always has the configured reserve threshold available.

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
        return "resource"

    @property
    def system_prompt(self) -> str:
        return _RESOURCE_SYSTEM_PROMPT

    @property
    def response_schema(self) -> type[ResourceBudget]:
        return ResourceBudget

    # ------------------------------------------------------------------
    # Context enrichment
    # ------------------------------------------------------------------

    async def run(self, context: dict) -> ResourceBudget:
        """Run the resource agent, injecting the configured return reserve.

        Adds ``min_return_battery_pct`` from ``config.py`` into the context
        dict before delegating to ``BaseAgent.run()``.  This means the LLM
        always sees the authoritative threshold and cannot be given a
        different value by the caller.

        Parameters
        ----------
        context:
            Must contain ``battery_pct``, ``battery_capacity_wh``,
            ``candidate_waypoints``, ``rover_speed_mps``,
            ``power_consumption_w``.  Any existing ``min_return_battery_pct``
            key is silently overwritten with the value from ``config.py``.

        Returns
        -------
        ResourceBudget
            Validated Pydantic model from the LLM response.
        """
        enriched = {
            **context,
            "min_return_battery_pct": config.MIN_RETURN_BATTERY_PCT,
        }
        return await super().run(enriched)
