"""
Mission Commander — orchestration layer for MissionMind.

Responsibility
--------------
The Mission Commander drives the three specialist agents in the correct order,
synthesises their outputs into a structured ``MissionPlanOutput`` via a direct
LLM call, converts that to a ``MissionPlan`` domain object, and submits it to
the deterministic ``SafetyValidator``.

If the validator rejects the plan the commander automatically prunes the
candidate waypoint list (removing the highest-risk / most energy-expensive
waypoints) and retries.  After exhausting ``config.MAX_PLANNING_RETRIES``
total attempts it raises ``PlanningFailedError``.

Architecture note — no new LLM client
--------------------------------------
The commander makes its own LLM synthesis call but does NOT introduce a new
HTTP stack.  It re-uses ``BaseAgent`` exactly as all other agents do: subclass,
supply a ``system_prompt`` and ``response_schema``, and let ``BaseAgent.run()``
handle JSON parsing, Pydantic validation, and transient-failure retries.

Orchestration flow
------------------
::

    context dict
        │
        ├─ [parallel]  ScienceAgent.run()    → ScienceAnalysis
        ├─ [parallel]  ResourceAgent.run()   → ResourceBudget
        │         (asyncio.gather)
        └─ SafetyAgent.run()                 → RiskAssessment
              │
              └─ _CommanderSynthesisAgent.run()  → MissionPlanOutput
                    │       (BaseAgent subclass — reuses existing LLM stack)
                    └─ SafetyValidator.validate()
                          ├─ PASS  → return MissionPlan (status=ACTIVE)
                          └─ FAIL  → prune candidates, retry
                                         └─ exhausted → PlanningFailedError
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import Any

from missionmind import config
from missionmind.agents.base_agent import AgentResponseError, BaseAgent
from missionmind.agents.client import LLMClient, WatsonxClient
from missionmind.agents.resource_agent import ResourceAgent
from missionmind.agents.safety_agent import SafetyAgent
from missionmind.agents.science_agent import ScienceAgent
from missionmind.models.mission import MissionPlan, MissionStatus, Waypoint
from missionmind.safety.validator import SafetyValidator, ValidationResult
from missionmind.schemas.outputs import (
    MissionPlanOutput,
    ResourceBudget,
    RiskAssessment,
    ScienceAnalysis,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt loading — loaded once at module import
# ---------------------------------------------------------------------------

_PROMPT_PATH = pathlib.Path(__file__).parent.parent / "prompts" / "commander_prompt.md"
_COMMANDER_SYSTEM_PROMPT: str = _PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# PlanningFailedError
# ---------------------------------------------------------------------------

class PlanningFailedError(RuntimeError):
    """Raised when the MissionCommander cannot produce a validated plan.

    Raised after all ``config.MAX_PLANNING_RETRIES + 1`` attempts have been
    made and every generated plan was rejected by the ``SafetyValidator``.

    Attributes
    ----------
    violations:
        Violation strings from the *last* rejected plan's ``ValidationResult``.
        Callers can inspect these to understand why planning failed.
    attempts:
        Total number of planning attempts that were made.
    """

    def __init__(
        self,
        message: str,
        *,
        violations: list[str] | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.violations: list[str] = violations if violations is not None else []
        self.attempts: int = attempts

    def __str__(self) -> str:
        base = super().__str__()
        suffix = f"attempts={self.attempts}"
        if self.violations:
            suffix += " | violations: " + "; ".join(self.violations)
        return f"{base} | {suffix}"


# ---------------------------------------------------------------------------
# _CommanderSynthesisAgent — private BaseAgent subclass for the synthesis call
# ---------------------------------------------------------------------------

class _CommanderSynthesisAgent(BaseAgent[MissionPlanOutput]):
    """Private agent that issues the commander's synthesis LLM call.

    Re-uses ``BaseAgent`` so that JSON parsing, Pydantic validation, and
    transient LLM failure retries are handled by the existing shared stack.
    This is intentionally private; external code uses ``MissionCommander``.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    @property
    def name(self) -> str:
        return "commander"

    @property
    def system_prompt(self) -> str:
        return _COMMANDER_SYSTEM_PROMPT

    @property
    def response_schema(self) -> type[MissionPlanOutput]:
        return MissionPlanOutput


# ---------------------------------------------------------------------------
# MissionCommander
# ---------------------------------------------------------------------------

class MissionCommander:
    """Orchestrates specialist agents to produce a validated ``MissionPlan``.

    Constructor
    -----------
    science_agent : ScienceAgent
    resource_agent : ResourceAgent
    safety_agent : SafetyAgent
    validator : SafetyValidator
    llm_client : LLMClient | None
        Client used for the commander's own synthesis call.  Defaults to a
        ``WatsonxClient()`` built from environment variables.  Inject a mock
        in tests to prevent real network calls.
    max_attempts : int | None
        Maximum TOTAL planning attempts (including the first).  Default: 3
        (from ``config.MAX_PLANNING_RETRIES``).  A value of 1 means one
        attempt with no retries; 3 means up to three full
        generate-validate cycles before raising ``PlanningFailedError``.
    retry_delay : float
        Initial back-off in seconds passed through to ``BaseAgent``.
    """

    def __init__(
        self,
        *,
        science_agent: ScienceAgent,
        resource_agent: ResourceAgent,
        safety_agent: SafetyAgent,
        validator: SafetyValidator,
        llm_client: LLMClient | None = None,
        max_attempts: int | None = None,
        retry_delay: float = 1.0,
    ) -> None:
        self._science_agent = science_agent
        self._resource_agent = resource_agent
        self._safety_agent = safety_agent
        self._validator = validator
        self._llm_client: LLMClient = llm_client or WatsonxClient()
        # _max_attempts is the TOTAL number of generate-validate cycles allowed.
        # config.MAX_PLANNING_RETRIES (default 3) is used directly as the total.
        self._max_attempts: int = (
            max_attempts if max_attempts is not None else config.MAX_PLANNING_RETRIES
        )
        # The synthesis agent is constructed once and reused across retries.
        self._synthesis_agent = _CommanderSynthesisAgent(
            llm_client=self._llm_client,
            max_retries=2,
            retry_delay=retry_delay,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def plan(self, context: dict[str, Any]) -> MissionPlan:
        """Produce a safety-validated ``MissionPlan`` from the given context.

        Steps
        -----
        1. Run ``ScienceAgent`` and ``ResourceAgent`` in parallel.
        2. Run ``SafetyAgent`` with the results from step 1.
        3. Run the commander synthesis LLM call.
        4. Convert ``MissionPlanOutput`` → ``MissionPlan``.
        5. Validate with ``SafetyValidator``.
        6. On failure: prune candidates and retry (up to ``max_attempts - 1`` times).
        7. On exhaustion: raise ``PlanningFailedError``.

        Parameters
        ----------
        context:
            Must contain at least ``candidate_waypoints`` (list of waypoint
            dicts) and ``rover_state`` (dict with ``battery_pct`` and
            ``battery_capacity_wh``).  All other keys are forwarded to agents.

        Returns
        -------
        MissionPlan
            A validated plan with ``status=ACTIVE``.

        Raises
        ------
        PlanningFailedError
            If all attempts produce plans rejected by the validator.
        AgentResponseError
            If any agent returns an unparseable LLM response (not retried at
            the commander level — structural failures propagate immediately).
        """
        # Work with a mutable copy of the candidate list so retries can prune
        candidate_waypoints: list[dict] = list(
            context.get("candidate_waypoints", [])
        )
        rover_state: dict = context.get("rover_state", {})

        last_violations: list[str] = []

        for attempt in range(1, self._max_attempts + 1):
            logger.info(
                "[commander] Attempt %d/%d — %d candidate waypoint(s).",
                attempt,
                self._max_attempts,
                len(candidate_waypoints),
            )

            # Build the context for this attempt with the current candidate list
            attempt_context = {**context, "candidate_waypoints": candidate_waypoints}

            # ── Step 1: Science + Resource in parallel ─────────────────────
            science_result, resource_result = await asyncio.gather(
                self._science_agent.run(attempt_context),
                self._resource_agent.run(attempt_context),
            )

            # ── Step 2: Safety (needs science + resource results) ──────────
            safety_context = {
                **attempt_context,
                "science_analysis": science_result.model_dump(),
                "resource_budget": resource_result.model_dump(),
            }
            risk_result = await self._safety_agent.run(safety_context)

            # ── Step 3: Commander synthesis LLM call ───────────────────────
            synthesis_context = {
                **attempt_context,
                "science_analysis": science_result.model_dump(),
                "resource_budget": resource_result.model_dump(),
                "risk_assessment": risk_result.model_dump(),
            }
            plan_output: MissionPlanOutput = await self._synthesis_agent.run(
                synthesis_context
            )

            # ── Step 4: Convert to MissionPlan ─────────────────────────────
            mission_plan = self._convert_to_mission_plan(
                plan_output, candidate_waypoints
            )

            # ── Step 5: Deterministic safety validation ────────────────────
            validation: ValidationResult = self._validator.validate(
                mission_plan, rover_state
            )

            if validation.passed:
                logger.info(
                    "[commander] Plan %s validated on attempt %d.",
                    mission_plan.plan_id,
                    attempt,
                )
                mission_plan.status = MissionStatus.ACTIVE
                return mission_plan

            # ── Step 6: Prune and prepare for retry ────────────────────────
            last_violations = validation.violations
            logger.warning(
                "[commander] Attempt %d rejected — %d violation(s): %s",
                attempt,
                len(last_violations),
                "; ".join(last_violations),
            )

            if attempt < self._max_attempts:
                candidate_waypoints = self._prune_waypoints(
                    candidate_waypoints, last_violations, risk_result, resource_result
                )

        # ── Step 7: All attempts exhausted ────────────────────────────────
        raise PlanningFailedError(
            f"Mission planning failed after {self._max_attempts} attempt(s).",
            violations=last_violations,
            attempts=self._max_attempts,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_mission_plan(
        plan_output: MissionPlanOutput,
        candidate_waypoints: list[dict],
    ) -> MissionPlan:
        """Convert a validated ``MissionPlanOutput`` to a ``MissionPlan``.

        Looks up each planned waypoint ID in ``candidate_waypoints`` to
        reconstruct full ``Waypoint`` objects with terrain / energy data.
        Entries whose IDs are not found in candidates are skipped with a
        warning (defensive — should not normally happen with a well-behaved LLM).

        The returned plan has ``status=PENDING``; the caller sets it to
        ``ACTIVE`` once the ``SafetyValidator`` approves it.
        """
        # Build lookup: id → raw dict
        lookup: dict[str, dict] = {
            str(wp.get("id", "")): wp for wp in candidate_waypoints
        }

        # Sort by visit_order to guarantee sequence regardless of LLM output order
        ordered = sorted(
            plan_output.planned_waypoints, key=lambda e: e.visit_order
        )

        waypoints: list[Waypoint] = []
        for entry in ordered:
            raw = lookup.get(entry.waypoint_id)
            if raw is None:
                logger.warning(
                    "[commander] Waypoint '%s' from plan not in candidates; skipping.",
                    entry.waypoint_id,
                )
                continue
            waypoints.append(
                Waypoint(
                    id=entry.waypoint_id,
                    x=float(raw.get("x", 0.0)),
                    y=float(raw.get("y", 0.0)),
                    scientific_value=float(entry.expected_science_value),
                    terrain_risk=float(raw.get("terrain_risk", 0.0)),
                    estimated_travel_time_minutes=float(
                        raw.get("estimated_travel_time_minutes", 0.0)
                    ),
                    estimated_energy_wh=float(entry.expected_energy_wh),
                    is_base=bool(raw.get("is_base", False)),
                    label=str(raw.get("label", "")),
                )
            )

        total_energy = sum(w.estimated_energy_wh for w in waypoints)
        total_time = sum(w.estimated_travel_time_minutes for w in waypoints)

        return MissionPlan(
            waypoints=waypoints,
            total_energy_wh=total_energy,
            total_time_minutes=total_time,
            status=MissionStatus.PENDING,
            reasoning=plan_output.reasoning,
            confidence=plan_output.confidence,
        )

    @staticmethod
    def _prune_waypoints(
        candidate_waypoints: list[dict],
        violations: list[str],
        risk_result: RiskAssessment,
        resource_result: ResourceBudget,
    ) -> list[dict]:
        """Remove the most problematic non-base waypoint from the candidate list.

        Strategy (in priority order):
        1. Always keep every waypoint with ``is_base=True``.
        2. Remove waypoints that the Safety Agent explicitly recommended
           excluding (``risk_result.recommended_exclusions``).
        3. If an ENERGY BUDGET violation is present, also remove the single
           non-base waypoint with the highest energy cost according to
           ``resource_result.energy_per_waypoint`` (falling back to the
           waypoint's own ``estimated_energy_wh`` field).
        4. If a TERRAIN RISK violation is present, also remove the non-base
           waypoint with the highest ``terrain_risk`` value.
        5. If none of 3/4 apply but violations still exist (e.g. duration),
           remove the *last* non-base waypoint (simplest safe fallback).

        Returns the pruned list.  Base waypoints are always preserved so that
        subsequent validation can produce the canonical ``MISSING RETURN TO
        BASE`` violation rather than silently hanging.
        """
        exclusions: set[str] = set(risk_result.recommended_exclusions)
        violation_text = " ".join(violations).upper()

        base_wps = [w for w in candidate_waypoints if w.get("is_base", False)]
        non_base = [w for w in candidate_waypoints if not w.get("is_base", False)]

        if not non_base:
            return candidate_waypoints  # nothing left to prune

        to_remove: set[str] = set()

        # Always apply safety exclusions first
        to_remove.update(
            str(w.get("id", "")) for w in non_base
            if str(w.get("id", "")) in exclusions
        )

        # Energy violation → remove most expensive remaining non-base waypoint
        if "ENERGY BUDGET EXCEEDED" in violation_text:
            eligible = [
                w for w in non_base
                if str(w.get("id", "")) not in to_remove
            ]
            if eligible:
                def _energy_cost(w: dict) -> float:
                    wid = str(w.get("id", ""))
                    return resource_result.energy_per_waypoint.get(
                        wid, float(w.get("estimated_energy_wh", 0.0))
                    )
                worst = max(eligible, key=_energy_cost)
                to_remove.add(str(worst.get("id", "")))

        # Terrain risk violation → remove highest-risk remaining non-base waypoint
        if "TERRAIN RISK EXCEEDED" in violation_text:
            eligible = [
                w for w in non_base
                if str(w.get("id", "")) not in to_remove
            ]
            if eligible:
                worst = max(
                    eligible, key=lambda w: float(w.get("terrain_risk", 0.0))
                )
                to_remove.add(str(worst.get("id", "")))

        # Generic fallback — drop the last non-base waypoint
        if not to_remove:
            to_remove.add(str(non_base[-1].get("id", "")))

        pruned_non_base = [
            w for w in non_base if str(w.get("id", "")) not in to_remove
        ]
        return pruned_non_base + base_wps
