"""
Deterministic Safety Validator — the hard safety gate for MissionMind.

Architecture
------------
This module is intentionally LLM-free.  It contains zero AI calls, zero
network calls, and zero randomness.  Given the same ``MissionPlan``,
``rover_state``, and threshold values it will ALWAYS produce the same result.

Why no LLM here?
    AI models are probabilistic.  A safety gate that might say "OK" on one
    call and "fail" on the next is not a safety gate at all.  Hard rules must
    be enforced by deterministic code that can be reasoned about, audited, and
    unit-tested exhaustively.

Separation of concerns
    ┌─────────────────────────────┐   ┌──────────────────────────────────────┐
    │  SafetyAgent (Sub-Task 5)   │   │  SafetyValidator (this module)       │
    │  AI soft-constraint analysis│   │  Pure-Python hard-constraint checker  │
    │  Probabilistic, advisory    │   │  Deterministic, mandatory             │
    │  Says "I'd recommend …"     │   │  Says "REJECTED — reason: …"         │
    └─────────────────────────────┘   └──────────────────────────────────────┘

Hard rules enforced (in order)
-------------------------------
1. ENERGY BUDGET
   ``plan.total_energy_wh`` must not exceed the usable energy available:
   ``usable_wh = battery_pct × battery_capacity_wh × (1 − MIN_RETURN_BATTERY_PCT)``

   ``battery_pct`` is a fraction in [0, 1] (e.g. 0.80 = 80 %).
   ``battery_capacity_wh`` is the total capacity in watt-hours.
   The reserve fraction ``MIN_RETURN_BATTERY_PCT`` is always subtracted first.

2. TERRAIN RISK
   No single ``Waypoint.terrain_risk`` may exceed ``MAX_TERRAIN_RISK_SCORE``.

3. MISSION DURATION
   ``plan.total_time_minutes`` must not exceed
   ``MAX_MISSION_DURATION_HOURS × 60``.

4. RETURN TO BASE
   The final waypoint in ``plan.waypoints`` must have ``is_base=True``.

5. SCIENCE OBJECTIVE
   The plan must contain at least one non-base waypoint
   (``plan.science_waypoints()`` must be non-empty).

rover_state keys
-----------------
battery_pct : float
    Current battery charge as a fraction of full capacity (0.0 – 1.0).
battery_capacity_wh : float
    Total battery capacity in watt-hours.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from missionmind import config
from missionmind.models.mission import MissionPlan, Waypoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """The outcome of a ``SafetyValidator.validate()`` call.

    Attributes
    ----------
    passed:
        ``True`` if the plan satisfies all hard constraints, ``False``
        otherwise.
    violations:
        Human-readable descriptions of every constraint that was violated.
        Empty when ``passed`` is ``True``.  May contain multiple entries
        when several rules fail simultaneously.
    """

    passed: bool
    violations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


# ---------------------------------------------------------------------------
# SafetyValidator
# ---------------------------------------------------------------------------

class SafetyValidator:
    """Pure-Python deterministic safety gate for MissionMind.

    Checks a ``MissionPlan`` against five hard constraints before it is
    handed to the simulation or executed on the rover.  Any plan that fails
    one or more rules is rejected with a ``ValidationResult(passed=False, …)``.

    All threshold values default to the constants defined in ``config.py`` but
    can be overridden via constructor arguments — this makes unit tests
    independent of environment variables.

    Parameters
    ----------
    min_return_battery_pct:
        Minimum battery fraction that must remain after the mission for the
        rover to return to base safely.  Default: ``config.MIN_RETURN_BATTERY_PCT``.
    max_terrain_risk_score:
        Maximum allowed ``terrain_risk`` for any single waypoint.
        Default: ``config.MAX_TERRAIN_RISK_SCORE``.
    max_mission_duration_hours:
        Maximum allowed total mission duration in hours.
        Default: ``config.MAX_MISSION_DURATION_HOURS``.
    """

    def __init__(
        self,
        *,
        min_return_battery_pct: float | None = None,
        max_terrain_risk_score: float | None = None,
        max_mission_duration_hours: float | None = None,
    ) -> None:
        self._min_return_battery_pct: float = (
            min_return_battery_pct
            if min_return_battery_pct is not None
            else config.MIN_RETURN_BATTERY_PCT
        )
        self._max_terrain_risk_score: float = (
            max_terrain_risk_score
            if max_terrain_risk_score is not None
            else config.MAX_TERRAIN_RISK_SCORE
        )
        self._max_mission_duration_hours: float = (
            max_mission_duration_hours
            if max_mission_duration_hours is not None
            else config.MAX_MISSION_DURATION_HOURS
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(self, plan: MissionPlan, rover_state: dict) -> ValidationResult:
        """Validate a ``MissionPlan`` against all hard safety constraints.

        Runs every rule regardless of intermediate failures, so all violations
        are reported in a single pass.

        Parameters
        ----------
        plan:
            The ``MissionPlan`` to validate.
        rover_state:
            Dict with at least:
            - ``battery_pct`` (float, 0.0–1.0)
            - ``battery_capacity_wh`` (float, Wh)

        Returns
        -------
        ValidationResult
            ``passed=True`` and empty violations if all rules pass.
            ``passed=False`` and one or more violation strings if any rule fails.
        """
        violations: list[str] = []

        violations.extend(self._check_energy_budget(plan, rover_state))
        violations.extend(self._check_terrain_risk(plan))
        violations.extend(self._check_mission_duration(plan))
        violations.extend(self._check_return_to_base(plan))
        violations.extend(self._check_science_objective(plan))

        result = ValidationResult(
            passed=len(violations) == 0,
            violations=violations,
        )

        if result.passed:
            logger.info("Plan %s passed all safety checks.", plan.plan_id)
        else:
            logger.warning(
                "Plan %s FAILED safety validation: %d violation(s): %s",
                plan.plan_id,
                len(violations),
                "; ".join(violations),
            )

        return result

    # ------------------------------------------------------------------
    # Private rule checkers — each returns a (possibly empty) list of
    # violation strings.  Returning an empty list means the rule passed.
    # ------------------------------------------------------------------

    def _check_energy_budget(
        self, plan: MissionPlan, rover_state: dict
    ) -> list[str]:
        """Rule 1 — total plan energy must not exceed the usable budget.

        Usable energy = battery_pct × battery_capacity_wh × (1 − MIN_RETURN_BATTERY_PCT)

        ``battery_pct`` is a fraction in [0, 1] as established by the Resource
        Agent contract (Sub-Task 4).  The plan's ``total_energy_wh`` field is
        compared against this budget.
        """
        battery_pct: float = float(rover_state.get("battery_pct", 0.0))
        battery_capacity_wh: float = float(
            rover_state.get("battery_capacity_wh", 0.0)
        )

        current_charge_wh = battery_pct * battery_capacity_wh
        return_reserve_wh = self._min_return_battery_pct * battery_capacity_wh
        usable_energy_wh = max(0.0, current_charge_wh - return_reserve_wh)

        if plan.total_energy_wh > usable_energy_wh:
            return [
                f"ENERGY BUDGET EXCEEDED: plan requires {plan.total_energy_wh:.1f} Wh "
                f"but only {usable_energy_wh:.1f} Wh is available after reserving "
                f"{return_reserve_wh:.1f} Wh ({self._min_return_battery_pct * 100:.0f}%) "
                f"for return (current charge: {current_charge_wh:.1f} Wh)."
            ]
        return []

    def _check_terrain_risk(self, plan: MissionPlan) -> list[str]:
        """Rule 2 — no waypoint may exceed MAX_TERRAIN_RISK_SCORE."""
        violations: list[str] = []
        for wp in plan.waypoints:
            if wp.terrain_risk > self._max_terrain_risk_score:
                violations.append(
                    f"TERRAIN RISK EXCEEDED: waypoint '{wp.id}' "
                    f"(label='{wp.label}') has terrain_risk={wp.terrain_risk:.3f}, "
                    f"which exceeds the maximum allowed "
                    f"{self._max_terrain_risk_score:.3f}."
                )
        return violations

    def _check_mission_duration(self, plan: MissionPlan) -> list[str]:
        """Rule 3 — total plan time must not exceed MAX_MISSION_DURATION_HOURS × 60."""
        max_minutes = self._max_mission_duration_hours * 60.0
        if plan.total_time_minutes > max_minutes:
            return [
                f"MISSION DURATION EXCEEDED: plan takes "
                f"{plan.total_time_minutes:.1f} minutes "
                f"but the maximum allowed is {max_minutes:.1f} minutes "
                f"({self._max_mission_duration_hours:.1f} hours)."
            ]
        return []

    def _check_return_to_base(self, plan: MissionPlan) -> list[str]:
        """Rule 4 — the final waypoint must be the base station."""
        if not plan.has_return_waypoint():
            if not plan.waypoints:
                return [
                    "MISSING RETURN TO BASE: the mission plan contains no waypoints."
                ]
            last = plan.waypoints[-1]
            return [
                f"MISSING RETURN TO BASE: the final waypoint '{last.id}' "
                f"(label='{last.label}') does not have is_base=True. "
                f"Every plan must end with a return-to-base waypoint."
            ]
        return []

    def _check_science_objective(self, plan: MissionPlan) -> list[str]:
        """Rule 5 — the plan must contain at least one science waypoint."""
        if len(plan.science_waypoints()) == 0:
            return [
                "NO SCIENCE WAYPOINTS: the mission plan contains no science "
                "objectives (all waypoints are base waypoints or the plan is empty). "
                "At least one non-base waypoint is required."
            ]
        return []
