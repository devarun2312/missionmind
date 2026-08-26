"""
Tests for MissionCommander — Sub-Task 7.

All four scenarios required by the approved plan are covered, plus
additional tests for important behaviours.

No real LLM/network calls are made in any test — all AI clients are mocked.

Test classes
------------
TestHappyPath                 — approved Test 1
TestRetrySucceeds             — approved Test 2
TestAllRetriesExhausted       — approved Test 3
TestParallelExecution         — approved Test 4
TestSafetyAgentOrdering       — safety agent runs after science+resource
TestConvertToMissionPlan      — _convert_to_mission_plan unit tests
TestPruneWaypoints            — _prune_waypoints unit tests
TestPlanningFailedError       — PlanningFailedError standalone tests
TestContextPropagation        — agents receive correct context keys
TestMalformedCommanderOutput  — AgentResponseError propagates immediately
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missionmind.agents.base_agent import AgentResponseError
from missionmind.agents.mission_commander import (
    MissionCommander,
    PlanningFailedError,
    _CommanderSynthesisAgent,
)
from missionmind.models.mission import MissionPlan, MissionStatus, Waypoint
from missionmind.safety.validator import SafetyValidator, ValidationResult
from missionmind.schemas.outputs import (
    MissionPlanOutput,
    PlannedWaypointEntry,
    ResourceBudget,
    RiskAssessment,
    RiskLevel,
    ScienceAnalysis,
    ScoredTarget,
    WaypointRisk,
)


# ============================================================
# Shared test data
# ============================================================

_BASE_DICT = {
    "id": "base-001",
    "x": 0.0, "y": 0.0,
    "terrain_risk": 0.0,
    "is_base": True,
    "label": "BASE",
    "estimated_travel_time_minutes": 0.0,
    "estimated_energy_wh": 0.0,
}

_WP_A = {
    "id": "wp-A",
    "x": 100.0, "y": 50.0,
    "terrain_risk": 0.2,
    "is_base": False,
    "label": "crater-A",
    "estimated_travel_time_minutes": 30.0,
    "estimated_energy_wh": 20.0,
}

_WP_B = {
    "id": "wp-B",
    "x": 200.0, "y": 80.0,
    "terrain_risk": 0.3,
    "is_base": False,
    "label": "outcrop-B",
    "estimated_travel_time_minutes": 40.0,
    "estimated_energy_wh": 30.0,
}

_WP_C_RISKY = {
    "id": "wp-C",
    "x": 300.0, "y": 10.0,
    "terrain_risk": 0.75,      # above default MAX_TERRAIN_RISK_SCORE=0.70
    "is_base": False,
    "label": "ridge-C",
    "estimated_travel_time_minutes": 50.0,
    "estimated_energy_wh": 50.0,
}

_CANDIDATES = [_WP_A, _WP_B, _BASE_DICT]

_ROVER_STATE = {
    "battery_pct": 0.90,
    "battery_capacity_wh": 500.0,
    "position_x": 0.0,
    "position_y": 0.0,
}

_CONTEXT = {
    "candidate_waypoints": _CANDIDATES,
    "rover_state": _ROVER_STATE,
    "weather_forecast": {"dust_storm_probability": 0.05},
    "comm_windows": [],
    "terrain_map": {},
}


# ============================================================
# Helper factories
# ============================================================

def _science_analysis(ids=("wp-A", "wp-B")) -> ScienceAnalysis:
    return ScienceAnalysis(
        scored_targets=[
            ScoredTarget(waypoint_id=i, scientific_value=0.8) for i in ids
        ],
        priority_order=list(ids),
        reasoning="good targets",
    )


def _resource_budget() -> ResourceBudget:
    return ResourceBudget(
        available_energy_wh=250.0,
        available_time_minutes=300.0,
        recommended_waypoints=["wp-A", "wp-B"],
        energy_per_waypoint={"wp-A": 20.0, "wp-B": 30.0},
        reasoning="fits budget",
    )


def _risk_assessment(exclusions: list[str] | None = None) -> RiskAssessment:
    return RiskAssessment(
        waypoint_risks=[
            WaypointRisk(waypoint_id="wp-A", risk_score=0.2),
            WaypointRisk(waypoint_id="wp-B", risk_score=0.3),
        ],
        overall_risk_level=RiskLevel.LOW,
        recommended_exclusions=exclusions or [],
        reasoning="low risk",
    )


def _plan_output(
    waypoints: tuple[str, ...] = ("wp-A", "wp-B", "base-001"),
    energy_wh: float = 50.0,
    time_min: float = 70.0,
) -> MissionPlanOutput:
    entries = [
        PlannedWaypointEntry(
            waypoint_id=wid,
            visit_order=i + 1,
            expected_science_value=0.0 if wid == "base-001" else 0.8,
            expected_energy_wh=0.0 if wid == "base-001" else 20.0,
        )
        for i, wid in enumerate(waypoints)
    ]
    return MissionPlanOutput(
        planned_waypoints=entries,
        total_estimated_energy_wh=energy_wh,
        total_estimated_time_minutes=time_min,
        confidence=0.9,
        reasoning="balanced",
    )


def _build_commander(
    *,
    plan_output: MissionPlanOutput | None = None,
    validator_results: list[ValidationResult] | None = None,
    max_attempts: int = 3,
    science_return: ScienceAnalysis | None = None,
    resource_return: ResourceBudget | None = None,
    risk_return: RiskAssessment | None = None,
):
    """Build a MissionCommander with all external dependencies mocked.

    Returns (commander, science_mock, resource_mock, safety_mock, validator_mock).
    max_attempts is the TOTAL number of generate-validate cycles allowed.
    """
    science_mock = MagicMock()
    science_mock.run = AsyncMock(
        return_value=science_return or _science_analysis()
    )

    resource_mock = MagicMock()
    resource_mock.run = AsyncMock(
        return_value=resource_return or _resource_budget()
    )

    safety_mock = MagicMock()
    safety_mock.run = AsyncMock(
        return_value=risk_return or _risk_assessment()
    )

    validator_mock = MagicMock(spec=SafetyValidator)
    if validator_results is not None:
        validator_mock.validate.side_effect = validator_results
    else:
        validator_mock.validate.return_value = ValidationResult(passed=True)

    llm_mock = MagicMock()

    commander = MissionCommander(
        science_agent=science_mock,
        resource_agent=resource_mock,
        safety_agent=safety_mock,
        validator=validator_mock,
        llm_client=llm_mock,
        max_attempts=max_attempts,
        retry_delay=0.0,
    )

    # Patch the synthesis agent's run() so no real LLM call occurs
    po = plan_output if plan_output is not None else _plan_output()
    commander._synthesis_agent.run = AsyncMock(return_value=po)

    return commander, science_mock, resource_mock, safety_mock, validator_mock


# ============================================================
# Approved Test 1 — Happy path
# ============================================================

class TestHappyPath:
    """All agents and validator mocked; validator approves the plan."""

    async def test_returns_mission_plan(self):
        commander, *_ = _build_commander()
        result = await commander.plan(_CONTEXT)
        assert isinstance(result, MissionPlan)

    async def test_status_is_active(self):
        commander, *_ = _build_commander()
        result = await commander.plan(_CONTEXT)
        assert result.status is MissionStatus.ACTIVE

    async def test_science_agent_called(self):
        commander, science_mock, *_ = _build_commander()
        await commander.plan(_CONTEXT)
        science_mock.run.assert_called_once()

    async def test_resource_agent_called(self):
        commander, _, resource_mock, *_ = _build_commander()
        await commander.plan(_CONTEXT)
        resource_mock.run.assert_called_once()

    async def test_safety_agent_called(self):
        commander, _, _, safety_mock, _ = _build_commander()
        await commander.plan(_CONTEXT)
        safety_mock.run.assert_called_once()

    async def test_validator_called(self):
        commander, *_, validator_mock = _build_commander()
        await commander.plan(_CONTEXT)
        validator_mock.validate.assert_called_once()

    async def test_synthesis_agent_called(self):
        commander, *_ = _build_commander()
        await commander.plan(_CONTEXT)
        commander._synthesis_agent.run.assert_called_once()

    async def test_validator_called_once_on_first_pass(self):
        commander, *_, validator_mock = _build_commander()
        await commander.plan(_CONTEXT)
        assert validator_mock.validate.call_count == 1

    async def test_plan_ends_with_base_waypoint(self):
        commander, *_ = _build_commander()
        result = await commander.plan(_CONTEXT)
        assert result.waypoints, "Plan should have waypoints"
        assert result.waypoints[-1].is_base is True

    async def test_plan_has_science_waypoints(self):
        commander, *_ = _build_commander()
        result = await commander.plan(_CONTEXT)
        assert len(result.science_waypoints()) >= 1

    async def test_reasoning_propagated_from_plan_output(self):
        po = _plan_output()
        commander, *_ = _build_commander(plan_output=po)
        result = await commander.plan(_CONTEXT)
        assert result.reasoning == po.reasoning

    async def test_confidence_propagated_from_plan_output(self):
        po = _plan_output()
        commander, *_ = _build_commander(plan_output=po)
        result = await commander.plan(_CONTEXT)
        assert result.confidence == po.confidence


# ============================================================
# Approved Test 2 — Validator rejects first plan, retry succeeds
# ============================================================

class TestRetrySucceeds:
    """First validation fails; second attempt is approved."""

    async def test_returns_active_plan_on_second_attempt(self):
        commander, *_ = _build_commander(
            validator_results=[
                ValidationResult(passed=False,
                                 violations=["ENERGY BUDGET EXCEEDED: too much"]),
                ValidationResult(passed=True),
            ],
        )
        result = await commander.plan(_CONTEXT)
        assert result.status is MissionStatus.ACTIVE

    async def test_rejected_plan_not_returned(self):
        """The commander must never return a plan that the validator rejected."""
        commander, *_, validator_mock = _build_commander(
            validator_results=[
                ValidationResult(passed=False,
                                 violations=["ENERGY BUDGET EXCEEDED: too much"]),
                ValidationResult(passed=True),
            ],
        )
        result = await commander.plan(_CONTEXT)
        # If a rejected plan had been returned its status would still be PENDING
        assert result.status is MissionStatus.ACTIVE

    async def test_validator_called_twice(self):
        commander, *_, validator_mock = _build_commander(
            validator_results=[
                ValidationResult(passed=False,
                                 violations=["ENERGY BUDGET EXCEEDED: too much"]),
                ValidationResult(passed=True),
            ],
        )
        await commander.plan(_CONTEXT)
        assert validator_mock.validate.call_count == 2

    async def test_science_agent_called_twice(self):
        """Each attempt re-runs all agents."""
        commander, science_mock, *_ = _build_commander(
            validator_results=[
                ValidationResult(passed=False,
                                 violations=["ENERGY BUDGET EXCEEDED: too much"]),
                ValidationResult(passed=True),
            ],
        )
        await commander.plan(_CONTEXT)
        assert science_mock.run.call_count == 2

    async def test_pruning_occurs_between_attempts(self):
        """Candidate list passed to agents on attempt 2 should be shorter."""
        seen_candidates: list[list] = []

        async def capture_science(ctx):
            seen_candidates.append(list(ctx.get("candidate_waypoints", [])))
            return _science_analysis()

        science_mock = MagicMock()
        science_mock.run = capture_science

        resource_mock = MagicMock()
        resource_mock.run = AsyncMock(return_value=_resource_budget())

        safety_mock = MagicMock()
        safety_mock.run = AsyncMock(return_value=_risk_assessment())

        validator_mock = MagicMock(spec=SafetyValidator)
        validator_mock.validate.side_effect = [
            ValidationResult(passed=False,
                             violations=["ENERGY BUDGET EXCEEDED: too much"]),
            ValidationResult(passed=True),
        ]

        commander = MissionCommander(
            science_agent=science_mock,
            resource_agent=resource_mock,
            safety_agent=safety_mock,
            validator=validator_mock,
            llm_client=MagicMock(),
            max_attempts=3,
            retry_delay=0.0,
        )
        commander._synthesis_agent.run = AsyncMock(return_value=_plan_output())

        # Include a risky candidate so pruning has something to remove
        context_with_extra = {
            **_CONTEXT,
            "candidate_waypoints": _CANDIDATES + [_WP_C_RISKY],
        }
        await commander.plan(context_with_extra)

        assert len(seen_candidates) == 2
        assert len(seen_candidates[1]) < len(seen_candidates[0]), (
            "Candidate list should be shorter after pruning"
        )


# ============================================================
# Approved Test 3 — All attempts exhausted → PlanningFailedError
# ============================================================

class TestAllRetriesExhausted:
    """Validator rejects every generated plan.

    max_attempts=3 → exactly 3 total generate-validate cycles, then failure.
    """

    async def test_raises_planning_failed_error(self):
        # max_attempts=3 → exactly 3 total attempts, all rejected
        commander, *_ = _build_commander(
            validator_results=[
                ValidationResult(passed=False,
                                 violations=["ENERGY BUDGET EXCEEDED: v1"]),
                ValidationResult(passed=False,
                                 violations=["ENERGY BUDGET EXCEEDED: v2"]),
                ValidationResult(passed=False,
                                 violations=["ENERGY BUDGET EXCEEDED: v3"]),
            ],
            max_attempts=3,
        )
        with pytest.raises(PlanningFailedError):
            await commander.plan(_CONTEXT)

    async def test_does_not_return_rejected_plan(self):
        commander, *_ = _build_commander(
            validator_results=[
                ValidationResult(passed=False, violations=["ENERGY BUDGET EXCEEDED: x"])
            ] * 3,
            max_attempts=3,
        )
        with pytest.raises(PlanningFailedError):
            await commander.plan(_CONTEXT)
        # If we reach here the error was raised — no plan was returned silently

    async def test_error_contains_last_violations(self):
        last_violation = "ENERGY BUDGET EXCEEDED: final"
        commander, *_ = _build_commander(
            validator_results=[
                ValidationResult(passed=False, violations=["ENERGY BUDGET EXCEEDED: early"]),
                ValidationResult(passed=False, violations=["ENERGY BUDGET EXCEEDED: middle"]),
                ValidationResult(passed=False, violations=[last_violation]),
            ],
            max_attempts=3,
        )
        with pytest.raises(PlanningFailedError) as exc_info:
            await commander.plan(_CONTEXT)
        assert last_violation in exc_info.value.violations

    async def test_error_attempts_equals_max_attempts(self):
        # max_attempts=3 → exactly 3 total attempts reported
        commander, *_ = _build_commander(
            validator_results=[
                ValidationResult(passed=False, violations=["ENERGY BUDGET EXCEEDED: x"])
            ] * 3,
            max_attempts=3,
        )
        with pytest.raises(PlanningFailedError) as exc_info:
            await commander.plan(_CONTEXT)
        assert exc_info.value.attempts == 3

    async def test_validator_called_exactly_max_attempts_times(self):
        commander, *_, validator_mock = _build_commander(
            validator_results=[
                ValidationResult(passed=False, violations=["ENERGY BUDGET EXCEEDED: x"])
            ] * 3,
            max_attempts=3,
        )
        with pytest.raises(PlanningFailedError):
            await commander.plan(_CONTEXT)
        assert validator_mock.validate.call_count == 3

    async def test_max_attempts_one_means_one_attempt_no_retry(self):
        # max_attempts=1 → one attempt only, then PlanningFailedError
        commander, *_, validator_mock = _build_commander(
            validator_results=[
                ValidationResult(passed=False, violations=["ENERGY BUDGET EXCEEDED: x"]),
            ],
            max_attempts=1,
        )
        with pytest.raises(PlanningFailedError) as exc_info:
            await commander.plan(_CONTEXT)
        assert exc_info.value.attempts == 1
        assert validator_mock.validate.call_count == 1


# ============================================================
# Approved Test 4 — Parallel execution of Science + Resource
# ============================================================

class TestParallelExecution:
    """Science and Resource agents must run concurrently via asyncio.gather."""

    async def test_asyncio_gather_is_called(self):
        """Spy on asyncio.gather to confirm it is used (not sequential awaits)."""
        with patch(
            "missionmind.agents.mission_commander.asyncio.gather",
            wraps=asyncio.gather,
        ) as gather_spy:
            commander, *_ = _build_commander()
            await commander.plan(_CONTEXT)
            assert gather_spy.called, "asyncio.gather must be called"

    async def test_science_and_resource_run_concurrently(self):
        """Use an asyncio Event to prove both agents are in flight simultaneously.

        The test replaces both agents with coroutines that:
        1. Record their start.
        2. Wait on a shared asyncio.Event.
        3. Record their completion after the event is set.

        The main coroutine sets the event *after* both agents have started.
        If execution were sequential, the first agent would never advance past
        the wait (because the event would not be set until after it completes),
        causing a deadlock.  Successful completion proves concurrency.
        """
        started: set[str] = set()
        completed: set[str] = set()
        both_started = asyncio.Event()

        async def science_side(_ctx):
            started.add("science")
            if "resource" in started:
                both_started.set()
            await both_started.wait()
            completed.add("science")
            return _science_analysis()

        async def resource_side(_ctx):
            started.add("resource")
            if "science" in started:
                both_started.set()
            await both_started.wait()
            completed.add("resource")
            return _resource_budget()

        science_mock = MagicMock()
        science_mock.run = science_side
        resource_mock = MagicMock()
        resource_mock.run = resource_side

        safety_mock = MagicMock()
        safety_mock.run = AsyncMock(return_value=_risk_assessment())

        validator_mock = MagicMock(spec=SafetyValidator)
        validator_mock.validate.return_value = ValidationResult(passed=True)

        commander = MissionCommander(
            science_agent=science_mock,
            resource_agent=resource_mock,
            safety_agent=safety_mock,
            validator=validator_mock,
            llm_client=MagicMock(),
            max_attempts=3,
            retry_delay=0.0,
        )
        commander._synthesis_agent.run = AsyncMock(return_value=_plan_output())

        await commander.plan(_CONTEXT)

        assert "science" in completed
        assert "resource" in completed

    async def test_safety_agent_runs_after_gather(self):
        """Safety agent must be called AFTER science and resource have completed."""
        call_order: list[str] = []

        async def science_side(_ctx):
            call_order.append("science")
            return _science_analysis()

        async def resource_side(_ctx):
            call_order.append("resource")
            return _resource_budget()

        async def safety_side(_ctx):
            call_order.append("safety")
            return _risk_assessment()

        science_mock = MagicMock()
        science_mock.run = science_side
        resource_mock = MagicMock()
        resource_mock.run = resource_side
        safety_mock = MagicMock()
        safety_mock.run = safety_side

        validator_mock = MagicMock(spec=SafetyValidator)
        validator_mock.validate.return_value = ValidationResult(passed=True)

        commander = MissionCommander(
            science_agent=science_mock,
            resource_agent=resource_mock,
            safety_agent=safety_mock,
            validator=validator_mock,
            llm_client=MagicMock(),
            max_attempts=3,
            retry_delay=0.0,
        )
        commander._synthesis_agent.run = AsyncMock(return_value=_plan_output())

        await commander.plan(_CONTEXT)

        # Both science and resource appear before safety in the call order
        safety_pos = call_order.index("safety")
        assert call_order.index("science") < safety_pos
        assert call_order.index("resource") < safety_pos


# ============================================================
# TestSafetyAgentOrdering — additional context content checks
# ============================================================

class TestSafetyAgentOrdering:
    async def test_safety_context_contains_science_analysis(self):
        captured: dict = {}

        async def safety_side(ctx):
            captured.update(ctx)
            return _risk_assessment()

        safety_mock = MagicMock()
        safety_mock.run = safety_side

        science_mock = MagicMock()
        science_mock.run = AsyncMock(return_value=_science_analysis())
        resource_mock = MagicMock()
        resource_mock.run = AsyncMock(return_value=_resource_budget())

        validator_mock = MagicMock(spec=SafetyValidator)
        validator_mock.validate.return_value = ValidationResult(passed=True)

        commander = MissionCommander(
            science_agent=science_mock,
            resource_agent=resource_mock,
            safety_agent=safety_mock,
            validator=validator_mock,
            llm_client=MagicMock(),
            max_attempts=3,
            retry_delay=0.0,
        )
        commander._synthesis_agent.run = AsyncMock(return_value=_plan_output())
        await commander.plan(_CONTEXT)

        assert "science_analysis" in captured

    async def test_safety_context_contains_resource_budget(self):
        captured: dict = {}

        async def safety_side(ctx):
            captured.update(ctx)
            return _risk_assessment()

        safety_mock = MagicMock()
        safety_mock.run = safety_side

        science_mock = MagicMock()
        science_mock.run = AsyncMock(return_value=_science_analysis())
        resource_mock = MagicMock()
        resource_mock.run = AsyncMock(return_value=_resource_budget())

        validator_mock = MagicMock(spec=SafetyValidator)
        validator_mock.validate.return_value = ValidationResult(passed=True)

        commander = MissionCommander(
            science_agent=science_mock,
            resource_agent=resource_mock,
            safety_agent=safety_mock,
            validator=validator_mock,
            llm_client=MagicMock(),
            max_attempts=3,
            retry_delay=0.0,
        )
        commander._synthesis_agent.run = AsyncMock(return_value=_plan_output())
        await commander.plan(_CONTEXT)

        assert "resource_budget" in captured

    async def test_synthesis_context_contains_all_three_analyses(self):
        """Commander synthesis receives science, resource, and risk results."""
        captured: dict = {}

        async def synthesis_side(ctx):
            captured.update(ctx)
            return _plan_output()

        commander, *_ = _build_commander()
        commander._synthesis_agent.run = synthesis_side

        await commander.plan(_CONTEXT)

        assert "science_analysis" in captured
        assert "resource_budget" in captured
        assert "risk_assessment" in captured

    async def test_validator_receives_mission_plan(self):
        """The validator must receive a MissionPlan, not raw output."""
        commander, *_, validator_mock = _build_commander()
        await commander.plan(_CONTEXT)
        plan_arg = validator_mock.validate.call_args[0][0]
        assert isinstance(plan_arg, MissionPlan)

    async def test_validator_receives_rover_state(self):
        commander, *_, validator_mock = _build_commander()
        await commander.plan(_CONTEXT)
        rover_state_arg = validator_mock.validate.call_args[0][1]
        assert isinstance(rover_state_arg, dict)
        assert "battery_pct" in rover_state_arg


# ============================================================
# TestConvertToMissionPlan
# ============================================================

class TestConvertToMissionPlan:
    def test_waypoints_in_visit_order(self):
        po = MissionPlanOutput(
            planned_waypoints=[
                PlannedWaypointEntry(waypoint_id="base-001", visit_order=3,
                                     expected_science_value=0.0, expected_energy_wh=0.0),
                PlannedWaypointEntry(waypoint_id="wp-A", visit_order=1,
                                     expected_science_value=0.8, expected_energy_wh=20.0),
                PlannedWaypointEntry(waypoint_id="wp-B", visit_order=2,
                                     expected_science_value=0.6, expected_energy_wh=30.0),
            ],
            total_estimated_energy_wh=50.0,
            total_estimated_time_minutes=70.0,
            confidence=0.8,
            reasoning="test",
        )
        plan = MissionCommander._convert_to_mission_plan(po, _CANDIDATES)
        assert plan.waypoints[0].id == "wp-A"
        assert plan.waypoints[1].id == "wp-B"
        assert plan.waypoints[2].id == "base-001"

    def test_unknown_waypoint_id_skipped(self):
        po = MissionPlanOutput(
            planned_waypoints=[
                PlannedWaypointEntry(waypoint_id="ghost-id", visit_order=1,
                                     expected_science_value=0.5, expected_energy_wh=10.0),
                PlannedWaypointEntry(waypoint_id="base-001", visit_order=2,
                                     expected_science_value=0.0, expected_energy_wh=0.0),
            ],
            total_estimated_energy_wh=10.0,
            total_estimated_time_minutes=30.0,
            confidence=0.5,
            reasoning="skip unknown",
        )
        plan = MissionCommander._convert_to_mission_plan(po, _CANDIDATES)
        ids = [w.id for w in plan.waypoints]
        assert "ghost-id" not in ids
        assert "base-001" in ids

    def test_total_energy_computed_from_waypoints(self):
        po = _plan_output(waypoints=("wp-A", "base-001"))
        plan = MissionCommander._convert_to_mission_plan(po, _CANDIDATES)
        expected = sum(w.estimated_energy_wh for w in plan.waypoints)
        assert plan.total_energy_wh == pytest.approx(expected)

    def test_base_waypoint_flag_preserved(self):
        po = _plan_output(waypoints=("wp-A", "base-001"))
        plan = MissionCommander._convert_to_mission_plan(po, _CANDIDATES)
        base_wps = [w for w in plan.waypoints if w.id == "base-001"]
        assert base_wps and base_wps[0].is_base is True

    def test_terrain_risk_comes_from_candidate_dict(self):
        po = _plan_output(waypoints=("wp-A", "base-001"))
        plan = MissionCommander._convert_to_mission_plan(po, _CANDIDATES)
        wp_a = next(w for w in plan.waypoints if w.id == "wp-A")
        assert wp_a.terrain_risk == pytest.approx(_WP_A["terrain_risk"])

    def test_science_value_comes_from_plan_output(self):
        po = _plan_output(waypoints=("wp-A", "base-001"))
        plan = MissionCommander._convert_to_mission_plan(po, _CANDIDATES)
        wp_a = next(w for w in plan.waypoints if w.id == "wp-A")
        assert wp_a.scientific_value == pytest.approx(0.8)

    def test_status_is_pending_before_validation(self):
        po = _plan_output()
        plan = MissionCommander._convert_to_mission_plan(po, _CANDIDATES)
        assert plan.status is MissionStatus.PENDING


# ============================================================
# TestPruneWaypoints
# ============================================================

class TestPruneWaypoints:
    def _base_risk(self) -> RiskAssessment:
        return _risk_assessment()

    def _base_budget(self) -> ResourceBudget:
        return _resource_budget()

    def test_removes_safety_exclusions(self):
        risk = _risk_assessment(exclusions=["wp-B"])
        result = MissionCommander._prune_waypoints(
            _CANDIDATES, [], risk, self._base_budget()
        )
        ids = [w["id"] for w in result]
        assert "wp-B" not in ids

    def test_base_always_kept(self):
        risk = _risk_assessment(exclusions=["wp-A", "wp-B"])
        result = MissionCommander._prune_waypoints(
            _CANDIDATES, [], risk, self._base_budget()
        )
        ids = [w["id"] for w in result]
        assert "base-001" in ids

    def test_energy_violation_removes_most_expensive(self):
        # wp-B costs 30 Wh (more than wp-A at 20 Wh)
        budget = ResourceBudget(
            available_energy_wh=5.0,
            available_time_minutes=300.0,
            recommended_waypoints=["wp-A", "wp-B"],
            energy_per_waypoint={"wp-A": 20.0, "wp-B": 30.0},
            reasoning="tight",
        )
        violations = ["ENERGY BUDGET EXCEEDED: too much"]
        result = MissionCommander._prune_waypoints(
            _CANDIDATES, violations, _risk_assessment(), budget
        )
        ids = [w["id"] for w in result]
        assert "wp-B" not in ids  # most expensive removed
        assert "wp-A" in ids       # cheaper kept

    def test_terrain_violation_removes_highest_risk(self):
        # wp-B has terrain_risk=0.3 > wp-A's 0.2
        violations = ["TERRAIN RISK EXCEEDED: wp-B above threshold"]
        result = MissionCommander._prune_waypoints(
            _CANDIDATES, violations, _risk_assessment(), self._base_budget()
        )
        ids = [w["id"] for w in result]
        assert "wp-B" not in ids

    def test_generic_violation_removes_last_nonbase(self):
        candidates = [_WP_A, _WP_B, _BASE_DICT]
        violations = ["MISSION DURATION EXCEEDED: too long"]
        result = MissionCommander._prune_waypoints(
            candidates, violations, _risk_assessment(), self._base_budget()
        )
        ids = [w["id"] for w in result]
        # _WP_B is the last non-base entry
        assert "wp-B" not in ids

    def test_no_nonbase_candidates_returns_unchanged(self):
        only_base = [_BASE_DICT]
        result = MissionCommander._prune_waypoints(
            only_base, ["ENERGY BUDGET EXCEEDED: x"],
            _risk_assessment(), self._base_budget()
        )
        assert result == only_base


# ============================================================
# TestPlanningFailedError — standalone class tests
# ============================================================

class TestPlanningFailedError:
    def test_is_runtime_error(self):
        err = PlanningFailedError("oops")
        assert isinstance(err, RuntimeError)

    def test_violations_attribute(self):
        vs = ["ENERGY BUDGET EXCEEDED: x", "TERRAIN RISK EXCEEDED: y"]
        err = PlanningFailedError("failed", violations=vs, attempts=3)
        assert err.violations == vs

    def test_attempts_attribute(self):
        err = PlanningFailedError("failed", violations=[], attempts=4)
        assert err.attempts == 4

    def test_str_includes_attempts(self):
        err = PlanningFailedError("failed", violations=[], attempts=2)
        assert "attempts=2" in str(err)

    def test_str_includes_violations_text(self):
        err = PlanningFailedError("failed",
                                  violations=["ENERGY BUDGET EXCEEDED: x"],
                                  attempts=1)
        assert "ENERGY BUDGET EXCEEDED: x" in str(err)

    def test_empty_violations_default(self):
        err = PlanningFailedError("failed")
        assert err.violations == []


# ============================================================
# TestContextPropagation — agents receive correct keys
# ============================================================

class TestContextPropagation:
    async def test_candidate_waypoints_passed_to_science(self):
        commander, science_mock, *_ = _build_commander()
        await commander.plan(_CONTEXT)
        ctx = science_mock.run.call_args[0][0]
        assert "candidate_waypoints" in ctx

    async def test_candidate_waypoints_passed_to_resource(self):
        commander, _, resource_mock, *_ = _build_commander()
        await commander.plan(_CONTEXT)
        ctx = resource_mock.run.call_args[0][0]
        assert "candidate_waypoints" in ctx


# ============================================================
# TestMalformedCommanderOutput — AgentResponseError propagates
# ============================================================

class TestMalformedCommanderOutput:
    async def test_agent_response_error_propagates_immediately(self):
        """If the commander synthesis LLM returns bad JSON, the error
        must propagate immediately — the commander must NOT silently retry."""
        commander, *_ = _build_commander()
        commander._synthesis_agent.run = AsyncMock(
            side_effect=AgentResponseError(
                "bad JSON from commander",
                agent_name="commander",
                raw_response="not json",
            )
        )
        with pytest.raises(AgentResponseError):
            await commander.plan(_CONTEXT)

    async def test_science_agent_response_error_propagates(self):
        science_mock = MagicMock()
        science_mock.run = AsyncMock(
            side_effect=AgentResponseError("bad science", agent_name="science")
        )
        resource_mock = MagicMock()
        resource_mock.run = AsyncMock(return_value=_resource_budget())
        safety_mock = MagicMock()
        safety_mock.run = AsyncMock(return_value=_risk_assessment())
        validator_mock = MagicMock(spec=SafetyValidator)
        validator_mock.validate.return_value = ValidationResult(passed=True)

        commander = MissionCommander(
            science_agent=science_mock,
            resource_agent=resource_mock,
            safety_agent=safety_mock,
            validator=validator_mock,
            llm_client=MagicMock(),
            max_attempts=3,
            retry_delay=0.0,
        )
        commander._synthesis_agent.run = AsyncMock(return_value=_plan_output())

        with pytest.raises(AgentResponseError):
            await commander.plan(_CONTEXT)
