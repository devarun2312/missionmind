"""
Tests for Sub-Task 6: Deterministic Safety Validator.

CRITICAL: Every test in this file must be:
  - Completely offline — zero network calls, zero AI/LLM calls.
  - Deterministic — same inputs always produce the same result.
  - Fast — pure Python only, no async, no mocking needed.

Covers:
- ValidationResult dataclass attributes and bool behaviour.
- SafetyValidator constructor uses config defaults.
- SafetyValidator constructor thresholds are injectable (override for tests).
- validate() returns ValidationResult(passed=True, violations=[]) for a valid plan.
- Rule 1: energy budget — plan.total_energy_wh > usable budget → violation.
- Rule 1: energy at exactly the budget limit → passes.
- Rule 1: rover_state keys interact correctly with MIN_RETURN_BATTERY_PCT.
- Rule 2: terrain_risk > MAX_TERRAIN_RISK_SCORE → violation.
- Rule 2: terrain_risk exactly at threshold → passes.
- Rule 2: multiple high-risk waypoints produce multiple violations.
- Rule 3: total_time_minutes > MAX_MISSION_DURATION_HOURS × 60 → violation.
- Rule 3: time exactly at limit → passes.
- Rule 4: plan not ending with is_base=True → violation.
- Rule 4: empty plan → violation.
- Rule 5: plan with no science waypoints → violation.
- Rule 5: plan with only a base waypoint → violation.
- Multiple rules failing simultaneously → all violations accumulated.
- validate() never stops after the first violation.
- SafetyValidator importable from missionmind.safety.
- No LLM / BaseAgent / WatsonxClient imported or used.
"""

from __future__ import annotations

import pytest

from missionmind import config
from missionmind.models.mission import MissionPlan, MissionStatus, Waypoint
from missionmind.safety.validator import SafetyValidator, ValidationResult


# ---------------------------------------------------------------------------
# Helpers — plan and rover_state factories
# ---------------------------------------------------------------------------

def _science_wp(
    *,
    id: str = "wp-sci-1",
    terrain_risk: float = 0.10,
    energy_wh: float = 30.0,
    time_minutes: float = 20.0,
    label: str = "science-site",
) -> Waypoint:
    """Return a non-base science waypoint."""
    return Waypoint(
        id=id,
        x=100.0, y=50.0,
        terrain_risk=terrain_risk,
        estimated_energy_wh=energy_wh,
        estimated_travel_time_minutes=time_minutes,
        scientific_value=0.8,
        is_base=False,
        label=label,
    )


def _base_wp(
    *,
    id: str = "wp-base",
    energy_wh: float = 10.0,
    time_minutes: float = 15.0,
) -> Waypoint:
    """Return a base-station waypoint (is_base=True)."""
    return Waypoint(
        id=id,
        x=0.0, y=0.0,
        terrain_risk=0.0,
        estimated_energy_wh=energy_wh,
        estimated_travel_time_minutes=time_minutes,
        is_base=True,
        label="BASE",
    )


def _valid_plan(
    *,
    science_energy_wh: float = 30.0,
    science_time_minutes: float = 20.0,
    base_energy_wh: float = 10.0,
    base_time_minutes: float = 15.0,
    terrain_risk: float = 0.10,
) -> MissionPlan:
    """Return a plan that satisfies all five hard rules under HEALTHY_ROVER."""
    sci = _science_wp(
        energy_wh=science_energy_wh,
        time_minutes=science_time_minutes,
        terrain_risk=terrain_risk,
    )
    base = _base_wp(
        energy_wh=base_energy_wh,
        time_minutes=base_time_minutes,
    )
    return MissionPlan(
        waypoints=[sci, base],
        total_energy_wh=science_energy_wh + base_energy_wh,
        total_time_minutes=science_time_minutes + base_time_minutes,
    )


# Rover with comfortable battery headroom:
# 80 % of 500 Wh = 400 Wh present, reserve = 20 % × 500 = 100 Wh → 300 Wh usable.
HEALTHY_ROVER = {
    "battery_pct": 0.80,
    "battery_capacity_wh": 500.0,
}

# Thresholds extracted for easy use in assertions.
_MIN_RESERVE = config.MIN_RETURN_BATTERY_PCT   # default 0.20
_MAX_RISK    = config.MAX_TERRAIN_RISK_SCORE    # default 0.70
_MAX_HOURS   = config.MAX_MISSION_DURATION_HOURS  # default 8.0
_MAX_MINUTES = _MAX_HOURS * 60.0                   # default 480.0


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_passed_true_no_violations(self):
        r = ValidationResult(passed=True, violations=[])
        assert r.passed is True
        assert r.violations == []

    def test_passed_false_with_violations(self):
        r = ValidationResult(passed=False, violations=["rule A", "rule B"])
        assert r.passed is False
        assert len(r.violations) == 2

    def test_bool_true_when_passed(self):
        r = ValidationResult(passed=True)
        assert bool(r) is True

    def test_bool_false_when_failed(self):
        r = ValidationResult(passed=False, violations=["fail"])
        assert bool(r) is False

    def test_default_violations_is_empty_list(self):
        r = ValidationResult(passed=True)
        assert r.violations == []

    def test_violations_accumulate_independently(self):
        r1 = ValidationResult(passed=False, violations=["a"])
        r2 = ValidationResult(passed=False, violations=["b"])
        r1.violations.append("c")
        assert r2.violations == ["b"]   # r2 not affected


# ---------------------------------------------------------------------------
# SafetyValidator construction & config injection
# ---------------------------------------------------------------------------

class TestSafetyValidatorConstruction:
    def test_uses_config_defaults(self):
        v = SafetyValidator()
        assert v._min_return_battery_pct == pytest.approx(_MIN_RESERVE)
        assert v._max_terrain_risk_score == pytest.approx(_MAX_RISK)
        assert v._max_mission_duration_hours == pytest.approx(_MAX_HOURS)

    def test_thresholds_injectable(self):
        v = SafetyValidator(
            min_return_battery_pct=0.15,
            max_terrain_risk_score=0.50,
            max_mission_duration_hours=4.0,
        )
        assert v._min_return_battery_pct == pytest.approx(0.15)
        assert v._max_terrain_risk_score == pytest.approx(0.50)
        assert v._max_mission_duration_hours == pytest.approx(4.0)

    def test_partial_override_leaves_others_as_config(self):
        v = SafetyValidator(min_return_battery_pct=0.05)
        assert v._min_return_battery_pct == pytest.approx(0.05)
        assert v._max_terrain_risk_score == pytest.approx(_MAX_RISK)
        assert v._max_mission_duration_hours == pytest.approx(_MAX_HOURS)

    def test_no_ai_imports_in_module(self):
        """The validator module must not import BaseAgent, LLMClient, or WatsonxClient."""
        import missionmind.safety.validator as mod
        import inspect
        source = inspect.getsource(mod)
        assert "BaseAgent" not in source
        assert "WatsonxClient" not in source
        assert "LLMClient" not in source

    def test_importable_from_safety_package(self):
        from missionmind.safety import SafetyValidator as SV, ValidationResult as VR
        assert SV is SafetyValidator
        assert VR is ValidationResult


# ---------------------------------------------------------------------------
# Rule 0 — valid plan passes all checks (plan requirement test 1)
# ---------------------------------------------------------------------------

class TestValidPlanPassesAllRules:
    def test_valid_plan_passes(self):
        v = SafetyValidator()
        plan = _valid_plan()
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is True
        assert result.violations == []

    def test_valid_plan_with_multiple_science_waypoints_passes(self):
        v = SafetyValidator()
        sci1 = _science_wp(id="wp-sci-1", energy_wh=20.0, time_minutes=15.0)
        sci2 = _science_wp(id="wp-sci-2", energy_wh=25.0, time_minutes=18.0)
        base = _base_wp(energy_wh=10.0, time_minutes=12.0)
        plan = MissionPlan(
            waypoints=[sci1, sci2, base],
            total_energy_wh=55.0,
            total_time_minutes=45.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is True

    def test_passes_with_different_rover_state(self):
        """Full charge rover → more headroom; plan still passes."""
        v = SafetyValidator()
        full_rover = {"battery_pct": 1.0, "battery_capacity_wh": 500.0}
        plan = _valid_plan(science_energy_wh=150.0)   # 160 Wh total, 300 Wh usable at 80%
        # 1.0 × 500 × (1 - 0.20) = 400 Wh usable
        result = v.validate(plan, full_rover)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Rule 1 — energy budget (plan requirement test 2)
# ---------------------------------------------------------------------------

class TestEnergyBudgetRule:
    def _validator_and_usable(self, rover: dict, reserve: float = _MIN_RESERVE):
        """Helper: create validator and compute expected usable Wh.

        Formula matches the validator:
          current_charge_wh = battery_pct × battery_capacity_wh
          return_reserve_wh = reserve × battery_capacity_wh
          usable_wh         = max(0, current_charge_wh − return_reserve_wh)
        """
        v = SafetyValidator(min_return_battery_pct=reserve)
        current = rover["battery_pct"] * rover["battery_capacity_wh"]
        reserve_wh = reserve * rover["battery_capacity_wh"]
        usable = max(0.0, current - reserve_wh)
        return v, usable

    def test_energy_over_budget_fails(self):
        v, usable = self._validator_and_usable(HEALTHY_ROVER)
        # Plan that uses more than the usable budget.
        plan = _valid_plan(science_energy_wh=usable + 50.0, base_energy_wh=0.0)
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert any("ENERGY BUDGET" in msg for msg in result.violations)

    def test_energy_violation_message_is_human_readable(self):
        v, usable = self._validator_and_usable(HEALTHY_ROVER)
        plan = _valid_plan(science_energy_wh=usable + 1.0, base_energy_wh=0.0)
        result = v.validate(plan, HEALTHY_ROVER)
        msg = result.violations[0]
        assert "Wh" in msg or "wh" in msg.lower()
        assert "ENERGY" in msg.upper()

    def test_energy_exactly_at_budget_passes(self):
        v, usable = self._validator_and_usable(HEALTHY_ROVER)
        # total_energy_wh == usable exactly → should pass
        plan = _valid_plan(science_energy_wh=usable - 10.0, base_energy_wh=10.0)
        # Adjust: set total_energy_wh to exactly usable
        plan2 = MissionPlan(
            waypoints=plan.waypoints,
            total_energy_wh=usable,
            total_time_minutes=35.0,
        )
        result = v.validate(plan2, HEALTHY_ROVER)
        assert result.passed is True

    def test_zero_battery_makes_everything_over_budget(self):
        v = SafetyValidator()
        flat_rover = {"battery_pct": 0.0, "battery_capacity_wh": 500.0}
        plan = _valid_plan(science_energy_wh=0.1)   # even tiny energy fails
        result = v.validate(plan, flat_rover)
        assert result.passed is False
        assert any("ENERGY BUDGET" in m for m in result.violations)

    def test_reserve_subtracted_before_comparison(self):
        """Verify the formula uses (1 - reserve), not just reserve."""
        # 50 % charged, 100 Wh capacity, 20 % reserve → 30 Wh usable
        rover = {"battery_pct": 0.50, "battery_capacity_wh": 100.0}
        v = SafetyValidator(min_return_battery_pct=0.20)
        # Plan needing exactly 30 Wh → should pass
        plan = _valid_plan(science_energy_wh=20.0, base_energy_wh=10.0)
        result = v.validate(plan, rover)
        assert result.passed is True

    def test_energy_violation_includes_reserve_info(self):
        v, usable = self._validator_and_usable(HEALTHY_ROVER)
        plan = _valid_plan(science_energy_wh=usable + 50.0, base_energy_wh=0.0)
        result = v.validate(plan, HEALTHY_ROVER)
        # Message should mention the reserve percentage
        msg = " ".join(result.violations)
        assert "reserve" in msg.lower() or "%" in msg or str(int(_MIN_RESERVE * 100)) in msg


# ---------------------------------------------------------------------------
# Rule 2 — terrain risk (plan requirement test 3)
# ---------------------------------------------------------------------------

class TestTerrainRiskRule:
    def test_high_terrain_risk_fails(self):
        v = SafetyValidator()
        plan = _valid_plan(terrain_risk=_MAX_RISK + 0.01)
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert any("TERRAIN RISK" in m for m in result.violations)

    def test_terrain_risk_violation_message_is_human_readable(self):
        v = SafetyValidator()
        plan = _valid_plan(terrain_risk=0.95)
        result = v.validate(plan, HEALTHY_ROVER)
        msg = result.violations[0]
        # Should name the waypoint and include the score
        assert "0.95" in msg or "terrain" in msg.lower()

    def test_terrain_risk_exactly_at_threshold_passes(self):
        v = SafetyValidator()
        plan = _valid_plan(terrain_risk=_MAX_RISK)
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is True

    def test_multiple_high_risk_waypoints_produce_multiple_violations(self):
        v = SafetyValidator()
        sci1 = _science_wp(id="wp-risky-1", terrain_risk=0.90, energy_wh=10.0, time_minutes=10.0)
        sci2 = _science_wp(id="wp-risky-2", terrain_risk=0.95, energy_wh=10.0, time_minutes=10.0)
        base = _base_wp()
        plan = MissionPlan(
            waypoints=[sci1, sci2, base],
            total_energy_wh=20.0,
            total_time_minutes=35.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        terrain_violations = [m for m in result.violations if "TERRAIN RISK" in m]
        assert len(terrain_violations) == 2

    def test_high_terrain_risk_on_base_waypoint_also_fails(self):
        """Even the return waypoint must comply with terrain rules."""
        v = SafetyValidator()
        sci = _science_wp(terrain_risk=0.10)
        bad_base = Waypoint(
            id="wp-bad-base", x=0.0, y=0.0,
            terrain_risk=0.95, is_base=True, label="BASE"
        )
        plan = MissionPlan(
            waypoints=[sci, bad_base],
            total_energy_wh=30.0,
            total_time_minutes=35.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert any("TERRAIN RISK" in m for m in result.violations)

    def test_custom_threshold_respected(self):
        v = SafetyValidator(max_terrain_risk_score=0.30)
        plan = _valid_plan(terrain_risk=0.35)   # 0.35 > 0.30 custom threshold
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert any("TERRAIN RISK" in m for m in result.violations)


# ---------------------------------------------------------------------------
# Rule 3 — mission duration (plan requirement test 4)
# ---------------------------------------------------------------------------

class TestMissionDurationRule:
    def test_over_duration_fails(self):
        v = SafetyValidator()
        plan = _valid_plan(
            science_time_minutes=_MAX_MINUTES + 1.0,
            base_time_minutes=0.0,
        )
        plan2 = MissionPlan(
            waypoints=plan.waypoints,
            total_energy_wh=40.0,
            total_time_minutes=_MAX_MINUTES + 1.0,
        )
        result = v.validate(plan2, HEALTHY_ROVER)
        assert result.passed is False
        assert any("DURATION" in m or "TIME" in m for m in result.violations)

    def test_duration_violation_message_is_human_readable(self):
        v = SafetyValidator()
        plan = MissionPlan(
            waypoints=[_science_wp(), _base_wp()],
            total_energy_wh=40.0,
            total_time_minutes=_MAX_MINUTES + 30.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        msg = " ".join(result.violations)
        assert "minute" in msg.lower() or "hour" in msg.lower()

    def test_exactly_at_max_duration_passes(self):
        v = SafetyValidator()
        plan = MissionPlan(
            waypoints=[_science_wp(), _base_wp()],
            total_energy_wh=40.0,
            total_time_minutes=_MAX_MINUTES,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is True

    def test_custom_duration_threshold_respected(self):
        v = SafetyValidator(max_mission_duration_hours=2.0)
        plan = MissionPlan(
            waypoints=[_science_wp(), _base_wp()],
            total_energy_wh=40.0,
            total_time_minutes=130.0,   # 130 min > 2h (120 min)
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert any("DURATION" in m or "TIME" in m for m in result.violations)


# ---------------------------------------------------------------------------
# Rule 4 — return to base (plan requirement test 5)
# ---------------------------------------------------------------------------

class TestReturnToBaseRule:
    def test_plan_not_ending_with_base_fails(self):
        v = SafetyValidator()
        # End with a science waypoint instead of a base waypoint.
        sci = _science_wp()
        plan = MissionPlan(
            waypoints=[sci],
            total_energy_wh=30.0,
            total_time_minutes=20.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert any("RETURN TO BASE" in m for m in result.violations)

    def test_return_to_base_violation_message_is_human_readable(self):
        v = SafetyValidator()
        sci = _science_wp()
        plan = MissionPlan(waypoints=[sci], total_energy_wh=30.0, total_time_minutes=20.0)
        result = v.validate(plan, HEALTHY_ROVER)
        assert any("base" in m.lower() or "return" in m.lower() for m in result.violations)

    def test_empty_plan_fails_return_to_base(self):
        v = SafetyValidator()
        plan = MissionPlan(waypoints=[], total_energy_wh=0.0, total_time_minutes=0.0)
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert any("RETURN TO BASE" in m or "MISSING" in m for m in result.violations)

    def test_base_waypoint_in_middle_fails(self):
        """Base must be the FINAL waypoint, not in the middle."""
        v = SafetyValidator()
        base = _base_wp()
        sci = _science_wp()
        plan = MissionPlan(
            waypoints=[base, sci],   # base first, science last — wrong order
            total_energy_wh=40.0,
            total_time_minutes=35.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert any("RETURN TO BASE" in m for m in result.violations)

    def test_plan_ending_with_base_passes_this_rule(self):
        """Specifically test Rule 4 isolation: science + base → Rule 4 passes."""
        v = SafetyValidator()
        plan = _valid_plan()
        # Check Rule 4 directly.
        violations = v._check_return_to_base(plan)
        assert violations == []


# ---------------------------------------------------------------------------
# Rule 5 — science objective (plan requirement test 6)
# ---------------------------------------------------------------------------

class TestScienceObjectiveRule:
    def test_no_science_waypoints_fails(self):
        v = SafetyValidator()
        base = _base_wp()
        plan = MissionPlan(
            waypoints=[base],
            total_energy_wh=10.0,
            total_time_minutes=15.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert any("SCIENCE" in m for m in result.violations)

    def test_science_violation_message_is_human_readable(self):
        v = SafetyValidator()
        plan = MissionPlan(waypoints=[_base_wp()], total_energy_wh=10.0, total_time_minutes=15.0)
        result = v.validate(plan, HEALTHY_ROVER)
        msg = " ".join(result.violations)
        assert "science" in msg.lower() or "objective" in msg.lower()

    def test_empty_plan_fails_science_rule(self):
        v = SafetyValidator()
        plan = MissionPlan(waypoints=[], total_energy_wh=0.0, total_time_minutes=0.0)
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert any("SCIENCE" in m for m in result.violations)

    def test_single_science_waypoint_satisfies_rule(self):
        """Rule 5 isolation: plan with one science + base passes Rule 5."""
        v = SafetyValidator()
        plan = _valid_plan()
        violations = v._check_science_objective(plan)
        assert violations == []

    def test_only_base_waypoints_fails_science_rule(self):
        v = SafetyValidator()
        base1 = _base_wp(id="base-a")
        base2 = _base_wp(id="base-b")
        plan = MissionPlan(
            waypoints=[base1, base2],
            total_energy_wh=20.0,
            total_time_minutes=30.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert any("SCIENCE" in m for m in result.violations)


# ---------------------------------------------------------------------------
# Multiple violations simultaneously (plan requirement test 7)
# ---------------------------------------------------------------------------

class TestMultipleViolations:
    def test_energy_and_terrain_both_violated(self):
        v = SafetyValidator()
        # Over budget (will use 99999 Wh) AND high terrain risk AND no return.
        sci = _science_wp(terrain_risk=0.99, energy_wh=0.0, time_minutes=10.0)
        plan = MissionPlan(
            waypoints=[sci],           # no base → Rule 4 fails
            total_energy_wh=99999.0,   # over budget → Rule 1 fails
            total_time_minutes=10.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert len(result.violations) >= 2

    def test_all_five_rules_can_fail_simultaneously(self):
        """Construct a plan that violates every single rule."""
        v = SafetyValidator(
            min_return_battery_pct=0.20,
            max_terrain_risk_score=0.70,
            max_mission_duration_hours=8.0,
        )
        rover = {"battery_pct": 0.80, "battery_capacity_wh": 500.0}
        # Usable = 0.80 × 500 × 0.80 = 320 Wh

        bad_sci = _science_wp(
            terrain_risk=0.99,           # Rule 2: exceeds 0.70
            energy_wh=0.0,
            time_minutes=0.0,
        )
        bad_base_mid = Waypoint(         # is_base but not at the end
            id="base-mid", x=0.0, y=0.0, is_base=True
        )
        bad_sci2 = _science_wp(
            id="wp-sci-end",             # science at end → Rule 4 fails
            terrain_risk=0.05,
            energy_wh=0.0,
            time_minutes=0.0,
        )
        plan = MissionPlan(
            waypoints=[bad_sci, bad_base_mid, bad_sci2],
            total_energy_wh=99999.0,     # Rule 1: over budget
            total_time_minutes=99999.0,  # Rule 3: over duration
        )
        # Rule 5 is actually OK here (there are science waypoints) but
        # Rules 1, 2, 3, 4 all fail.
        result = v.validate(plan, rover)
        assert result.passed is False
        assert len(result.violations) >= 3

    def test_violations_accumulate_not_short_circuit(self):
        """validate() must NOT stop at the first violation."""
        v = SafetyValidator()
        # Both Rule 1 (energy) and Rule 2 (terrain) will fail.
        sci = _science_wp(terrain_risk=0.99, energy_wh=0.0, time_minutes=10.0)
        base = _base_wp()
        plan = MissionPlan(
            waypoints=[sci, base],
            total_energy_wh=99999.0,
            total_time_minutes=35.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        # At minimum energy AND terrain violations must both be present.
        has_energy   = any("ENERGY" in m for m in result.violations)
        has_terrain  = any("TERRAIN" in m for m in result.violations)
        assert has_energy,  "Energy violation missing from accumulated list"
        assert has_terrain, "Terrain violation missing from accumulated list"

    def test_all_violations_returned_in_single_result(self):
        """A single validate() call returns ALL violations at once."""
        v = SafetyValidator(max_mission_duration_hours=1.0)
        # Terrain + duration
        sci = _science_wp(terrain_risk=0.99, time_minutes=200.0)
        base = _base_wp()
        plan = MissionPlan(
            waypoints=[sci, base],
            total_energy_wh=40.0,
            total_time_minutes=200.0,
        )
        result = v.validate(plan, HEALTHY_ROVER)
        assert result.passed is False
        assert len(result.violations) >= 2


# ---------------------------------------------------------------------------
# Rule isolation — test each _check_* method directly
# ---------------------------------------------------------------------------

class TestRuleIsolation:
    """Test the private rule methods directly to ensure each one works in
    isolation without being masked by other failing rules."""

    def test_check_energy_budget_pass(self):
        v = SafetyValidator()
        plan = _valid_plan(science_energy_wh=50.0, base_energy_wh=10.0)
        # 80 % × 500 Wh × 0.80 = 320 Wh usable; 60 Wh < 320 → pass
        violations = v._check_energy_budget(plan, HEALTHY_ROVER)
        assert violations == []

    def test_check_energy_budget_fail(self):
        v = SafetyValidator()
        plan = _valid_plan(science_energy_wh=400.0, base_energy_wh=0.0)
        violations = v._check_energy_budget(plan, HEALTHY_ROVER)
        assert len(violations) == 1
        assert "ENERGY BUDGET" in violations[0]

    def test_check_terrain_risk_pass(self):
        v = SafetyValidator()
        plan = _valid_plan(terrain_risk=0.10)
        violations = v._check_terrain_risk(plan)
        assert violations == []

    def test_check_terrain_risk_fail(self):
        v = SafetyValidator()
        plan = _valid_plan(terrain_risk=0.99)
        violations = v._check_terrain_risk(plan)
        assert len(violations) == 1
        assert "TERRAIN RISK" in violations[0]

    def test_check_mission_duration_pass(self):
        v = SafetyValidator()
        plan = _valid_plan(science_time_minutes=60.0, base_time_minutes=30.0)
        violations = v._check_mission_duration(plan)
        assert violations == []

    def test_check_mission_duration_fail(self):
        v = SafetyValidator(max_mission_duration_hours=1.0)
        plan = MissionPlan(
            waypoints=[_science_wp(), _base_wp()],
            total_energy_wh=40.0,
            total_time_minutes=120.0,  # 120 > 60 (1 hour)
        )
        violations = v._check_mission_duration(plan)
        assert len(violations) == 1

    def test_check_return_to_base_pass(self):
        v = SafetyValidator()
        plan = _valid_plan()
        violations = v._check_return_to_base(plan)
        assert violations == []

    def test_check_return_to_base_fail(self):
        v = SafetyValidator()
        plan = MissionPlan(
            waypoints=[_science_wp()],
            total_energy_wh=30.0,
            total_time_minutes=20.0,
        )
        violations = v._check_return_to_base(plan)
        assert len(violations) == 1
        assert "RETURN TO BASE" in violations[0]

    def test_check_science_objective_pass(self):
        v = SafetyValidator()
        plan = _valid_plan()
        violations = v._check_science_objective(plan)
        assert violations == []

    def test_check_science_objective_fail(self):
        v = SafetyValidator()
        plan = MissionPlan(
            waypoints=[_base_wp()],
            total_energy_wh=10.0,
            total_time_minutes=15.0,
        )
        violations = v._check_science_objective(plan)
        assert len(violations) == 1
        assert "SCIENCE" in violations[0]


# ---------------------------------------------------------------------------
# Determinism guarantee
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_result_multiple_calls(self):
        v = SafetyValidator()
        plan = _valid_plan()
        results = [v.validate(plan, HEALTHY_ROVER) for _ in range(5)]
        assert all(r.passed is True for r in results)

    def test_different_plans_independent_results(self):
        v = SafetyValidator()
        good = _valid_plan()
        bad  = _valid_plan(terrain_risk=0.99)
        r1 = v.validate(good, HEALTHY_ROVER)
        r2 = v.validate(bad,  HEALTHY_ROVER)
        assert r1.passed is True
        assert r2.passed is False
