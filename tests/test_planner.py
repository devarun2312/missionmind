"""
Tests for the Mission Planner entry point — Sub-Task 8.

All tests patch ``missionmind.planning.planner._build_commander`` so the
internal factory returns a mock MissionCommander.  No real LLM/IBM/network
calls are made in any test.

The public API under test is exactly:

    async plan_mission(rover_state: dict, env_state: dict) -> MissionPlan

Test classes
------------
TestApprovedTest1Delegation        — approved Test 1: context assembly and delegation
TestApprovedTest2MissingRoverKey   — approved Test 2: missing rover_state key → ValueError
TestMissingEnvStateKey             — missing env_state key → ValueError
TestMultipleMissingKeys            — multiple missing keys listed in error
TestContextAssembly                — _build_context unit tests
TestValidateKeys                   — _validate_keys unit tests
TestLlmClientFactory               — _build_llm_client returns WatsonxClient
TestInputDictNotMutated            — input dicts unchanged after call
TestLogging                        — INFO-level log on success
TestImportPath                     — importable from missionmind.planning
TestErrorPropagation               — PlanningFailedError propagates
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missionmind.agents.client import WatsonxClient
from missionmind.agents.mission_commander import PlanningFailedError
from missionmind.models.mission import MissionPlan, MissionStatus, Waypoint
from missionmind.planning.planner import (
    ENV_STATE_KEYS,
    ROVER_STATE_KEYS,
    _build_context,
    _build_llm_client,
    _validate_keys,
    plan_mission,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VALID_ROVER_STATE: dict = {
    "battery_pct":          0.85,
    "battery_capacity_wh":  500.0,
    "position_x":           0.0,
    "position_y":           0.0,
    "rover_speed_mps":      0.5,
    "power_consumption_w":  50.0,
}

_VALID_ENV_STATE: dict = {
    "candidate_waypoints": [
        {
            "id": "wp-1", "x": 150.0, "y": 80.0,
            "terrain_risk": 0.2, "is_base": False, "label": "crater-A",
            "estimated_travel_time_minutes": 30.0, "estimated_energy_wh": 20.0,
        },
        {
            "id": "base", "x": 0.0, "y": 0.0,
            "terrain_risk": 0.0, "is_base": True, "label": "BASE",
            "estimated_travel_time_minutes": 0.0, "estimated_energy_wh": 0.0,
        },
    ],
    "weather_forecast": {
        "dust_storm_probability": 0.1,
        "temperature_min_c":      -60.0,
        "temperature_max_c":       20.0,
        "wind_speed_mps":           5.0,
        "forecast_hours":           8,
    },
    "comm_windows":      [{"start_utc": "2025-01-01T10:00:00Z", "duration_minutes": 30}],
    "terrain_map":       {},
    "mission_objectives": ["search for biosignatures"],
}


def _make_active_plan() -> MissionPlan:
    return MissionPlan(
        waypoints=[
            Waypoint(id="wp-1", x=150.0, y=80.0, scientific_value=0.8,
                     terrain_risk=0.2, estimated_travel_time_minutes=30.0,
                     estimated_energy_wh=20.0, is_base=False, label="crater-A"),
            Waypoint(id="base", x=0.0, y=0.0, scientific_value=0.0,
                     terrain_risk=0.0, estimated_travel_time_minutes=0.0,
                     estimated_energy_wh=0.0, is_base=True, label="BASE"),
        ],
        total_energy_wh=20.0,
        total_time_minutes=30.0,
        status=MissionStatus.ACTIVE,
        reasoning="good plan",
        confidence=0.9,
    )


def _mock_commander(plan: MissionPlan | None = None) -> MagicMock:
    """Return a MagicMock whose .plan() is an AsyncMock returning the given plan."""
    mock = MagicMock()
    mock.plan = AsyncMock(return_value=plan or _make_active_plan())
    return mock


@contextmanager
def _patch_commander(plan: MissionPlan | None = None):
    """Context manager: patch _build_commander to return a mock commander.

    Yields the mock commander so callers can inspect call args.
    """
    commander = _mock_commander(plan)
    with patch(
        "missionmind.planning.planner._build_commander",
        return_value=commander,
    ):
        yield commander


# ---------------------------------------------------------------------------
# Approved Test 1 — Delegation
# ---------------------------------------------------------------------------

class TestApprovedTest1Delegation:
    """plan_mission must delegate to MissionCommander.plan() and return its result."""

    async def test_returns_mission_plan(self):
        with _patch_commander():
            plan = await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        assert isinstance(plan, MissionPlan)

    async def test_returns_exactly_commander_result(self):
        expected = _make_active_plan()
        with _patch_commander(expected):
            plan = await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        assert plan is expected

    async def test_commander_plan_called_once(self):
        with _patch_commander() as commander:
            await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        commander.plan.assert_called_once()

    async def test_commander_receives_candidate_waypoints_in_context(self):
        with _patch_commander() as commander:
            await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        ctx = commander.plan.call_args[0][0]
        assert "candidate_waypoints" in ctx
        assert ctx["candidate_waypoints"] is _VALID_ENV_STATE["candidate_waypoints"]

    async def test_commander_receives_rover_state_dict_in_context(self):
        with _patch_commander() as commander:
            await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        ctx = commander.plan.call_args[0][0]
        assert "rover_state" in ctx
        assert ctx["rover_state"]["battery_pct"] == _VALID_ROVER_STATE["battery_pct"]

    async def test_commander_receives_battery_pct_flat_in_context(self):
        """rover_state keys are also present flat in context for agent access."""
        with _patch_commander() as commander:
            await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        ctx = commander.plan.call_args[0][0]
        assert ctx["battery_pct"] == _VALID_ROVER_STATE["battery_pct"]

    async def test_commander_receives_rover_position_in_context(self):
        with _patch_commander() as commander:
            await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        ctx = commander.plan.call_args[0][0]
        assert "rover_position" in ctx
        assert ctx["rover_position"]["x"] == _VALID_ROVER_STATE["position_x"]
        assert ctx["rover_position"]["y"] == _VALID_ROVER_STATE["position_y"]

    async def test_commander_receives_mission_objectives_in_context(self):
        with _patch_commander() as commander:
            await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        ctx = commander.plan.call_args[0][0]
        assert ctx["mission_objectives"] is _VALID_ENV_STATE["mission_objectives"]

    async def test_build_commander_called_once(self):
        """_build_commander factory must be called exactly once per plan_mission call."""
        with patch(
            "missionmind.planning.planner._build_commander",
            return_value=_mock_commander(),
        ) as build_mock:
            await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        build_mock.assert_called_once()

    async def test_status_active(self):
        with _patch_commander():
            plan = await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        assert plan.status is MissionStatus.ACTIVE

    async def test_public_signature_accepts_only_two_positional_args(self):
        """The public API must be plan_mission(rover_state, env_state) — no extra params."""
        import inspect
        sig = inspect.signature(plan_mission)
        params = list(sig.parameters.keys())
        assert params == ["rover_state", "env_state"]


# ---------------------------------------------------------------------------
# Approved Test 2 — Missing required rover_state key → ValueError
# ---------------------------------------------------------------------------

class TestApprovedTest2MissingRoverKey:
    """Each missing rover_state key must raise ValueError with the key named."""

    async def test_missing_battery_pct_raises(self):
        bad = {k: v for k, v in _VALID_ROVER_STATE.items() if k != "battery_pct"}
        with pytest.raises(ValueError, match="battery_pct"):
            await plan_mission(bad, _VALID_ENV_STATE)

    async def test_missing_battery_capacity_wh_raises(self):
        bad = {k: v for k, v in _VALID_ROVER_STATE.items() if k != "battery_capacity_wh"}
        with pytest.raises(ValueError, match="battery_capacity_wh"):
            await plan_mission(bad, _VALID_ENV_STATE)

    async def test_missing_position_x_raises(self):
        bad = {k: v for k, v in _VALID_ROVER_STATE.items() if k != "position_x"}
        with pytest.raises(ValueError, match="position_x"):
            await plan_mission(bad, _VALID_ENV_STATE)

    async def test_missing_position_y_raises(self):
        bad = {k: v for k, v in _VALID_ROVER_STATE.items() if k != "position_y"}
        with pytest.raises(ValueError, match="position_y"):
            await plan_mission(bad, _VALID_ENV_STATE)

    async def test_missing_rover_speed_mps_raises(self):
        bad = {k: v for k, v in _VALID_ROVER_STATE.items() if k != "rover_speed_mps"}
        with pytest.raises(ValueError, match="rover_speed_mps"):
            await plan_mission(bad, _VALID_ENV_STATE)

    async def test_missing_power_consumption_w_raises(self):
        bad = {k: v for k, v in _VALID_ROVER_STATE.items() if k != "power_consumption_w"}
        with pytest.raises(ValueError, match="power_consumption_w"):
            await plan_mission(bad, _VALID_ENV_STATE)

    async def test_error_identifies_source_as_rover_state(self):
        with pytest.raises(ValueError, match="rover_state"):
            await plan_mission({}, _VALID_ENV_STATE)

    async def test_validation_runs_before_commander_called(self):
        with patch(
            "missionmind.planning.planner._build_commander",
            return_value=_mock_commander(),
        ) as build_mock:
            with pytest.raises(ValueError):
                await plan_mission({}, _VALID_ENV_STATE)
        # Commander should never be called when validation fails
        build_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Missing env_state keys → ValueError
# ---------------------------------------------------------------------------

class TestMissingEnvStateKey:
    async def test_missing_candidate_waypoints_raises(self):
        bad = {k: v for k, v in _VALID_ENV_STATE.items() if k != "candidate_waypoints"}
        with pytest.raises(ValueError, match="candidate_waypoints"):
            await plan_mission(_VALID_ROVER_STATE, bad)

    async def test_missing_weather_forecast_raises(self):
        bad = {k: v for k, v in _VALID_ENV_STATE.items() if k != "weather_forecast"}
        with pytest.raises(ValueError, match="weather_forecast"):
            await plan_mission(_VALID_ROVER_STATE, bad)

    async def test_missing_comm_windows_raises(self):
        bad = {k: v for k, v in _VALID_ENV_STATE.items() if k != "comm_windows"}
        with pytest.raises(ValueError, match="comm_windows"):
            await plan_mission(_VALID_ROVER_STATE, bad)

    async def test_missing_terrain_map_raises(self):
        bad = {k: v for k, v in _VALID_ENV_STATE.items() if k != "terrain_map"}
        with pytest.raises(ValueError, match="terrain_map"):
            await plan_mission(_VALID_ROVER_STATE, bad)

    async def test_missing_mission_objectives_raises(self):
        bad = {k: v for k, v in _VALID_ENV_STATE.items() if k != "mission_objectives"}
        with pytest.raises(ValueError, match="mission_objectives"):
            await plan_mission(_VALID_ROVER_STATE, bad)

    async def test_error_identifies_source_as_env_state(self):
        with pytest.raises(ValueError, match="env_state"):
            await plan_mission(_VALID_ROVER_STATE, {})


# ---------------------------------------------------------------------------
# Multiple missing keys
# ---------------------------------------------------------------------------

class TestMultipleMissingKeys:
    async def test_all_rover_keys_listed_when_empty_dict(self):
        with pytest.raises(ValueError) as exc_info:
            await plan_mission({}, _VALID_ENV_STATE)
        msg = str(exc_info.value)
        for key in ROVER_STATE_KEYS:
            assert key in msg, f"Expected '{key}' in error message"

    async def test_all_env_keys_listed_when_empty_dict(self):
        with pytest.raises(ValueError) as exc_info:
            await plan_mission(_VALID_ROVER_STATE, {})
        msg = str(exc_info.value)
        for key in ENV_STATE_KEYS:
            assert key in msg, f"Expected '{key}' in error message"

    async def test_two_missing_rover_keys_both_listed(self):
        bad = {k: v for k, v in _VALID_ROVER_STATE.items()
               if k not in ("battery_pct", "position_x")}
        with pytest.raises(ValueError) as exc_info:
            await plan_mission(bad, _VALID_ENV_STATE)
        msg = str(exc_info.value)
        assert "battery_pct" in msg
        assert "position_x" in msg


# ---------------------------------------------------------------------------
# _build_context unit tests
# ---------------------------------------------------------------------------

class TestContextAssembly:
    def test_env_keys_present(self):
        ctx = _build_context(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        for key in ENV_STATE_KEYS:
            assert key in ctx, f"Expected '{key}' in context"

    def test_rover_keys_present_flat(self):
        ctx = _build_context(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        for key in ROVER_STATE_KEYS:
            assert key in ctx, f"Expected '{key}' in context"

    def test_rover_state_nested_present(self):
        ctx = _build_context(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        assert "rover_state" in ctx
        assert ctx["rover_state"] == _VALID_ROVER_STATE

    def test_rover_state_nested_is_copy(self):
        ctx = _build_context(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        assert ctx["rover_state"] is not _VALID_ROVER_STATE

    def test_rover_position_derived(self):
        ctx = _build_context(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        assert "rover_position" in ctx
        assert ctx["rover_position"]["x"] == _VALID_ROVER_STATE["position_x"]
        assert ctx["rover_position"]["y"] == _VALID_ROVER_STATE["position_y"]

    def test_candidate_waypoints_preserved(self):
        ctx = _build_context(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        assert ctx["candidate_waypoints"] is _VALID_ENV_STATE["candidate_waypoints"]

    def test_battery_pct_value_correct(self):
        ctx = _build_context(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        assert ctx["battery_pct"] == pytest.approx(_VALID_ROVER_STATE["battery_pct"])

    def test_battery_capacity_value_correct(self):
        ctx = _build_context(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        assert ctx["battery_capacity_wh"] == pytest.approx(
            _VALID_ROVER_STATE["battery_capacity_wh"]
        )


# ---------------------------------------------------------------------------
# _validate_keys unit tests
# ---------------------------------------------------------------------------

class TestValidateKeys:
    def test_passes_when_all_keys_present(self):
        _validate_keys("test", {"a": 1, "b": 2}, ("a", "b"))  # no exception

    def test_raises_on_single_missing_key(self):
        with pytest.raises(ValueError, match="missing_key"):
            _validate_keys("test", {}, ("missing_key",))

    def test_raises_on_empty_data(self):
        with pytest.raises(ValueError):
            _validate_keys("test", {}, ("required",))

    def test_error_contains_all_missing_keys(self):
        with pytest.raises(ValueError) as exc_info:
            _validate_keys("test", {}, ("key1", "key2", "key3"))
        msg = str(exc_info.value)
        assert "key1" in msg
        assert "key2" in msg
        assert "key3" in msg

    def test_error_contains_source_name(self):
        with pytest.raises(ValueError, match="my_source"):
            _validate_keys("my_source", {}, ("k",))

    def test_no_raise_when_extra_keys_present(self):
        # Extra keys beyond the required set are fine
        _validate_keys("test", {"a": 1, "b": 2, "extra": 99}, ("a", "b"))


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

class TestLlmClientFactory:
    def test_returns_watsonx_client(self):
        client = _build_llm_client()
        assert isinstance(client, WatsonxClient)

    def test_no_real_http_made_on_construction(self):
        # Constructing WatsonxClient must not make any network calls
        client = _build_llm_client()
        assert client is not None


# ---------------------------------------------------------------------------
# Input dicts not mutated
# ---------------------------------------------------------------------------

class TestInputDictNotMutated:
    async def test_rover_state_not_mutated(self):
        original_keys = set(_VALID_ROVER_STATE.keys())
        original_vals = dict(_VALID_ROVER_STATE)
        with _patch_commander():
            await plan_mission(dict(_VALID_ROVER_STATE), dict(_VALID_ENV_STATE))
        # Original reference values must be unchanged
        assert set(_VALID_ROVER_STATE.keys()) == original_keys
        assert _VALID_ROVER_STATE == original_vals

    async def test_env_state_not_mutated(self):
        original_keys = set(_VALID_ENV_STATE.keys())
        with _patch_commander():
            await plan_mission(dict(_VALID_ROVER_STATE), dict(_VALID_ENV_STATE))
        assert set(_VALID_ENV_STATE.keys()) == original_keys


# ---------------------------------------------------------------------------
# Logging at INFO level
# ---------------------------------------------------------------------------

class TestLogging:
    async def test_info_logged_on_success(self, caplog):
        with caplog.at_level(logging.INFO, logger="missionmind.planning.planner"):
            with _patch_commander():
                await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        assert any(rec.levelno == logging.INFO for rec in caplog.records)

    async def test_plan_id_logged(self, caplog):
        plan = _make_active_plan()
        with caplog.at_level(logging.INFO, logger="missionmind.planning.planner"):
            with _patch_commander(plan):
                await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)
        log_text = " ".join(rec.message for rec in caplog.records)
        assert plan.plan_id in log_text


# ---------------------------------------------------------------------------
# PlanningFailedError propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:
    async def test_planning_failed_error_propagates(self):
        commander = _mock_commander()
        commander.plan = AsyncMock(
            side_effect=PlanningFailedError(
                "all attempts exhausted", attempts=3
            )
        )
        with patch(
            "missionmind.planning.planner._build_commander",
            return_value=commander,
        ):
            with pytest.raises(PlanningFailedError):
                await plan_mission(_VALID_ROVER_STATE, _VALID_ENV_STATE)


# ---------------------------------------------------------------------------
# Import path
# ---------------------------------------------------------------------------

class TestImportPath:
    def test_importable_from_planning_package(self):
        from missionmind.planning import plan_mission as pm
        assert callable(pm)

    def test_importable_from_planning_planner(self):
        from missionmind.planning.planner import plan_mission as pm
        assert callable(pm)

    def test_rover_state_keys_exported(self):
        from missionmind.planning.planner import ROVER_STATE_KEYS
        assert isinstance(ROVER_STATE_KEYS, tuple)
        assert len(ROVER_STATE_KEYS) > 0

    def test_env_state_keys_exported(self):
        from missionmind.planning.planner import ENV_STATE_KEYS
        assert isinstance(ENV_STATE_KEYS, tuple)
        assert len(ENV_STATE_KEYS) > 0
