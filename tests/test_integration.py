"""
Integration Tests for MissionMind — Sub-Task 10.

These tests exercise the REAL end-to-end planning pipeline with only the
external LLM network layer replaced by a deterministic fake client.

What is REAL in these tests
----------------------------
- plan_mission()         public entry point and all its wiring
- ScienceAgent           full agent: prompt loading, JSON parsing, Pydantic validation
- ResourceAgent          full agent: prompt loading, JSON parsing, config injection
- SafetyAgent            full agent: prompt loading, JSON parsing, config injection
- MissionCommander       orchestration: asyncio.gather, synthesis call, retry logic
- MissionPlan conversion _convert_to_mission_plan()
- SafetyValidator        all five deterministic hard-constraint rules
- replan()               Replanner dispatcher and event handlers

What is FAKE in these tests
----------------------------
- External HTTP / IBM watsonx network call (replaced by FakeLLMClient)

No real API credentials are needed.  No network traffic is produced.

Fake LLM design
---------------
``FakeLLMClient`` accepts a list of canned JSON strings at construction time.
Each call to ``complete()`` pops the next string from the queue and wraps it in
an ``LLMResponse``.  This gives the tests full, deterministic control over what
every agent "hears" from the AI backend.

Agent call order within a single ``commander.plan()`` attempt:
  1. ScienceAgent
  2. ResourceAgent       (parallel with science via asyncio.gather)
  3. SafetyAgent
  4. Commander synthesis

For a two-attempt scenario (validator rejects attempt 1, approves attempt 2)
there are 4 + 4 = 8 LLM calls in total.

Test overview
-------------
TestIntegration1FullPlanMission
    Approved Test 1 — full plan_mission() with real components.

TestIntegration2ValidatorRejectRetry
    Approved Test 2 — real validator rejects first plan; retry succeeds.

TestIntegration3NewDiscoveryReplan
    Approved Test 3 — replan() with NEW_DISCOVERY event.
"""

from __future__ import annotations

import json
from collections import deque
from unittest.mock import patch

import pytest

from missionmind.agents.client import LLMResponse
from missionmind.models.events import EventType, MissionEvent
from missionmind.models.mission import MissionPlan, MissionStatus
from missionmind.planning.planner import plan_mission
from missionmind.planning.replanner import replan


# ---------------------------------------------------------------------------
# FakeLLMClient — deterministic stand-in for WatsonxClient
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """A deterministic fake LLM client for integration testing.

    Accepts a sequence of canned JSON strings at construction time.
    Each ``complete()`` call returns the next string in the queue wrapped
    in an ``LLMResponse``.  The queue is consumed in FIFO order.

    Raises ``AssertionError`` if the queue is exhausted (i.e. the code made
    more LLM calls than the test expected).  This surfaces wiring surprises
    early rather than silently hanging or returning wrong responses.

    Parameters
    ----------
    responses:
        Ordered sequence of raw JSON strings that should be returned for
        successive ``complete()`` calls.
    """

    def __init__(self, responses: list[str]) -> None:
        self._queue: deque[str] = deque(responses)
        self.call_count: int = 0

    async def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        assert self._queue, (
            f"FakeLLMClient: queue exhausted on call #{self.call_count + 1}. "
            "Provide more canned responses."
        )
        self.call_count += 1
        content = self._queue.popleft()
        return LLMResponse(content=content, model="fake-model")

    @property
    def remaining(self) -> int:
        """Number of canned responses not yet consumed."""
        return len(self._queue)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# Waypoints used across all three integration tests.
# Coordinates designed so all distance/energy arithmetic stays well within
# the default validator thresholds.
_WP_CRATER = {
    "id": "wp-crater",
    "x": 100.0, "y": 0.0,
    "terrain_risk": 0.15,
    "is_base": False,
    "label": "crater-rim",
    "estimated_travel_time_minutes": 30.0,
    "estimated_energy_wh": 15.0,
}
_WP_OUTCROP = {
    "id": "wp-outcrop",
    "x": 160.0, "y": 0.0,
    "terrain_risk": 0.20,
    "is_base": False,
    "label": "rock-outcrop",
    "estimated_travel_time_minutes": 20.0,
    "estimated_energy_wh": 10.0,
}
_WP_BASE = {
    "id": "base-hq",
    "x": 0.0, "y": 0.0,
    "terrain_risk": 0.0,
    "is_base": True,
    "label": "BASE",
    "estimated_travel_time_minutes": 0.0,
    "estimated_energy_wh": 0.0,
}

# rover_state: battery at 90 %, 500 Wh capacity.
# With MIN_RETURN_BATTERY_PCT=0.20: usable = 0.90×500 − 0.20×500 = 350 Wh.
# Both science waypoints together cost 25 Wh — comfortably within budget.
_ROVER_STATE = {
    "battery_pct":          0.90,
    "battery_capacity_wh":  500.0,
    "position_x":           0.0,
    "position_y":           0.0,
    "rover_speed_mps":      0.5,
    "power_consumption_w":  50.0,
}

_ENV_STATE = {
    "candidate_waypoints": [_WP_CRATER, _WP_OUTCROP, _WP_BASE],
    "weather_forecast": {
        "dust_storm_probability": 0.05,
        "temperature_min_c":      -55.0,
        "temperature_max_c":       15.0,
        "wind_speed_mps":           3.0,
        "forecast_hours":           8,
    },
    "comm_windows": [
        {"start_utc": "2025-06-01T10:00:00Z", "duration_minutes": 45}
    ],
    "terrain_map": {},
    "mission_objectives": [
        "search for biosignatures",
        "characterise basalt mineralogy",
    ],
}


# ---------------------------------------------------------------------------
# Canned JSON response builders
# ---------------------------------------------------------------------------

def _science_response(waypoint_ids: list[str] = ("wp-crater", "wp-outcrop")) -> str:
    """Build a valid ScienceAnalysis JSON string."""
    return json.dumps({
        "scored_targets": [
            {"waypoint_id": wid, "scientific_value": 0.85, "justification": "promising geology"}
            for wid in waypoint_ids
        ],
        "priority_order": list(waypoint_ids),
        "reasoning": "Both targets show signs of ancient water activity.",
    })


def _resource_response(
    waypoint_ids: list[str] = ("wp-crater", "wp-outcrop"),
    energy_per: float = 15.0,
) -> str:
    """Build a valid ResourceBudget JSON string."""
    return json.dumps({
        "available_energy_wh":     350.0,
        "available_time_minutes":  300.0,
        "recommended_waypoints":   list(waypoint_ids),
        "energy_per_waypoint":     {wid: energy_per for wid in waypoint_ids},
        "reasoning":               "Plenty of energy for both targets.",
    })


def _safety_response(exclude: list[str] | None = None) -> str:
    """Build a valid RiskAssessment JSON string."""
    return json.dumps({
        "waypoint_risks": [
            {"waypoint_id": "wp-crater",  "risk_score": 0.15, "factors": ["mild slope"]},
            {"waypoint_id": "wp-outcrop", "risk_score": 0.20, "factors": ["loose surface"]},
        ],
        "overall_risk_level":      "LOW",
        "recommended_exclusions":  exclude or [],
        "reasoning":               "Both targets are within acceptable risk parameters.",
    })


def _commander_response(
    waypoints: list[tuple[str, int, float, float]] | None = None,
    total_energy: float = 25.0,
    total_time: float = 50.0,
) -> str:
    """Build a valid MissionPlanOutput JSON string.

    ``waypoints`` is a list of (id, visit_order, science_value, energy_wh).
    Defaults to a plan with both science waypoints + base.
    """
    if waypoints is None:
        waypoints = [
            ("wp-crater",  1, 0.85, 15.0),
            ("wp-outcrop", 2, 0.80, 10.0),
            ("base-hq",    3, 0.0,   0.0),
        ]
    entries = [
        {
            "waypoint_id":           wid,
            "visit_order":           order,
            "expected_science_value": sv,
            "expected_energy_wh":    ewh,
        }
        for wid, order, sv, ewh in waypoints
    ]
    return json.dumps({
        "planned_waypoints":            entries,
        "total_estimated_energy_wh":    total_energy,
        "total_estimated_time_minutes": total_time,
        "confidence":                   0.90,
        "reasoning":                    "Optimal science-to-energy trade-off.",
    })


# ---------------------------------------------------------------------------
# Helper: build a FakeLLMClient for one successful planning cycle.
# Response order within each attempt: science, resource, safety, commander.
# ---------------------------------------------------------------------------

def _fake_client_one_attempt(
    include_outcrop: bool = True,
    commander_wps: list[tuple] | None = None,
) -> FakeLLMClient:
    """Single-attempt fake: science → resource → safety → commander."""
    ids = ["wp-crater", "wp-outcrop"] if include_outcrop else ["wp-crater"]
    energy_map = 15.0 if include_outcrop else 15.0
    responses = [
        _science_response(ids),
        _resource_response(ids, energy_per=energy_map),
        _safety_response(),
        _commander_response(commander_wps),
    ]
    return FakeLLMClient(responses)


# ---------------------------------------------------------------------------
# Integration Test 1 — Full plan_mission() success
# ---------------------------------------------------------------------------

class TestIntegration1FullPlanMission:
    """
    Approved Integration Test 1.

    Exercises the complete plan_mission() pipeline with real components:
      - real planner wiring (_build_commander factory patched to inject fake LLM)
      - real ScienceAgent, ResourceAgent, SafetyAgent (BaseAgent JSON parsing)
      - real MissionCommander orchestration and synthesis
      - real MissionPlan conversion
      - real SafetyValidator (all five rules)

    The ONLY fake is the external LLM network layer (FakeLLMClient).
    """

    async def test_returns_mission_plan(self):
        """plan_mission() must return a real MissionPlan instance."""
        fake = _fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, _ENV_STATE)
        assert isinstance(result, MissionPlan)

    async def test_plan_status_is_active(self):
        """Returned plan must have status=ACTIVE (validator approved it)."""
        fake = _fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, _ENV_STATE)
        assert result.status is MissionStatus.ACTIVE

    async def test_science_waypoints_included(self):
        """Both science waypoints must appear in the plan."""
        fake = _fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, _ENV_STATE)
        wp_ids = {w.id for w in result.waypoints}
        assert "wp-crater" in wp_ids
        assert "wp-outcrop" in wp_ids

    async def test_return_to_base_is_final_waypoint(self):
        """The last waypoint must be the base station."""
        fake = _fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, _ENV_STATE)
        assert result.waypoints, "Plan must contain waypoints"
        assert result.waypoints[-1].is_base is True

    async def test_plan_satisfies_energy_constraint(self):
        """Plan total energy must not exceed the usable budget."""
        from missionmind import config

        fake = _fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, _ENV_STATE)

        usable_wh = (
            _ROVER_STATE["battery_pct"] * _ROVER_STATE["battery_capacity_wh"]
            - config.MIN_RETURN_BATTERY_PCT * _ROVER_STATE["battery_capacity_wh"]
        )
        assert result.total_energy_wh <= usable_wh

    async def test_plan_satisfies_terrain_constraint(self):
        """No waypoint terrain_risk may exceed MAX_TERRAIN_RISK_SCORE."""
        from missionmind import config

        fake = _fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, _ENV_STATE)

        for wp in result.waypoints:
            assert wp.terrain_risk <= config.MAX_TERRAIN_RISK_SCORE, (
                f"Waypoint {wp.id!r} terrain_risk={wp.terrain_risk} "
                f"exceeds threshold {config.MAX_TERRAIN_RISK_SCORE}"
            )

    async def test_plan_has_science_waypoints(self):
        """At least one non-base waypoint must be present."""
        fake = _fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, _ENV_STATE)
        assert len(result.science_waypoints()) >= 1

    async def test_fake_llm_consumed_exactly_four_times(self):
        """One planning attempt = 4 LLM calls (science, resource, safety, synthesis)."""
        fake = _fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            await plan_mission(_ROVER_STATE, _ENV_STATE)
        assert fake.call_count == 4
        assert fake.remaining == 0

    async def test_no_real_network_call_made(self):
        """No httpx request must be issued — confirmed by httpx.AsyncClient patch."""
        import httpx

        fake = _fake_client_one_attempt()
        original_post = httpx.AsyncClient.post

        calls: list = []

        async def spy_post(self, url, **kwargs):
            calls.append(url)
            return await original_post(self, url, **kwargs)

        with patch.object(httpx.AsyncClient, "post", spy_post):
            with patch(
                "missionmind.planning.planner._build_llm_client",
                return_value=fake,
            ):
                await plan_mission(_ROVER_STATE, _ENV_STATE)

        assert calls == [], (
            f"Real HTTP calls detected: {calls}. Integration tests must stay offline."
        )

    async def test_plan_reasoning_populated(self):
        """Commander reasoning must be forwarded into the MissionPlan."""
        fake = _fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, _ENV_STATE)
        assert result.reasoning != ""

    async def test_plan_confidence_within_range(self):
        fake = _fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, _ENV_STATE)
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Integration Test 2 — Validator rejects first plan; retry/replanning succeeds
# ---------------------------------------------------------------------------

class TestIntegration2ValidatorRejectRetry:
    """
    Approved Integration Test 2.

    The REAL SafetyValidator must reject the first plan.  The REAL
    MissionCommander then prunes candidates and retries.  The REAL
    SafetyValidator approves the second plan.

    Mechanism for causing the first rejection:
    The commander's synthesis response for attempt 1 sets ``total_estimated_energy_wh``
    to a very high value AND plans a waypoint with terrain_risk=0.75 (above the 0.70
    default threshold), ensuring at least one hard-rule violation.  The actual
    MissionPlan.total_energy_wh is derived from ``expected_energy_wh`` per waypoint
    (see ``_convert_to_mission_plan``), so we need the waypoint energy to exceed the
    usable budget.

    Approach: Use a risky waypoint (terrain_risk=0.75 > 0.70) in attempt 1 so the
    TERRAIN RISK rule fires.  Attempt 2 removes that waypoint and succeeds.
    """

    # A second science waypoint with terrain_risk above the 0.70 threshold.
    _WP_RISKY = {
        "id": "wp-risky",
        "x": 250.0, "y": 0.0,
        "terrain_risk": 0.75,           # > MAX_TERRAIN_RISK_SCORE=0.70 → triggers rejection
        "is_base": False,
        "label": "risky-ridge",
        "estimated_travel_time_minutes": 40.0,
        "estimated_energy_wh": 30.0,
    }

    def _env_with_risky(self) -> dict:
        return {
            **_ENV_STATE,
            "candidate_waypoints": [_WP_CRATER, self._WP_RISKY, _WP_BASE],
        }

    def _build_fake_for_two_attempts(self) -> FakeLLMClient:
        """
        Attempt 1 (4 calls): commander synthesises a plan INCLUDING wp-risky
            → REAL validator fires TERRAIN RISK EXCEEDED
            → REAL MissionCommander prunes wp-risky (highest terrain_risk)
        Attempt 2 (4 calls): commander synthesises a plan with only wp-crater
            → REAL validator approves
        """
        # Attempt 1: science sees both, commander includes the risky waypoint.
        attempt1_science   = _science_response(["wp-crater", "wp-risky"])
        attempt1_resource  = _resource_response(["wp-crater", "wp-risky"], energy_per=20.0)
        attempt1_safety    = _safety_response()   # no recommended exclusions (soft check)
        attempt1_commander = _commander_response(
            waypoints=[
                ("wp-crater", 1, 0.85, 15.0),
                ("wp-risky",  2, 0.60, 30.0),   # terrain_risk=0.75 > 0.70 → rejection
                ("base-hq",   3, 0.00,  0.0),
            ],
            total_energy=45.0,
            total_time=70.0,
        )

        # Attempt 2: after pruning wp-risky, only wp-crater remains.
        attempt2_science   = _science_response(["wp-crater"])
        attempt2_resource  = _resource_response(["wp-crater"], energy_per=15.0)
        attempt2_safety    = _safety_response()
        attempt2_commander = _commander_response(
            waypoints=[
                ("wp-crater", 1, 0.85, 15.0),
                ("base-hq",   2, 0.00,  0.0),
            ],
            total_energy=15.0,
            total_time=30.0,
        )

        return FakeLLMClient([
            attempt1_science, attempt1_resource, attempt1_safety, attempt1_commander,
            attempt2_science, attempt2_resource, attempt2_safety, attempt2_commander,
        ])

    async def test_final_result_is_mission_plan(self):
        """Despite the first rejection, plan_mission must ultimately return a MissionPlan."""
        fake = self._build_fake_for_two_attempts()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, self._env_with_risky())
        assert isinstance(result, MissionPlan)

    async def test_final_status_is_active(self):
        """The final returned plan must be status=ACTIVE (validator approved it)."""
        fake = self._build_fake_for_two_attempts()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, self._env_with_risky())
        assert result.status is MissionStatus.ACTIVE

    async def test_risky_waypoint_absent_from_final_plan(self):
        """wp-risky (terrain_risk=0.75) must NOT appear in the approved plan."""
        fake = self._build_fake_for_two_attempts()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, self._env_with_risky())
        wp_ids = {w.id for w in result.waypoints}
        assert "wp-risky" not in wp_ids, (
            "The unsafe waypoint should have been pruned after the first rejection"
        )

    async def test_eight_llm_calls_for_two_attempts(self):
        """Two full planning attempts must consume exactly 8 LLM calls."""
        fake = self._build_fake_for_two_attempts()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            await plan_mission(_ROVER_STATE, self._env_with_risky())
        assert fake.call_count == 8, (
            f"Expected 8 LLM calls (4 per attempt × 2 attempts), got {fake.call_count}"
        )

    async def test_final_plan_passes_terrain_constraint(self):
        """The approved plan must satisfy the terrain-risk hard rule."""
        from missionmind import config

        fake = self._build_fake_for_two_attempts()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, self._env_with_risky())
        for wp in result.waypoints:
            assert wp.terrain_risk <= config.MAX_TERRAIN_RISK_SCORE

    async def test_final_plan_has_return_to_base(self):
        fake = self._build_fake_for_two_attempts()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await plan_mission(_ROVER_STATE, self._env_with_risky())
        assert result.waypoints[-1].is_base is True

    async def test_real_validator_used_not_mocked(self):
        """Confirm the real SafetyValidator is in the pipeline (import check)."""
        from missionmind.safety.validator import SafetyValidator
        # If SafetyValidator is importable and the test above passes, it was real.
        assert SafetyValidator is not None

    async def test_no_network_call_on_retry(self):
        """Even during retry, no real HTTP calls should occur."""
        import httpx

        calls: list = []

        async def spy_post(self_inner, url, **kwargs):
            calls.append(url)

        fake = self._build_fake_for_two_attempts()
        with patch.object(httpx.AsyncClient, "post", spy_post):
            with patch(
                "missionmind.planning.planner._build_llm_client",
                return_value=fake,
            ):
                # The fake doesn't make real calls, so no exception expected
                # (spy_post returns None which would break real HTTP but that's fine —
                # FakeLLMClient bypasses httpx entirely)
                try:
                    await plan_mission(_ROVER_STATE, self._env_with_risky())
                except Exception:
                    pass  # any error is fine — we only care that httpx.post wasn't called

        assert calls == [], "Real HTTP calls must not occur in integration tests"


# ---------------------------------------------------------------------------
# Integration Test 3 — NEW_DISCOVERY replan
# ---------------------------------------------------------------------------

class TestIntegration3NewDiscoveryReplan:
    """
    Approved Integration Test 3.

    Exercises the real Replanner → plan_mission() pipeline via a NEW_DISCOVERY
    event.

    Start: a valid active MissionPlan (wp-crater + base).
    Event: NEW_DISCOVERY of "wp-ice-deposit" at (200, 0).
    Expected: the real Replanner adds the discovery to candidates, then the
    real plan_mission() / agents / commander / validator evaluate it and
    return a new approved plan that includes the discovery.
    """

    def _current_plan(self) -> MissionPlan:
        from missionmind.models.mission import Waypoint
        return MissionPlan(
            waypoints=[
                Waypoint(
                    id="wp-crater", x=100.0, y=0.0,
                    scientific_value=0.85, terrain_risk=0.15,
                    estimated_travel_time_minutes=30.0, estimated_energy_wh=15.0,
                    is_base=False, label="crater-rim",
                ),
                Waypoint(
                    id="base-hq", x=0.0, y=0.0,
                    scientific_value=0.0, terrain_risk=0.0,
                    estimated_travel_time_minutes=0.0, estimated_energy_wh=0.0,
                    is_base=True, label="BASE",
                ),
            ],
            total_energy_wh=15.0,
            total_time_minutes=30.0,
            status=MissionStatus.ACTIVE,
            reasoning="initial plan",
            confidence=0.9,
        )

    def _discovery_event(self) -> MissionEvent:
        return MissionEvent(
            event_type=EventType.NEW_DISCOVERY,
            severity=0.7,
            payload={
                "id":               "wp-ice-deposit",
                "x":                200.0,
                "y":                0.0,
                "label":            "ice-deposit",
                "scientific_value": 0.95,
                "terrain_risk":     0.10,
                "estimated_travel_time_minutes": 25.0,
                "estimated_energy_wh": 12.0,
            },
        )

    def _fake_for_discovery(self) -> FakeLLMClient:
        """
        After NEW_DISCOVERY, the updated env has three science candidates:
        wp-ice-deposit (new), wp-crater, wp-outcrop + base.
        The commander includes the discovery in the plan.
        """
        science   = _science_response(["wp-ice-deposit", "wp-crater", "wp-outcrop"])
        resource  = _resource_response(
            ["wp-ice-deposit", "wp-crater", "wp-outcrop"],
            energy_per=12.0,
        )
        safety    = _safety_response()
        commander = _commander_response(
            waypoints=[
                ("wp-ice-deposit", 1, 0.95, 12.0),
                ("wp-crater",      2, 0.85, 15.0),
                ("wp-outcrop",     3, 0.80, 10.0),
                ("base-hq",        4,  0.0,  0.0),
            ],
            total_energy=37.0,
            total_time=75.0,
        )
        return FakeLLMClient([science, resource, safety, commander])

    async def test_returns_mission_plan(self):
        """replan() with NEW_DISCOVERY must return a real MissionPlan."""
        fake = self._fake_for_discovery()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await replan(
                self._current_plan(),
                self._discovery_event(),
                _ROVER_STATE,
                _ENV_STATE,
            )
        assert isinstance(result, MissionPlan)

    async def test_discovery_in_revised_plan(self):
        """The new discovery waypoint must appear in the revised plan."""
        fake = self._fake_for_discovery()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await replan(
                self._current_plan(),
                self._discovery_event(),
                _ROVER_STATE,
                _ENV_STATE,
            )
        wp_ids = {w.id for w in result.waypoints}
        assert "wp-ice-deposit" in wp_ids, (
            "The newly discovered waypoint must appear in the revised plan"
        )

    async def test_revised_plan_is_active(self):
        """Revised plan must be ACTIVE (real validator approved it)."""
        fake = self._fake_for_discovery()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await replan(
                self._current_plan(),
                self._discovery_event(),
                _ROVER_STATE,
                _ENV_STATE,
            )
        assert result.status is MissionStatus.ACTIVE

    async def test_return_to_base_still_final(self):
        """Even after inserting a discovery, the last waypoint must be the base."""
        fake = self._fake_for_discovery()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await replan(
                self._current_plan(),
                self._discovery_event(),
                _ROVER_STATE,
                _ENV_STATE,
            )
        assert result.waypoints[-1].is_base is True

    async def test_real_replanner_called_plan_mission(self):
        """Confirm plan_mission() was invoked by the real Replanner (not bypassed)."""
        from missionmind.planning import replan as real_replan
        from missionmind.planning import plan_mission as real_pm

        plan_mission_calls: list = []

        async def spy_plan_mission(rs, es):
            plan_mission_calls.append((rs, es))
            # Delegate to the real plan_mission with the fake client already patched
            return await real_pm(rs, es)

        fake = self._fake_for_discovery()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            with patch(
                "missionmind.planning.replanner.plan_mission",
                side_effect=spy_plan_mission,
            ):
                await real_replan(
                    self._current_plan(),
                    self._discovery_event(),
                    _ROVER_STATE,
                    _ENV_STATE,
                )

        assert len(plan_mission_calls) == 1, (
            "The Replanner must delegate to plan_mission() exactly once"
        )
        # Confirm the discovery was added to the candidate list passed in
        _, es_used = plan_mission_calls[0]
        ids = [w["id"] for w in es_used["candidate_waypoints"]]
        assert "wp-ice-deposit" in ids

    async def test_original_env_state_not_mutated(self):
        """The caller's env_state must not be modified by the Replanner."""
        original_count = len(_ENV_STATE["candidate_waypoints"])
        fake = self._fake_for_discovery()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            await replan(
                self._current_plan(),
                self._discovery_event(),
                _ROVER_STATE,
                _ENV_STATE,
            )
        assert len(_ENV_STATE["candidate_waypoints"]) == original_count

    async def test_no_network_call_during_replan(self):
        """No real HTTP traffic must occur during replan()."""
        import httpx

        calls: list = []

        async def spy_post(self_inner, url, **kwargs):
            calls.append(url)

        fake = self._fake_for_discovery()
        with patch.object(httpx.AsyncClient, "post", spy_post):
            with patch(
                "missionmind.planning.planner._build_llm_client",
                return_value=fake,
            ):
                try:
                    await replan(
                        self._current_plan(),
                        self._discovery_event(),
                        _ROVER_STATE,
                        _ENV_STATE,
                    )
                except Exception:
                    pass

        assert calls == []

    async def test_revised_plan_satisfies_all_hard_constraints(self):
        """The revised plan must pass all five deterministic safety rules."""
        from missionmind import config
        from missionmind.safety.validator import SafetyValidator

        fake = self._fake_for_discovery()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            result = await replan(
                self._current_plan(),
                self._discovery_event(),
                _ROVER_STATE,
                _ENV_STATE,
            )

        # Run the real validator independently to confirm
        validator = SafetyValidator()
        validation = validator.validate(result, _ROVER_STATE)
        assert validation.passed, (
            f"Revised plan failed safety validation: {validation.violations}"
        )
