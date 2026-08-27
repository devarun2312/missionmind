"""
Tests for POST /api/mission/replan.

All tests run fully offline — no real uvicorn process, no network calls.
Deterministic events (RETURN_TO_BASE, critical BATTERY_FAILURE) require no
LLM patch because they never call _build_llm_client.
Non-deterministic events (NEW_DISCOVERY, COMM_LOSS, TERRAIN_HAZARD, and
non-critical BATTERY_FAILURE) patch _build_llm_client via make_fake_client_one_attempt.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from missionmind.agents.base_agent import AgentResponseError
from missionmind.agents.mission_commander import PlanningFailedError
from missionmind.models.mission import MissionStatus

from tests.test_api.conftest import (
    VALID_ROVER_STATE,
    VALID_ENV_STATE,
    FakeLLMClient,
    make_fake_client_one_attempt,
)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# A minimal active MissionPlan dict to serve as current_plan in replan requests.
_BASE_WAYPOINT = {
    "id": "wp-base",
    "x": 0.0,
    "y": 0.0,
    "scientific_value": 0.0,
    "terrain_risk": 0.0,
    "estimated_travel_time_minutes": 0.0,
    "estimated_energy_wh": 0.0,
    "is_base": True,
    "label": "BASE",
}

_CRATER_WAYPOINT = {
    "id": "wp-crater",
    "x": 100.0,
    "y": 0.0,
    "scientific_value": 0.85,
    "terrain_risk": 0.15,
    "estimated_travel_time_minutes": 30.0,
    "estimated_energy_wh": 15.0,
    "is_base": False,
    "label": "crater-rim",
}

_CURRENT_PLAN: dict = {
    "plan_id": "test-plan-001",
    "waypoints": [_CRATER_WAYPOINT, _BASE_WAYPOINT],
    "total_energy_wh": 15.0,
    "total_time_minutes": 30.0,
    "status": "ACTIVE",
    "reasoning": "nominal plan",
    "confidence": 0.9,
}


def _replan_body(event: dict) -> dict:
    """Compose a full replan request body around a given event dict."""
    return {
        "current_plan": _CURRENT_PLAN,
        "event": event,
        "rover_state": VALID_ROVER_STATE,
        "env_state": VALID_ENV_STATE,
    }


# ---------------------------------------------------------------------------
# Deterministic RETURN_TO_BASE — no LLM call required
# ---------------------------------------------------------------------------

class TestReplanReturnToBase:
    async def test_returns_200(self, client):
        event = {"event_type": "RETURN_TO_BASE", "severity": 1.0, "payload": {}}
        resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 200

    async def test_status_is_active(self, client):
        event = {"event_type": "RETURN_TO_BASE", "severity": 1.0, "payload": {}}
        body = (await client.post("/api/mission/replan", json=_replan_body(event))).json()
        assert body["status"] == MissionStatus.ACTIVE.value

    async def test_only_base_waypoint_in_plan(self, client):
        event = {"event_type": "RETURN_TO_BASE", "severity": 1.0, "payload": {}}
        body = (await client.post("/api/mission/replan", json=_replan_body(event))).json()
        assert len(body["waypoints"]) == 1
        assert body["waypoints"][0]["is_base"] is True

    async def test_content_type_is_json(self, client):
        event = {"event_type": "RETURN_TO_BASE", "severity": 1.0, "payload": {}}
        resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Deterministic critical BATTERY_FAILURE — no LLM call required
# ---------------------------------------------------------------------------

class TestReplanCriticalBatteryFailure:
    async def test_critical_battery_returns_200(self, client):
        # 0.04 is below CRITICAL_BATTERY_PCT (0.10)
        event = {
            "event_type": "BATTERY_FAILURE",
            "severity": 1.0,
            "payload": {"battery_pct": 0.04},
        }
        resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 200

    async def test_critical_battery_only_base_waypoint(self, client):
        event = {
            "event_type": "BATTERY_FAILURE",
            "severity": 1.0,
            "payload": {"battery_pct": 0.04},
        }
        body = (await client.post("/api/mission/replan", json=_replan_body(event))).json()
        assert len(body["waypoints"]) == 1
        assert body["waypoints"][0]["is_base"] is True

    async def test_critical_battery_plan_is_active(self, client):
        event = {
            "event_type": "BATTERY_FAILURE",
            "severity": 1.0,
            "payload": {"battery_pct": 0.04},
        }
        body = (await client.post("/api/mission/replan", json=_replan_body(event))).json()
        assert body["status"] == MissionStatus.ACTIVE.value


# ---------------------------------------------------------------------------
# Non-critical BATTERY_FAILURE — triggers full LLM pipeline
# ---------------------------------------------------------------------------

class TestReplanNonCriticalBatteryFailure:
    async def test_returns_200_and_active_plan(self, client):
        fake = make_fake_client_one_attempt()
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            # 0.50 is above CRITICAL_BATTERY_PCT (0.10) — goes to LLM pipeline
            event = {
                "event_type": "BATTERY_FAILURE",
                "severity": 0.4,
                "payload": {"battery_pct": 0.50},
            }
            resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 200
        assert resp.json()["status"] == MissionStatus.ACTIVE.value


# ---------------------------------------------------------------------------
# NEW_DISCOVERY — triggers full LLM pipeline
# ---------------------------------------------------------------------------

class TestReplanNewDiscovery:
    async def test_returns_200(self, client):
        fake = make_fake_client_one_attempt(
            science_wids=["discovery-wp", "wp-crater", "wp-base"],
            commander_entries=[
                ("discovery-wp", 1, 0.9,  12.0),
                ("wp-crater",    2, 0.85, 15.0),
                ("wp-base",      3, 0.0,   0.0),
            ],
        )
        with patch(
            "missionmind.planning.planner._build_llm_client",
            return_value=fake,
        ):
            event = {
                "event_type": "NEW_DISCOVERY",
                "severity": 0.5,
                "payload": {
                    "id": "discovery-wp",
                    "x": 200.0,
                    "y": 50.0,
                    "label": "ice-pocket",
                    "scientific_value": 0.9,
                },
            }
            resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == MissionStatus.ACTIVE.value
        assert len(body["waypoints"]) >= 1


# ---------------------------------------------------------------------------
# Validation / 422 tests
# ---------------------------------------------------------------------------

class TestReplanValidation:
    async def test_missing_current_plan_returns_422(self, client):
        resp = await client.post(
            "/api/mission/replan",
            json={
                "event": {"event_type": "RETURN_TO_BASE", "severity": 1.0},
                "rover_state": VALID_ROVER_STATE,
                "env_state": VALID_ENV_STATE,
            },
        )
        assert resp.status_code == 422

    async def test_missing_event_returns_422(self, client):
        resp = await client.post(
            "/api/mission/replan",
            json={
                "current_plan": _CURRENT_PLAN,
                "rover_state": VALID_ROVER_STATE,
                "env_state": VALID_ENV_STATE,
            },
        )
        assert resp.status_code == 422

    async def test_invalid_event_type_returns_422(self, client):
        event = {"event_type": "NOT_A_REAL_EVENT", "severity": 0.5}
        resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 422

    async def test_severity_above_1_returns_422(self, client):
        event = {"event_type": "RETURN_TO_BASE", "severity": 1.5}
        resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 422

    async def test_severity_below_0_returns_422(self, client):
        event = {"event_type": "RETURN_TO_BASE", "severity": -0.1}
        resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error-mapping tests (502, 503, 500)
# ---------------------------------------------------------------------------

class TestReplanErrorMapping:
    async def test_agent_response_error_returns_502(self, client):
        err = AgentResponseError("bad JSON from LLM")
        with patch(
            "missionmind.api.routes.replanning.replan",
            side_effect=err,
        ):
            event = {"event_type": "RETURN_TO_BASE", "severity": 1.0, "payload": {}}
            resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 502
        body = resp.json()
        assert body["detail"]["error"] == "ai_response_error"

    async def test_planning_failed_error_returns_503(self, client):
        err = PlanningFailedError("planning failed", attempts=3, violations=["max_terrain_risk_exceeded"])
        with patch(
            "missionmind.api.routes.replanning.replan",
            side_effect=err,
        ):
            event = {"event_type": "RETURN_TO_BASE", "severity": 1.0, "payload": {}}
            resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "planning_failed"
        assert body["detail"]["attempts"] == 3

    async def test_value_error_returns_400(self, client):
        with patch(
            "missionmind.api.routes.replanning.replan",
            side_effect=ValueError("unknown event type"),
        ):
            event = {"event_type": "RETURN_TO_BASE", "severity": 1.0, "payload": {}}
            resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 400
        body = resp.json()
        assert body["detail"]["error"] == "invalid_request"

    async def test_unexpected_exception_returns_500(self, client):
        with patch(
            "missionmind.api.routes.replanning.replan",
            side_effect=RuntimeError("unexpected crash"),
        ):
            event = {"event_type": "RETURN_TO_BASE", "severity": 1.0, "payload": {}}
            resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"]["error"] == "internal_error"

    async def test_agent_response_error_not_caught_as_value_error(self, client):
        """AgentResponseError must map to 502, NOT 400, even though it's a ValueError subclass."""
        err = AgentResponseError("structural failure")
        with patch(
            "missionmind.api.routes.replanning.replan",
            side_effect=err,
        ):
            event = {"event_type": "RETURN_TO_BASE", "severity": 1.0, "payload": {}}
            resp = await client.post("/api/mission/replan", json=_replan_body(event))
        assert resp.status_code == 502
