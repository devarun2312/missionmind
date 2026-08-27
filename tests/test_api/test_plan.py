"""
Tests for POST /api/mission/plan.

All tests run fully offline — no real uvicorn process, no network calls.
``_build_llm_client`` is patched via the ``fake_llm`` fixture in conftest.py.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from missionmind.agents.base_agent import AgentResponseError
from missionmind.agents.mission_commander import PlanningFailedError
from missionmind.models.mission import MissionStatus

from tests.test_api.conftest import (
    VALID_PLAN_BODY,
    VALID_ROVER_STATE,
    VALID_ENV_STATE,
    FakeLLMClient,
    make_fake_client_one_attempt,
)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestPlanHappyPath:
    async def test_returns_200(self, client, fake_llm):
        resp = await client.post("/api/mission/plan", json=VALID_PLAN_BODY)
        assert resp.status_code == 200

    async def test_content_type_is_json(self, client, fake_llm):
        resp = await client.post("/api/mission/plan", json=VALID_PLAN_BODY)
        assert "application/json" in resp.headers["content-type"]

    async def test_response_is_valid_mission_plan(self, client, fake_llm):
        body = (await client.post("/api/mission/plan", json=VALID_PLAN_BODY)).json()
        assert "plan_id" in body
        assert "waypoints" in body
        assert isinstance(body["waypoints"], list)
        assert "status" in body
        assert "total_energy_wh" in body
        assert "total_time_minutes" in body
        assert "confidence" in body

    async def test_status_is_active(self, client, fake_llm):
        body = (await client.post("/api/mission/plan", json=VALID_PLAN_BODY)).json()
        assert body["status"] == MissionStatus.ACTIVE.value

    async def test_has_waypoints(self, client, fake_llm):
        body = (await client.post("/api/mission/plan", json=VALID_PLAN_BODY)).json()
        assert len(body["waypoints"]) >= 1

    async def test_last_waypoint_is_base(self, client, fake_llm):
        body = (await client.post("/api/mission/plan", json=VALID_PLAN_BODY)).json()
        assert body["waypoints"][-1]["is_base"] is True

    async def test_all_llm_responses_consumed(self, client, fake_llm):
        await client.post("/api/mission/plan", json=VALID_PLAN_BODY)
        assert fake_llm.call_count == 4  # science, resource, safety, commander


# ---------------------------------------------------------------------------
# Validation / 400 tests
# ---------------------------------------------------------------------------

class TestPlanValidation:
    async def test_missing_rover_state_returns_422(self, client):
        resp = await client.post(
            "/api/mission/plan",
            json={"env_state": VALID_ENV_STATE},
        )
        assert resp.status_code == 422

    async def test_missing_env_state_returns_422(self, client):
        resp = await client.post(
            "/api/mission/plan",
            json={"rover_state": VALID_ROVER_STATE},
        )
        assert resp.status_code == 422

    async def test_battery_pct_above_1_returns_422(self, client):
        bad_body = {
            **VALID_PLAN_BODY,
            "rover_state": {**VALID_ROVER_STATE, "battery_pct": 1.5},
        }
        resp = await client.post("/api/mission/plan", json=bad_body)
        assert resp.status_code == 422

    async def test_battery_pct_below_0_returns_422(self, client):
        bad_body = {
            **VALID_PLAN_BODY,
            "rover_state": {**VALID_ROVER_STATE, "battery_pct": -0.1},
        }
        resp = await client.post("/api/mission/plan", json=bad_body)
        assert resp.status_code == 422

    async def test_empty_body_returns_422(self, client):
        resp = await client.post("/api/mission/plan", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error-mapping tests (502, 503, 500)
# ---------------------------------------------------------------------------

class TestPlanErrorMapping:
    async def test_agent_response_error_returns_502(self, client, app):
        with patch(
            "missionmind.planning.planner._build_llm_client",
            side_effect=AgentResponseError("bad JSON from LLM"),
        ):
            resp = await client.post("/api/mission/plan", json=VALID_PLAN_BODY)
        assert resp.status_code == 502
        body = resp.json()
        assert body["detail"]["error"] == "ai_response_error"

    async def test_planning_failed_error_returns_503(self, client):
        err = PlanningFailedError("planning failed", attempts=3, violations=["terrain_risk_too_high"])
        with patch(
            "missionmind.api.routes.planning.plan_mission",
            side_effect=err,
        ):
            resp = await client.post("/api/mission/plan", json=VALID_PLAN_BODY)
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["error"] == "planning_failed"
        assert body["detail"]["attempts"] == 3
        assert "terrain_risk_too_high" in body["detail"]["violations"]

    async def test_value_error_returns_400(self, client):
        with patch(
            "missionmind.api.routes.planning.plan_mission",
            side_effect=ValueError("missing required key"),
        ):
            resp = await client.post("/api/mission/plan", json=VALID_PLAN_BODY)
        assert resp.status_code == 400
        body = resp.json()
        assert body["detail"]["error"] == "invalid_request"

    async def test_unexpected_exception_returns_500(self, client):
        with patch(
            "missionmind.api.routes.planning.plan_mission",
            side_effect=RuntimeError("disk full"),
        ):
            resp = await client.post("/api/mission/plan", json=VALID_PLAN_BODY)
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"]["error"] == "internal_error"

    async def test_agent_response_error_not_caught_as_value_error(self, client):
        """AgentResponseError must map to 502, NOT 400, even though it's a ValueError subclass."""
        err = AgentResponseError("structural LLM failure")
        with patch(
            "missionmind.api.routes.planning.plan_mission",
            side_effect=err,
        ):
            resp = await client.post("/api/mission/plan", json=VALID_PLAN_BODY)
        assert resp.status_code == 502
