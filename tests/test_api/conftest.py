"""
Shared fixtures for MissionMind API tests.

All API tests run fully offline:
- The FastAPI app is exercised via httpx.AsyncClient + ASGITransport
  (no real uvicorn process, no network sockets).
- ``missionmind.planning.planner._build_llm_client`` is patched with
  ``FakeLLMClient`` for tests that exercise the full planning pipeline,
  so no real IBM watsonx calls occur.

FakeLLMClient is a self-contained FIFO queue of canned JSON strings,
identical in behaviour to the one used in ``tests/test_integration.py``
but defined here independently to avoid cross-test-module coupling.
"""

from __future__ import annotations

import json
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import patch

import pytest
import httpx

from missionmind.agents.client import LLMResponse
from missionmind.api.app import create_app


# ---------------------------------------------------------------------------
# FakeLLMClient — deterministic stand-in for WatsonxClient
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """FIFO queue of canned JSON strings returned as LLM responses.

    Each ``complete()`` call pops the next string from the queue and wraps
    it in an ``LLMResponse``.  An ``AssertionError`` is raised if the queue
    is exhausted unexpectedly.
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
            f"FakeLLMClient: queue exhausted on call #{self.call_count + 1}."
        )
        self.call_count += 1
        return LLMResponse(content=self._queue.popleft(), model="fake-model")


# ---------------------------------------------------------------------------
# Canned JSON builders — one-attempt happy path
# ---------------------------------------------------------------------------

def _science_json(wids: list[str]) -> str:
    return json.dumps({
        "scored_targets": [
            {"waypoint_id": w, "scientific_value": 0.85, "justification": "good"}
            for w in wids
        ],
        "priority_order": wids,
        "reasoning": "promising targets",
    })


def _resource_json(wids: list[str], energy: float = 15.0) -> str:
    return json.dumps({
        "available_energy_wh": 350.0,
        "available_time_minutes": 300.0,
        "recommended_waypoints": wids,
        "energy_per_waypoint": {w: energy for w in wids},
        "reasoning": "within budget",
    })


def _safety_json() -> str:
    return json.dumps({
        "waypoint_risks": [],
        "overall_risk_level": "LOW",
        "recommended_exclusions": [],
        "reasoning": "safe",
    })


def _commander_json(
    entries: list[tuple[str, int, float, float]],
    total_energy: float = 15.0,
    total_time: float = 30.0,
) -> str:
    """entries: [(waypoint_id, visit_order, science_value, energy_wh)]"""
    return json.dumps({
        "planned_waypoints": [
            {
                "waypoint_id": wid,
                "visit_order": order,
                "expected_science_value": sv,
                "expected_energy_wh": ewh,
            }
            for wid, order, sv, ewh in entries
        ],
        "total_estimated_energy_wh": total_energy,
        "total_estimated_time_minutes": total_time,
        "confidence": 0.9,
        "reasoning": "optimal plan",
    })


def make_fake_client_one_attempt(
    science_wids: list[str] | None = None,
    commander_entries: list[tuple[str, int, float, float]] | None = None,
) -> FakeLLMClient:
    """Build a FakeLLMClient pre-loaded for a single successful planning cycle.

    Response order: science → resource → safety → commander (4 total).
    """
    wids = science_wids or ["wp-crater", "wp-base"]
    science_ids = [w for w in wids if w != "wp-base"]
    entries = commander_entries or [
        ("wp-crater", 1, 0.85, 15.0),
        ("wp-base",   2, 0.0,   0.0),
    ]
    return FakeLLMClient([
        _science_json(science_ids),
        _resource_json(science_ids),
        _safety_json(),
        _commander_json(entries),
    ])


# ---------------------------------------------------------------------------
# Standard test inputs
# ---------------------------------------------------------------------------

VALID_ROVER_STATE: dict = {
    "battery_pct":         0.90,
    "battery_capacity_wh": 500.0,
    "position_x":          0.0,
    "position_y":          0.0,
    "rover_speed_mps":     0.5,
    "power_consumption_w": 50.0,
}

VALID_ENV_STATE: dict = {
    "candidate_waypoints": [
        {
            "id": "wp-crater",
            "x": 100.0, "y": 0.0,
            "terrain_risk": 0.15,
            "is_base": False,
            "label": "crater-rim",
            "estimated_travel_time_minutes": 30.0,
            "estimated_energy_wh": 15.0,
        },
        {
            "id": "wp-base",
            "x": 0.0, "y": 0.0,
            "terrain_risk": 0.0,
            "is_base": True,
            "label": "BASE",
            "estimated_travel_time_minutes": 0.0,
            "estimated_energy_wh": 0.0,
        },
    ],
    "weather_forecast": {
        "dust_storm_probability": 0.05,
        "temperature_min_c": -55.0,
        "temperature_max_c": 15.0,
        "wind_speed_mps": 3.0,
        "forecast_hours": 8,
    },
    "comm_windows": [{"start_utc": "2025-06-01T10:00:00Z", "duration_minutes": 45}],
    "terrain_map": {},
    "mission_objectives": ["search for biosignatures"],
}

VALID_PLAN_BODY: dict = {
    "rover_state": VALID_ROVER_STATE,
    "env_state": VALID_ENV_STATE,
}


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Fresh FastAPI application instance for each test."""
    return create_app()


@pytest.fixture
async def client(app) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client wired directly to the FastAPI app (no sockets)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
def fake_llm():
    """Patch _build_llm_client with a single-attempt FakeLLMClient."""
    fake = make_fake_client_one_attempt()
    with patch(
        "missionmind.planning.planner._build_llm_client",
        return_value=fake,
    ):
        yield fake
