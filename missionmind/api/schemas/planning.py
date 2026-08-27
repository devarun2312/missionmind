"""
HTTP request schemas for POST /api/mission/plan.

These Pydantic v2 models translate an incoming JSON request body into the
plain Python dicts that ``plan_mission()`` requires.  They mirror the exact
keys defined in ``planner.ROVER_STATE_KEYS`` and ``planner.ENV_STATE_KEYS``.

Battery convention (IMPORTANT)
-------------------------------
``battery_pct`` is a **fraction**, not a percentage:
- ``0.85`` means 85 %
- ``0.10`` means 10 %

Values outside [0.0, 1.0] are rejected with HTTP 422.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Rover state
# ---------------------------------------------------------------------------

class RoverStateInput(BaseModel):
    """Current rover status.  Maps 1-to-1 with ``planner.ROVER_STATE_KEYS``.

    All six keys are required; there are no optional fields because the
    backend will raise a ``ValueError`` for any that are missing.
    """

    battery_pct: float = Field(
        ge=0.0, le=1.0,
        description="Current battery charge as a fraction (0.0–1.0).  "
                    "0.85 means 85 %, NOT 85.",
    )
    battery_capacity_wh: float = Field(
        gt=0.0,
        description="Total battery capacity in watt-hours.",
    )
    position_x: float = Field(
        description="Rover X coordinate in metres from base station.",
    )
    position_y: float = Field(
        description="Rover Y coordinate in metres from base station.",
    )
    rover_speed_mps: float = Field(
        gt=0.0,
        description="Nominal driving speed in metres per second.",
    )
    power_consumption_w: float = Field(
        gt=0.0,
        description="Power draw during driving in watts.",
    )


# ---------------------------------------------------------------------------
# Waypoint input
# ---------------------------------------------------------------------------

class WaypointInput(BaseModel):
    """A single candidate waypoint the rover may visit.

    Matches the dictionary keys read by the MissionCommander, ScienceAgent,
    ResourceAgent, and SafetyAgent.  ``scientific_value`` is optional because
    the AI agents assign it themselves; pre-seeding it is allowed but not
    required.
    """

    id: str = Field(description="Unique waypoint identifier.")
    x: float = Field(description="X coordinate in metres from base.")
    y: float = Field(description="Y coordinate in metres from base.")
    terrain_risk: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="Hazard level 0.0 (safe) – 1.0 (impassable).",
    )
    is_base: bool = Field(
        default=False,
        description="True if this waypoint is the home base station.",
    )
    label: str = Field(default="", description="Human-readable waypoint name.")
    estimated_travel_time_minutes: float = Field(
        ge=0.0, default=0.0,
        description="Expected travel time from the previous waypoint in minutes.",
    )
    estimated_energy_wh: float = Field(
        ge=0.0, default=0.0,
        description="Expected energy consumption for the drive in watt-hours.",
    )
    # Optional — agents may use a pre-scored value if present; they will
    # overwrite it with their own assessment regardless.
    scientific_value: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description="Pre-assessed scientific interest (0.0–1.0).  "
                    "AI agents will assign their own scores.",
    )


# ---------------------------------------------------------------------------
# Environment state sub-models
# ---------------------------------------------------------------------------

class WeatherForecast(BaseModel):
    """Weather conditions forwarded to the SafetyAgent."""

    dust_storm_probability: float = Field(ge=0.0, le=1.0, default=0.1)
    temperature_min_c: float = -60.0
    temperature_max_c: float = 20.0
    wind_speed_mps: float = Field(ge=0.0, default=5.0)
    forecast_hours: int = Field(ge=1, default=8)


class CommWindow(BaseModel):
    """A single communication window with mission control."""

    start_utc: str = Field(description="ISO-8601 UTC datetime string.")
    duration_minutes: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

class EnvStateInput(BaseModel):
    """Current environment and mission context.
    Maps 1-to-1 with ``planner.ENV_STATE_KEYS``.
    """

    candidate_waypoints: list[WaypointInput] = Field(
        min_length=1,
        description="Waypoints the rover may visit.  Must include at least "
                    "one waypoint with ``is_base=True``.",
    )
    weather_forecast: WeatherForecast
    comm_windows: list[CommWindow] = Field(default_factory=list)
    terrain_map: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional terrain data keyed by waypoint id.",
    )
    mission_objectives: list[str] = Field(
        min_length=1,
        description="Plain-text mission goal strings.",
    )


# ---------------------------------------------------------------------------
# Top-level plan request
# ---------------------------------------------------------------------------

class PlanRequest(BaseModel):
    """Request body for ``POST /api/mission/plan``."""

    rover_state: RoverStateInput
    env_state: EnvStateInput


# ---------------------------------------------------------------------------
# Health response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response body for ``GET /api/health``."""

    status: str = "ok"
    backend: str = "missionmind"
    version: str = "0.1.0"
