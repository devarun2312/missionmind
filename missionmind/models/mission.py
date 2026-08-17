"""
Core mission domain models.

These are the central data structures shared across every layer of the system:
agents, planner, replanner, validator, and the simulation/frontend integration.

All models are Pydantic BaseModel so they serialise cleanly to JSON.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# MissionStatus
# ---------------------------------------------------------------------------

class MissionStatus(str, Enum):
    """Lifecycle state of a MissionPlan.

    PENDING     Plan has been created but not yet approved or dispatched.
    ACTIVE      Plan is currently being executed by the rover.
    REPLANNING  A mid-mission event has triggered a new planning cycle.
    ABORTED     Plan was rejected by the safety validator or abandoned.
    COMPLETE    All waypoints visited and rover has returned to base.
    """

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    REPLANNING = "REPLANNING"
    ABORTED = "ABORTED"
    COMPLETE = "COMPLETE"


# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------

class Waypoint(BaseModel):
    """A single navigation target on the Martian surface.

    Attributes
    ----------
    id:
        Unique identifier for this waypoint (auto-generated UUID if not given).
    x, y:
        2-D surface coordinates in metres relative to the base station.
    scientific_value:
        AI-assessed scientific interest of this location, 0.0 (none) to 1.0 (max).
    terrain_risk:
        Hazard level for traversal, 0.0 (safe) to 1.0 (impassable).
    estimated_travel_time_minutes:
        Expected time to drive to this waypoint from the previous one.
    estimated_energy_wh:
        Expected energy consumption for the drive in watt-hours.
    is_base:
        True if this waypoint represents the home base (return destination).
    label:
        Optional human-readable name, e.g. "crater-rim-A" or "BASE".
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    x: float
    y: float
    scientific_value: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    terrain_risk: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    estimated_travel_time_minutes: Annotated[float, Field(ge=0.0)] = 0.0
    estimated_energy_wh: Annotated[float, Field(ge=0.0)] = 0.0
    is_base: bool = False
    label: str = ""

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# MissionPlan
# ---------------------------------------------------------------------------

class MissionPlan(BaseModel):
    """A complete, ordered mission plan produced by the MissionCommander.

    Attributes
    ----------
    plan_id:
        Unique identifier for this plan (auto-generated UUID).
    waypoints:
        Ordered list of waypoints the rover will visit.  The last entry
        MUST be a waypoint with ``is_base=True``.
    total_energy_wh:
        Sum of ``estimated_energy_wh`` across all waypoints.
    total_time_minutes:
        Sum of ``estimated_travel_time_minutes`` across all waypoints.
    status:
        Current lifecycle state of the plan.
    created_at:
        UTC timestamp of plan creation.
    reasoning:
        Free-text explanation of why this plan was chosen (from the LLM).
    confidence:
        Commander's self-assessed confidence in the plan, 0.0 – 1.0.
    """

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    waypoints: list[Waypoint] = Field(default_factory=list)
    total_energy_wh: Annotated[float, Field(ge=0.0)] = 0.0
    total_time_minutes: Annotated[float, Field(ge=0.0)] = 0.0
    status: MissionStatus = MissionStatus.PENDING
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    reasoning: str = ""
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    @field_validator("waypoints")
    @classmethod
    def _waypoints_must_be_list(cls, v: list[Waypoint]) -> list[Waypoint]:
        # Pydantic already enforces the type; this exists as an extension point.
        return v

    def science_waypoints(self) -> list[Waypoint]:
        """Return all non-base waypoints (those with scientific objectives)."""
        return [w for w in self.waypoints if not w.is_base]

    def has_return_waypoint(self) -> bool:
        """True if the final waypoint is the base station."""
        return bool(self.waypoints) and self.waypoints[-1].is_base

    model_config = {"frozen": False}
