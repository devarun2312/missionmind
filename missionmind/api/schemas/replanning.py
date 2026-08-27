"""
HTTP request schema for POST /api/mission/replan.

``MissionEvent`` in the backend is a Python ``@dataclass``, not a Pydantic
model.  ``MissionEventInput`` is the Pydantic representation used for HTTP
deserialization.  The route handler converts it into a real ``MissionEvent``
dataclass before calling ``replan()``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from missionmind.api.schemas.planning import EnvStateInput, RoverStateInput
from missionmind.models.events import EventType
from missionmind.models.mission import MissionPlan


# ---------------------------------------------------------------------------
# Event input
# ---------------------------------------------------------------------------

class MissionEventInput(BaseModel):
    """Pydantic representation of a ``MissionEvent`` for HTTP deserialization.

    Event payload conventions (forwarded verbatim to the Replanner):

    BATTERY_FAILURE  → ``{"battery_pct": 0.08}``
    COMM_LOSS        → ``{"safe_comm_radius_m": 500.0}``
    TERRAIN_HAZARD   → ``{"waypoint_id": "wp-123", "neighbour_ids": [...]}``
    NEW_DISCOVERY    → ``{"x": 450.0, "y": -120.0, "label": "ice-deposit",
                          "scientific_value": 0.9}``
    RETURN_TO_BASE   → ``{}``
    """

    event_type: EventType = Field(
        description="Type of mid-mission event.  Must be one of: "
                    + ", ".join(e.value for e in EventType),
    )
    severity: float = Field(
        ge=0.0, le=1.0,
        description="Event severity 0.0 (minor) – 1.0 (critical).",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-type-specific data.  See payload conventions above.",
    )


# ---------------------------------------------------------------------------
# Replan request
# ---------------------------------------------------------------------------

class ReplanRequest(BaseModel):
    """Request body for ``POST /api/mission/replan``."""

    current_plan: MissionPlan = Field(
        description="The MissionPlan currently being executed by the rover.",
    )
    event: MissionEventInput = Field(
        description="The mid-mission event that triggered replanning.",
    )
    rover_state: RoverStateInput = Field(
        description="Current rover status (same keys as the plan endpoint).",
    )
    env_state: EnvStateInput = Field(
        description="Current environment state (same keys as the plan endpoint).",
    )
