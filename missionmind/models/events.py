"""
Mission event models.

Events are emitted by the simulation layer during a live mission and consumed
by the Replanner to trigger mid-mission plan adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """All event types that can trigger a replanning cycle.

    BATTERY_FAILURE   Battery level has dropped unexpectedly.
    COMM_LOSS         Communication with mission control has been lost.
    TERRAIN_HAZARD    A new terrain obstacle or hazard has been detected.
    NEW_DISCOVERY     An unexpected scientific target has been identified.
    RETURN_TO_BASE    An explicit command to return to base was issued.
    """

    BATTERY_FAILURE = "BATTERY_FAILURE"
    COMM_LOSS = "COMM_LOSS"
    TERRAIN_HAZARD = "TERRAIN_HAZARD"
    NEW_DISCOVERY = "NEW_DISCOVERY"
    RETURN_TO_BASE = "RETURN_TO_BASE"


# ---------------------------------------------------------------------------
# MissionEvent
# ---------------------------------------------------------------------------

@dataclass
class MissionEvent:
    """A single mid-mission event that may require replanning.

    Attributes
    ----------
    event_type:
        The category of event (see ``EventType``).
    severity:
        How serious the event is, 0.0 (minor) to 1.0 (critical).
    payload:
        Arbitrary key-value data specific to the event type.  Examples:

        BATTERY_FAILURE  → {"battery_pct": 0.08}
        COMM_LOSS        → {"last_contact_seconds_ago": 320}
        TERRAIN_HAZARD   → {"waypoint_id": "abc-123", "obstacle": "boulder"}
        NEW_DISCOVERY    → {"x": 450.0, "y": -120.0, "label": "ice-deposit"}
        RETURN_TO_BASE   → {}
    timestamp:
        UTC datetime when the event was detected (auto-set if not provided).
    """

    event_type: EventType
    severity: float  # 0.0 – 1.0
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(
                f"severity must be between 0.0 and 1.0, got {self.severity}"
            )
