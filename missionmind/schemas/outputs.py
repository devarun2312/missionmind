"""
Structured AI output schemas.

Each Pydantic model here represents the exact JSON structure an AI agent is
expected to return.  Using these schemas achieves two things:

1. The LLM is given the schema as part of its prompt so it knows the required
   output format.
2. The agent layer validates the LLM response against the schema before
   passing it upstream — bad outputs are caught early, not silently ignored.

All models use Pydantic v2 and are JSON-serialisable.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Coarse-grained risk classification used in RiskAssessment."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# ScienceAgent output
# ---------------------------------------------------------------------------

class ScoredTarget(BaseModel):
    """A single waypoint scored by the Science Agent.

    Attributes
    ----------
    waypoint_id:
        References ``Waypoint.id`` from the candidate list.
    scientific_value:
        Assigned score, 0.0 (no interest) to 1.0 (exceptional value).
    justification:
        Short free-text reason for the score (geology, atmosphere, etc.).
    """

    waypoint_id: str
    scientific_value: Annotated[float, Field(ge=0.0, le=1.0)]
    justification: str = ""


class ScienceAnalysis(BaseModel):
    """Full output of the Science Agent.

    Attributes
    ----------
    scored_targets:
        Every candidate waypoint with its assigned scientific score.
    priority_order:
        Waypoint IDs listed from highest to lowest scientific priority.
    reasoning:
        Overall reasoning narrative from the agent.
    """

    scored_targets: list[ScoredTarget] = Field(default_factory=list)
    priority_order: list[str] = Field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# ResourceAgent output
# ---------------------------------------------------------------------------

class ResourceBudget(BaseModel):
    """Full output of the Resource / Energy Agent.

    Attributes
    ----------
    available_energy_wh:
        Total energy available for the mission after deducting the return reserve.
    available_time_minutes:
        Total time budget available for the mission.
    recommended_waypoints:
        Ordered list of waypoint IDs that fit within the energy/time budget.
    energy_per_waypoint:
        Estimated energy cost (Wh) for each waypoint, keyed by waypoint ID.
    reasoning:
        Free-text explanation of how the budget was calculated.
    """

    available_energy_wh: Annotated[float, Field(ge=0.0)]
    available_time_minutes: Annotated[float, Field(ge=0.0)]
    recommended_waypoints: list[str] = Field(default_factory=list)
    energy_per_waypoint: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# SafetyAgent output
# ---------------------------------------------------------------------------

class WaypointRisk(BaseModel):
    """Risk assessment for a single waypoint.

    Attributes
    ----------
    waypoint_id:
        References ``Waypoint.id``.
    risk_score:
        Assessed hazard level, 0.0 (safe) to 1.0 (dangerous).
    factors:
        List of contributing risk factors (e.g. "steep slope", "dust storm risk").
    """

    waypoint_id: str
    risk_score: Annotated[float, Field(ge=0.0, le=1.0)]
    factors: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Full output of the Safety Agent.

    Attributes
    ----------
    waypoint_risks:
        Per-waypoint risk detail.
    overall_risk_level:
        Coarse mission-level risk classification.
    recommended_exclusions:
        Waypoint IDs the Safety Agent recommends removing from the plan.
    reasoning:
        Overall safety narrative from the agent.
    """

    waypoint_risks: list[WaypointRisk] = Field(default_factory=list)
    overall_risk_level: RiskLevel = RiskLevel.LOW
    recommended_exclusions: list[str] = Field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# MissionCommander output
# ---------------------------------------------------------------------------

class PlannedWaypointEntry(BaseModel):
    """A single entry in the commander's ordered waypoint sequence.

    Attributes
    ----------
    waypoint_id:
        References ``Waypoint.id``.
    visit_order:
        1-based position in the planned sequence.
    expected_science_value:
        Commander's estimate of science return at this waypoint.
    expected_energy_wh:
        Commander's estimate of energy consumed reaching this waypoint.
    """

    waypoint_id: str
    visit_order: Annotated[int, Field(ge=1)]
    expected_science_value: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    expected_energy_wh: Annotated[float, Field(ge=0.0)] = 0.0


class MissionPlanOutput(BaseModel):
    """Full structured plan emitted by the MissionCommander LLM call.

    This is the AI-generated plan *before* it is converted into a ``MissionPlan``
    domain object and validated by the deterministic SafetyValidator.

    Attributes
    ----------
    planned_waypoints:
        Ordered waypoint entries chosen by the commander.
    total_estimated_energy_wh:
        Sum of energy estimates across all planned waypoints.
    total_estimated_time_minutes:
        Sum of time estimates across all planned waypoints.
    confidence:
        Commander's self-assessed confidence in the plan quality, 0.0 – 1.0.
    reasoning:
        Narrative explanation of the plan's trade-offs and priorities.
    """

    planned_waypoints: list[PlannedWaypointEntry] = Field(default_factory=list)
    total_estimated_energy_wh: Annotated[float, Field(ge=0.0)] = 0.0
    total_estimated_time_minutes: Annotated[float, Field(ge=0.0)] = 0.0
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    reasoning: str = ""
