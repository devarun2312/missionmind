"""
Mission Replanner — mid-mission event handler for MissionMind.

Purpose
-------
The Replanner listens for ``MissionEvent`` objects emitted by the simulation
layer and responds to mid-mission changes by producing a revised
``MissionPlan``.

Architecture
------------
The Replanner answers the question:

    "What changed, and how should the input to normal planning be adjusted?"

It is NOT a second MissionCommander.  For all event types that require
re-optimisation it adjusts a *copy* of the rover/environment state and then
delegates back to the existing ``plan_mission()`` entry point so the full
agent pipeline (ScienceAgent → ResourceAgent → SafetyAgent → MissionCommander
→ SafetyValidator) runs exactly once on the updated context.

The two exceptions are deterministic emergency returns:

* ``BATTERY_FAILURE`` below ``CRITICAL_BATTERY_PCT`` — abort immediately.
* ``RETURN_TO_BASE``                                 — return to base now.

These are handled without any LLM call because the decision is unambiguous
and must be instant and reliable.

Public API
----------
::

    from missionmind.planning import replan

    revised_plan = await replan(current_plan, event, rover_state, env_state)

Required imports
----------------
``MissionEvent`` and ``EventType`` live in ``missionmind.models.events``.
``MissionPlan`` and ``Waypoint`` live in ``missionmind.models.mission``.

Event payload conventions
--------------------------
BATTERY_FAILURE  → payload may contain ``{"battery_pct": <float>}``
                   If present the updated battery level is used; otherwise
                   the value from rover_state is kept.
COMM_LOSS        → payload may contain ``{"safe_comm_radius_m": <float>}``
                   Waypoints whose Euclidean distance from base (0, 0)
                   exceeds this radius are removed.  If the key is absent
                   all candidate waypoints are preserved.
TERRAIN_HAZARD   → payload must contain ``{"waypoint_id": "<id>"}``
                   The named waypoint is blacklisted.  Optionally
                   ``{"neighbour_ids": ["id1", "id2", ...]}`` lists
                   adjacent waypoints to blacklist as well.
NEW_DISCOVERY    → payload must contain ``{"x": <float>, "y": <float>}``
                   Optional keys: ``"id"``, ``"label"``,
                   ``"scientific_value"`` (default 0.9),
                   ``"terrain_risk"`` (default 0.1),
                   ``"estimated_travel_time_minutes"`` (default 0.0),
                   ``"estimated_energy_wh"`` (default 0.0).
RETURN_TO_BASE   → payload ignored; generates an immediate return plan.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from missionmind import config
from missionmind.models.events import EventType, MissionEvent
from missionmind.models.mission import MissionPlan, MissionStatus, Waypoint
from missionmind.planning.planner import plan_mission

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ReplanContext
# ---------------------------------------------------------------------------


@dataclass
class ReplanContext:
    """Bundles together everything the Replanner needs for one replanning cycle.

    Attributes
    ----------
    current_plan:
        The ``MissionPlan`` that was active when the event occurred.
    event:
        The ``MissionEvent`` that triggered replanning.
    rover_state:
        A *copy* of the rover's current state dict.  Handlers may mutate
        this copy without affecting the caller's original dict.
    env_state:
        A *copy* of the environment state dict.  Handlers may mutate this
        copy without affecting the caller's original dict.
    """

    current_plan: MissionPlan
    event: MissionEvent
    rover_state: dict[str, Any]
    env_state: dict[str, Any]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def replan(
    current_plan: MissionPlan,
    event: MissionEvent,
    rover_state: dict[str, Any],
    env_state: dict[str, Any],
) -> MissionPlan:
    """Produce a revised ``MissionPlan`` in response to a mid-mission event.

    The function dispatches to a dedicated handler for each supported
    ``EventType``.  Handlers that require full re-optimisation modify a copy
    of the state and then delegate back to ``plan_mission()``.  Emergency
    handlers return a deterministic plan immediately without any LLM call.

    Parameters
    ----------
    current_plan:
        The plan currently being executed by the rover.
    event:
        The mid-mission event that requires a planning adjustment.
    rover_state:
        Current rover status dict (must satisfy ``plan_mission`` key
        requirements for non-emergency handlers).  Not mutated.
    env_state:
        Current environment state dict (must satisfy ``plan_mission`` key
        requirements for non-emergency handlers).  Not mutated.

    Returns
    -------
    MissionPlan
        A revised plan appropriate to the event.

    Raises
    ------
    ValueError
        If ``event.event_type`` is not a supported ``EventType``.
    """
    # Build working copies so we never mutate caller-owned dicts.
    ctx = ReplanContext(
        current_plan=current_plan,
        event=event,
        rover_state=dict(rover_state),
        env_state=dict(env_state),
    )

    logger.info(
        "replan: handling event=%s severity=%.2f",
        event.event_type,
        event.severity,
    )

    if event.event_type == EventType.BATTERY_FAILURE:
        return await _handle_battery_failure(ctx)
    elif event.event_type == EventType.COMM_LOSS:
        return await _handle_comm_loss(ctx)
    elif event.event_type == EventType.TERRAIN_HAZARD:
        return await _handle_terrain_hazard(ctx)
    elif event.event_type == EventType.NEW_DISCOVERY:
        return await _handle_new_discovery(ctx)
    elif event.event_type == EventType.RETURN_TO_BASE:
        return _handle_return_to_base(ctx)
    else:
        raise ValueError(
            f"replan: unsupported event type '{event.event_type}'. "
            f"Supported types: {[e.value for e in EventType]}"
        )


# ---------------------------------------------------------------------------
# Private handlers
# ---------------------------------------------------------------------------


async def _handle_battery_failure(ctx: ReplanContext) -> MissionPlan:
    """Handle a BATTERY_FAILURE event.

    If the updated battery level is below ``CRITICAL_BATTERY_PCT``:
        → Return an immediate deterministic return-to-base plan.
          No LLM call is made — the decision is unambiguous.

    Otherwise:
        → Update the rover state with the new battery level and delegate
          to ``plan_mission()`` so the normal pipeline re-evaluates the
          mission scope with reduced energy budget.
    """
    # Apply updated battery level from the event payload if provided.
    new_battery_pct: float = ctx.event.payload.get(
        "battery_pct", ctx.rover_state.get("battery_pct", 0.0)
    )
    ctx.rover_state["battery_pct"] = new_battery_pct

    if new_battery_pct < config.CRITICAL_BATTERY_PCT:
        logger.warning(
            "replan: BATTERY_FAILURE — battery %.1f%% is below critical %.1f%%. "
            "Issuing immediate return-to-base.",
            new_battery_pct * 100,
            config.CRITICAL_BATTERY_PCT * 100,
        )
        return _build_emergency_return_plan(ctx, reasoning="BATTERY CRITICAL — returning to base immediately.")

    # Battery is low but not critical — replan with reduced scope.
    logger.info(
        "replan: BATTERY_FAILURE — battery %.1f%% is above critical %.1f%%. "
        "Replanning with reduced energy budget.",
        new_battery_pct * 100,
        config.CRITICAL_BATTERY_PCT * 100,
    )
    return await plan_mission(ctx.rover_state, ctx.env_state)


async def _handle_comm_loss(ctx: ReplanContext) -> MissionPlan:
    """Handle a COMM_LOSS event.

    Removes candidate waypoints that are outside the safe communication
    radius.  If no radius is provided in the event payload all candidates
    are preserved and normal replanning is still triggered.

    The safe communication radius is read from
    ``event.payload["safe_comm_radius_m"]``.  If the key is absent, all
    candidates are kept (safe default — no false blacklisting).

    Waypoint distance is the Euclidean distance from base (0, 0).
    """
    safe_radius: float | None = ctx.event.payload.get("safe_comm_radius_m")

    candidates: list[dict[str, Any]] = list(
        ctx.env_state.get("candidate_waypoints", [])
    )

    if safe_radius is not None:
        filtered = [
            wp for wp in candidates
            if wp.get("is_base", False) or _distance_from_base(wp) <= safe_radius
        ]
        logger.info(
            "replan: COMM_LOSS — safe_comm_radius_m=%.0f | "
            "candidates before=%d after=%d",
            safe_radius,
            len(candidates),
            len(filtered),
        )
        ctx.env_state = {**ctx.env_state, "candidate_waypoints": filtered}
    else:
        logger.info(
            "replan: COMM_LOSS — no safe_comm_radius_m in payload; "
            "all %d candidates preserved.",
            len(candidates),
        )

    return await plan_mission(ctx.rover_state, ctx.env_state)


async def _handle_terrain_hazard(ctx: ReplanContext) -> MissionPlan:
    """Handle a TERRAIN_HAZARD event.

    Blacklists the affected waypoint (``event.payload["waypoint_id"]``) and
    any explicitly named neighbours (``event.payload.get("neighbour_ids", [])``).
    Then delegates to ``plan_mission()``.
    """
    affected_id: str = ctx.event.payload.get("waypoint_id", "")
    neighbour_ids: list[str] = list(ctx.event.payload.get("neighbour_ids", []))

    blacklisted: set[str] = {affected_id} | set(neighbour_ids)
    blacklisted.discard("")  # guard against missing waypoint_id

    candidates: list[dict[str, Any]] = list(
        ctx.env_state.get("candidate_waypoints", [])
    )
    filtered = [
        wp for wp in candidates
        if wp.get("id") not in blacklisted
    ]

    logger.info(
        "replan: TERRAIN_HAZARD — blacklisted=%s | "
        "candidates before=%d after=%d",
        blacklisted,
        len(candidates),
        len(filtered),
    )

    ctx.env_state = {**ctx.env_state, "candidate_waypoints": filtered}
    return await plan_mission(ctx.rover_state, ctx.env_state)


async def _handle_new_discovery(ctx: ReplanContext) -> MissionPlan:
    """Handle a NEW_DISCOVERY event.

    Adds the newly discovered waypoint to the candidate set and delegates
    to ``plan_mission()`` so the full agent pipeline (ScienceAgent,
    ResourceAgent, SafetyAgent, MissionCommander) can decide whether to
    include it given the current resource budget.
    """
    payload = ctx.event.payload
    new_wp: dict[str, Any] = {
        "id":                             payload.get("id", f"discovery-{id(ctx.event)}"),
        "x":                              float(payload["x"]),
        "y":                              float(payload["y"]),
        "label":                          payload.get("label", "new-discovery"),
        "scientific_value":               float(payload.get("scientific_value", 0.9)),
        "terrain_risk":                   float(payload.get("terrain_risk", 0.1)),
        "is_base":                        False,
        "estimated_travel_time_minutes":  float(
            payload.get("estimated_travel_time_minutes", 0.0)
        ),
        "estimated_energy_wh":            float(
            payload.get("estimated_energy_wh", 0.0)
        ),
    }

    candidates: list[dict[str, Any]] = list(
        ctx.env_state.get("candidate_waypoints", [])
    )
    candidates = [new_wp] + candidates  # prepend so it is visible to agents

    logger.info(
        "replan: NEW_DISCOVERY — added waypoint id='%s' at (%.1f, %.1f) "
        "science_value=%.2f",
        new_wp["id"],
        new_wp["x"],
        new_wp["y"],
        new_wp["scientific_value"],
    )

    ctx.env_state = {**ctx.env_state, "candidate_waypoints": candidates}
    return await plan_mission(ctx.rover_state, ctx.env_state)


def _handle_return_to_base(ctx: ReplanContext) -> MissionPlan:
    """Handle an explicit RETURN_TO_BASE command.

    Generates a minimal, deterministic return plan without any LLM call.
    The plan contains only the base waypoint.
    """
    logger.info("replan: RETURN_TO_BASE — building immediate return plan.")
    return _build_emergency_return_plan(
        ctx, reasoning="RETURN_TO_BASE command received — returning to base."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_emergency_return_plan(ctx: ReplanContext, reasoning: str) -> MissionPlan:
    """Build a minimal deterministic return-to-base ``MissionPlan``.

    Finds the base waypoint in the current plan or environment candidates and
    returns a plan containing only that waypoint.  No LLM call is made.

    The plan is intentionally *not* passed through ``SafetyValidator`` because
    the normal validator requires at least one science waypoint — a rule that
    is correct for normal mission plans but should not block an emergency abort.

    Parameters
    ----------
    ctx:
        The active replanning context.
    reasoning:
        Free-text explanation stored on the returned plan.

    Returns
    -------
    MissionPlan
        A plan whose sole waypoint is the base station, with
        ``status=MissionStatus.ACTIVE``.
    """
    base_wp = _find_base_waypoint(ctx)

    return MissionPlan(
        waypoints=[base_wp],
        total_energy_wh=base_wp.estimated_energy_wh,
        total_time_minutes=base_wp.estimated_travel_time_minutes,
        status=MissionStatus.ACTIVE,
        reasoning=reasoning,
        confidence=1.0,
    )


def _find_base_waypoint(ctx: ReplanContext) -> Waypoint:
    """Locate the base waypoint from the current plan or environment candidates.

    Search order:
    1. The last waypoint in ``current_plan`` that has ``is_base=True``.
    2. Any waypoint in ``env_state["candidate_waypoints"]`` with ``is_base=True``.
    3. A synthesised fallback at (0, 0) labelled "BASE".

    Returns
    -------
    Waypoint
        The base-station waypoint.
    """
    # 1. Check current plan
    for wp in reversed(ctx.current_plan.waypoints):
        if wp.is_base:
            return wp

    # 2. Check environment candidates
    for candidate in ctx.env_state.get("candidate_waypoints", []):
        if candidate.get("is_base", False):
            return Waypoint(
                id=candidate.get("id", "base"),
                x=float(candidate.get("x", 0.0)),
                y=float(candidate.get("y", 0.0)),
                scientific_value=0.0,
                terrain_risk=float(candidate.get("terrain_risk", 0.0)),
                estimated_travel_time_minutes=float(
                    candidate.get("estimated_travel_time_minutes", 0.0)
                ),
                estimated_energy_wh=float(
                    candidate.get("estimated_energy_wh", 0.0)
                ),
                is_base=True,
                label=candidate.get("label", "BASE"),
            )

    # 3. Synthesised fallback
    logger.warning(
        "replan: no base waypoint found in current plan or candidates; "
        "synthesising fallback at (0, 0)."
    )
    return Waypoint(
        id="base",
        x=0.0,
        y=0.0,
        scientific_value=0.0,
        terrain_risk=0.0,
        estimated_travel_time_minutes=0.0,
        estimated_energy_wh=0.0,
        is_base=True,
        label="BASE",
    )


def _distance_from_base(wp: dict[str, Any]) -> float:
    """Return the Euclidean distance of a waypoint dict from base (0, 0)."""
    x = float(wp.get("x", 0.0))
    y = float(wp.get("y", 0.0))
    return math.sqrt(x * x + y * y)
