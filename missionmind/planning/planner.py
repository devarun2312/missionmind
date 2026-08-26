"""
Mission Planner — public entry point for the MissionMind AI backend.

Purpose
-------
This module is the single, clean surface that the simulation and frontend
teams call.  Callers supply two plain Python dicts (``rover_state`` and
``env_state``) and receive a validated, ready-to-execute ``MissionPlan``.

All internal wiring — constructing the LLM client, instantiating the four
specialist agents, creating the ``SafetyValidator``, and building the
``MissionCommander`` — is handled here so callers never need to touch those
classes directly.

Public API
----------
::

    from missionmind.planning import plan_mission

    plan = await plan_mission(rover_state, env_state)

Required state keys are documented in ``ROVER_STATE_KEYS`` and
``ENV_STATE_KEYS``.

Dependency flow
---------------
::

    _build_llm_client()
          │
          ├── ScienceAgent(llm_client)
          ├── ResourceAgent(llm_client)
          ├── SafetyAgent(llm_client)
          └── MissionCommander(
                  science_agent, resource_agent, safety_agent,
                  validator=SafetyValidator(),
                  llm_client=llm_client,
              )
                  │
                  └── await commander.plan(context) → MissionPlan

LLM client
----------
``_build_llm_client()`` returns the existing ``WatsonxClient`` from
``missionmind.agents.client``.  Credentials are read from environment
variables by ``WatsonxClient`` itself (``IBM_WATSONX_API_KEY``,
``IBM_WATSONX_URL``, ``IBM_WATSONX_PROJECT_ID``).  No credentials are
handled or stored in this module.

Tests
-----
Tests inject a pre-built ``MissionCommander`` via the ``_commander``
parameter of ``plan_mission`` to avoid real LLM calls.  Alternatively,
``_build_llm_client`` can be patched at the module level.
"""

from __future__ import annotations

import logging
from typing import Any

from missionmind.agents.client import WatsonxClient
from missionmind.agents.mission_commander import MissionCommander
from missionmind.agents.resource_agent import ResourceAgent
from missionmind.agents.safety_agent import SafetyAgent
from missionmind.agents.science_agent import ScienceAgent
from missionmind.models.mission import MissionPlan
from missionmind.safety.validator import SafetyValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Required state-key constants
# ---------------------------------------------------------------------------

#: Keys that MUST be present in the ``rover_state`` dict.
#:
#: battery_pct         — current charge fraction (0.0–1.0)
#: battery_capacity_wh — total battery capacity in watt-hours
#: position_x          — rover X coordinate in metres from base
#: position_y          — rover Y coordinate in metres from base
#: rover_speed_mps     — nominal driving speed in metres per second
#: power_consumption_w — power draw during driving in watts
ROVER_STATE_KEYS: tuple[str, ...] = (
    "battery_pct",
    "battery_capacity_wh",
    "position_x",
    "position_y",
    "rover_speed_mps",
    "power_consumption_w",
)

#: Keys that MUST be present in the ``env_state`` dict.
#:
#: candidate_waypoints — list of waypoint dicts the rover may visit
#: weather_forecast    — dict: dust_storm_probability, temperatures, wind_speed_mps, forecast_hours
#: comm_windows        — list of {"start_utc": str, "duration_minutes": int}
#: terrain_map         — dict keyed by waypoint id with slope/surface data
#: mission_objectives  — list of plain-text mission goal strings
ENV_STATE_KEYS: tuple[str, ...] = (
    "candidate_waypoints",
    "weather_forecast",
    "comm_windows",
    "terrain_map",
    "mission_objectives",
)


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def _build_llm_client() -> WatsonxClient:
    """Construct the production LLM client.

    Returns the existing ``WatsonxClient`` which reads all IBM watsonx
    credentials from environment variables.  No credentials are handled
    here.  Tests can patch this function at the module level to inject a
    mock without touching the rest of the wiring.

    Returns
    -------
    WatsonxClient
        The shared production LLM backend used by all agents.
    """
    return WatsonxClient()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def plan_mission(
    rover_state: dict[str, Any],
    env_state: dict[str, Any],
) -> MissionPlan:
    """Generate an autonomous mission plan for the Mars rover.

    This is the primary entry point for the MissionMind AI backend.
    It validates inputs, assembles the planning context, wires all
    agents together, and returns a safety-validated ``MissionPlan``.

    Parameters
    ----------
    rover_state:
        Current rover status.  Must contain all keys in ``ROVER_STATE_KEYS``.

        Example::

            {
                "battery_pct":         0.85,   # fraction 0.0–1.0 (e.g. 0.85 = 85 %)
                "battery_capacity_wh": 500.0,  # Wh
                "position_x":          0.0,    # metres from base
                "position_y":          0.0,
                "rover_speed_mps":     0.5,    # m/s
                "power_consumption_w": 50.0,   # W during driving
            }

    env_state:
        Current environment and mission context.  Must contain all keys
        in ``ENV_STATE_KEYS``.

        Example::

            {
                "candidate_waypoints": [
                    {"id": "wp-1", "x": 150.0, "y": 80.0,
                     "terrain_risk": 0.2, "is_base": False,
                     "label": "crater-A",
                     "estimated_travel_time_minutes": 30.0,
                     "estimated_energy_wh": 20.0},
                    {"id": "base", "x": 0.0, "y": 0.0,
                     "terrain_risk": 0.0, "is_base": True,
                     "label": "BASE",
                     "estimated_travel_time_minutes": 0.0,
                     "estimated_energy_wh": 0.0},
                ],
                "weather_forecast": {
                    "dust_storm_probability": 0.1,
                    "temperature_min_c":      -60.0,
                    "temperature_max_c":       20.0,
                    "wind_speed_mps":           5.0,
                    "forecast_hours":           8,
                },
                "comm_windows": [
                    {"start_utc": "2025-01-01T10:00:00Z",
                     "duration_minutes": 30}
                ],
                "terrain_map":       {},
                "mission_objectives": ["search for biosignatures",
                                       "characterise subsurface water-ice"],
            }

    Returns
    -------
    MissionPlan
        A safety-validated plan with ``status=ACTIVE``.

    Raises
    ------
    ValueError
        If any required key is absent from ``rover_state`` or ``env_state``.
    PlanningFailedError
        If the commander cannot produce a valid plan within the allowed
        number of attempts.
    AgentResponseError
        If a specialist agent returns an unparseable LLM response.
    """
    _validate_keys("rover_state", rover_state, ROVER_STATE_KEYS)
    _validate_keys("env_state", env_state, ENV_STATE_KEYS)

    commander = _build_commander()

    context = _build_context(rover_state, env_state)

    logger.info(
        "plan_mission: starting | battery=%.0f%% | candidates=%d",
        rover_state["battery_pct"] * 100,
        len(env_state.get("candidate_waypoints", [])),
    )

    plan = await commander.plan(context)

    logger.info(
        "plan_mission: plan %s created | waypoints=%d | energy=%.1f Wh | time=%.1f min",
        plan.plan_id,
        len(plan.waypoints),
        plan.total_energy_wh,
        plan.total_time_minutes,
    )

    return plan


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_keys(
    source_name: str,
    data: dict[str, Any],
    required: tuple[str, ...],
) -> None:
    """Raise ``ValueError`` if any required key is missing from ``data``.

    Parameters
    ----------
    source_name:
        Human-readable name of the dict being checked, used in the error
        message (e.g. ``"rover_state"``).
    data:
        The dict to check.
    required:
        Tuple of key names that must all be present.

    Raises
    ------
    ValueError
        If one or more keys are missing.  The message lists every missing
        key so the caller can fix all problems in one go.
    """
    missing = [k for k in required if k not in data]
    if missing:
        keys_str = ", ".join(repr(k) for k in missing)
        raise ValueError(
            f"plan_mission: '{source_name}' is missing required key(s): {keys_str}"
        )


def _build_context(
    rover_state: dict[str, Any],
    env_state: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the planning context dict from rover and environment state.

    The context is a flat dict that merges all keys from both inputs,
    plus a ``rover_position`` sub-dict derived from ``position_x`` /
    ``position_y`` (required by the ScienceAgent's context contract).

    The original ``rover_state`` and ``env_state`` dicts are never mutated.

    Parameters
    ----------
    rover_state:
        Validated rover state (all ``ROVER_STATE_KEYS`` present).
    env_state:
        Validated environment state (all ``ENV_STATE_KEYS`` present).

    Returns
    -------
    dict
        Merged context ready to pass to ``MissionCommander.plan()``.
    """
    return {
        # All env_state keys (candidate_waypoints, weather_forecast, etc.)
        **env_state,
        # All rover_state keys (battery_pct, battery_capacity_wh, etc.)
        **rover_state,
        # rover_state nested as a dict for SafetyValidator and downstream use
        "rover_state": dict(rover_state),
        # Convenience rover_position sub-dict expected by ScienceAgent
        "rover_position": {
            "x": rover_state["position_x"],
            "y": rover_state["position_y"],
        },
    }


def _build_commander() -> MissionCommander:
    """Wire all agents and validators into a ``MissionCommander``.

    This is the production factory called by ``plan_mission`` when no
    ``_commander`` override is provided.  All agents share a single
    ``WatsonxClient`` instance so credentials are read from the environment
    only once.

    Returns
    -------
    MissionCommander
        Fully wired commander, ready to call ``.plan()``.
    """
    llm_client = _build_llm_client()

    science_agent = ScienceAgent(llm_client=llm_client)
    resource_agent = ResourceAgent(llm_client=llm_client)
    safety_agent = SafetyAgent(llm_client=llm_client)
    validator = SafetyValidator()

    return MissionCommander(
        science_agent=science_agent,
        resource_agent=resource_agent,
        safety_agent=safety_agent,
        validator=validator,
        llm_client=llm_client,
    )
