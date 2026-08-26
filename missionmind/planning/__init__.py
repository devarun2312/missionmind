"""
MissionMind planning package.

Public API
----------
plan_mission(rover_state, env_state) -> MissionPlan
    Generate an autonomous mission plan for the Mars rover.
    This is the primary entry point for simulation and frontend layers.

replan(current_plan, event, rover_state, env_state) -> MissionPlan
    Produce a revised MissionPlan in response to a mid-mission event.
    Dispatches to per-event handlers; delegates to plan_mission() for
    events that require full re-optimisation.

ReplanContext
    Dataclass bundling the inputs for one replanning cycle.

Usage
-----
::

    from missionmind.planning import plan_mission, replan

    plan = await plan_mission(rover_state, env_state)

    revised = await replan(plan, event, rover_state, env_state)
"""

from missionmind.planning.planner import plan_mission
from missionmind.planning.replanner import ReplanContext, replan

__all__ = ["plan_mission", "replan", "ReplanContext"]
