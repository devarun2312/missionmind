"""
MissionMind planning package.

Public API
----------
plan_mission(rover_state, env_state) -> MissionPlan
    Generate an autonomous mission plan for the Mars rover.
    This is the primary entry point for simulation and frontend layers.

Usage
-----
::

    from missionmind.planning import plan_mission

    plan = await plan_mission(rover_state, env_state)
"""

from missionmind.planning.planner import plan_mission

__all__ = ["plan_mission"]
