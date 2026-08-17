"""
MissionMind — AI-powered autonomous mission planning for a simulated Mars rover.

Package layout
--------------
missionmind.config          Runtime constants and configurable thresholds.
missionmind.models          Domain data models (MissionPlan, Waypoint, events …).
missionmind.schemas         Pydantic schemas for structured AI agent outputs.
missionmind.agents          AI agents (science, resource, safety, commander).
missionmind.safety          Deterministic safety validator.
missionmind.planning        Mission planner and replanner entry points.
"""

__version__ = "0.1.0"
