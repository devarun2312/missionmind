"""
AI agents for MissionMind.

Available agents (implemented in subsequent sub-tasks):
    ScienceAgent      — scores candidate waypoints by scientific value.
    ResourceAgent     — models energy/time budgets and recommends feasible waypoints.
    SafetyAgent       — assesses terrain and environmental risk.
    MissionCommander  — orchestrates the above agents into a final MissionPlan.

Shared infrastructure (this sub-task):
    LLMClient         — protocol / interface every agent uses to call the AI backend.
    WatsonxClient     — production implementation calling IBM watsonx AI.
    BaseAgent         — abstract base class all agents inherit from.
    AgentResponseError — raised when the AI returns an unparseable response.
"""

from missionmind.agents.base_agent import AgentResponseError, BaseAgent
from missionmind.agents.client import LLMClient, LLMResponse, WatsonxClient

__all__ = [
    "AgentResponseError",
    "BaseAgent",
    "LLMClient",
    "LLMResponse",
    "WatsonxClient",
]
