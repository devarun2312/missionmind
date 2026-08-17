"""
AI agents for MissionMind.

Available agents:
    ScienceAgent      — scores candidate waypoints by scientific value.
    ResourceAgent     — models energy/time budgets and recommends feasible waypoints.
                        (implemented in Sub-Task 4)
    SafetyAgent       — assesses terrain and environmental risk.
                        (implemented in Sub-Task 5)
    MissionCommander  — orchestrates the above agents into a final MissionPlan.
                        (implemented in Sub-Task 7)

Shared infrastructure:
    LLMClient         — protocol / interface every agent uses to call the AI backend.
    WatsonxClient     — production implementation calling IBM watsonx AI.
    BaseAgent         — abstract base class all agents inherit from.
    AgentResponseError — raised when the AI returns an unparseable response.
"""

from missionmind.agents.base_agent import AgentResponseError, BaseAgent
from missionmind.agents.client import LLMClient, LLMResponse, WatsonxClient
from missionmind.agents.resource_agent import ResourceAgent
from missionmind.agents.safety_agent import SafetyAgent
from missionmind.agents.science_agent import ScienceAgent

__all__ = [
    "AgentResponseError",
    "BaseAgent",
    "LLMClient",
    "LLMResponse",
    "ResourceAgent",
    "SafetyAgent",
    "ScienceAgent",
    "WatsonxClient",
]
