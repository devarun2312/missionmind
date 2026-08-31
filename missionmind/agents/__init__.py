"""
AI agents for MissionMind.

Available agents
----------------
ScienceAgent      — scores candidate waypoints by scientific value.
ResourceAgent     — models energy/time budgets and recommends feasible waypoints.
SafetyAgent       — assesses terrain and environmental risk (AI soft-constraint).
MissionCommander  — orchestrates the above agents into a validated MissionPlan.

Shared infrastructure
---------------------
LLMClient           — protocol / interface every agent uses to call the AI backend.
WatsonxClient       — production implementation calling IBM watsonx AI.
BaseAgent           — abstract base class all agents inherit from.
AgentResponseError  — raised when the AI returns an unparseable response.
PlanningFailedError — raised when MissionCommander exhausts all retry attempts.
"""

from missionmind.agents.base_agent import AgentResponseError, BaseAgent
from missionmind.agents.client import LLMClient, LLMResponse, OllamaClient, WatsonxClient
from missionmind.agents.mission_commander import MissionCommander, PlanningFailedError
from missionmind.agents.resource_agent import ResourceAgent
from missionmind.agents.safety_agent import SafetyAgent
from missionmind.agents.science_agent import ScienceAgent

__all__ = [
    "AgentResponseError",
    "BaseAgent",
    "LLMClient",
    "LLMResponse",
    "MissionCommander",
    "OllamaClient",
    "PlanningFailedError",
    "ResourceAgent",
    "SafetyAgent",
    "ScienceAgent",
    "WatsonxClient",
]
