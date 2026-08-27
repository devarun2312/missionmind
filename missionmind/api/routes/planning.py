"""
POST /api/mission/plan — invoke the MissionMind AI planning pipeline.

Thin HTTP bridge: converts the validated Pydantic request models into the
plain Python dicts that ``plan_mission()`` expects, calls the backend, and
returns the resulting ``MissionPlan`` as JSON.

Error mapping
-------------
AgentResponseError   → 502 Bad Gateway  (structural LLM failure)
PlanningFailedError  → 503 Service Unavailable
ValueError           → 400 Bad Request
Exception            → 500 Internal Server Error
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from missionmind.agents.base_agent import AgentResponseError
from missionmind.agents.mission_commander import PlanningFailedError
from missionmind.api.schemas.planning import PlanRequest
from missionmind.models.mission import MissionPlan
from missionmind.planning.planner import plan_mission

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/plan",
    response_model=MissionPlan,
    summary="Generate a mission plan",
    description=(
        "Run the full MissionMind AI planning pipeline: ScienceAgent, "
        "ResourceAgent, SafetyAgent, MissionCommander, and SafetyValidator.  "
        "Returns a safety-validated MissionPlan with status=ACTIVE."
    ),
    responses={
        400: {"description": "Missing or invalid state keys"},
        502: {"description": "AI agent returned an unparseable response"},
        503: {"description": "Planning failed after all retry attempts"},
    },
)
async def plan(body: PlanRequest) -> MissionPlan:
    """Convert HTTP request to backend dicts, call plan_mission(), return plan."""
    rover_state = body.rover_state.model_dump()
    env_state = body.env_state.model_dump()
    # Convert nested WaypointInput objects to plain dicts
    env_state["candidate_waypoints"] = [
        wp.model_dump() for wp in body.env_state.candidate_waypoints
    ]

    try:
        return await plan_mission(rover_state, env_state)

    except AgentResponseError as exc:
        # Must be caught BEFORE ValueError because AgentResponseError is a
        # subclass of ValueError.
        logger.warning("AgentResponseError during planning: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={"error": "ai_response_error", "message": str(exc)},
        ) from exc

    except PlanningFailedError as exc:
        logger.warning(
            "PlanningFailedError: attempts=%d violations=%s",
            exc.attempts,
            exc.violations,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "planning_failed",
                "message": str(exc),
                "violations": exc.violations,
                "attempts": exc.attempts,
            },
        ) from exc

    except ValueError as exc:
        logger.warning("ValueError during planning: %s", exc)
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": str(exc)},
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error during planning")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "An unexpected error occurred."},
        ) from exc
