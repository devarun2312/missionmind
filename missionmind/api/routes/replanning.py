"""
POST /api/mission/replan — invoke the MissionMind event-driven replanner.

Thin HTTP bridge: converts the Pydantic request body into the types that
``replan()`` expects (including constructing a real ``MissionEvent``
dataclass from the ``MissionEventInput`` Pydantic model), then calls the
backend and returns the revised ``MissionPlan`` as JSON.

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
from missionmind.api.schemas.replanning import ReplanRequest
from missionmind.models.events import MissionEvent
from missionmind.models.mission import MissionPlan
from missionmind.planning.replanner import replan

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/replan",
    response_model=MissionPlan,
    summary="Replan mission in response to a mid-mission event",
    description=(
        "Handle a MissionEvent (BATTERY_FAILURE, COMM_LOSS, TERRAIN_HAZARD, "
        "NEW_DISCOVERY, or RETURN_TO_BASE) and produce a revised MissionPlan.  "
        "Emergency events (critical battery, RETURN_TO_BASE) are handled "
        "deterministically without an LLM call."
    ),
    responses={
        400: {"description": "Missing or invalid state keys or unknown event type"},
        502: {"description": "AI agent returned an unparseable response"},
        503: {"description": "Replanning failed after all retry attempts"},
    },
)
async def replan_mission(body: ReplanRequest) -> MissionPlan:
    """Convert HTTP request to backend types, call replan(), return revised plan."""
    # Construct a real MissionEvent dataclass from the Pydantic HTTP model
    event = MissionEvent(
        event_type=body.event.event_type,
        severity=body.event.severity,
        payload=body.event.payload,
    )

    rover_state = body.rover_state.model_dump()
    env_state = body.env_state.model_dump()
    env_state["candidate_waypoints"] = [
        wp.model_dump() for wp in body.env_state.candidate_waypoints
    ]

    try:
        return await replan(
            current_plan=body.current_plan,
            event=event,
            rover_state=rover_state,
            env_state=env_state,
        )

    except AgentResponseError as exc:
        # Must be caught BEFORE ValueError because AgentResponseError is a
        # subclass of ValueError.
        logger.warning("AgentResponseError during replanning: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={"error": "ai_response_error", "message": str(exc)},
        ) from exc

    except PlanningFailedError as exc:
        logger.warning(
            "PlanningFailedError during replanning: attempts=%d violations=%s",
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
        logger.warning("ValueError during replanning: %s", exc)
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": str(exc)},
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error during replanning")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "An unexpected error occurred."},
        ) from exc
