"""Health check endpoint — GET /api/health."""

from __future__ import annotations

from fastapi import APIRouter

from missionmind.api.schemas.planning import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns a simple liveness response.  "
                "Does not test the LLM backend.",
)
async def health() -> HealthResponse:
    return HealthResponse()
