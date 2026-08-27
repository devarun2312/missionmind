"""
FastAPI application factory for the MissionMind API.

Usage
-----
The application is exposed via ``missionmind.api.main``.  Build it yourself
only when testing:

::

    from missionmind.api.app import create_app
    app = create_app()
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from missionmind.api.routes.health import router as health_router
from missionmind.api.routes.planning import router as planning_router
from missionmind.api.routes.replanning import router as replanning_router


# ---------------------------------------------------------------------------
# CORS origins for local React development
# ---------------------------------------------------------------------------

_CORS_ORIGINS: list[str] = [
    "http://localhost:3000",    # Create React App default
    "http://127.0.0.1:3000",
    "http://localhost:5173",    # Vite default
    "http://127.0.0.1:5173",
    "http://localhost:5174",    # Vite secondary port
    "http://127.0.0.1:5174",
]


def create_app() -> FastAPI:
    """Build and configure the MissionMind FastAPI application.

    This factory is called once at startup (by ``main.py``) and can also be
    called in tests to get a fresh, isolated application instance.

    Returns
    -------
    FastAPI
        Fully configured application with CORS middleware and all routers
        mounted under ``/api``.
    """
    app = FastAPI(
        title="MissionMind API",
        description=(
            "HTTP integration layer for the MissionMind AI-powered Mars rover "
            "mission planning backend.  Exposes ``plan_mission()`` and "
            "``replan()`` over JSON/HTTP for use by a React frontend."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — permissive for local hackathon development, NOT for production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=False,   # no cookies/auth — keep simple
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    # Mount routers
    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(planning_router, prefix="/api/mission", tags=["planning"])
    app.include_router(replanning_router, prefix="/api/mission", tags=["replanning"])

    return app
