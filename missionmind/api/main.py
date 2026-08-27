"""
MissionMind API entry point.

Usage
-----
::

    # Recommended (supports --reload for development)
    uvicorn missionmind.api.main:app --reload

    # Alternative (runs uvicorn programmatically)
    python -m missionmind.api.main

Interactive API docs
--------------------
http://127.0.0.1:8000/docs

Endpoints
---------
GET  /api/health
POST /api/mission/plan
POST /api/mission/replan
"""

from __future__ import annotations

from missionmind.api.app import create_app

# Module-level app instance — used by uvicorn:
#   uvicorn missionmind.api.main:app --reload
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "missionmind.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
