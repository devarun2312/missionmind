"""
MissionMind API — FastAPI integration layer.

Exposes the MissionMind AI planning backend over HTTP so a React (or any
other) frontend can request mission plans and trigger replanning.

Public surface
--------------
``create_app()`` — build and configure the FastAPI application.
``app``          — pre-built instance used by uvicorn and tests.

Start the server
----------------
::

    uvicorn missionmind.api.main:app --reload

Interactive docs
----------------
http://127.0.0.1:8000/docs
"""
