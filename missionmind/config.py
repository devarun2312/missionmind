"""
MissionMind runtime configuration.

All values are sourced from environment variables when present so that
teammates can override them without touching source code.  Sensible
defaults are provided for local development.

Environment variable reference
-------------------------------
LLM_MODEL_NAME          Watsonx / OpenAI model identifier.
IBM_WATSONX_API_KEY     API key for IBM watsonx AI (never hard-coded here).
IBM_WATSONX_URL         Watsonx endpoint URL.
OPENAI_API_KEY          Fallback OpenAI key for local development.
MIN_RETURN_BATTERY_PCT  Fraction of battery that must be reserved for return.
CRITICAL_BATTERY_PCT    Fraction below which the rover must abort immediately.
MAX_MISSION_DURATION_HOURS
COMM_TIMEOUT_SECONDS
MAX_TERRAIN_RISK_SCORE
"""

import os

# ---------------------------------------------------------------------------
# LLM / AI model
# ---------------------------------------------------------------------------

# The model ID used for all agent calls.
# Override to point at a watsonx model, e.g. "ibm/granite-13b-instruct-v2".
LLM_MODEL_NAME: str = os.environ.get("LLM_MODEL_NAME", "gpt-4o")

# ---------------------------------------------------------------------------
# Battery / energy thresholds
# ---------------------------------------------------------------------------

# Minimum battery percentage that MUST remain when the rover returns to base.
# Expressed as a fraction (0.0 – 1.0).  Default: 20 %.
MIN_RETURN_BATTERY_PCT: float = float(
    os.environ.get("MIN_RETURN_BATTERY_PCT", "0.20")
)

# Battery percentage below which the rover must abort the current plan and
# return to base immediately, regardless of scientific objectives.
# Expressed as a fraction (0.0 – 1.0).  Default: 10 %.
CRITICAL_BATTERY_PCT: float = float(
    os.environ.get("CRITICAL_BATTERY_PCT", "0.10")
)

# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------

# Maximum total mission duration (driving + science stops).
MAX_MISSION_DURATION_HOURS: float = float(
    os.environ.get("MAX_MISSION_DURATION_HOURS", "8.0")
)

# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------

# Number of seconds without an uplink before the rover is considered out of
# communication contact and replanning is triggered.
COMM_TIMEOUT_SECONDS: int = int(
    os.environ.get("COMM_TIMEOUT_SECONDS", "300")
)

# ---------------------------------------------------------------------------
# Terrain / hazard
# ---------------------------------------------------------------------------

# Maximum terrain risk score (0.0 – 1.0) allowed for any single waypoint in
# an approved mission plan.  Waypoints above this threshold must be excluded.
MAX_TERRAIN_RISK_SCORE: float = float(
    os.environ.get("MAX_TERRAIN_RISK_SCORE", "0.70")
)

# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

# Maximum number of replanning attempts the MissionCommander will make before
# raising a PlanningFailedError.
MAX_PLANNING_RETRIES: int = int(
    os.environ.get("MAX_PLANNING_RETRIES", "3")
)
