"""
Tests for Sub-Task 5: Safety Agent.

Covers:
- SafetyAgent is a concrete, instantiable subclass of BaseAgent.
- SafetyAgent.name returns "safety".
- SafetyAgent.response_schema returns RiskAssessment.
- SafetyAgent.system_prompt is non-empty and loaded from safety_prompt.md.
- System prompt references key safety concepts (risk, hazard, terrain, exclusion).
- System prompt references LOW / MEDIUM / HIGH risk levels.
- System prompt specifies JSON output.

Happy path:
- run() returns a RiskAssessment instance.
- All waypoint_risks entries have risk_score in [0.0, 1.0].
- overall_risk_level is a valid RiskLevel enum value.
- reasoning is populated.

High-risk waypoint test:
- A RiskAssessment with HIGH overall_risk_level is returned.
- Waypoints flagged HIGH appear in recommended_exclusions.
- Waypoints not flagged do not appear in recommended_exclusions.

All-low-risk test:
- When all waypoints are LOW risk, recommended_exclusions is empty.
- overall_risk_level is LOW.

Malformed AI response:
- Malformed JSON raises AgentResponseError (agent_name="safety").
- Schema-invalid response (risk_score > 1.0) raises AgentResponseError.
- Structural failures are NOT retried.

Context forwarding:
- candidate_waypoints, weather_forecast, comm_windows, terrain_map forwarded.
- max_terrain_risk_score from config is injected automatically.
- Caller-supplied max_terrain_risk_score is overwritten by config value.
- system_prompt is sent as the system message.
- No real API calls are made.

RiskAssessment schema:
- risk_score boundary values 0.0 and 1.0 accepted.
- risk_score > 1.0 rejected by Pydantic.
- risk_score < 0.0 rejected by Pydantic.
- RiskLevel enum values LOW / MEDIUM / HIGH accepted.
- Invalid overall_risk_level rejected.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from missionmind import config
from missionmind.agents.base_agent import AgentResponseError, BaseAgent
from missionmind.agents.client import LLMResponse
from missionmind.agents.safety_agent import SafetyAgent
from missionmind.schemas.outputs import RiskAssessment, RiskLevel, WaypointRisk


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

CANDIDATE_WAYPOINTS = [
    {
        "id": "wp-safe",
        "x": 40.0, "y": 20.0,
        "terrain_risk": 0.10,
        "label": "flat-plains",
        "is_base": False,
        "scientific_value": 0.5,
        "estimated_travel_time_minutes": 8.0,
        "estimated_energy_wh": 12.0,
    },
    {
        "id": "wp-medium",
        "x": 180.0, "y": -60.0,
        "terrain_risk": 0.45,
        "label": "crater-slope",
        "is_base": False,
        "scientific_value": 0.75,
        "estimated_travel_time_minutes": 20.0,
        "estimated_energy_wh": 30.0,
    },
    {
        "id": "wp-dangerous",
        "x": 400.0, "y": 250.0,
        "terrain_risk": 0.85,    # exceeds MAX_TERRAIN_RISK_SCORE (0.70 default)
        "label": "icy-ridge",
        "is_base": False,
        "scientific_value": 0.9,
        "estimated_travel_time_minutes": 50.0,
        "estimated_energy_wh": 80.0,
    },
]

WEATHER_FORECAST = {
    "dust_storm_probability": 0.15,
    "temperature_min_c": -80.0,
    "temperature_max_c": -20.0,
    "wind_speed_mps": 12.0,
    "forecast_hours": 8,
}

COMM_WINDOWS = [
    {"start_utc": "2025-06-01T08:00:00Z", "duration_minutes": 45},
    {"start_utc": "2025-06-01T20:30:00Z", "duration_minutes": 30},
]

TERRAIN_MAP = {
    "wp-safe":      {"slope_degrees": 3.0,  "surface_type": "basalt", "surveyed": True},
    "wp-medium":    {"slope_degrees": 18.0, "surface_type": "sand",   "surveyed": True},
    "wp-dangerous": {"slope_degrees": 28.0, "surface_type": "ice",    "surveyed": False},
}

FULL_CONTEXT = {
    "candidate_waypoints": CANDIDATE_WAYPOINTS,
    "weather_forecast":    WEATHER_FORECAST,
    "comm_windows":        COMM_WINDOWS,
    "terrain_map":         TERRAIN_MAP,
}

# --- Canned valid responses ---

HIGH_RISK_RESPONSE = {
    "waypoint_risks": [
        {
            "waypoint_id": "wp-safe",
            "risk_score": 0.10,
            "factors": ["flat terrain", "well-surveyed"],
        },
        {
            "waypoint_id": "wp-medium",
            "risk_score": 0.45,
            "factors": ["18° slope", "loose sand surface"],
        },
        {
            "waypoint_id": "wp-dangerous",
            "risk_score": 0.88,
            "factors": ["28° slope", "unsurveyed ice surface", "exceeds terrain risk threshold"],
        },
    ],
    "overall_risk_level": "HIGH",
    "recommended_exclusions": ["wp-dangerous"],
    "reasoning": (
        "wp-dangerous exceeds the terrain risk threshold (0.85 > 0.70) and has a steep "
        "icy slope of 28° with no prior survey data. Exclusion is strongly recommended. "
        "wp-medium is marginal but acceptable with caution. wp-safe is clear."
    ),
}

ALL_LOW_RESPONSE = {
    "waypoint_risks": [
        {"waypoint_id": "wp-safe",      "risk_score": 0.08, "factors": ["nominal"]},
        {"waypoint_id": "wp-medium",    "risk_score": 0.22, "factors": ["minor slope"]},
        {"waypoint_id": "wp-dangerous", "risk_score": 0.28, "factors": ["treated as low after re-survey"]},
    ],
    "overall_risk_level": "LOW",
    "recommended_exclusions": [],
    "reasoning": "All waypoints assessed as low risk under current conditions.",
}

MEDIUM_RISK_RESPONSE = {
    "waypoint_risks": [
        {"waypoint_id": "wp-safe",      "risk_score": 0.12, "factors": []},
        {"waypoint_id": "wp-medium",    "risk_score": 0.50, "factors": ["moderate slope"]},
        {"waypoint_id": "wp-dangerous", "risk_score": 0.55, "factors": ["ice present"]},
    ],
    "overall_risk_level": "MEDIUM",
    "recommended_exclusions": [],
    "reasoning": "Mission is MEDIUM risk overall; no exclusions required.",
}


def _make_mock_client(payload: dict | str | None = None) -> MagicMock:
    """Return a mock LLM client — no network calls, no credentials needed."""
    if payload is None:
        content = json.dumps(HIGH_RISK_RESPONSE)
    elif isinstance(payload, dict):
        content = json.dumps(payload)
    else:
        content = payload  # raw str — may be intentionally malformed

    mock = MagicMock()
    mock.complete = AsyncMock(return_value=LLMResponse(
        content=content,
        model="test-model",
        prompt_tokens=100,
        completion_tokens=220,
    ))
    return mock


# ---------------------------------------------------------------------------
# Class identity & interface
# ---------------------------------------------------------------------------

class TestSafetyAgentInterface:
    def test_is_subclass_of_base_agent(self):
        assert issubclass(SafetyAgent, BaseAgent)

    def test_is_instantiable_with_mock_client(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        assert agent is not None

    def test_name_is_safety(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        assert agent.name == "safety"

    def test_response_schema_is_risk_assessment(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        assert agent.response_schema is RiskAssessment

    def test_system_prompt_is_non_empty_string(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        prompt = agent.system_prompt
        assert isinstance(prompt, str) and len(prompt) > 100

    def test_system_prompt_references_safety_officer_role(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        lower = agent.system_prompt.lower()
        assert "safety" in lower or "hazard" in lower

    def test_system_prompt_references_risk_levels(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        prompt = agent.system_prompt
        assert "LOW" in prompt and "MEDIUM" in prompt and "HIGH" in prompt

    def test_system_prompt_mentions_terrain_risk(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        assert "terrain" in agent.system_prompt.lower()

    def test_system_prompt_mentions_exclusion(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        assert "exclusion" in agent.system_prompt.lower() or "exclude" in agent.system_prompt.lower()

    def test_system_prompt_specifies_json_output(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        assert "json" in agent.system_prompt.lower()

    def test_system_prompt_loaded_from_file_consistently(self):
        a = SafetyAgent(llm_client=_make_mock_client())
        b = SafetyAgent(llm_client=_make_mock_client())
        assert a.system_prompt == b.system_prompt

    def test_can_import_from_agents_package(self):
        from missionmind.agents import SafetyAgent as SA  # noqa: F401
        assert SA is SafetyAgent


# ---------------------------------------------------------------------------
# Happy path — general valid response
# ---------------------------------------------------------------------------

class TestSafetyAgentHappyPath:
    async def test_run_returns_risk_assessment_instance(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        result = await agent.run(FULL_CONTEXT)
        assert isinstance(result, RiskAssessment)

    async def test_all_risk_scores_within_valid_range(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        result = await agent.run(FULL_CONTEXT)
        for wr in result.waypoint_risks:
            assert 0.0 <= wr.risk_score <= 1.0, (
                f"risk_score {wr.risk_score} out of range for {wr.waypoint_id}"
            )

    async def test_overall_risk_level_is_valid_enum(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        result = await agent.run(FULL_CONTEXT)
        assert result.overall_risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)

    async def test_reasoning_is_populated(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        result = await agent.run(FULL_CONTEXT)
        assert isinstance(result.reasoning, str) and len(result.reasoning) > 0

    async def test_waypoint_risks_list_is_populated(self):
        agent = SafetyAgent(llm_client=_make_mock_client())
        result = await agent.run(FULL_CONTEXT)
        assert len(result.waypoint_risks) == 3


# ---------------------------------------------------------------------------
# High-risk waypoint test (plan requirement)
# ---------------------------------------------------------------------------

class TestSafetyAgentHighRiskWaypoint:
    async def test_high_risk_overall_level(self):
        agent = SafetyAgent(llm_client=_make_mock_client(HIGH_RISK_RESPONSE))
        result = await agent.run(FULL_CONTEXT)
        assert result.overall_risk_level == RiskLevel.HIGH

    async def test_dangerous_waypoint_in_recommended_exclusions(self):
        """A waypoint above MAX_TERRAIN_RISK_SCORE must appear in exclusions."""
        agent = SafetyAgent(llm_client=_make_mock_client(HIGH_RISK_RESPONSE))
        result = await agent.run(FULL_CONTEXT)
        assert "wp-dangerous" in result.recommended_exclusions

    async def test_safe_waypoint_not_in_exclusions(self):
        agent = SafetyAgent(llm_client=_make_mock_client(HIGH_RISK_RESPONSE))
        result = await agent.run(FULL_CONTEXT)
        assert "wp-safe" not in result.recommended_exclusions

    async def test_medium_waypoint_not_in_exclusions(self):
        agent = SafetyAgent(llm_client=_make_mock_client(HIGH_RISK_RESPONSE))
        result = await agent.run(FULL_CONTEXT)
        assert "wp-medium" not in result.recommended_exclusions

    async def test_dangerous_waypoint_has_high_risk_score(self):
        agent = SafetyAgent(llm_client=_make_mock_client(HIGH_RISK_RESPONSE))
        result = await agent.run(FULL_CONTEXT)
        wp_scores = {wr.waypoint_id: wr.risk_score for wr in result.waypoint_risks}
        assert wp_scores["wp-dangerous"] > 0.60

    async def test_dangerous_waypoint_has_factors(self):
        agent = SafetyAgent(llm_client=_make_mock_client(HIGH_RISK_RESPONSE))
        result = await agent.run(FULL_CONTEXT)
        wp_factors = {wr.waypoint_id: wr.factors for wr in result.waypoint_risks}
        assert len(wp_factors["wp-dangerous"]) > 0


# ---------------------------------------------------------------------------
# All-low-risk test (plan requirement)
# ---------------------------------------------------------------------------

class TestSafetyAgentAllLowRisk:
    async def test_all_low_risk_level_is_low(self):
        agent = SafetyAgent(llm_client=_make_mock_client(ALL_LOW_RESPONSE))
        result = await agent.run(FULL_CONTEXT)
        assert result.overall_risk_level == RiskLevel.LOW

    async def test_all_low_exclusions_empty(self):
        """When all risks are LOW, no waypoints should be recommended for exclusion."""
        agent = SafetyAgent(llm_client=_make_mock_client(ALL_LOW_RESPONSE))
        result = await agent.run(FULL_CONTEXT)
        assert result.recommended_exclusions == []

    async def test_all_low_risk_scores_below_threshold(self):
        agent = SafetyAgent(llm_client=_make_mock_client(ALL_LOW_RESPONSE))
        result = await agent.run(FULL_CONTEXT)
        for wr in result.waypoint_risks:
            assert wr.risk_score <= 0.30, (
                f"{wr.waypoint_id} scored {wr.risk_score} but response claimed LOW"
            )

    async def test_medium_risk_level_with_no_exclusions(self):
        """MEDIUM risk with no exclusions is also a valid outcome."""
        agent = SafetyAgent(llm_client=_make_mock_client(MEDIUM_RISK_RESPONSE))
        result = await agent.run(FULL_CONTEXT)
        assert result.overall_risk_level == RiskLevel.MEDIUM
        assert result.recommended_exclusions == []


# ---------------------------------------------------------------------------
# MAX_TERRAIN_RISK_SCORE config injection
# ---------------------------------------------------------------------------

class TestSafetyAgentConfigInjection:
    async def test_max_terrain_risk_score_injected_into_user_message(self):
        mock_client = _make_mock_client()
        agent = SafetyAgent(llm_client=mock_client)
        await agent.run(FULL_CONTEXT)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        assert "max_terrain_risk_score" in parsed
        assert parsed["max_terrain_risk_score"] == pytest.approx(
            config.MAX_TERRAIN_RISK_SCORE
        )

    async def test_caller_cannot_override_max_terrain_risk_score(self):
        """Even if caller supplies a different value, config value wins."""
        mock_client = _make_mock_client()
        agent = SafetyAgent(llm_client=mock_client)
        tampered = {**FULL_CONTEXT, "max_terrain_risk_score": 0.0}
        await agent.run(tampered)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        assert parsed["max_terrain_risk_score"] == pytest.approx(
            config.MAX_TERRAIN_RISK_SCORE
        )
        assert parsed["max_terrain_risk_score"] > 0.0

    def test_max_terrain_risk_score_default_is_positive(self):
        assert config.MAX_TERRAIN_RISK_SCORE > 0.0


# ---------------------------------------------------------------------------
# Context forwarding
# ---------------------------------------------------------------------------

class TestSafetyAgentContextForwarding:
    async def test_candidate_waypoints_forwarded(self):
        mock_client = _make_mock_client()
        agent = SafetyAgent(llm_client=mock_client)
        await agent.run(FULL_CONTEXT)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        wp_ids = [wp["id"] for wp in parsed["candidate_waypoints"]]
        assert "wp-safe" in wp_ids
        assert "wp-dangerous" in wp_ids

    async def test_weather_forecast_forwarded(self):
        mock_client = _make_mock_client()
        agent = SafetyAgent(llm_client=mock_client)
        await agent.run(FULL_CONTEXT)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        assert "weather_forecast" in parsed
        assert parsed["weather_forecast"]["dust_storm_probability"] == pytest.approx(0.15)

    async def test_comm_windows_forwarded(self):
        mock_client = _make_mock_client()
        agent = SafetyAgent(llm_client=mock_client)
        await agent.run(FULL_CONTEXT)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        assert "comm_windows" in parsed
        assert len(parsed["comm_windows"]) == 2

    async def test_terrain_map_forwarded(self):
        mock_client = _make_mock_client()
        agent = SafetyAgent(llm_client=mock_client)
        await agent.run(FULL_CONTEXT)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        assert "terrain_map" in parsed
        assert "wp-safe" in parsed["terrain_map"]

    async def test_system_prompt_sent_to_llm(self):
        mock_client = _make_mock_client()
        agent = SafetyAgent(llm_client=mock_client)
        await agent.run(FULL_CONTEXT)
        assert mock_client.complete.call_args.kwargs["system_prompt"] == agent.system_prompt

    async def test_no_real_api_calls_made(self):
        mock_client = _make_mock_client()
        agent = SafetyAgent(llm_client=mock_client)
        await agent.run(FULL_CONTEXT)
        mock_client.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Malformed / invalid AI responses (plan requirement)
# ---------------------------------------------------------------------------

class TestSafetyAgentErrorHandling:
    async def test_malformed_json_raises_agent_response_error(self):
        mock_client = _make_mock_client("not valid json {{ broken")
        agent = SafetyAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run(FULL_CONTEXT)
        assert exc_info.value.agent_name == "safety"

    async def test_malformed_json_error_contains_raw_response(self):
        raw = "completely invalid"
        mock_client = _make_mock_client(raw)
        agent = SafetyAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run(FULL_CONTEXT)
        assert raw in exc_info.value.raw_response

    async def test_risk_score_above_one_raises_agent_response_error(self):
        """risk_score > 1.0 violates RiskAssessment schema — must be rejected."""
        bad_payload = {
            "waypoint_risks": [
                {"waypoint_id": "wp-safe", "risk_score": 2.5, "factors": ["impossible score"]}
            ],
            "overall_risk_level": "HIGH",
            "recommended_exclusions": ["wp-safe"],
            "reasoning": "bad data",
        }
        mock_client = _make_mock_client(bad_payload)
        agent = SafetyAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run(FULL_CONTEXT)
        assert exc_info.value.agent_name == "safety"

    async def test_negative_risk_score_raises_agent_response_error(self):
        bad_payload = {
            "waypoint_risks": [
                {"waypoint_id": "wp-safe", "risk_score": -0.1, "factors": []}
            ],
            "overall_risk_level": "LOW",
            "recommended_exclusions": [],
            "reasoning": "negative score",
        }
        mock_client = _make_mock_client(bad_payload)
        agent = SafetyAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError):
            await agent.run(FULL_CONTEXT)

    async def test_invalid_risk_level_raises_agent_response_error(self):
        """overall_risk_level must be LOW, MEDIUM, or HIGH — nothing else."""
        bad_payload = {
            "waypoint_risks": [],
            "overall_risk_level": "EXTREME",   # not a valid RiskLevel
            "recommended_exclusions": [],
            "reasoning": "invalid level",
        }
        mock_client = _make_mock_client(bad_payload)
        agent = SafetyAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError):
            await agent.run(FULL_CONTEXT)

    async def test_structural_failure_not_retried(self):
        mock_client = _make_mock_client("{ broken json")
        agent = SafetyAgent(llm_client=mock_client, max_retries=3, retry_delay=0.0)
        with pytest.raises(AgentResponseError):
            await agent.run(FULL_CONTEXT)
        mock_client.complete.assert_called_once()

    async def test_empty_string_response_raises_agent_response_error(self):
        mock_client = _make_mock_client("")
        agent = SafetyAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError):
            await agent.run(FULL_CONTEXT)


# ---------------------------------------------------------------------------
# RiskAssessment / WaypointRisk schema boundary values
# ---------------------------------------------------------------------------

class TestRiskAssessmentSchema:
    def test_risk_score_zero_is_valid(self):
        wr = WaypointRisk(waypoint_id="wp-x", risk_score=0.0)
        assert wr.risk_score == pytest.approx(0.0)

    def test_risk_score_one_is_valid(self):
        wr = WaypointRisk(waypoint_id="wp-x", risk_score=1.0)
        assert wr.risk_score == pytest.approx(1.0)

    def test_risk_score_above_one_rejected(self):
        with pytest.raises(ValidationError):
            WaypointRisk(waypoint_id="wp-x", risk_score=1.001)

    def test_risk_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            WaypointRisk(waypoint_id="wp-x", risk_score=-0.001)

    def test_overall_risk_level_defaults_to_low(self):
        ra = RiskAssessment()
        assert ra.overall_risk_level == RiskLevel.LOW

    def test_all_three_risk_levels_accepted(self):
        for level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH):
            ra = RiskAssessment(overall_risk_level=level)
            assert ra.overall_risk_level == level

    def test_empty_exclusions_is_valid(self):
        ra = RiskAssessment(recommended_exclusions=[])
        assert ra.recommended_exclusions == []

    def test_empty_waypoint_risks_is_valid(self):
        ra = RiskAssessment(waypoint_risks=[])
        assert ra.waypoint_risks == []

    async def test_boundary_risk_zero_passes_full_pipeline(self):
        payload = {
            "waypoint_risks": [
                {"waypoint_id": "wp-safe", "risk_score": 0.0, "factors": []}
            ],
            "overall_risk_level": "LOW",
            "recommended_exclusions": [],
            "reasoning": "Minimum risk.",
        }
        mock_client = _make_mock_client(payload)
        agent = SafetyAgent(llm_client=mock_client)
        result = await agent.run(FULL_CONTEXT)
        assert result.waypoint_risks[0].risk_score == pytest.approx(0.0)

    async def test_boundary_risk_one_passes_full_pipeline(self):
        payload = {
            "waypoint_risks": [
                {"waypoint_id": "wp-dangerous", "risk_score": 1.0,
                 "factors": ["critical hazard"]}
            ],
            "overall_risk_level": "HIGH",
            "recommended_exclusions": ["wp-dangerous"],
            "reasoning": "Maximum risk.",
        }
        mock_client = _make_mock_client(payload)
        agent = SafetyAgent(llm_client=mock_client)
        result = await agent.run(FULL_CONTEXT)
        assert result.waypoint_risks[0].risk_score == pytest.approx(1.0)
        assert "wp-dangerous" in result.recommended_exclusions


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSafetyAgentEdgeCases:
    async def test_empty_candidate_list_accepted(self):
        payload = {
            "waypoint_risks": [],
            "overall_risk_level": "LOW",
            "recommended_exclusions": [],
            "reasoning": "No candidates to assess.",
        }
        empty_context = {
            "candidate_waypoints": [],
            "weather_forecast": WEATHER_FORECAST,
            "comm_windows": COMM_WINDOWS,
            "terrain_map": {},
        }
        mock_client = _make_mock_client(payload)
        agent = SafetyAgent(llm_client=mock_client)
        result = await agent.run(empty_context)
        assert result.waypoint_risks == []
        assert result.recommended_exclusions == []

    async def test_all_waypoints_excluded_is_valid(self):
        """All candidates being excluded is a valid (if extreme) outcome."""
        all_excluded = {
            "waypoint_risks": [
                {"waypoint_id": "wp-safe",      "risk_score": 0.95, "factors": ["sensor error"]},
                {"waypoint_id": "wp-medium",    "risk_score": 0.92, "factors": ["sensor error"]},
                {"waypoint_id": "wp-dangerous", "risk_score": 0.99, "factors": ["sensor error"]},
            ],
            "overall_risk_level": "HIGH",
            "recommended_exclusions": ["wp-safe", "wp-medium", "wp-dangerous"],
            "reasoning": "Sensor malfunction — all waypoints flagged.",
        }
        mock_client = _make_mock_client(all_excluded)
        agent = SafetyAgent(llm_client=mock_client)
        result = await agent.run(FULL_CONTEXT)
        assert len(result.recommended_exclusions) == 3
        assert result.overall_risk_level == RiskLevel.HIGH

    async def test_single_waypoint_low_risk(self):
        single = [CANDIDATE_WAYPOINTS[0]]
        payload = {
            "waypoint_risks": [
                {"waypoint_id": "wp-safe", "risk_score": 0.08, "factors": ["flat terrain"]}
            ],
            "overall_risk_level": "LOW",
            "recommended_exclusions": [],
            "reasoning": "Single safe waypoint.",
        }
        context = {**FULL_CONTEXT, "candidate_waypoints": single}
        mock_client = _make_mock_client(payload)
        agent = SafetyAgent(llm_client=mock_client)
        result = await agent.run(context)
        assert len(result.waypoint_risks) == 1
        assert result.recommended_exclusions == []
