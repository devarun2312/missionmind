"""
Tests for Sub-Task 3: Science Agent.

Covers:
- ScienceAgent is a concrete, instantiable subclass of BaseAgent.
- ScienceAgent.name returns "science".
- ScienceAgent.response_schema returns ScienceAnalysis.
- ScienceAgent.system_prompt is non-empty and loaded from the prompt file.
- ScienceAgent.run() with a mocked client returning valid ScienceAnalysis JSON:
    * Returns a ScienceAnalysis instance.
    * Scores are within [0.0, 1.0].
    * priority_order and scored_targets are populated.
    * reasoning is present.
- ScienceAgent.run() with a mocked client returning malformed JSON raises AgentResponseError.
- Pydantic schema rejects scientific_value outside [0.0, 1.0].
- Candidate waypoints are forwarded into the context user message.
- rover_position and mission_objectives are forwarded into the context user message.
- The science system prompt is sent as the system message.
- No real external API / network calls are made in any test.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from missionmind.agents.base_agent import AgentResponseError, BaseAgent
from missionmind.agents.science_agent import ScienceAgent
from missionmind.agents.client import LLMResponse
from missionmind.schemas.outputs import ScienceAnalysis, ScoredTarget


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

# Minimal valid ScienceAnalysis JSON the mock LLM will return.
VALID_RESPONSE = {
    "scored_targets": [
        {
            "waypoint_id": "wp-crater-a",
            "scientific_value": 0.92,
            "justification": (
                "Exposed phyllosilicate deposits and proximity to ancient "
                "fluvial channels indicate strong biosignature potential."
            ),
        },
        {
            "waypoint_id": "wp-plain-b",
            "scientific_value": 0.15,
            "justification": (
                "Featureless basaltic plain with no hydration signatures; "
                "low mineralogical diversity."
            ),
        },
        {
            "waypoint_id": "wp-dune-c",
            "scientific_value": 0.48,
            "justification": (
                "Active aeolian dunes offer atmospheric and grain-size data "
                "but limited direct biosignature opportunity."
            ),
        },
    ],
    "priority_order": ["wp-crater-a", "wp-dune-c", "wp-plain-b"],
    "reasoning": (
        "The crater site dominates due to confirmed hydrated minerals. "
        "Dunes offer secondary atmospheric value. The plain has minimal interest."
    ),
}

CANDIDATE_WAYPOINTS = [
    {
        "id": "wp-crater-a",
        "x": 320.0,
        "y": -145.0,
        "scientific_value": 0.0,
        "terrain_risk": 0.3,
        "label": "crater-rim-A",
        "is_base": False,
        "estimated_travel_time_minutes": 25.0,
        "estimated_energy_wh": 38.0,
    },
    {
        "id": "wp-plain-b",
        "x": 110.0,
        "y": 60.0,
        "scientific_value": 0.0,
        "terrain_risk": 0.05,
        "label": "basalt-plain-B",
        "is_base": False,
        "estimated_travel_time_minutes": 10.0,
        "estimated_energy_wh": 14.0,
    },
    {
        "id": "wp-dune-c",
        "x": 205.0,
        "y": -80.0,
        "scientific_value": 0.0,
        "terrain_risk": 0.2,
        "label": "dune-field-C",
        "is_base": False,
        "estimated_travel_time_minutes": 18.0,
        "estimated_energy_wh": 26.0,
    },
]

ROVER_POSITION = {"x": 0.0, "y": 0.0}

MISSION_OBJECTIVES = [
    "search for biosignatures in ancient lake deposits",
    "characterise subsurface water-ice distribution",
    "collect atmospheric dust samples",
]


def _make_mock_client(payload: dict | str | None = None) -> MagicMock:
    """Return a mock LLM client with no real network activity.

    Parameters
    ----------
    payload:
        If a dict, it is serialised to JSON and used as the response content.
        If a str, it is used directly as the response content (allows malformed JSON).
        If None, defaults to VALID_RESPONSE.
    """
    if payload is None:
        content = json.dumps(VALID_RESPONSE)
    elif isinstance(payload, dict):
        content = json.dumps(payload)
    else:
        content = payload  # raw string — may be intentionally malformed

    mock = MagicMock()
    mock.complete = AsyncMock(return_value=LLMResponse(
        content=content,
        model="test-model",
        prompt_tokens=80,
        completion_tokens=200,
    ))
    return mock


# ---------------------------------------------------------------------------
# Class identity & interface
# ---------------------------------------------------------------------------

class TestScienceAgentInterface:
    def test_is_subclass_of_base_agent(self):
        assert issubclass(ScienceAgent, BaseAgent)

    def test_is_instantiable_with_mock_client(self):
        agent = ScienceAgent(llm_client=_make_mock_client())
        assert agent is not None

    def test_name_is_science(self):
        agent = ScienceAgent(llm_client=_make_mock_client())
        assert agent.name == "science"

    def test_response_schema_is_science_analysis(self):
        agent = ScienceAgent(llm_client=_make_mock_client())
        assert agent.response_schema is ScienceAnalysis

    def test_system_prompt_is_non_empty_string(self):
        agent = ScienceAgent(llm_client=_make_mock_client())
        prompt = agent.system_prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # substantial prompt, not a placeholder

    def test_system_prompt_contains_planetary_scientist_role(self):
        """The prompt must declare the planetary geologist / astrobiologist role."""
        agent = ScienceAgent(llm_client=_make_mock_client())
        prompt_lower = agent.system_prompt.lower()
        assert "geologist" in prompt_lower or "astrobiologist" in prompt_lower

    def test_system_prompt_references_scientific_value_range(self):
        """The prompt must tell the LLM the valid score range."""
        agent = ScienceAgent(llm_client=_make_mock_client())
        assert "0.0" in agent.system_prompt and "1.0" in agent.system_prompt

    def test_system_prompt_specifies_json_output(self):
        """The prompt must instruct the LLM to respond with JSON."""
        agent = ScienceAgent(llm_client=_make_mock_client())
        prompt_lower = agent.system_prompt.lower()
        assert "json" in prompt_lower

    def test_system_prompt_loaded_from_file(self):
        """Two independently created agents must have the same prompt content,
        confirming it is loaded from the prompt file, not generated dynamically."""
        agent_a = ScienceAgent(llm_client=_make_mock_client())
        agent_b = ScienceAgent(llm_client=_make_mock_client())
        assert agent_a.system_prompt == agent_b.system_prompt

    def test_can_import_from_agents_package(self):
        from missionmind.agents import ScienceAgent as SA  # noqa: F401
        assert SA is ScienceAgent


# ---------------------------------------------------------------------------
# Happy path — valid AI response
# ---------------------------------------------------------------------------

class TestScienceAgentHappyPath:
    async def test_run_returns_science_analysis_instance(self):
        agent = ScienceAgent(llm_client=_make_mock_client())
        context = {
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        }
        result = await agent.run(context)
        assert isinstance(result, ScienceAnalysis)

    async def test_run_returns_all_scored_targets(self):
        agent = ScienceAgent(llm_client=_make_mock_client())
        result = await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        assert len(result.scored_targets) == 3
        ids = {t.waypoint_id for t in result.scored_targets}
        assert ids == {"wp-crater-a", "wp-plain-b", "wp-dune-c"}

    async def test_run_scores_are_within_valid_range(self):
        agent = ScienceAgent(llm_client=_make_mock_client())
        result = await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        for target in result.scored_targets:
            assert 0.0 <= target.scientific_value <= 1.0, (
                f"Score {target.scientific_value} out of range for {target.waypoint_id}"
            )

    async def test_run_priority_order_is_populated(self):
        agent = ScienceAgent(llm_client=_make_mock_client())
        result = await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        assert len(result.priority_order) == 3
        assert result.priority_order[0] == "wp-crater-a"  # highest scored

    async def test_run_reasoning_is_populated(self):
        agent = ScienceAgent(llm_client=_make_mock_client())
        result = await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    async def test_run_justification_populated_per_target(self):
        agent = ScienceAgent(llm_client=_make_mock_client())
        result = await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        for target in result.scored_targets:
            assert isinstance(target.justification, str)

    async def test_run_highest_priority_has_highest_score(self):
        """priority_order[0] should correspond to the highest scientific_value."""
        agent = ScienceAgent(llm_client=_make_mock_client())
        result = await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        scores = {t.waypoint_id: t.scientific_value for t in result.scored_targets}
        top_id = result.priority_order[0]
        assert scores[top_id] == max(scores.values())


# ---------------------------------------------------------------------------
# Context forwarding — what gets sent to the LLM
# ---------------------------------------------------------------------------

class TestScienceAgentContextForwarding:
    async def test_candidate_waypoints_included_in_user_message(self):
        mock_client = _make_mock_client()
        agent = ScienceAgent(llm_client=mock_client)
        await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        call_kwargs = mock_client.complete.call_args.kwargs
        user_msg = call_kwargs["user_message"]
        parsed = json.loads(user_msg)
        assert "candidate_waypoints" in parsed
        wp_ids = [wp["id"] for wp in parsed["candidate_waypoints"]]
        assert "wp-crater-a" in wp_ids

    async def test_rover_position_included_in_user_message(self):
        mock_client = _make_mock_client()
        agent = ScienceAgent(llm_client=mock_client)
        await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        call_kwargs = mock_client.complete.call_args.kwargs
        parsed = json.loads(call_kwargs["user_message"])
        assert "rover_position" in parsed
        assert parsed["rover_position"]["x"] == pytest.approx(0.0)

    async def test_mission_objectives_included_in_user_message(self):
        mock_client = _make_mock_client()
        agent = ScienceAgent(llm_client=mock_client)
        await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        call_kwargs = mock_client.complete.call_args.kwargs
        parsed = json.loads(call_kwargs["user_message"])
        assert "mission_objectives" in parsed
        assert MISSION_OBJECTIVES[0] in parsed["mission_objectives"]

    async def test_science_system_prompt_sent_to_llm(self):
        mock_client = _make_mock_client()
        agent = ScienceAgent(llm_client=mock_client)
        await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["system_prompt"] == agent.system_prompt

    async def test_no_real_api_calls_are_made(self):
        """The mock must be the only thing called — no live network activity."""
        mock_client = _make_mock_client()
        agent = ScienceAgent(llm_client=mock_client)
        await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        # If a real call happened, complete would NOT have been called on our mock.
        mock_client.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Malformed / invalid AI responses
# ---------------------------------------------------------------------------

class TestScienceAgentErrorHandling:
    async def test_malformed_json_raises_agent_response_error(self):
        mock_client = _make_mock_client("this is not json {{ broken")
        agent = ScienceAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run({
                "candidate_waypoints": CANDIDATE_WAYPOINTS,
                "rover_position": ROVER_POSITION,
                "mission_objectives": MISSION_OBJECTIVES,
            })
        assert exc_info.value.agent_name == "science"

    async def test_malformed_json_error_contains_raw_response(self):
        raw = "not json at all"
        mock_client = _make_mock_client(raw)
        agent = ScienceAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run({"candidate_waypoints": [], "rover_position": {}, "mission_objectives": []})
        assert raw in exc_info.value.raw_response

    async def test_valid_json_wrong_schema_raises_agent_response_error(self):
        """scientific_value > 1.0 violates the Pydantic schema — must be rejected."""
        bad_payload = {
            "scored_targets": [
                {"waypoint_id": "wp-1", "scientific_value": 9.9,
                 "justification": "extreme value"}
            ],
            "priority_order": ["wp-1"],
            "reasoning": "out of range",
        }
        mock_client = _make_mock_client(bad_payload)
        agent = ScienceAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run({"candidate_waypoints": [], "rover_position": {}, "mission_objectives": []})
        assert exc_info.value.agent_name == "science"

    async def test_scientific_value_below_zero_rejected_by_schema(self):
        bad_payload = {
            "scored_targets": [
                {"waypoint_id": "wp-1", "scientific_value": -0.5,
                 "justification": "negative value"}
            ],
            "priority_order": ["wp-1"],
            "reasoning": "negative",
        }
        mock_client = _make_mock_client(bad_payload)
        agent = ScienceAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError):
            await agent.run({"candidate_waypoints": [], "rover_position": {}, "mission_objectives": []})

    async def test_llm_called_once_on_structural_failure(self):
        """Structural JSON failures must NOT trigger retry logic."""
        mock_client = _make_mock_client("{ broken json")
        agent = ScienceAgent(llm_client=mock_client, max_retries=3, retry_delay=0.0)
        with pytest.raises(AgentResponseError):
            await agent.run({"candidate_waypoints": [], "rover_position": {}, "mission_objectives": []})
        mock_client.complete.assert_called_once()

    async def test_empty_string_response_raises_agent_response_error(self):
        mock_client = _make_mock_client("")
        agent = ScienceAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError):
            await agent.run({"candidate_waypoints": [], "rover_position": {}, "mission_objectives": []})


# ---------------------------------------------------------------------------
# Schema boundary values — Pydantic clamping behaviour
# ---------------------------------------------------------------------------

class TestScienceAnalysisSchemaValidation:
    def test_score_at_zero_is_valid(self):
        target = ScoredTarget(waypoint_id="wp-x", scientific_value=0.0)
        assert target.scientific_value == pytest.approx(0.0)

    def test_score_at_one_is_valid(self):
        target = ScoredTarget(waypoint_id="wp-x", scientific_value=1.0)
        assert target.scientific_value == pytest.approx(1.0)

    def test_score_above_one_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ScoredTarget(waypoint_id="wp-x", scientific_value=1.001)

    def test_score_below_zero_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ScoredTarget(waypoint_id="wp-x", scientific_value=-0.001)

    def test_science_analysis_with_no_targets_is_valid(self):
        """An empty result is structurally valid — no waypoints to score."""
        analysis = ScienceAnalysis(
            scored_targets=[],
            priority_order=[],
            reasoning="No candidates provided.",
        )
        assert analysis.scored_targets == []

    async def test_boundary_score_exactly_zero_passes_full_pipeline(self):
        """Score of exactly 0.0 must pass Pydantic validation end-to-end."""
        payload = {
            "scored_targets": [
                {"waypoint_id": "wp-z", "scientific_value": 0.0,
                 "justification": "featureless plain"}
            ],
            "priority_order": ["wp-z"],
            "reasoning": "Minimum score.",
        }
        mock_client = _make_mock_client(payload)
        agent = ScienceAgent(llm_client=mock_client)
        result = await agent.run({"candidate_waypoints": [], "rover_position": {}, "mission_objectives": []})
        assert result.scored_targets[0].scientific_value == pytest.approx(0.0)

    async def test_boundary_score_exactly_one_passes_full_pipeline(self):
        """Score of exactly 1.0 must pass Pydantic validation end-to-end."""
        payload = {
            "scored_targets": [
                {"waypoint_id": "wp-top", "scientific_value": 1.0,
                 "justification": "maximum scientific interest"}
            ],
            "priority_order": ["wp-top"],
            "reasoning": "Maximum score.",
        }
        mock_client = _make_mock_client(payload)
        agent = ScienceAgent(llm_client=mock_client)
        result = await agent.run({"candidate_waypoints": [], "rover_position": {}, "mission_objectives": []})
        assert result.scored_targets[0].scientific_value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Minimal context — edge cases
# ---------------------------------------------------------------------------

class TestScienceAgentEdgeCases:
    async def test_empty_candidate_list_accepted(self):
        """An empty waypoint list is valid input; agent should handle it."""
        payload = {
            "scored_targets": [],
            "priority_order": [],
            "reasoning": "No candidates to evaluate.",
        }
        mock_client = _make_mock_client(payload)
        agent = ScienceAgent(llm_client=mock_client)
        result = await agent.run({
            "candidate_waypoints": [],
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        assert result.scored_targets == []
        assert result.priority_order == []

    async def test_single_waypoint_returns_single_target(self):
        single_wp = [CANDIDATE_WAYPOINTS[0]]
        payload = {
            "scored_targets": [
                {"waypoint_id": "wp-crater-a", "scientific_value": 0.92,
                 "justification": "Strong biosignature potential."}
            ],
            "priority_order": ["wp-crater-a"],
            "reasoning": "Single candidate; scored on its own merits.",
        }
        mock_client = _make_mock_client(payload)
        agent = ScienceAgent(llm_client=mock_client)
        result = await agent.run({
            "candidate_waypoints": single_wp,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        assert len(result.scored_targets) == 1
        assert result.scored_targets[0].waypoint_id == "wp-crater-a"

    async def test_context_with_only_required_keys_works(self):
        """Agent must work with the minimum required context keys."""
        mock_client = _make_mock_client()
        agent = ScienceAgent(llm_client=mock_client)
        # All three required keys present
        result = await agent.run({
            "candidate_waypoints": CANDIDATE_WAYPOINTS,
            "rover_position": ROVER_POSITION,
            "mission_objectives": MISSION_OBJECTIVES,
        })
        assert isinstance(result, ScienceAnalysis)
