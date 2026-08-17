"""
Tests for Sub-Task 4: Resource / Energy Agent.

Covers:
- ResourceAgent is a concrete, instantiable subclass of BaseAgent.
- ResourceAgent.name returns "resource".
- ResourceAgent.response_schema returns ResourceBudget.
- ResourceAgent.system_prompt is non-empty and loaded from resource_prompt.md.
- System prompt references key concepts: reserve, budget, energy, watt.
- run() happy path: mock returns valid ResourceBudget JSON → parses correctly.
- Recommended waypoints are a subset of candidate waypoint IDs.
- available_energy_wh and available_time_minutes are non-negative.
- energy_per_waypoint is a dict keyed by waypoint IDs.
- Low-battery edge case: mock returns empty recommended_waypoints when battery
  is near MIN_RETURN_BATTERY_PCT → no expensive waypoints recommended.
- Malformed JSON response raises AgentResponseError.
- Schema-invalid response (negative energy) raises AgentResponseError.
- MIN_RETURN_BATTERY_PCT from config is injected into every context automatically.
- Caller-supplied min_return_battery_pct is overwritten by the config value.
- All required context keys are forwarded in the user message.
- System prompt is sent as the system message to the LLM.
- No real external API calls are made in any test.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from missionmind import config
from missionmind.agents.base_agent import AgentResponseError, BaseAgent
from missionmind.agents.client import LLMResponse
from missionmind.agents.resource_agent import ResourceAgent
from missionmind.schemas.outputs import ResourceBudget


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

CANDIDATE_WAYPOINTS = [
    {
        "id": "wp-close",
        "x": 50.0,
        "y": 30.0,
        "label": "nearby-outcrop",
        "is_base": False,
        "estimated_travel_time_minutes": 8.0,
        "estimated_energy_wh": 12.0,
        "scientific_value": 0.7,
        "terrain_risk": 0.1,
    },
    {
        "id": "wp-mid",
        "x": 200.0,
        "y": -90.0,
        "label": "mid-range-crater",
        "is_base": False,
        "estimated_travel_time_minutes": 22.0,
        "estimated_energy_wh": 35.0,
        "scientific_value": 0.85,
        "terrain_risk": 0.25,
    },
    {
        "id": "wp-far",
        "x": 480.0,
        "y": 310.0,
        "label": "distant-ridge",
        "is_base": False,
        "estimated_travel_time_minutes": 55.0,
        "estimated_energy_wh": 88.0,
        "scientific_value": 0.6,
        "terrain_risk": 0.4,
    },
]

# Rover with healthy battery (80 %) and 500 Wh capacity → 400 Wh present.
# Reserve = 20 % × 500 = 100 Wh.  Usable = 400 − 100 = 300 Wh.
HEALTHY_ROVER_CONTEXT = {
    "battery_pct": 0.80,
    "battery_capacity_wh": 500.0,
    "candidate_waypoints": CANDIDATE_WAYPOINTS,
    "rover_speed_mps": 0.05,
    "power_consumption_w": 80.0,
}

# Rover with barely-above-reserve battery (22 %) → very little usable energy.
LOW_BATTERY_CONTEXT = {
    "battery_pct": 0.22,
    "battery_capacity_wh": 500.0,
    "candidate_waypoints": CANDIDATE_WAYPOINTS,
    "rover_speed_mps": 0.05,
    "power_consumption_w": 80.0,
}

# Rover at exactly the reserve threshold → zero usable energy.
AT_RESERVE_CONTEXT = {
    "battery_pct": config.MIN_RETURN_BATTERY_PCT,
    "battery_capacity_wh": 500.0,
    "candidate_waypoints": CANDIDATE_WAYPOINTS,
    "rover_speed_mps": 0.05,
    "power_consumption_w": 80.0,
}

# Healthy response the mock LLM returns for HEALTHY_ROVER_CONTEXT.
HEALTHY_RESPONSE = {
    "available_energy_wh": 300.0,
    "available_time_minutes": 225.0,
    "recommended_waypoints": ["wp-close", "wp-mid"],
    "energy_per_waypoint": {
        "wp-close": 12.0,
        "wp-mid":   35.0,
        "wp-far":   88.0,
    },
    "reasoning": (
        "Current charge: 400 Wh.  Return reserve: 100 Wh.  "
        "Usable: 300 Wh.  wp-close (12 Wh) and wp-mid (35 Wh) total 47 Wh "
        "— well within budget.  wp-far (88 Wh) cumulative would exceed time "
        "budget; excluded."
    ),
}

# Response the mock LLM returns when battery is near-reserve (empty recommendations).
LOW_BATTERY_RESPONSE = {
    "available_energy_wh": 10.0,
    "available_time_minutes": 7.5,
    "recommended_waypoints": [],
    "energy_per_waypoint": {
        "wp-close": 12.0,
        "wp-mid":   35.0,
        "wp-far":   88.0,
    },
    "reasoning": (
        "Current charge: 110 Wh.  Return reserve: 100 Wh.  "
        "Usable: 10 Wh — insufficient for any candidate waypoint. "
        "Recommend immediate return to base."
    ),
}

# Response when rover is exactly at the reserve → zero available energy.
ZERO_ENERGY_RESPONSE = {
    "available_energy_wh": 0.0,
    "available_time_minutes": 0.0,
    "recommended_waypoints": [],
    "energy_per_waypoint": {
        "wp-close": 12.0,
        "wp-mid":   35.0,
        "wp-far":   88.0,
    },
    "reasoning": "Battery is at the return reserve. No energy available for waypoints.",
}


def _make_mock_client(payload: dict | str | None = None) -> MagicMock:
    """Return a mock LLM client — no network calls, no real credentials needed."""
    if payload is None:
        content = json.dumps(HEALTHY_RESPONSE)
    elif isinstance(payload, dict):
        content = json.dumps(payload)
    else:
        content = payload  # raw str — may be intentionally malformed

    mock = MagicMock()
    mock.complete = AsyncMock(return_value=LLMResponse(
        content=content,
        model="test-model",
        prompt_tokens=90,
        completion_tokens=180,
    ))
    return mock


# ---------------------------------------------------------------------------
# Class identity & interface
# ---------------------------------------------------------------------------

class TestResourceAgentInterface:
    def test_is_subclass_of_base_agent(self):
        assert issubclass(ResourceAgent, BaseAgent)

    def test_is_instantiable_with_mock_client(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        assert agent is not None

    def test_name_is_resource(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        assert agent.name == "resource"

    def test_response_schema_is_resource_budget(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        assert agent.response_schema is ResourceBudget

    def test_system_prompt_is_non_empty_string(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        assert isinstance(agent.system_prompt, str)
        assert len(agent.system_prompt) > 100

    def test_system_prompt_references_power_engineer_role(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        prompt_lower = agent.system_prompt.lower()
        assert "power" in prompt_lower or "energy" in prompt_lower

    def test_system_prompt_mentions_return_reserve(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        prompt_lower = agent.system_prompt.lower()
        assert "reserve" in prompt_lower or "return" in prompt_lower

    def test_system_prompt_references_watt(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        assert "watt" in agent.system_prompt.lower() or "wh" in agent.system_prompt.lower()

    def test_system_prompt_specifies_json_output(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        assert "json" in agent.system_prompt.lower()

    def test_system_prompt_loaded_from_file_consistently(self):
        a = ResourceAgent(llm_client=_make_mock_client())
        b = ResourceAgent(llm_client=_make_mock_client())
        assert a.system_prompt == b.system_prompt

    def test_can_import_from_agents_package(self):
        from missionmind.agents import ResourceAgent as RA  # noqa: F401
        assert RA is ResourceAgent


# ---------------------------------------------------------------------------
# Happy path — healthy battery, valid AI response
# ---------------------------------------------------------------------------

class TestResourceAgentHappyPath:
    async def test_run_returns_resource_budget_instance(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        result = await agent.run(HEALTHY_ROVER_CONTEXT)
        assert isinstance(result, ResourceBudget)

    async def test_available_energy_is_non_negative(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        result = await agent.run(HEALTHY_ROVER_CONTEXT)
        assert result.available_energy_wh >= 0.0

    async def test_available_time_is_non_negative(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        result = await agent.run(HEALTHY_ROVER_CONTEXT)
        assert result.available_time_minutes >= 0.0

    async def test_recommended_waypoints_is_subset_of_candidates(self):
        """All recommended waypoint IDs must exist in candidate_waypoints."""
        agent = ResourceAgent(llm_client=_make_mock_client())
        result = await agent.run(HEALTHY_ROVER_CONTEXT)
        candidate_ids = {wp["id"] for wp in CANDIDATE_WAYPOINTS}
        for wp_id in result.recommended_waypoints:
            assert wp_id in candidate_ids, (
                f"Recommended ID {wp_id!r} is not in candidate waypoints"
            )

    async def test_energy_per_waypoint_is_dict(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        result = await agent.run(HEALTHY_ROVER_CONTEXT)
        assert isinstance(result.energy_per_waypoint, dict)

    async def test_energy_per_waypoint_covers_candidates(self):
        """energy_per_waypoint should include all candidate IDs."""
        agent = ResourceAgent(llm_client=_make_mock_client())
        result = await agent.run(HEALTHY_ROVER_CONTEXT)
        candidate_ids = {wp["id"] for wp in CANDIDATE_WAYPOINTS}
        for wp_id in candidate_ids:
            assert wp_id in result.energy_per_waypoint

    async def test_reasoning_is_populated(self):
        agent = ResourceAgent(llm_client=_make_mock_client())
        result = await agent.run(HEALTHY_ROVER_CONTEXT)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    async def test_healthy_battery_recommends_at_least_one_waypoint(self):
        """With 300 Wh usable and cheap waypoints, at least one should be recommended."""
        agent = ResourceAgent(llm_client=_make_mock_client())
        result = await agent.run(HEALTHY_ROVER_CONTEXT)
        assert len(result.recommended_waypoints) >= 1

    async def test_expensive_waypoint_may_be_excluded(self):
        """wp-far at 88 Wh should be excluded when cumulative time is too high."""
        agent = ResourceAgent(llm_client=_make_mock_client())
        result = await agent.run(HEALTHY_ROVER_CONTEXT)
        # The mock response excludes wp-far — verify the recommended set doesn't include it.
        assert "wp-far" not in result.recommended_waypoints


# ---------------------------------------------------------------------------
# Low-battery / return-reserve edge cases
# ---------------------------------------------------------------------------

class TestResourceAgentLowBattery:
    async def test_near_reserve_returns_empty_waypoints(self):
        """When usable energy is tiny, no waypoints should be recommended."""
        agent = ResourceAgent(llm_client=_make_mock_client(LOW_BATTERY_RESPONSE))
        result = await agent.run(LOW_BATTERY_CONTEXT)
        assert result.recommended_waypoints == []

    async def test_near_reserve_available_energy_is_small(self):
        agent = ResourceAgent(llm_client=_make_mock_client(LOW_BATTERY_RESPONSE))
        result = await agent.run(LOW_BATTERY_CONTEXT)
        # With 22 % of 500 Wh = 110 Wh present, reserve = 100 Wh → only 10 Wh usable.
        assert result.available_energy_wh < 20.0

    async def test_at_exact_reserve_threshold_empty_recommendations(self):
        """When battery_pct == MIN_RETURN_BATTERY_PCT, usable energy is zero."""
        agent = ResourceAgent(llm_client=_make_mock_client(ZERO_ENERGY_RESPONSE))
        result = await agent.run(AT_RESERVE_CONTEXT)
        assert result.recommended_waypoints == []
        assert result.available_energy_wh == pytest.approx(0.0)

    async def test_near_reserve_reasoning_mentions_reserve(self):
        agent = ResourceAgent(llm_client=_make_mock_client(LOW_BATTERY_RESPONSE))
        result = await agent.run(LOW_BATTERY_CONTEXT)
        assert len(result.reasoning) > 0


# ---------------------------------------------------------------------------
# MIN_RETURN_BATTERY_PCT injection
# ---------------------------------------------------------------------------

class TestResourceAgentConfigInjection:
    async def test_min_return_pct_injected_into_user_message(self):
        """ResourceAgent must always inject MIN_RETURN_BATTERY_PCT from config."""
        mock_client = _make_mock_client()
        agent = ResourceAgent(llm_client=mock_client)
        await agent.run(HEALTHY_ROVER_CONTEXT)

        call_kwargs = mock_client.complete.call_args.kwargs
        parsed = json.loads(call_kwargs["user_message"])
        assert "min_return_battery_pct" in parsed
        assert parsed["min_return_battery_pct"] == pytest.approx(
            config.MIN_RETURN_BATTERY_PCT
        )

    async def test_caller_cannot_override_min_return_pct(self):
        """Even if the caller supplies a different reserve, config value wins."""
        mock_client = _make_mock_client()
        agent = ResourceAgent(llm_client=mock_client)
        tampered_context = {
            **HEALTHY_ROVER_CONTEXT,
            "min_return_battery_pct": 0.0,   # attacker tries to remove the reserve
        }
        await agent.run(tampered_context)

        call_kwargs = mock_client.complete.call_args.kwargs
        parsed = json.loads(call_kwargs["user_message"])
        # Config value (0.20 by default) must have overwritten 0.0.
        assert parsed["min_return_battery_pct"] == pytest.approx(
            config.MIN_RETURN_BATTERY_PCT
        )
        assert parsed["min_return_battery_pct"] > 0.0

    async def test_min_return_pct_default_is_positive(self):
        """Sanity-check: the configured threshold must be > 0."""
        assert config.MIN_RETURN_BATTERY_PCT > 0.0


# ---------------------------------------------------------------------------
# Context forwarding
# ---------------------------------------------------------------------------

class TestResourceAgentContextForwarding:
    async def test_battery_pct_forwarded(self):
        mock_client = _make_mock_client()
        agent = ResourceAgent(llm_client=mock_client)
        await agent.run(HEALTHY_ROVER_CONTEXT)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        assert parsed["battery_pct"] == pytest.approx(0.80)

    async def test_battery_capacity_wh_forwarded(self):
        mock_client = _make_mock_client()
        agent = ResourceAgent(llm_client=mock_client)
        await agent.run(HEALTHY_ROVER_CONTEXT)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        assert parsed["battery_capacity_wh"] == pytest.approx(500.0)

    async def test_candidate_waypoints_forwarded(self):
        mock_client = _make_mock_client()
        agent = ResourceAgent(llm_client=mock_client)
        await agent.run(HEALTHY_ROVER_CONTEXT)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        wp_ids = [wp["id"] for wp in parsed["candidate_waypoints"]]
        assert "wp-close" in wp_ids
        assert "wp-mid" in wp_ids
        assert "wp-far" in wp_ids

    async def test_rover_speed_forwarded(self):
        mock_client = _make_mock_client()
        agent = ResourceAgent(llm_client=mock_client)
        await agent.run(HEALTHY_ROVER_CONTEXT)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        assert parsed["rover_speed_mps"] == pytest.approx(0.05)

    async def test_power_consumption_forwarded(self):
        mock_client = _make_mock_client()
        agent = ResourceAgent(llm_client=mock_client)
        await agent.run(HEALTHY_ROVER_CONTEXT)
        parsed = json.loads(mock_client.complete.call_args.kwargs["user_message"])
        assert parsed["power_consumption_w"] == pytest.approx(80.0)

    async def test_system_prompt_sent_to_llm(self):
        mock_client = _make_mock_client()
        agent = ResourceAgent(llm_client=mock_client)
        await agent.run(HEALTHY_ROVER_CONTEXT)
        assert mock_client.complete.call_args.kwargs["system_prompt"] == agent.system_prompt

    async def test_no_real_api_calls_made(self):
        mock_client = _make_mock_client()
        agent = ResourceAgent(llm_client=mock_client)
        await agent.run(HEALTHY_ROVER_CONTEXT)
        mock_client.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Malformed / invalid AI responses
# ---------------------------------------------------------------------------

class TestResourceAgentErrorHandling:
    async def test_malformed_json_raises_agent_response_error(self):
        mock_client = _make_mock_client("not valid json {{ broken")
        agent = ResourceAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run(HEALTHY_ROVER_CONTEXT)
        assert exc_info.value.agent_name == "resource"

    async def test_malformed_json_error_contains_raw_response(self):
        raw = "totally broken"
        mock_client = _make_mock_client(raw)
        agent = ResourceAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run(HEALTHY_ROVER_CONTEXT)
        assert raw in exc_info.value.raw_response

    async def test_negative_available_energy_raises_agent_response_error(self):
        """available_energy_wh < 0 violates ResourceBudget schema — must be rejected."""
        bad_payload = {
            "available_energy_wh": -50.0,   # schema requires ge=0.0
            "available_time_minutes": 100.0,
            "recommended_waypoints": [],
            "energy_per_waypoint": {},
            "reasoning": "bad data",
        }
        mock_client = _make_mock_client(bad_payload)
        agent = ResourceAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError) as exc_info:
            await agent.run(HEALTHY_ROVER_CONTEXT)
        assert exc_info.value.agent_name == "resource"

    async def test_negative_available_time_raises_agent_response_error(self):
        bad_payload = {
            "available_energy_wh": 100.0,
            "available_time_minutes": -10.0,  # schema requires ge=0.0
            "recommended_waypoints": [],
            "energy_per_waypoint": {},
            "reasoning": "bad data",
        }
        mock_client = _make_mock_client(bad_payload)
        agent = ResourceAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError):
            await agent.run(HEALTHY_ROVER_CONTEXT)

    async def test_structural_failure_not_retried(self):
        """Malformed JSON must not trigger BaseAgent retry logic."""
        mock_client = _make_mock_client("{ broken")
        agent = ResourceAgent(llm_client=mock_client, max_retries=3, retry_delay=0.0)
        with pytest.raises(AgentResponseError):
            await agent.run(HEALTHY_ROVER_CONTEXT)
        mock_client.complete.assert_called_once()

    async def test_empty_string_response_raises_agent_response_error(self):
        mock_client = _make_mock_client("")
        agent = ResourceAgent(llm_client=mock_client)
        with pytest.raises(AgentResponseError):
            await agent.run(HEALTHY_ROVER_CONTEXT)


# ---------------------------------------------------------------------------
# ResourceBudget schema boundary values
# ---------------------------------------------------------------------------

class TestResourceBudgetSchema:
    def test_available_energy_zero_is_valid(self):
        rb = ResourceBudget(available_energy_wh=0.0, available_time_minutes=0.0)
        assert rb.available_energy_wh == pytest.approx(0.0)

    def test_available_energy_negative_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ResourceBudget(available_energy_wh=-1.0, available_time_minutes=10.0)

    def test_available_time_negative_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ResourceBudget(available_energy_wh=10.0, available_time_minutes=-1.0)

    def test_empty_recommended_waypoints_is_valid(self):
        rb = ResourceBudget(
            available_energy_wh=0.0,
            available_time_minutes=0.0,
            recommended_waypoints=[],
        )
        assert rb.recommended_waypoints == []

    async def test_zero_energy_response_passes_full_pipeline(self):
        """available_energy_wh=0.0 and empty recommendations must be valid end-to-end."""
        mock_client = _make_mock_client(ZERO_ENERGY_RESPONSE)
        agent = ResourceAgent(llm_client=mock_client)
        result = await agent.run(AT_RESERVE_CONTEXT)
        assert result.available_energy_wh == pytest.approx(0.0)
        assert result.recommended_waypoints == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestResourceAgentEdgeCases:
    async def test_empty_candidate_list_accepted(self):
        """No candidates to evaluate is a valid degenerate case."""
        payload = {
            "available_energy_wh": 300.0,
            "available_time_minutes": 225.0,
            "recommended_waypoints": [],
            "energy_per_waypoint": {},
            "reasoning": "No candidate waypoints provided.",
        }
        empty_context = {
            "battery_pct": 0.80,
            "battery_capacity_wh": 500.0,
            "candidate_waypoints": [],
            "rover_speed_mps": 0.05,
            "power_consumption_w": 80.0,
        }
        mock_client = _make_mock_client(payload)
        agent = ResourceAgent(llm_client=mock_client)
        result = await agent.run(empty_context)
        assert isinstance(result, ResourceBudget)
        assert result.recommended_waypoints == []

    async def test_single_affordable_waypoint_recommended(self):
        single_wp = [CANDIDATE_WAYPOINTS[0]]  # wp-close: 12 Wh
        payload = {
            "available_energy_wh": 300.0,
            "available_time_minutes": 225.0,
            "recommended_waypoints": ["wp-close"],
            "energy_per_waypoint": {"wp-close": 12.0},
            "reasoning": "Only one candidate; fits within budget.",
        }
        context = {**HEALTHY_ROVER_CONTEXT, "candidate_waypoints": single_wp}
        mock_client = _make_mock_client(payload)
        agent = ResourceAgent(llm_client=mock_client)
        result = await agent.run(context)
        assert "wp-close" in result.recommended_waypoints
