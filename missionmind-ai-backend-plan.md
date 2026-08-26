# MissionMind — AI & Mission-Intelligence Backend Plan

## Scope

**Owner:** Person 1 (AI & mission-intelligence backend)

**Goal:** Build the AI-powered autonomous mission planning backend for a simulated Mars rover.
The system generates mission plans via specialised AI agents, validates them with
deterministic safety rules, executes them against a simulation interface, and replans in
real-time when the environment changes.

**Out of scope for this plan:** Frontend, simulation engine internals, final integration
wiring. The backend exposes clean Python APIs that the simulation and frontend teams will call.

---

## Architecture Overview

```
missionmind/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # shared agent contract
│   ├── science_agent.py       # scores scientific targets
│   ├── resource_agent.py      # models energy / battery / time budgets
│   ├── safety_agent.py        # soft-constraint risk assessment
│   └── mission_commander.py   # orchestrates sub-agents, emits MissionPlan
├── safety/
│   ├── __init__.py
│   └── validator.py           # deterministic hard-constraint checker
├── planning/
│   ├── __init__.py
│   ├── planner.py             # builds the initial MissionPlan
│   └── replanner.py           # reacts to environmental events
├── models/
│   ├── __init__.py
│   ├── mission.py             # MissionPlan, Waypoint, MissionStatus dataclasses
│   └── events.py              # BatteryFailure, CommLoss, TerrainHazard, NewDiscovery
├── schemas/
│   ├── __init__.py
│   └── outputs.py             # Pydantic models for all structured AI responses
├── config.py                  # runtime constants, thresholds, LLM model names
├── prompts/
│   ├── science_prompt.md
│   ├── resource_prompt.md
│   ├── safety_prompt.md
│   └── commander_prompt.md
└── tests/
    ├── __init__.py
    ├── test_science_agent.py
    ├── test_resource_agent.py
    ├── test_safety_agent.py
    ├── test_mission_commander.py
    ├── test_validator.py
    ├── test_planner.py
    └── test_replanner.py
```

Additional root-level files:
```
missionmind/
├── requirements.txt
├── pyproject.toml
└── README.md                  (expand existing)
```

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffold & Shared Models

**Intent**
Establish the Python package structure, install configuration, shared data models, and
structured output schemas before any agent or planner code is written. Every subsequent
sub-task depends on these types being stable.

**Expected Outcomes**
- `missionmind/` is a proper Python package (`pyproject.toml` / `requirements.txt`).
- All domain types (`MissionPlan`, `Waypoint`, `MissionStatus`, event types) exist as
  validated dataclasses or Pydantic models.
- All structured AI response schemas exist as Pydantic models in `schemas/outputs.py`.
- A `config.py` with editable thresholds and model names is present.
- The package imports cleanly (`python -c "import missionmind"`).

**Todo List**
1. Create `missionmind/pyproject.toml` declaring the package, Python ≥ 3.11 requirement,
   and dependencies: `pydantic`, `openai` (IBM watsonx-compatible), `pytest`, `pytest-asyncio`.
2. Create `missionmind/requirements.txt` mirroring the runtime deps for teammates.
3. Create `missionmind/__init__.py`.
4. Create `missionmind/config.py` with:
   - LLM model name constant (IBM watsonx model ID or `gpt-4o` placeholder).
   - Battery thresholds: `MIN_RETURN_BATTERY_PCT`, `CRITICAL_BATTERY_PCT`.
   - Time budget constants: `MAX_MISSION_DURATION_HOURS`.
   - Communication blackout timeout: `COMM_TIMEOUT_SECONDS`.
   - Max terrain risk score: `MAX_TERRAIN_RISK_SCORE`.
5. Create `missionmind/models/__init__.py` and `missionmind/models/mission.py` with:
   - `Waypoint` — id, coordinates (x, y), scientific_value (0-1), terrain_risk (0-1),
     estimated_travel_time_minutes, estimated_energy_wh.
   - `MissionStatus` — enum: PENDING, ACTIVE, REPLANNING, ABORTED, COMPLETE.
   - `MissionPlan` — plan_id, waypoints list, total_energy_wh, total_time_minutes,
     status, created_at, reasoning (str).
6. Create `missionmind/models/events.py` with:
   - `EventType` enum: BATTERY_FAILURE, COMM_LOSS, TERRAIN_HAZARD, NEW_DISCOVERY, RETURN_TO_BASE.
   - `MissionEvent` dataclass with event_type, severity (0-1), payload (dict), timestamp.
7. Create `missionmind/schemas/__init__.py` and `missionmind/schemas/outputs.py` with Pydantic
   models for each agent's structured response:
   - `ScienceAnalysis` — scored targets list, recommended priority order, reasoning.
   - `ResourceBudget` — available_energy_wh, available_time_minutes, recommended_waypoints,
     energy_per_waypoint dict, reasoning.
   - `RiskAssessment` — per-waypoint risk scores, overall_risk_level (LOW/MEDIUM/HIGH),
     recommended_exclusions list, reasoning.
   - `MissionPlanOutput` — the full structured plan that MissionCommander emits, mirrors
     `MissionPlan` fields plus a confidence score.
8. Create `missionmind/prompts/` directory with placeholder `.md` files for each agent's
   system prompt (content filled in Sub-Task 3).

**Relevant Context**
- No existing code; greenfield.
- Pydantic v2 is preferred for schema validation.
- All models must be serialisable to JSON for the frontend/simulation teams.

**Dependencies**
- None (this is the foundation).

**Tests**
- `tests/test_models.py` — instantiate each model/dataclass with valid data, assert field
  types and defaults, test Pydantic validation errors on bad input.

**Status:** [x] done

---

### Sub-Task 2 — Base Agent Contract

**Intent**
Define a shared abstract base class that all agents inherit from, establishing a consistent
async interface, LLM call helper, and structured-output parsing. This prevents duplicated
LLM plumbing in every agent file.

**Expected Outcomes**
- `agents/base_agent.py` contains `BaseAgent` abstract class.
- LLM call logic is centralised (model name, temperature, JSON response format).
- Each concrete agent only needs to supply a system prompt and a response model.
- Agents can be instantiated and called with `await agent.run(context: dict) -> BaseModel`.

**Todo List**
1. Create `missionmind/agents/__init__.py`.
2. Create `missionmind/agents/base_agent.py`:
   - `BaseAgent(ABC)` with abstract properties: `system_prompt: str`,
     `response_schema: type[BaseModel]`.
   - Concrete async method `run(context: dict) -> BaseModel`:
     - Serialises context to a user message string.
     - Calls LLM with `response_format={"type": "json_object"}` (or IBM equivalent).
     - Parses and validates the JSON response against `response_schema`.
     - Raises `AgentResponseError` on parse failure.
   - `AgentResponseError` custom exception.
   - Constructor accepts an optional `llm_client` parameter (dependency-injectable for
     testing without live LLM calls).

**Relevant Context**
- `config.py` (Sub-Task 1) provides the model name.
- `schemas/outputs.py` (Sub-Task 1) provides the response models.
- IBM watsonx AI uses an OpenAI-compatible client interface — the client is injected so
  tests can pass a mock.

**Dependencies**
- Sub-Task 1 (config, schemas).

**Tests**
- `tests/test_base_agent.py` — create a minimal concrete subclass, inject a mock LLM
  client that returns valid JSON, assert `run()` returns a parsed Pydantic model.
  Test that malformed JSON raises `AgentResponseError`.

**Status:** [x] done

---

### Sub-Task 3 — Science Agent

**Intent**
The Science Agent analyses a list of candidate waypoints and scores them by scientific
value (geology, atmosphere, water-ice proximity) to recommend a prioritised target list
for the mission plan.

**Expected Outcomes**
- `agents/science_agent.py` is a concrete `BaseAgent` subclass.
- `prompts/science_prompt.md` contains the system prompt instructing the LLM to act as a
  planetary scientist scoring targets.
- Given a context dict with candidate waypoints and rover position, the agent returns a
  `ScienceAnalysis` Pydantic object.

**Todo List**
1. Write `missionmind/prompts/science_prompt.md`:
   - Role: planetary geologist / astrobiologist.
   - Input: list of candidate waypoints with coordinates and known terrain data.
   - Task: score each waypoint 0.0–1.0 for scientific value and justify each score.
   - Output: strict JSON matching `ScienceAnalysis` schema.
2. Create `missionmind/agents/science_agent.py`:
   - `ScienceAgent(BaseAgent)`.
   - `system_prompt` property reads `science_prompt.md`.
   - `response_schema` property returns `ScienceAnalysis`.
   - `run(context)` context keys: `candidate_waypoints`, `rover_position`, `mission_objectives`.

**Relevant Context**
- `schemas/outputs.py` → `ScienceAnalysis`.
- `models/mission.py` → `Waypoint`.

**Dependencies**
- Sub-Task 1 (schemas), Sub-Task 2 (BaseAgent).

**Tests**
- `tests/test_science_agent.py`:
  - Mock LLM returns a valid `ScienceAnalysis` JSON blob → assert parsed correctly.
  - Mock returns malformed JSON → assert `AgentResponseError` raised.
  - Test that science scores are clamped to [0, 1] (Pydantic validator).

**Status:** [x] done

---

### Sub-Task 4 — Resource / Energy Agent

**Intent**
The Resource Agent calculates the rover's available energy and time budget, estimates the
energy cost of each waypoint in the candidate plan, and recommends which waypoints are
reachable within the safe return margin.

**Expected Outcomes**
- `agents/resource_agent.py` is a concrete `BaseAgent` subclass.
- `prompts/resource_prompt.md` instructs the LLM on rover power specs, battery state,
  and travel energy estimation.
- Given current battery state and candidate waypoints, returns a `ResourceBudget` object
  recommending a feasible subset.

**Todo List**
1. Write `missionmind/prompts/resource_prompt.md`:
   - Role: power systems engineer / mission controller.
   - Input: current battery %, capacity Wh, candidate waypoints with distances, rover power
     consumption model.
   - Task: estimate energy per waypoint, compute remaining budget after mandatory return
     reserve, recommend feasible waypoints.
   - Output: strict JSON matching `ResourceBudget` schema.
2. Create `missionmind/agents/resource_agent.py`:
   - `ResourceAgent(BaseAgent)`.
   - `system_prompt` reads `resource_prompt.md`.
   - `response_schema` returns `ResourceBudget`.
   - Context keys: `battery_pct`, `battery_capacity_wh`, `candidate_waypoints`,
     `rover_speed_mps`, `power_consumption_w`.

**Relevant Context**
- `schemas/outputs.py` → `ResourceBudget`.
- `config.py` → `MIN_RETURN_BATTERY_PCT`.

**Dependencies**
- Sub-Task 1, Sub-Task 2.

**Tests**
- `tests/test_resource_agent.py`:
  - Happy path: mock returns valid `ResourceBudget`, verify recommended_waypoints subset.
  - Edge case: battery already near minimum → recommended_waypoints should be empty or
    only base return.
  - Malformed response raises `AgentResponseError`.

**Status:** [x] done

---

### Sub-Task 5 — Safety Agent

**Intent**
The Safety Agent performs a soft-constraint risk assessment — it reasons about terrain
hazards, dust storm probability, communication windows, and slope gradients to assign a
per-waypoint risk level. This is the AI layer; hard constraints are enforced by the
deterministic validator in Sub-Task 6.

**Expected Outcomes**
- `agents/safety_agent.py` is a concrete `BaseAgent` subclass.
- `prompts/safety_prompt.md` instructs the LLM to act as a mission safety officer.
- Returns `RiskAssessment` with per-waypoint risk scores and recommended exclusions.

**Todo List**
1. Write `missionmind/prompts/safety_prompt.md`:
   - Role: Mars mission safety officer.
   - Input: waypoints with terrain_risk scores, current weather data, comm windows.
   - Task: assess overall mission risk, flag waypoints exceeding acceptable thresholds,
     provide recommended exclusions with justification.
   - Output: strict JSON matching `RiskAssessment` schema.
2. Create `missionmind/agents/safety_agent.py`:
   - `SafetyAgent(BaseAgent)`.
   - `system_prompt` reads `safety_prompt.md`.
   - `response_schema` returns `RiskAssessment`.
   - Context keys: `candidate_waypoints`, `weather_forecast`, `comm_windows`,
     `terrain_map`.

**Relevant Context**
- `schemas/outputs.py` → `RiskAssessment`.
- `config.py` → `MAX_TERRAIN_RISK_SCORE`.

**Dependencies**
- Sub-Task 1, Sub-Task 2.

**Tests**
- `tests/test_safety_agent.py`:
  - Mock returns a `RiskAssessment` with some HIGH-risk waypoints; verify exclusion list.
  - Mock returns all LOW risk; verify empty exclusions.
  - Malformed response raises `AgentResponseError`.

**Status:** [x] done

---

### Sub-Task 6 — Deterministic Safety Validator

**Intent**
The Safety Validator is a pure-Python, LLM-free rule engine that enforces hard constraints
on any `MissionPlan`. It runs after the agents produce a plan and acts as the final gate:
any plan that fails a hard constraint is rejected outright. This is the key trust boundary
between AI-generated output and executable mission actions.

**Expected Outcomes**
- `safety/validator.py` contains `SafetyValidator` class with a `validate(plan)` method.
- Returns a `ValidationResult` (passed: bool, violations: list[str]).
- Hard rules enforced:
  1. Total energy must not exceed `(battery_pct / 100) * capacity_wh * (1 - MIN_RETURN_BATTERY_PCT)`.
  2. No individual waypoint terrain_risk may exceed `MAX_TERRAIN_RISK_SCORE`.
  3. Total mission time must not exceed `MAX_MISSION_DURATION_HOURS * 60`.
  4. Plan must include a `RETURN_TO_BASE` waypoint as the final step.
  5. Plan must have at least 1 science waypoint.
- Plans that fail are not executed; the system triggers replanning.

**Todo List**
1. Create `missionmind/safety/__init__.py`.
2. Create `missionmind/safety/validator.py`:
   - `ValidationResult` dataclass: `passed: bool`, `violations: list[str]`.
   - `SafetyValidator` class, constructor takes thresholds from `config.py` (injectable
     for testing).
   - `validate(plan: MissionPlan, rover_state: dict) -> ValidationResult` method
     implementing each rule as a separate private `_check_*` method.
   - All violation messages are human-readable strings suitable for logging/display.

**Relevant Context**
- `models/mission.py` → `MissionPlan`, `Waypoint`.
- `config.py` → all threshold constants.
- NO LLM calls — this must be deterministic and testable without any external service.

**Dependencies**
- Sub-Task 1 (models, config).

**Tests**
- `tests/test_validator.py`:
  - Valid plan passes all checks → `passed=True`, empty violations.
  - Plan exceeding energy budget → fails with energy violation string.
  - Plan with high-risk waypoint → fails with terrain violation string.
  - Plan exceeding max duration → fails with time violation string.
  - Plan missing RETURN_TO_BASE → fails.
  - Plan with no science waypoints → fails.
  - Multiple violations accumulate in `violations` list.

**Status:** [x] done

---

### Sub-Task 7 — Mission Commander

**Intent**
The Mission Commander is the orchestration layer that drives all three specialised agents
in sequence, synthesises their outputs into a coherent `MissionPlanOutput`, and submits it
to the Safety Validator. If the plan is rejected, it retries with modified constraints
(max N retries before aborting).

**Expected Outcomes**
- `agents/mission_commander.py` contains `MissionCommander` class.
- `MissionCommander.plan(context) -> MissionPlan` orchestrates agents and returns a
  validated plan or raises `PlanningFailedError` after exhausting retries.
- Retry logic: on validation failure, the commander narrows the waypoint list (removes
  highest-risk / most energy-expensive waypoints) and re-invokes agents.
- All agent calls are `async`; the commander uses `asyncio.gather` for parallel runs of
  independent agents.

**Expected Outcomes Detail**
- On success: returns a `MissionPlan` with `status=ACTIVE`.
- On failure after retries: raises `PlanningFailedError` with the accumulated violations.

**Todo List**
1. Write `missionmind/prompts/commander_prompt.md`:
   - Role: mission commander.
   - Input: science analysis, resource budget, risk assessment.
   - Task: select final waypoint sequence, balance scientific value vs energy vs risk,
     ensure return-to-base is final step.
   - Output: strict JSON matching `MissionPlanOutput` schema.
2. Create `missionmind/agents/mission_commander.py`:
   - `PlanningFailedError` custom exception.
   - `MissionCommander` class, constructor accepts instances of `ScienceAgent`,
     `ResourceAgent`, `SafetyAgent`, and `SafetyValidator`.
   - `async plan(context: dict) -> MissionPlan`:
     - Step 1: run `ScienceAgent` and `ResourceAgent` in parallel via `asyncio.gather`.
     - Step 2: run `SafetyAgent` with results incorporated.
     - Step 3: call LLM directly (commander prompt) to synthesise a `MissionPlanOutput`.
     - Step 4: convert to `MissionPlan`, run `SafetyValidator.validate()`.
     - Step 5: if validation fails, prune waypoints and retry (max 3 attempts).
     - Step 6: raise `PlanningFailedError` if all retries exhausted.
   - `_prune_waypoints(waypoints, violations) -> list[Waypoint]` helper.

**Relevant Context**
- All agents from Sub-Tasks 3–5.
- `safety/validator.py` (Sub-Task 6).
- `schemas/outputs.py` → `MissionPlanOutput`.
- `models/mission.py` → `MissionPlan`.

**Dependencies**
- Sub-Tasks 3, 4, 5, 6.

**Tests**
- `tests/test_mission_commander.py`:
  - All agents and validator mocked → happy path returns valid `MissionPlan`.
  - Validator rejects first plan, succeeds on retry → verify `plan()` resolves after pruning.
  - Validator rejects all retries → `PlanningFailedError` raised.
  - Verify `asyncio.gather` is actually used (parallel execution of science + resource
    agents); use mock call timing or an asyncio spy.

**Status:** [x] done

---

### Sub-Task 8 — Mission Planner (Entry Point)

**Intent**
The `planning/planner.py` module is the public entry point for the simulation and frontend
teams. It wraps `MissionCommander` and exposes a simple `async plan_mission(rover_state, env_state) -> MissionPlan` function. It is responsible for assembling the full context
dict that agents need and for constructing agent + commander instances with real (or
injected) LLM clients.

**Expected Outcomes**
- `planning/planner.py` exports `plan_mission(rover_state, env_state) -> MissionPlan`.
- All agent wiring is internal to the planner; callers do not touch agent classes.
- `rover_state` and `env_state` are plain dicts with documented keys.
- A `PLANNER_CONTEXT_KEYS` constant documents the required keys.

**Todo List**
1. Create `missionmind/planning/__init__.py`.
2. Create `missionmind/planning/planner.py`:
   - `ROVER_STATE_KEYS` and `ENV_STATE_KEYS` tuples documenting required dict keys.
   - `_build_llm_client() -> Any` factory reading IBM_WATSONX_API_KEY from env or falling
     back to OpenAI for local dev.
   - `async plan_mission(rover_state: dict, env_state: dict) -> MissionPlan`:
     - Validates presence of required keys.
     - Constructs LLM client, agents, commander, validator.
     - Calls `commander.plan(context)` and returns result.
     - Logs plan creation at INFO level.

**Relevant Context**
- `agents/mission_commander.py` (Sub-Task 7).
- IBM watsonx SDK or `openai` package for LLM client factory.

**Dependencies**
- Sub-Task 7.

**Tests**
- `tests/test_planner.py`:
  - Inject a mock commander → verify `plan_mission` delegates correctly.
  - Missing required key in `rover_state` → raises `ValueError`.

**Status:** [ ] pending

---

### Sub-Task 9 — Mission Replanner

**Intent**
The Replanner listens for `MissionEvent` objects emitted by the simulation layer and
responds to mid-mission changes: battery failure, communication loss, terrain hazards, and
new scientific discoveries. It produces a revised `MissionPlan` that replaces the current
one.

**Expected Outcomes**
- `planning/replanner.py` exports `async replan(current_plan, event, rover_state, env_state) -> MissionPlan`.
- Each event type has a dedicated handler that adjusts context before calling back into
  `MissionCommander.plan()`.
- `BATTERY_FAILURE`: forces immediate return-to-base if below critical threshold; otherwise
  shrinks waypoint budget.
- `COMM_LOSS`: removes waypoints outside the safe communication radius.
- `TERRAIN_HAZARD`: blacklists the affected waypoint and its neighbours.
- `NEW_DISCOVERY`: adds a new high-value waypoint and re-optimises if budget allows.
- `RETURN_TO_BASE`: generates a minimal energy return path.

**Todo List**
1. Create `missionmind/planning/replanner.py`:
   - `ReplanContext` dataclass: current_plan, event, rover_state, env_state.
   - `async replan(current_plan, event, rover_state, env_state) -> MissionPlan`.
   - Dispatcher: routes to `_handle_battery_failure`, `_handle_comm_loss`,
     `_handle_terrain_hazard`, `_handle_new_discovery`, `_handle_return_to_base`.
   - Each handler modifies `rover_state` / `env_state` appropriately and calls
     `plan_mission()` from the planner module.
   - `CRITICAL_BATTERY_PCT` threshold (from config) triggers immediate abort-and-return.

**Relevant Context**
- `models/events.py` → `MissionEvent`, `EventType`.
- `planning/planner.py` (Sub-Task 8).
- `config.py` → `CRITICAL_BATTERY_PCT`.

**Dependencies**
- Sub-Tasks 1, 8.

**Tests**
- `tests/test_replanner.py`:
  - `BATTERY_FAILURE` below critical → returned plan has only RETURN_TO_BASE waypoint.
  - `COMM_LOSS` → waypoints outside comm radius removed from plan.
  - `TERRAIN_HAZARD` → affected waypoint blacklisted.
  - `NEW_DISCOVERY` → new waypoint appears in revised plan when budget allows.
  - Unknown event type → raises `ValueError`.

**Status:** [ ] pending

---

### Sub-Task 10 — Integration Tests & AGENTS.md

**Intent**
Provide end-to-end integration tests that exercise the full pipeline from `plan_mission`
through to a validated `MissionPlan`, and document the AI architecture so all teammates
understand the backend interface.

**Expected Outcomes**
- `tests/test_integration.py` exercises the full planning pipeline with mocked LLM but
  real validator and real agent orchestration.
- `AGENTS.md` at the repo root documents every agent, its inputs/outputs, the validator
  rules, and the public API surface for teammates.

**Todo List**
1. Create `missionmind/tests/test_integration.py`:
   - Fixture: mock LLM client that returns canned valid responses for all agents.
   - Test: full `plan_mission` → returns a `MissionPlan` with `status=ACTIVE`.
   - Test: `plan_mission` → validator rejects → replanning succeeds.
   - Test: `replan` with `NEW_DISCOVERY` event → updated plan includes new waypoint.
2. Create `missionmind/AGENTS.md`:
   - System overview diagram (ASCII).
   - One section per agent: purpose, input context keys, output schema.
   - Validator rules (all 5 hard constraints listed).
   - Public API: `plan_mission(rover_state, env_state)` and `replan(...)` signatures with
     full key documentation.
   - Event types and their replanning behaviours.
   - How to run tests: `pytest missionmind/tests/`.

**Relevant Context**
- All prior sub-tasks.
- README.md exists (2 lines) — update it to link to AGENTS.md.

**Dependencies**
- Sub-Tasks 1–9 (should be last).

**Tests**
- The integration tests in this sub-task are themselves the validation artefact.

**Status:** [ ] pending

---

## Dependency Graph

```
Sub-Task 1 (Models + Schemas + Config)
    └── Sub-Task 2 (BaseAgent)
            ├── Sub-Task 3 (ScienceAgent)
            ├── Sub-Task 4 (ResourceAgent)
            └── Sub-Task 5 (SafetyAgent)
Sub-Task 1
    └── Sub-Task 6 (SafetyValidator)

Sub-Tasks 3 + 4 + 5 + 6
    └── Sub-Task 7 (MissionCommander)
            └── Sub-Task 8 (Planner)
                    └── Sub-Task 9 (Replanner)
                            └── Sub-Task 10 (Integration + AGENTS.md)
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM client | OpenAI SDK with IBM watsonx endpoint injection | IBM watsonx exposes an OpenAI-compatible API; same code works locally with GPT-4o |
| Structured outputs | Pydantic v2 + `response_format=json_object` | Guarantees agent outputs parse cleanly; validation errors are caught, not silently ignored |
| Safety enforcement | Two-layer: AI soft-assessment (SafetyAgent) + deterministic validator | AI provides reasoning; hard rules are never delegated to a probabilistic model |
| Async | All agent calls are `async` | Enables parallel execution of independent agents in the commander |
| Testing without LLM | All agents accept an injected `llm_client` | Unit tests run offline with zero API cost |
| Replanning | Event-driven, delegates back to the same planner | Single planning codepath, event handlers only mutate context |
