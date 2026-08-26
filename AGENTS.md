# AGENTS.md

This file provides guidance to agents working with code in this repository,
**and** serves as the definitive architecture/API reference for the MissionMind
AI backend.  Read this before starting any new sub-task or integration work.

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Commands (tests, install)](#commands)
3. [System architecture overview](#system-architecture-overview)
4. [ScienceAgent](#scienceagent)
5. [ResourceAgent](#resourceagent)
6. [SafetyAgent](#safetyagent)
7. [MissionCommander](#missioncommander)
8. [SafetyValidator](#safetyvalidator)
9. [Planner public API — `plan_mission()`](#planner-public-api)
10. [Replanner public API — `replan()`](#replanner-public-api)
11. [IBM watsonx integration](#ibm-watsonx-integration)
12. [Testing](#testing)
13. [Environment variables](#environment-variables)
14. [Critical non-obvious patterns](#critical-non-obvious-patterns)
15. [Implementation plan progress](#implementation-plan-progress)

---

## Repository layout

```
missionmind/                         ← repo root / outer package dir
├── pyproject.toml
├── requirements.txt
├── AGENTS.md                        ← this file
├── README.md
├── missionmind-ai-backend-plan.md   ← implementation plan (source of truth)
├── missionmind/                     ← importable Python package
│   ├── config.py                    ← all thresholds, env-var overridable
│   ├── models/
│   │   ├── mission.py               ← Waypoint, MissionPlan, MissionStatus
│   │   └── events.py                ← EventType, MissionEvent
│   ├── schemas/
│   │   └── outputs.py               ← ScienceAnalysis, ResourceBudget,
│   │                                   RiskAssessment, MissionPlanOutput
│   ├── agents/
│   │   ├── base_agent.py            ← BaseAgent(ABC), AgentResponseError
│   │   ├── client.py                ← LLMClient (Protocol), WatsonxClient,
│   │   │                               LLMResponse, LLMClientError
│   │   ├── science_agent.py
│   │   ├── resource_agent.py
│   │   ├── safety_agent.py
│   │   └── mission_commander.py     ← MissionCommander, PlanningFailedError
│   ├── safety/
│   │   └── validator.py             ← SafetyValidator, ValidationResult
│   ├── planning/
│   │   ├── planner.py               ← plan_mission() public entry point
│   │   └── replanner.py             ← replan() + ReplanContext
│   └── prompts/
│       ├── science_prompt.md
│       ├── resource_prompt.md
│       ├── safety_prompt.md
│       └── commander_prompt.md
└── tests/
    ├── test_models.py
    ├── test_base_agent.py
    ├── test_science_agent.py
    ├── test_resource_agent.py
    ├── test_safety_agent.py
    ├── test_validator.py
    ├── test_mission_commander.py
    ├── test_planner.py
    ├── test_replanner.py
    └── test_integration.py          ← end-to-end tests (Sub-Task 10)
```

The Python package lives one level deep: `missionmind/missionmind/`.
All tests live in `missionmind/tests/`.
`pyproject.toml` and `requirements.txt` live in `missionmind/` (the outer folder).

---

## Commands

**Python interpreter** — the system uses MSYS2's ucrt64 Python; `python` / `python3`
are not on the Windows PATH.  Use the full path or set `PYTHONPATH` explicitly:

```powershell
# Run all tests (PYTHONPATH must point at the OUTER missionmind/ folder)
$env:PYTHONPATH = "C:\Users\devar\OneDrive\Desktop\ibmbob\missionmind"
& "C:\msys64\ucrt64\bin\python.exe" -m pytest tests/ -v

# Run a single test file
& "C:\msys64\ucrt64\bin\python.exe" -m pytest tests/test_validator.py -v

# Run a single test by name
& "C:\msys64\ucrt64\bin\python.exe" -m pytest tests/test_validator.py::TestEnergyBudgetRule::test_energy_over_budget_fails -v

# Install dependencies — pip is NOT available in the MSYS2 environment;
# use pacman for pydantic/pytest, which installs system-wide:
& "C:\msys64\usr\bin\pacman.exe" -S --noconfirm "mingw-w64-ucrt-x86_64-python-pydantic" "mingw-w64-ucrt-x86_64-python-pytest" "mingw-w64-ucrt-x86_64-python-pytest-asyncio"
# httpx must also be installed via pacman:
& "C:\msys64\usr\bin\pacman.exe" -S --noconfirm "mingw-w64-ucrt-x86_64-python-httpx"
```

**pytest config** — `asyncio_mode = "auto"` is set in `pyproject.toml`, so all
`async def test_*` functions run automatically without any decorator.

> **Note on test directory:** Tests live in `tests/` (relative to the outer
> `missionmind/` folder).  The correct command is `pytest tests/`, not
> `pytest missionmind/tests/`.

---

## System architecture overview

```
Simulation / frontend layer
          │
          ▼
  plan_mission(rover_state, env_state)          ← planner.py
          │
          ▼
    _build_commander()  ─────────────────────────────────┐
          │                                              │
          ▼                                              ▼
   MissionCommander.plan(context)              WatsonxClient
          │                                  (IBM watsonx AI /
     ┌────┴─────────────────┐                 OpenAI-compat REST
     │    asyncio.gather    │                 via httpx)
     ▼                      ▼
ScienceAgent         ResourceAgent
  (LLM call)           (LLM call)
     └──────────┬──────────┘
                ▼
          SafetyAgent
           (LLM call)
                │
                ▼
     Commander synthesis
           (LLM call)
                │
                ▼
      _convert_to_mission_plan()
                │
                ▼
        SafetyValidator                ← deterministic, LLM-free
         ┌──────┴──────┐
       PASS           FAIL
         │              │
         ▼              ▼
  MissionPlan      _prune_waypoints()
  (status=ACTIVE)  then retry (up to
                   MAX_PLANNING_RETRIES
                   total attempts)
                         │
                   all fail → PlanningFailedError

─────────────────────────────────────────────────────
Mid-mission replanning flow:

MissionEvent (from simulation)
          │
          ▼
     replan(current_plan, event, rover_state, env_state)
          │
    ┌─────┴──────────────────────────────────────┐
    │   Event-type dispatch                      │
    │                                            │
    │  BATTERY_FAILURE                           │
    │    < CRITICAL_BATTERY_PCT → emergency      │
    │      return plan (deterministic, no LLM)   │
    │    ≥ CRITICAL_BATTERY_PCT → update battery │
    │      level, delegate to plan_mission()     │
    │                                            │
    │  COMM_LOSS → filter candidates outside     │
    │    safe_comm_radius_m, plan_mission()       │
    │                                            │
    │  TERRAIN_HAZARD → blacklist affected       │
    │    waypoint (+ neighbours), plan_mission() │
    │                                            │
    │  NEW_DISCOVERY → add discovered waypoint   │
    │    to candidates, plan_mission()           │
    │                                            │
    │  RETURN_TO_BASE → emergency return plan    │
    │    (deterministic, no LLM)                 │
    └────────────────────────────────────────────┘
          │
          ▼
     MissionPlan (revised)
```

---

## ScienceAgent

**Module:** `missionmind/agents/science_agent.py`
**Import:** `from missionmind.agents import ScienceAgent`

**Purpose:**
Evaluates the scientific value of each candidate waypoint.  It acts as a
planetary-science expert: it considers geological features, proximity to
scientifically interesting formations, and alignment with mission objectives.

ScienceAgent does **not** make resource, energy, or hard-safety decisions.
Those are handled by ResourceAgent and SafetyValidator respectively.

**Input context keys (passed to `run(context)`):**

| Key | Type | Description |
|---|---|---|
| `candidate_waypoints` | `list[dict]` | Candidate waypoint dicts (id, x, y, terrain_risk, label, …) |
| `rover_position` | `dict` | `{"x": float, "y": float}` — rover's current location |
| `mission_objectives` | `list[str]` | Plain-text mission goal strings |

**Output schema:** `ScienceAnalysis` (from `missionmind/schemas/outputs.py`)

```python
class ScoredTarget(BaseModel):
    waypoint_id: str
    scientific_value: float        # 0.0 (none) – 1.0 (exceptional)
    justification: str             # free-text reason

class ScienceAnalysis(BaseModel):
    scored_targets: list[ScoredTarget]
    priority_order: list[str]      # waypoint IDs, highest priority first
    reasoning: str
```

---

## ResourceAgent

**Module:** `missionmind/agents/resource_agent.py`
**Import:** `from missionmind.agents import ResourceAgent`

**Purpose:**
Models the rover's energy and time budget.  Determines which waypoints are
feasible given the current battery state and the reserve requirement, and
estimates per-waypoint energy costs.

**Battery convention — fractions, not percentages:**
`battery_pct = 0.80` means 80 %.  `MIN_RETURN_BATTERY_PCT = 0.20` means 20 %.
The usable energy formula is:
```
usable_wh = battery_pct × battery_capacity_wh − MIN_RETURN_BATTERY_PCT × battery_capacity_wh
```
This is the correct formula.  `battery_pct × battery_capacity_wh × (1 − reserve)` gives
a different answer when `battery_pct < 1.0` and must **not** be used.

ResourceAgent automatically injects `min_return_battery_pct` from `config.py` into
the context before calling the LLM.  A caller-supplied value for that key is
silently overwritten.

**Input context keys:**

| Key | Type | Description |
|---|---|---|
| `battery_pct` | `float` | Fraction of full charge (0.0–1.0) |
| `battery_capacity_wh` | `float` | Total battery capacity in Wh |
| `candidate_waypoints` | `list[dict]` | Candidate waypoints |
| `rover_speed_mps` | `float` | Nominal driving speed in m/s |
| `power_consumption_w` | `float` | Power draw during driving in W |

**Output schema:** `ResourceBudget` (from `missionmind/schemas/outputs.py`)

```python
class ResourceBudget(BaseModel):
    available_energy_wh: float           # usable Wh after reserving return energy
    available_time_minutes: float        # total time budget in minutes
    recommended_waypoints: list[str]     # waypoint IDs that fit the budget
    energy_per_waypoint: dict[str, float] # per-waypoint energy estimate in Wh
    reasoning: str
```

---

## SafetyAgent

**Module:** `missionmind/agents/safety_agent.py`
**Import:** `from missionmind.agents import SafetyAgent`

**Purpose:**
Performs an AI-based **soft** risk assessment.  It considers terrain conditions,
weather forecasts, and communication windows to identify potential hazards and
recommend which waypoints should be excluded from the plan.

**This is NOT the hard safety gate.**  SafetyAgent provides advisory
recommendations.  The final mandatory safety enforcement is done by the
deterministic `SafetyValidator` (see below), which runs after MissionCommander
produces a plan.

SafetyAgent automatically injects `max_terrain_risk_score` from `config.py`
into the context.  A caller-supplied value is silently overwritten.

**Input context keys:**

| Key | Type | Description |
|---|---|---|
| `candidate_waypoints` | `list[dict]` | Candidate waypoints with terrain_risk |
| `weather_forecast` | `dict` | Dust-storm probability, temperatures, wind speed, forecast hours |
| `comm_windows` | `list[dict]` | Communication window schedule |
| `terrain_map` | `dict` | Additional terrain data keyed by waypoint id |

**Output schema:** `RiskAssessment` (from `missionmind/schemas/outputs.py`)

```python
class WaypointRisk(BaseModel):
    waypoint_id: str
    risk_score: float              # 0.0 (safe) – 1.0 (dangerous)
    factors: list[str]             # contributing risk factors

class RiskAssessment(BaseModel):
    waypoint_risks: list[WaypointRisk]
    overall_risk_level: RiskLevel  # LOW | MEDIUM | HIGH
    recommended_exclusions: list[str]  # waypoint IDs to exclude
    reasoning: str
```

---

## MissionCommander

**Module:** `missionmind/agents/mission_commander.py`
**Import:** `from missionmind.agents import MissionCommander`

**Purpose:**
Orchestrates the three specialist agents into a validated `MissionPlan`.

**Orchestration flow (within one planning attempt):**

1. Run `ScienceAgent` and `ResourceAgent` **in parallel** via `asyncio.gather`.
2. Run `SafetyAgent` with science and resource results added to context.
3. Run the commander synthesis LLM call (`_CommanderSynthesisAgent`) with all
   three agent results in context → `MissionPlanOutput`.
4. Convert `MissionPlanOutput` → `MissionPlan` domain object.
5. Submit to `SafetyValidator` (deterministic hard rules).
6. **PASS** → set `status=ACTIVE`, return plan.
7. **FAIL** → call `_prune_waypoints()` to remove the most problematic candidate,
   then retry from step 1.
8. After **`MAX_PLANNING_RETRIES`** total attempts (default **3**), raise
   `PlanningFailedError` if no plan was approved.

**Output:**

- `MissionPlan` with `status=MissionStatus.ACTIVE` on success.
- `PlanningFailedError` (subclass of `RuntimeError`) on exhaustion.  Carries
  `violations` (from the last rejection) and `attempts` attributes.
- `AgentResponseError` propagates immediately if any agent returns structurally
  invalid JSON (not retried at the commander level).

**Constructor parameters (relevant for testing):**

```python
MissionCommander(
    science_agent=...,
    resource_agent=...,
    safety_agent=...,
    validator=...,
    llm_client=...,       # inject a fake in tests
    max_attempts=3,       # total generate-validate cycles
    retry_delay=1.0,
)
```

---

## SafetyValidator

**Module:** `missionmind/safety/validator.py`
**Import:** `from missionmind.safety import SafetyValidator, ValidationResult`

**Purpose:**
The final, mandatory hard-safety gate.  Pure Python.  Deterministic.
**Completely LLM-free** — contains zero AI calls, zero network calls, zero
randomness.  A source-code inspection test enforces this.

`validate(plan, rover_state) → ValidationResult`

**The five hard constraints (all must pass):**

1. **ENERGY BUDGET**
   `plan.total_energy_wh` must not exceed the usable energy:
   ```
   usable_wh = max(0, battery_pct × capacity_wh − MIN_RETURN_BATTERY_PCT × capacity_wh)
   ```
   Violation prefix: `"ENERGY BUDGET EXCEEDED"`

2. **TERRAIN RISK**
   No single `Waypoint.terrain_risk` may exceed `MAX_TERRAIN_RISK_SCORE` (default 0.70).
   Violation prefix: `"TERRAIN RISK EXCEEDED"`

3. **MISSION DURATION**
   `plan.total_time_minutes` must not exceed `MAX_MISSION_DURATION_HOURS × 60`
   (default 8 h = 480 min).
   Violation prefix: `"MISSION DURATION EXCEEDED"`

4. **RETURN TO BASE**
   The final waypoint in `plan.waypoints` must have `is_base=True`.
   Violation prefix: `"MISSING RETURN TO BASE"`

5. **SCIENCE OBJECTIVE**
   The plan must contain at least one non-base waypoint
   (`plan.science_waypoints()` must be non-empty).
   Violation prefix: `"NO SCIENCE WAYPOINTS"`

**Emergency return plans** produced by the Replanner (critical battery /
`RETURN_TO_BASE` event) deliberately contain **only** the base waypoint.  They
bypass the normal SafetyValidator because the "no science waypoints" rule is
intentionally correct for normal missions but must not block an emergency abort.
These plans are built deterministically by the Replanner without any LLM call.

**ValidationResult:**
```python
@dataclass
class ValidationResult:
    passed: bool
    violations: list[str]   # empty when passed=True
```
`bool(result)` returns `result.passed`.

All thresholds are injectable via constructor args, making the validator fully
testable without environment variables.

---

## Planner public API

**Module:** `missionmind/planning/planner.py`
**Import:** `from missionmind.planning import plan_mission`

```python
async def plan_mission(
    rover_state: dict[str, Any],
    env_state: dict[str, Any],
) -> MissionPlan
```

**Required `rover_state` keys** (`ROVER_STATE_KEYS`):

| Key | Type | Units / convention |
|---|---|---|
| `battery_pct` | `float` | Fraction 0.0–1.0.  `0.85` = 85 %. |
| `battery_capacity_wh` | `float` | Total battery capacity in watt-hours. |
| `position_x` | `float` | Rover X coordinate in metres from base. |
| `position_y` | `float` | Rover Y coordinate in metres from base. |
| `rover_speed_mps` | `float` | Nominal driving speed in metres per second. |
| `power_consumption_w` | `float` | Power draw during driving in watts. |

**Required `env_state` keys** (`ENV_STATE_KEYS`):

| Key | Type | Description |
|---|---|---|
| `candidate_waypoints` | `list[dict]` | Waypoints the rover may visit.  Each dict must include `id`, `x`, `y`, `terrain_risk`, `is_base`, `label`, `estimated_travel_time_minutes`, `estimated_energy_wh`. |
| `weather_forecast` | `dict` | `{dust_storm_probability, temperature_min_c, temperature_max_c, wind_speed_mps, forecast_hours}` |
| `comm_windows` | `list[dict]` | `[{"start_utc": "<ISO-8601>", "duration_minutes": int}]` |
| `terrain_map` | `dict` | Additional terrain data keyed by waypoint id (may be `{}`). |
| `mission_objectives` | `list[str]` | Plain-text mission goal strings. |

**Raises:**
- `ValueError` — if any required key is absent (all missing keys listed in the message).
- `PlanningFailedError` — if the commander cannot produce a valid plan within the allowed attempts.
- `AgentResponseError` — if a specialist agent returns an unparseable LLM response.

**Notes:**
- Input dicts are never mutated.
- In tests, inject a fake LLM client by patching `missionmind.planning.planner._build_llm_client`.

---

## Replanner public API

**Module:** `missionmind/planning/replanner.py`
**Import:** `from missionmind.planning import replan, ReplanContext`

```python
async def replan(
    current_plan: MissionPlan,
    event: MissionEvent,
    rover_state: dict[str, Any],
    env_state: dict[str, Any],
) -> MissionPlan
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `current_plan` | `MissionPlan` | The plan currently being executed by the rover. |
| `event` | `MissionEvent` | The mid-mission event triggering replanning. |
| `rover_state` | `dict` | Current rover state (same keys as `plan_mission`).  Not mutated. |
| `env_state` | `dict` | Current environment state.  Not mutated. |

**`MissionEvent` structure:**
```python
@dataclass
class MissionEvent:
    event_type: EventType     # one of the values below
    severity: float           # 0.0 (minor) – 1.0 (critical)
    payload: dict[str, Any]   # event-type-specific data (see below)
    timestamp: datetime       # auto-set to UTC now if not provided
```

**Event behaviors:**

| `EventType` | Behavior | `payload` keys used |
|---|---|---|
| `BATTERY_FAILURE` | If `battery_pct < CRITICAL_BATTERY_PCT` (default 0.10): deterministic immediate return plan, no LLM call. Otherwise: update rover `battery_pct` from payload and delegate to `plan_mission()`. | `battery_pct` (optional; falls back to rover_state) |
| `COMM_LOSS` | Remove candidate waypoints whose Euclidean distance from base (0,0) exceeds `safe_comm_radius_m`.  Then delegate to `plan_mission()`. If radius is absent from payload, all candidates are preserved. | `safe_comm_radius_m` (optional) |
| `TERRAIN_HAZARD` | Blacklist the named waypoint and any explicit neighbours.  Then delegate to `plan_mission()`. | `waypoint_id` (required), `neighbour_ids` (optional list) |
| `NEW_DISCOVERY` | Add the discovered waypoint to candidates.  Then delegate to `plan_mission()` so agents evaluate it against energy/safety constraints. | `x`, `y` (required), `id`, `label`, `scientific_value` (default 0.9), `terrain_risk` (default 0.1), `estimated_travel_time_minutes`, `estimated_energy_wh` (both default 0.0) |
| `RETURN_TO_BASE` | Deterministic immediate return plan, no LLM call. | payload ignored |

**Raises:**
- `ValueError` — for any unknown/unsupported `EventType`.

**`ReplanContext` dataclass** (internal bundle, also exported):
```python
@dataclass
class ReplanContext:
    current_plan: MissionPlan
    event: MissionEvent
    rover_state: dict[str, Any]   # working copy — may be mutated by handlers
    env_state: dict[str, Any]     # working copy — may be mutated by handlers
```

---

## IBM watsonx integration

**Module:** `missionmind/agents/client.py`

**`WatsonxClient`** is the production LLM backend.  It speaks the OpenAI-compatible
REST protocol (`/chat/completions`) via `httpx.AsyncClient`.  It does **not** use
the `openai` Python package (MSYS2 Python cannot build the `jiter` Rust extension).

**`LLMClient` protocol** is a `@runtime_checkable` structural interface:
```python
class LLMClient(Protocol):
    async def complete(
        self, *, system_prompt, user_message,
        model=None, temperature=0.2, max_tokens=2048,
    ) -> LLMResponse: ...
```
Any object implementing `complete()` with this signature is a valid client.
Plain `MagicMock` objects do **not** pass `isinstance(obj, LLMClient)` checks even
if `.complete` is set — test usability behaviourally instead.

**`LLMResponse` dataclass** (returned by `complete()`):
```python
@dataclass(frozen=True)
class LLMResponse:
    content: str             # raw text / JSON string from model
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
```

**Credentials must never be hardcoded.**  All secrets are read from environment
variables at construction time (see [Environment variables](#environment-variables)).

In **unit tests**: inject a `MagicMock` + `AsyncMock` as described below.
In **integration tests**: use a `FakeLLMClient` that returns canned JSON strings
(see `tests/test_integration.py` for a reference implementation).
Neither requires real IBM credentials.

---

## Testing

All tests run **offline** — zero real IBM watsonx or network calls are made.

**Unit tests** (Sub-Tasks 1–9):
Each agent accepts an injected `llm_client`, so tests supply a `MagicMock` + `AsyncMock`:
```python
from unittest.mock import AsyncMock, MagicMock
from missionmind.agents.client import LLMResponse

mock_client = MagicMock()
mock_client.complete = AsyncMock(
    return_value=LLMResponse(content='{"field": "value"}', model="test-model")
)
```

**Integration tests** (`tests/test_integration.py`, Sub-Task 10):
Use `FakeLLMClient` (defined in the test file) — a FIFO queue of canned JSON
strings.  Tests patch `missionmind.planning.planner._build_llm_client` to return
the fake, then exercise the complete real pipeline end-to-end.

**Run all tests:**
```powershell
$env:PYTHONPATH = "C:\Users\devar\OneDrive\Desktop\ibmbob\missionmind"
& "C:\msys64\ucrt64\bin\python.exe" -m pytest tests/ -v
```

**Run a specific test file:**
```powershell
& "C:\msys64\ucrt64\bin\python.exe" -m pytest tests/test_integration.py -v
& "C:\msys64\ucrt64\bin\python.exe" -m pytest tests/test_validator.py -v
```

---

## Environment variables (production / IBM watsonx)

| Variable | Purpose | Required for |
|---|---|---|
| `IBM_WATSONX_API_KEY` | IBM Cloud API key (primary auth) | Production |
| `IBM_WATSONX_URL` | Base URL, e.g. `https://us-south.ml.cloud.ibm.com` | Production |
| `IBM_WATSONX_PROJECT_ID` | Watsonx project ID (added to every payload) | Production |
| `LLM_MODEL_NAME` | Model ID, default `gpt-4o` | Both |
| `LLM_BASE_URL` | Override full completions URL (Ollama / OpenAI local dev) | Local dev |
| `OPENAI_API_KEY` | Fallback Bearer token when `IBM_WATSONX_API_KEY` is absent | Local dev |

**Credentials must never be hardcoded in source files.**

---

## Critical non-obvious patterns

### Package layout
`missionmind/` (outer) contains `pyproject.toml`, `requirements.txt`, `tests/`, and the
inner `missionmind/` package. When setting `PYTHONPATH`, point at the **outer** directory
so that `import missionmind` resolves to `missionmind/missionmind/`.

### Dependency situation — NO `openai` package
The `openai` package was intentionally replaced with `httpx` because MSYS2 Python cannot
build `jiter` (a Rust extension). `WatsonxClient` in `agents/client.py` speaks the
OpenAI-compatible REST protocol directly via `httpx.AsyncClient`. Do **not** add the
`openai` package back.

### Config values are always fractions, not percentages
`MIN_RETURN_BATTERY_PCT = 0.20` means 20 %. `battery_pct` in rover state is also a
fraction (0.0–1.0). The energy formula is:
```
usable_wh = battery_pct × capacity_wh − (MIN_RETURN_BATTERY_PCT × capacity_wh)
```
This is **not** `battery_pct × capacity_wh × (1 − reserve)` — those give different
numbers when `battery_pct < 1.0`.

### Agent mock pattern
LLM clients are never mocked with `AsyncMock(spec=["complete"])` — that creates a plain
`MagicMock` for `.complete`, not an `AsyncMock`. The correct pattern used throughout:

```python
mock = MagicMock()
mock.complete = AsyncMock(return_value=LLMResponse(content=json.dumps(payload), model="test-model"))
```

### ResourceAgent and SafetyAgent override `run()`
Both agents override `BaseAgent.run()` to inject a config value before calling
`super().run(enriched)`. A caller-supplied value for that key is **silently overwritten**:
- `ResourceAgent` injects `min_return_battery_pct`
- `SafetyAgent` injects `max_terrain_risk_score`

### `LLMClient` is a structural Protocol
`isinstance(obj, LLMClient)` only returns `True` for classes that explicitly implement
the protocol (e.g. `WatsonxClient`). Plain `MagicMock` objects do **not** pass
`isinstance` checks even if `.complete` is set. Test usability behaviourally instead.

### SafetyValidator is entirely LLM-free
`missionmind/safety/validator.py` must never import `BaseAgent`, `WatsonxClient`, or
`LLMClient`. A test (`test_no_ai_imports_in_module`) inspects the source code to enforce this.

### Violation strings use ALL-CAPS prefixes
Every `ValidationResult.violations` entry starts with a specific keyword:
`ENERGY BUDGET`, `TERRAIN RISK`, `MISSION DURATION`, `RETURN TO BASE`, `SCIENCE`.
Tests `assert any("ENERGY BUDGET" in m …)` — keep these prefixes stable.

### Prompt files are loaded at module import time
Each agent file caches its prompt as a module-level `_*_SYSTEM_PROMPT` string read once
from `missionmind/prompts/*.md`. Editing a `.md` file requires a Python process restart
to take effect; there is no hot-reload.

### Structural AI failures are never retried
`AgentResponseError` (bad JSON / schema mismatch) bypasses the retry loop entirely.
Only `LLMClientError` (network / HTTP errors) triggers exponential back-off. This is
intentional — retrying a structurally invalid response wastes quota.

### New agents follow an identical pattern
To add a new agent:
1. Create `missionmind/agents/<name>_agent.py` subclassing `BaseAgent[SchemaType]`.
2. Implement the three abstract properties: `name`, `system_prompt`, `response_schema`.
3. Write the prompt in `missionmind/prompts/<name>_prompt.md`.
4. Export from `missionmind/agents/__init__.py`.
5. If a config threshold must be injected, override `run()` and call `super().run(enriched)`.

---

## Implementation plan progress

See `missionmind/missionmind-ai-backend-plan.md` for the full sub-task list.
**All Sub-Tasks 1–10 are complete.**

| Sub-Task | Component | Status |
|---|---|---|
| 1 | Project scaffold, shared models, Pydantic schemas, `config.py` | ✅ |
| 2 | `BaseAgent`, `WatsonxClient`, `AgentResponseError` | ✅ |
| 3 | `ScienceAgent` | ✅ |
| 4 | `ResourceAgent` | ✅ |
| 5 | `SafetyAgent` | ✅ |
| 6 | `SafetyValidator` (deterministic, LLM-free) | ✅ |
| 7 | `MissionCommander` (orchestration + retry) | ✅ |
| 8 | `plan_mission()` public entry point | ✅ |
| 9 | `replan()` event-driven replanner | ✅ |
| 10 | Integration tests + AGENTS.md documentation | ✅ |
