# MissionMind

**AI-powered autonomous mission planning for a simulated Mars rover.**

Built for the IBM Bob space exploration hackathon by Person 1 (AI & mission-intelligence backend).

---

## What it does

MissionMind generates autonomous mission plans for a Mars rover by coordinating multiple specialised AI agents, validating the plans with deterministic safety rules, and replanning in real-time when the environment changes.

The system balances: scientific value · battery/energy · mission time · communication windows · environmental risk · safe return to base.

---

## Current implementation status

| Sub-Task | Component | Status |
|---|---|---|
| 1 | Project scaffold, shared models (`Waypoint`, `MissionPlan`, `MissionStatus`, `MissionEvent`), Pydantic schemas (`ScienceAnalysis`, `ResourceBudget`, `RiskAssessment`, `MissionPlanOutput`), `config.py` | ✅ Complete |
| 2 | `BaseAgent` (async LLM contract), `WatsonxClient` (IBM watsonx via `httpx`), `AgentResponseError` | ✅ Complete |
| 3 | `ScienceAgent` — scores candidate waypoints by scientific value, returns `ScienceAnalysis` | ✅ Complete |
| 4 | `ResourceAgent` — computes energy/time budget, returns `ResourceBudget` | ✅ Complete |
| 5 | `SafetyAgent` — AI soft-constraint risk assessment, returns `RiskAssessment` | ✅ Complete |
| 6 | `SafetyValidator` — deterministic hard-constraint gate, returns `ValidationResult` | ✅ Complete |
| 7 | `MissionCommander` — orchestrates agents, retries with pruned waypoints | ✅ Complete |
| 8 | `Planner` — public entry point `plan_mission(rover_state, env_state)` | ✅ Complete |
| 9 | `Replanner` — event-driven `replan(current_plan, event, …)` | ✅ Complete |
| 10 | Integration tests + final AGENTS.md documentation | ✅ Complete |
| API | FastAPI HTTP integration layer — `GET /api/health`, `POST /api/mission/plan`, `POST /api/mission/replan` | ✅ Complete |

**537 tests passing.** All tests run offline — zero real AI/API calls.

---

## Package structure

```
missionmind/                        ← repo root / outer package dir
├── pyproject.toml
├── requirements.txt
├── AGENTS.md                       ← agent guidance (read this first)
├── missionmind-ai-backend-plan.md  ← implementation plan (source of truth)
├── missionmind/                    ← importable Python package
│   ├── config.py                   ← all thresholds, env-var overridable
│   ├── models/
│   │   ├── mission.py              ← Waypoint, MissionPlan, MissionStatus
│   │   └── events.py               ← EventType, MissionEvent
│   ├── schemas/
│   │   └── outputs.py              ← ScienceAnalysis, ResourceBudget, RiskAssessment, MissionPlanOutput
│   ├── agents/
│   │   ├── base_agent.py           ← BaseAgent(ABC), AgentResponseError
│   │   ├── client.py               ← LLMClient (Protocol), WatsonxClient, LLMResponse
│   │   ├── science_agent.py
│   │   ├── resource_agent.py
│   │   ├── safety_agent.py
│   │   └── mission_commander.py    ← MissionCommander, PlanningFailedError
│   ├── safety/
│   │   └── validator.py            ← SafetyValidator, ValidationResult (LLM-free)
│   ├── api/                        ← FastAPI HTTP integration layer
│   │   ├── app.py                  ← create_app() factory, CORS middleware
│   │   ├── main.py                 ← app = create_app(); uvicorn entry point
│   │   ├── routes/
│   │   │   ├── health.py           ← GET /api/health
│   │   │   ├── planning.py         ← POST /api/mission/plan
│   │   │   └── replanning.py       ← POST /api/mission/replan
│   │   └── schemas/
│   │       ├── planning.py         ← RoverStateInput, WaypointInput, EnvStateInput, PlanRequest
│   │       └── replanning.py       ← MissionEventInput, ReplanRequest
│   ├── planning/
│   │   ├── planner.py              ← plan_mission() public entry point
│   │   └── replanner.py            ← replan() + ReplanContext
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
    └── test_integration.py         ← end-to-end pipeline tests
```

---

## Starting the API server

```powershell
$env:PYTHONPATH = "C:\Users\devar\OneDrive\Desktop\ibmbob\missionmind"
# Set your IBM watsonx credentials (or local LLM overrides) in the environment
$env:IBM_WATSONX_API_KEY = "..."
$env:IBM_WATSONX_URL     = "https://us-south.ml.cloud.ibm.com"
$env:IBM_WATSONX_PROJECT_ID = "..."

& "C:\msys64\ucrt64\bin\uvicorn.exe" missionmind.api.main:app --reload
# Server listens on http://127.0.0.1:8000
# OpenAPI docs: http://127.0.0.1:8000/docs
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Health check — returns `{"status":"ok","backend":"missionmind","version":"..."}` |
| `POST` | `/api/mission/plan` | Run the full planning pipeline; body: `{"rover_state":{…},"env_state":{…}}` |
| `POST` | `/api/mission/replan` | Replan in response to a mid-mission event; body: `{"current_plan":{…},"event":{…},"rover_state":{…},"env_state":{…}}` |

CORS is pre-configured for `localhost:3000`, `localhost:5173`, and `localhost:5174` (React/Vite dev servers).

---

## Running tests

```powershell
# Requires MSYS2 ucrt64 Python — 'python' is not on the Windows PATH
$env:PYTHONPATH = "C:\Users\devar\OneDrive\Desktop\ibmbob\missionmind"
& "C:\msys64\ucrt64\bin\python.exe" -m pytest tests/ -v

# Single test file
& "C:\msys64\ucrt64\bin\python.exe" -m pytest tests/test_validator.py::TestEnergyBudgetRule -v

# Integration tests only
& "C:\msys64\ucrt64\bin\python.exe" -m pytest tests/test_integration.py -v
```

For full architecture and API documentation see **[AGENTS.md](AGENTS.md)**.

---

## Key design decisions

| Decision | Implementation |
|---|---|
| **LLM backend** | `WatsonxClient` calls IBM watsonx's OpenAI-compatible REST endpoint via `httpx` — **no `openai` package** (MSYS2 cannot build its Rust extension `jiter`) |
| **Structured outputs** | Pydantic v2 validates every LLM response; schema failures raise `AgentResponseError` immediately |
| **Two-layer safety** | `SafetyAgent` = AI soft recommendations; `SafetyValidator` = deterministic hard rules (LLM-free, tested with source inspection) |
| **Async** | All agent calls are `async`; independent agents run in parallel via `asyncio.gather` |
| **Test isolation** | Every agent accepts `llm_client` injection; tests use `MagicMock` + `AsyncMock`, never real endpoints |
| **Config** | All thresholds in `config.py` are env-var overridable; values are **fractions** (0.0–1.0), not percentages |

---

## Environment variables

| Variable | Purpose | Required for |
|---|---|---|
| `IBM_WATSONX_API_KEY` | IBM Cloud API key | Production |
| `IBM_WATSONX_URL` | e.g. `https://us-south.ml.cloud.ibm.com` | Production |
| `IBM_WATSONX_PROJECT_ID` | Watsonx project ID | Production |
| `LLM_MODEL_NAME` | Model ID (default `gpt-4o`) | Both |
| `LLM_BASE_URL` | Override completions URL (Ollama / OpenAI) | Local dev |
| `OPENAI_API_KEY` | Fallback Bearer token | Local dev |

**Credentials must never be hardcoded in source files.**

---

## Architecture documentation

Full AI backend architecture, agent APIs, validator rules, event handling,
and testing guide are in **[AGENTS.md](AGENTS.md)**.
## MissionMind Demo

MissionMind is an AI-powered autonomous Mars rover mission commander built for the IBM Bob Hackathon.

It combines multiple specialist agents with deterministic safety validation to create and dynamically replan rover missions when conditions change.

### AI Architecture

MissionMind uses:

- **Science Agent** — prioritizes scientifically valuable targets.
- **Resource Agent** — evaluates battery, energy, and time constraints.
- **Safety Agent** — evaluates terrain, weather, communications, and mission risk.
- **Mission Commander** — combines specialist recommendations into a mission plan.
- **Deterministic Safety Validator** — independently enforces hard safety constraints before a plan is accepted.

The demo runs **IBM Granite 4.2 3B locally through Ollama**, allowing mission intelligence to operate locally without depending on a remote connection.

IBM Bob was used as the primary development assistant throughout the project.

### Autonomous Replanning Events

MissionMind can dynamically replan for:

- Battery Failure
- Terrain Hazard
- New Scientific Discovery
- Communication Loss
- Return to Base

Safety-critical conditions such as critical battery and explicit Return to Base requests use deterministic emergency behavior rather than relying on LLM judgment.

### Run the Backend

From the repository root:

```powershell
$env:MISSIONMIND_LLM_PROVIDER = "ollama"
$venvSite = (Resolve-Path ".venv\lib\python3.12\site-packages").Path
& "C:\msys64\ucrt64\bin\python.exe" -c "import sys; sys.path.append(r'$venvSite'); import uvicorn; uvicorn.run('missionmind.api.main:app', host='127.0.0.1', port=8000)"