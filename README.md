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
| 7 | `MissionCommander` — orchestrates agents, retries with pruned waypoints | 🔲 Pending |
| 8 | `Planner` — public entry point `plan_mission(rover_state, env_state)` | 🔲 Pending |
| 9 | `Replanner` — event-driven `replan(current_plan, event, …)` | 🔲 Pending |
| 10 | Integration tests + final AGENTS.md documentation | 🔲 Pending |

**296 tests passing.** All tests run offline — zero real AI/API calls.

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
│   │   └── safety_agent.py
│   ├── safety/
│   │   └── validator.py            ← SafetyValidator, ValidationResult (LLM-free)
│   └── prompts/
│       ├── science_prompt.md
│       ├── resource_prompt.md
│       ├── safety_prompt.md
│       └── commander_prompt.md     ← placeholder for Sub-Task 7
└── tests/
    ├── test_models.py
    ├── test_base_agent.py
    ├── test_science_agent.py
    ├── test_resource_agent.py
    ├── test_safety_agent.py
    └── test_validator.py
```

---

## Running tests

```powershell
# Requires MSYS2 ucrt64 Python — 'python' is not on the Windows PATH
$env:PYTHONPATH = "C:\Users\devar\OneDrive\Desktop\ibmbob\missionmind"
& "C:\msys64\ucrt64\bin\python.exe" -m pytest missionmind/tests/ -v

# Single test
& "C:\msys64\ucrt64\bin\python.exe" -m pytest missionmind/tests/test_validator.py::TestEnergyBudgetRule -v
```

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

## Continuing development

The approved implementation plan is in `missionmind-ai-backend-plan.md`.
Read it before starting any new sub-task. Next: **Sub-Task 7 — MissionCommander**.
