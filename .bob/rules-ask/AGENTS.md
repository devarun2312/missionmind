# AGENTS.md — Ask mode

This file provides guidance to agents when working with code in this repository.

## Key context for answering questions

### Where things actually live
- Package source: `missionmind/missionmind/` (double-nested — outer dir is the repo root for the package, inner is the importable package)
- Tests: `missionmind/tests/` (sibling to the inner package, NOT inside it)
- Prompts: `missionmind/missionmind/prompts/*.md` (Markdown files loaded at import time, not at call time)
- Implementation plan: `missionmind/missionmind-ai-backend-plan.md` (tracks sub-task completion)

### Two safety layers — they are NOT the same thing
- **SafetyAgent** (`agents/safety_agent.py`) — AI, async, probabilistic, returns `RiskAssessment`, recommends exclusions
- **SafetyValidator** (`safety/validator.py`) — pure Python, sync, deterministic, returns `ValidationResult`, enforces hard rules
- They are independent; SafetyValidator has a test that proves it contains zero AI imports

### `battery_pct` is always a fraction (0.0–1.0), never a percentage (0–100)
All config thresholds (`MIN_RETURN_BATTERY_PCT`, `CRITICAL_BATTERY_PCT`) are also fractions.

### IBM watsonx is the production AI backend
- Uses OpenAI-compatible `/ml/v1/text/chat` REST endpoint via `httpx`
- Requires `IBM_WATSONX_API_KEY`, `IBM_WATSONX_URL`, `IBM_WATSONX_PROJECT_ID`
- Falls back to `OPENAI_API_KEY` + `LLM_BASE_URL` for local dev
- No `openai` Python package — everything is raw HTTP through `WatsonxClient`

### All config is env-var overridable with sane defaults
`config.py` reads every threshold from an env var; changing behaviour for tests never
requires editing source — just pass injectable constructor args to `SafetyValidator` or
`BaseAgent(llm_client=mock)`.

### Sub-task completion status
Sub-Tasks 1–6 complete (models, BaseAgent, Science/Resource/Safety agents, SafetyValidator).
Sub-Tasks 7–10 pending (MissionCommander, Planner, Replanner, integration tests + AGENTS.md update).
