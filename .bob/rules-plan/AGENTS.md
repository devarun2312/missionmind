# AGENTS.md — Plan mode

This file provides guidance to agents when working with code in this repository.

## Non-obvious architectural constraints

### The two-layer safety architecture is intentional and load-bearing
```
SafetyAgent (AI, soft)  →  recommends exclusions in RiskAssessment
SafetyValidator (Python, hard)  →  rejects plans in ValidationResult
```
They must stay separate. SafetyValidator contains a self-test (`test_no_ai_imports_in_module`)
that reads its own source to prove it has zero AI imports. Merging them breaks this test
and the trust model.

### ResourceAgent and SafetyAgent silently overwrite caller-supplied config keys
`ResourceAgent.run()` always overwrites `min_return_battery_pct` with `config.MIN_RETURN_BATTERY_PCT`.
`SafetyAgent.run()` always overwrites `max_terrain_risk_score` with `config.MAX_TERRAIN_RISK_SCORE`.
This is intentional: callers cannot accidentally (or maliciously) weaken the thresholds.
Any future agent that injects a config threshold MUST follow this same pattern.

### `LLMClient` is a structural Protocol — not an ABC
`WatsonxClient` satisfies it without inheriting from it. `isinstance(mock, LLMClient)` will
return `False` for plain `MagicMock` objects. The design uses behavioural conformance, not
inheritance, so swapping backends requires no code changes.

### Structural AI failures must never be retried
`AgentResponseError` (JSON parse / Pydantic validation) exits the retry loop immediately.
Only `LLMClientError` (network) triggers retries. Changing this breaks the "one call on
structural failure" test invariant present in every agent test file.

### Prompts are loaded at module import, not at agent instantiation
Each agent module has a `_*_SYSTEM_PROMPT` module-level constant. This means all agents
of the same type share the same prompt string regardless of when they were created.
There is no per-instance prompt — do not design features that require per-instance variation.

### Energy formula in SafetyValidator differs from the obvious formulation
The validator computes:
```python
current_charge_wh = battery_pct * capacity_wh
reserve_wh = MIN_RETURN_BATTERY_PCT * capacity_wh
usable = max(0, current_charge_wh - reserve_wh)
```
**Not**: `battery_pct * capacity_wh * (1 - reserve_pct)` — that formula is arithmetically
wrong for partially-charged batteries and was caught by a test failure during development.

### Package install is impossible via pip on this system
MSYS2's Python (cpython-312, MINGW ABI) reports `Unsupported platform: 312` to maturin,
so any Rust-compiled wheel (pydantic-core, jiter) fails to build. All runtime dependencies
must be available as pre-built MSYS2 pacman packages. This rules out the `openai` SDK.

### `PYTHONPATH` must be the outer `missionmind/` directory
The project is double-nested (`missionmind/missionmind/`). Running pytest from the wrong
directory silently breaks all imports. The `testpaths = ["tests"]` in `pyproject.toml`
assumes CWD is the outer `missionmind/` directory.

### Sub-Tasks 7–10 are pending
7. MissionCommander (orchestrates 3 agents in parallel via `asyncio.gather`, calls SafetyValidator, retries with pruned waypoints)
8. Planner (public entry point: `plan_mission(rover_state, env_state) -> MissionPlan`)
9. Replanner (event-driven: `replan(current_plan, event, rover_state, env_state) -> MissionPlan`)
10. Integration tests + `AGENTS.md` content update
