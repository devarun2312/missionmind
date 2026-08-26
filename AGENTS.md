# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Repository layout

The Python package lives one level deep: `missionmind/missionmind/`.
All tests live in `missionmind/tests/`.
`pyproject.toml` and `requirements.txt` live in `missionmind/` (the outer folder).

## Commands

**Python interpreter** — the system uses MSYS2's ucrt64 Python; `python` / `python3` are
not on the Windows PATH. Use the full path or set `PYTHONPATH` explicitly:

```bash
# Run all tests (must set PYTHONPATH to the outer missionmind/ folder)
$env:PYTHONPATH = "C:\Users\devar\OneDrive\Desktop\ibmbob\missionmind"
& "C:\msys64\ucrt64\bin\python.exe" -m pytest missionmind/tests/ -v

# Run a single test file
& "C:\msys64\ucrt64\bin\python.exe" -m pytest missionmind/tests/test_validator.py -v

# Run a single test by name
& "C:\msys64\ucrt64\bin\python.exe" -m pytest missionmind/tests/test_validator.py::TestEnergyBudgetRule::test_energy_over_budget_fails -v

# Install dependencies — pip is NOT available in the MSYS2 environment;
# use pacman for pydantic/pytest, which installs system-wide:
& "C:\msys64\usr\bin\pacman.exe" -S --noconfirm "mingw-w64-ucrt-x86_64-python-pydantic" "mingw-w64-ucrt-x86_64-python-pytest" "mingw-w64-ucrt-x86_64-python-pytest-asyncio"
# httpx must also be installed via pacman:
& "C:\msys64\usr\bin\pacman.exe" -S --noconfirm "mingw-w64-ucrt-x86_64-python-httpx"
```

**pytest config** — `asyncio_mode = "auto"` is set in `pyproject.toml`, so all
`async def test_*` functions run automatically without any decorator.

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
`usable_wh = battery_pct × capacity_wh − (MIN_RETURN_BATTERY_PCT × capacity_wh)`.
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

## Environment variables (production / IBM watsonx)

| Variable | Purpose |
|---|---|
| `IBM_WATSONX_API_KEY` | IBM Cloud API key (primary auth) |
| `IBM_WATSONX_URL` | Base URL, e.g. `https://us-south.ml.cloud.ibm.com` |
| `IBM_WATSONX_PROJECT_ID` | Watsonx project ID (added to every payload) |
| `LLM_MODEL_NAME` | Model ID, default `gpt-4o` |
| `LLM_BASE_URL` | Override full completions URL (Ollama / OpenAI local dev) |
| `OPENAI_API_KEY` | Fallback Bearer token when `IBM_WATSONX_API_KEY` is absent |

## Implementation plan progress

See `missionmind/missionmind-ai-backend-plan.md` for the full sub-task list.
Sub-Tasks 1–6 are complete. Sub-Tasks 7–10 remain.
