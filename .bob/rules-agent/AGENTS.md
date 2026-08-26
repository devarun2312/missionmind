# AGENTS.md — Agent (coding) mode

This file provides guidance to agents when working with code in this repository.

## Non-obvious coding rules

### Always use `MagicMock` + `AsyncMock` for LLM client mocks
```python
mock = MagicMock()
mock.complete = AsyncMock(return_value=LLMResponse(content=json.dumps(payload), model="test-model"))
```
`AsyncMock(spec=["complete"])` makes `.complete` a plain `MagicMock` (not awaitable).

### Adding a new agent — exact checklist
1. `missionmind/agents/<name>_agent.py` — subclass `BaseAgent[YourSchema]`
2. Implement `name` (str), `system_prompt` (reads `_load_prompt()`), `response_schema` (returns schema class)
3. `missionmind/prompts/<name>_prompt.md` — must include JSON schema in output section
4. Export from `missionmind/agents/__init__.py`
5. If injecting a config value: override `run()`, merge into context dict, call `super().run(enriched)`
6. Test file: inject `_make_mock_client(payload)`, assert `isinstance(result, YourSchema)`

### Energy formula — two versions exist, only one is correct
Validator formula (correct): `current_wh - reserve_wh` where both are computed from `capacity_wh`.
**Wrong** version: `battery_pct × capacity_wh × (1 − reserve_pct)` — gives a different value when battery < 100%.
Always use: `usable = max(0, battery_pct * capacity_wh - MIN_RETURN_BATTERY_PCT * capacity_wh)`.

### SafetyValidator must never import AI code
The `test_no_ai_imports_in_module` test inspects source via `inspect.getsource()`.
Never add `BaseAgent`, `WatsonxClient`, or `LLMClient` to `safety/validator.py`.

### Violation message prefixes are tested literally
Tests use `assert any("ENERGY BUDGET" in m …)`. These exact prefixes must stay:
`ENERGY BUDGET EXCEEDED`, `TERRAIN RISK EXCEEDED`, `MISSION DURATION EXCEEDED`, `MISSING RETURN TO BASE`, `NO SCIENCE WAYPOINTS`.

### Prompt files are module-level cached strings
`_SCIENCE_SYSTEM_PROMPT`, `_RESOURCE_SYSTEM_PROMPT`, etc. are loaded once at import time.
Changes to `.md` files require a process restart — no dynamic reload.

### `asyncio_mode = "auto"` is active — no `@pytest.mark.asyncio` needed
All `async def test_*` functions run automatically. Adding the decorator is harmless but redundant.

### `PYTHONPATH` must point to the **outer** `missionmind/` directory
```powershell
$env:PYTHONPATH = "C:\Users\devar\OneDrive\Desktop\ibmbob\missionmind"
& "C:\msys64\ucrt64\bin\python.exe" -m pytest missionmind/tests/test_validator.py::TestClass::test_name -v
```

### `openai` package is banned — use `httpx` only
MSYS2 Python cannot build `jiter` (Rust). `WatsonxClient` calls the REST API directly.
Do not add `openai` to `pyproject.toml` or `requirements.txt`.
