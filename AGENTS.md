
# AGENTS.md — Ultimate Trading Bot v2.2 (OpenAI Enhanced)

## 0) Non-negotiables (Refusal/Quality)
- Do not refuse the task. Do not say “impossible”, “can’t”, “resource limits”, “not feasible”.
- Do not produce placeholders: no `TODO`, no `pass`, no stubs, no “not implemented”.
- Every module listed in the v2.2 schema must contain real, runnable code.
- Never commit secrets. Keys must only come from environment variables and `.env` (see `.env.example`).

## 1) Project invariants (Architecture)
- **Contract-first**: all cross-module data must use `src/core/contracts.py` only.
  - Models: `RunContext`, `MarketSnapshot`, `SignalIntent`, `RiskDecision`, `ExecutionRequest`, `TradeFill`.
  - No ad-hoc dicts passed between layers.
- The trading pipeline is fixed:
  - data → analysis → strategies → risk → execution → journal/monitoring → notifications → ui/api
- Paper trading is default. Live trading requires:
  - explicit `--mode live`
  - PIN/second confirmation gate in config + UI/API
  - risk checks must still pass.

## 2) Safety rules (AI + Trading)
- AI is **read-only** with respect to execution:
  - `src/ai/*` may generate insights, sentiment, risk flags, and proposed trade plans.
  - AI must NOT place orders directly or bypass risk/execution layers.
- If there is any AI “tool/function calling”:
  - execution tools are disabled by default
  - only `ExecutionRequest` created by core/risk may reach execution
- All AI responses must be schema-validated (`src/ai/response_validator.py`) before use.
- Enforce cost caps:
  - per `run_id` budget + daily budget in `src/ai/cost_tracker.py`
  - if cap exceeded: degrade mode (AI off) but bot continues safely.

## 3) Configuration rules
- Config is loaded from:
  - `config/settings.py` + `config/api_config.py` (+ `openai_config.py`, `prompts_config.py` when enabled)
- `.env.example` must be kept up-to-date with all required keys.
- Defaults:
  - database: SQLite (local) for quick start; must be configurable
  - logging: JSON + console, include `run_id` in every log record

## 4) Dependencies & compatibility
- Target runtime: Windows + Python 3.11+
- Prefer `pydantic` for typed validation.
- For networking: set timeouts everywhere (OpenAI, broker, data sources).
- OpenAI SDK: use the official `openai` package and modern API patterns; keep code compatible with pinned version in `requirements-ai.txt`.

## 5) Testing & acceptance gates (Must pass)
After **every meaningful change**, run:
- `python -m compileall .`
- `python -m pytest -q`

Do not stop until tests pass.
If a bug is fixed, add a regression test.

Minimum working checks required at each phase:
- Phase A (Core loop): `python -m scripts.run_bot --mode paper --dry-run --watchlist data/watchlist.csv` runs at least one cycle and logs cleanly.
- Phase B (UI/API): server starts; API endpoints respond; UI templates render; websocket feed runs locally.
- Phase C (OpenAI): `python -m scripts.test_openai_connection` passes with a key; if missing key, it must skip gracefully with clear logs (not fail).

## 6) Code quality conventions
- Keep modules cohesive: one responsibility per file; share logic via `src/utils/*`.
- Errors must be explicit with custom exceptions (`src/utils/exceptions.py`).
- Use type hints across public APIs; validate external inputs.
- Add docstrings for public classes/functions, especially in core/risk/execution/ai.
- Never silently swallow exceptions; log with stack trace and propagate where appropriate.

## 7) Repository operations (Edits)
- Preserve the v2.2 folder structure exactly.
- Do not delete modules listed in schema.
- If you must introduce a new helper module, put it under `src/utils/` and keep imports clean.
- When altering contracts/config, update all dependents in the same change.

## 8) Deliverables format (What to output at the end)
When completing a task, provide:
1) The commands executed and their outputs (compileall + pytest).
2) The run command executed and key log lines proving it worked.
3) A short list of files changed.
