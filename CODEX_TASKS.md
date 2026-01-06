# CODEX_TASKS.md — Ultimate Trading Bot v2.2 Execution Plan (Acceptance Criteria)

This document defines the required execution phases for building **Ultimate Trading Bot v2.2 (OpenAI Enhanced)**.
Codex (or any agent) must follow these phases sequentially. A phase is not complete until all acceptance criteria pass.

## Global Rules (Apply to all phases)
- No refusal language (no “impossible”, “can’t”, “resource limits”).
- No placeholders: no `TODO`, `pass`, stubs, “NotImplemented”.
- Never hardcode secrets. All secrets come from environment variables and `.env` (see `.env.example`).
- Cross-module data must use `src/core/contracts.py` only (no ad-hoc dict passing across layers).
- After any meaningful change run:
  - `python -m compileall .`
  - `python -m pytest -q`
- If a bug is fixed, add a regression test.
- Target runtime: Windows + Python 3.11+

---

## PHASE A — Core Trading Loop (Paper) + Persistence + Watchlist
**Goal:** A full end-to-end paper trading loop exists and runs at least one cycle, with deterministic tests.

### A.1 Required components (must exist and be non-empty)
- `config/`:
  - `settings.py`, `api_config.py`, `constants.py`, `logging_config.py`, `strategy_config.py`, `risk_config.py`
- `src/core/`:
  - `contracts.py`, `trading_engine.py`, `order_manager.py`, `position_manager.py`, `account_manager.py`,
    `session_manager.py`, `market_hours.py`, `symbol_manager.py`, `event_bus.py`, `state_machine.py`,
    `scheduler.py`, `heartbeat.py`
- `src/data/`:
  - `data_manager.py`, `market_data.py`, `historical_data.py`, `real_time_data.py`,
    `alpaca_data.py`, `data_validator.py`, `data_cleaner.py`, `data_transformer.py`,
    `cache_manager.py`, `database_manager.py`, `data_aggregator.py`
- `src/strategies/`:
  - `base_strategy.py`, `strategy_manager.py`, `signal_generator.py`, `position_sizer.py`,
    `rsi_strategy.py`, `moving_average_strategy.py`
- `src/risk/`:
  - `risk_manager.py`, `position_risk.py`, `portfolio_risk.py`, `drawdown_monitor.py`,
    `var_calculator.py`, `stop_loss_manager.py`, `take_profit_manager.py`, `trailing_stop.py`,
    `exposure_manager.py`, `correlation_risk.py`, `liquidity_risk.py`, `sector_exposure.py`,
    `daily_loss_limit.py`, `risk_reporter.py`
- `src/execution/`:
  - `execution_engine.py`, `order_router.py`, `order_types.py`, `fill_tracker.py`,
    `slippage_analyzer.py`, `transaction_cost.py`, `alpaca_executor.py`, `paper_executor.py`,
    `order_validator.py`, `execution_analytics.py`, `smart_order_router.py`,
    `order_aggregator.py`, `execution_logger.py`
- `src/watchlist/`:
  - `watchlist_manager.py`, `watchlist_storage.py`, `watchlist_importer.py`, `watchlist_exporter.py`,
    `watchlist_validator.py`, `watchlist_sync.py`
- `scripts/`:
  - `setup.py`, `run_bot.py`, `download_data.py`, `database_migrate.py`, `health_check.py`
- `tests/`:
  - `conftest.py`, `test_core.py`, `test_strategies.py`, `test_risk.py`, `test_execution.py`,
    `test_watchlist.py`, `test_integration.py`

### A.2 Contracts (hard requirement)
`src/core/contracts.py` must define and be used end-to-end:
- `RunContext(run_id, mode, timestamps, cost_budget, dry_run)`
- `MarketSnapshot`
- `SignalIntent`
- `RiskDecision`
- `ExecutionRequest`
- `TradeFill`

No module may pass raw dicts across layer boundaries (data→analysis→strategies→risk→execution).

### A.3 Persistence
- SQLite is the default local DB.
- `src/data/database_manager.py` must persist:
  - orders
  - fills
  - positions
  - runs (run_id, mode, timing)
  - journal entries
- `scripts/database_migrate.py` must initialize/upgrade schema idempotently.

### A.4 Watchlist (minimum)
- CSV import/export works.
- Validation: symbol format, duplicates, max-size enforcement (supports 250).
- A sample file exists: `data/watchlist.csv` (or equivalent) and is referenced by docs/scripts.

### A.5 Execution behavior (paper + dry-run)
- `scripts/run_bot.py` must support:
  - `--mode paper|live|backtest`
  - `--dry-run` (never sends orders)
  - `--watchlist <path>`
- Pipeline enforcement:
  - strategies output `SignalIntent`
  - risk outputs `RiskDecision` approve/veto with reasons
  - execution consumes `ExecutionRequest`
- Paper trading is default. Live trading is gated (PIN + explicit confirm).

### A.6 Phase A acceptance commands (must pass)
Run and show outputs:
1. `python -m compileall .`
2. `python -m pytest -q`
3. `python -m scripts.run_bot --mode paper --dry-run --watchlist data/watchlist.csv`
   - Must complete at least one loop cycle without crashing
   - Must log `run_id` and key stage transitions (data, strategy, risk, execution)

---

## PHASE B — API + UI + Websocket + Monitoring + Notifications
**Goal:** UI renders, API endpoints respond, websocket provides real-time feed, monitoring/alerts work.

### B.1 Required components (must exist and be non-empty)
- `src/api/`:
  - `api_server.py`, `auth_middleware.py`, `rate_limiter.py`, `serializers.py`, `validators.py`
  - `endpoints/`:
    - `trading_endpoints.py`, `data_endpoints.py`, `strategy_endpoints.py`, `backtest_endpoints.py`,
      `portfolio_endpoints.py`, `alert_endpoints.py`, `watchlist_endpoints.py`, `ai_endpoints.py`,
      `health_endpoints.py`
- `src/ui/`:
  - `app.py`, `routes.py`, `websocket_server.py`, `dashboard.py`, `charts.py`, `tables.py`, `forms.py`,
    `auth.py`, `watchlist_ui.py`, `symbol_search.py`, `bulk_import_ui.py`, `alert_dashboard.py`,
    `real_time_feed.py`, `strategy_builder.py`, `backtest_visualizer.py`, `portfolio_heatmap.py`,
    `trade_analytics.py`, `mobile_responsive.py`, `settings_ui.py`, `ai_chat_interface.py`,
    `ai_insights_panel.py`
  - `templates/` (all HTML listed in schema must exist and render)
  - `static/` (all CSS/JS listed in schema must exist and be served)
- `src/monitoring/`:
  - `system_monitor.py`, `performance_tracker.py`, `health_checker.py`, `metrics_collector.py`,
    `dashboard_data.py`, `alert_manager.py`, `log_analyzer.py`, `uptime_monitor.py`
- `src/notifications/`:
  - `notification_manager.py`, `email_sender.py`, `telegram_bot.py`, `discord_bot.py`, `sms_sender.py`,
    `advanced_alerts.py`, `ai_notification_writer.py`

### B.2 API requirements
- Health endpoint returns status and core subsystem health.
- Trading endpoints can:
  - fetch account/positions
  - fetch recent runs/journal
  - request a dry-run cycle
- Watchlist endpoints support:
  - list/add/remove/import/export
- Rate limiting is enforced.
- Authentication exists (basic session login is acceptable).

### B.3 UI requirements
- App starts locally.
- Pages render without template errors:
  - dashboard, watchlist, import, symbol search, strategy, backtest, portfolio, settings, alerts,
    trades, analytics, login, ai_chat, ai_insights
- Websocket feed works locally:
  - real-time updates visible on UI
- Mobile responsive layout is not broken (basic CSS rules present).

### B.4 Monitoring + alerts
- `scripts/health_check.py` returns subsystem checks:
  - DB connectivity
  - data fetch (mock/offline acceptable for dev)
  - scheduler/heartbeat
  - API/UI process health
- Alert manager can emit at least:
  - daily loss limit trigger
  - system health degrade trigger

### B.5 Phase B acceptance commands (must pass)
Run and show outputs:
1. `python -m compileall .`
2. `python -m pytest -q`
3. `python -m scripts.health_check`
4. Start servers:
   - API server start command (document exact command in docs)
   - UI start command (document exact command in docs)
   - Verify a health endpoint returns OK

---

## PHASE C — OpenAI Enhanced (Read-only AI) + Cost Control + Schema Validation + Vision
**Goal:** OpenAI features operate safely, validated, and cost-capped. AI never directly executes trades.

### C.1 Required components (must exist and be non-empty)
- `config/`:
  - `openai_config.py`, `prompts_config.py`
- `src/ai/` (all must be implemented):
  - `openai_client.py` (timeout + retry/backoff + circuit breaker)
  - `sentiment_analyzer.py`
  - `chart_analyzer.py` (Vision support)
  - `trading_agent.py` (function calling for READ/PLAN tools only; execution disabled)
  - `strategy_advisor.py`
  - `news_processor.py`
  - `risk_assessor.py`
  - `market_narrator.py`
  - `prompt_manager.py`
  - `cost_tracker.py` (daily + run caps; degrade mode)
  - `response_validator.py` (schema validation)
- AI integrations in other modules (must exist and be connected):
  - `src/data/ai_data_preprocessor.py`, `src/data/news_ai_enricher.py`
  - `src/analysis/ai_pattern_analyzer.py`, `src/analysis/ai_technical_interpreter.py`
  - `src/strategies/ai_sentiment_strategy.py`, `src/strategies/ai_consensus_strategy.py`
  - `src/sentiment/openai_sentiment.py`, `src/sentiment/sentiment_signal_generator.py`
  - `src/ml/llm_feature_extractor.py`, `src/ml/hybrid_ml_ai.py`
  - `src/backtesting/ai_backtest_analyzer.py`
  - `src/notifications/ai_notification_writer.py`
  - `src/api/endpoints/ai_endpoints.py`
  - `src/ui/ai_chat_interface.py`, `src/ui/ai_insights_panel.py`

### C.2 Safety requirements (hard)
- AI never calls execution directly.
- AI outputs must be validated by schema:
  - if invalid: retry once with stricter prompt
  - if still invalid: fallback to “no-op / neutral”
- Cost caps:
  - per run_id budget
  - daily budget
  - when exceeded: AI degrade mode (AI disabled, bot continues safely)

### C.3 Vision requirements
- `chart_analyzer.py` must accept an image input path/bytes, encode, and return structured result.
- Must work with Pillow installed.
- Must be tested with at least one local test image fixture in `tests/test_data/`.

### C.4 Scripts
- `scripts/test_openai_connection.py`:
  - If key exists: performs a minimal request and returns success
  - If key missing: exits gracefully and marks tests as skipped (not failed)
- `scripts/ai_cost_report.py` generates a report from stored cost logs.

### C.5 Tests (must pass)
- `tests/test_ai_sentiment.py`
- `tests/test_ai_agent.py`
- `tests/test_ai_chart.py`
- `tests/test_ai_integration.py`

### C.6 Phase C acceptance commands (must pass)
Run and show outputs:
1. `python -m compileall .`
2. `python -m pytest -q`
3. `python -m scripts.test_openai_connection`
4. If OpenAI key exists, run one dry-run cycle with AI enabled and show logs include:
   - cost tracking
   - schema validation status
   - degrade mode behavior (if cap reached)

---

## PHASE D — Backtesting + Optimization + ML Pipeline (Stability & Reports)
**Goal:** backtests, parameter optimization, and ML pipeline run reliably and produce reports.

### D.1 Required components
- Backtesting:
  - `src/backtesting/*` modules implemented and integrated
  - `scripts/run_backtest.py` works end-to-end
  - `backtest_reporter.py` outputs summary artifacts
- Optimization:
  - `src/optimization/*` runs grid/random/bayesian/genetic/PSO
  - `scripts/optimize_strategy.py` works
- ML:
  - `src/ml/*` training/evaluation/prediction + model_registry + drift detection
  - `scripts/train_ml_models.py` works

### D.2 Phase D acceptance commands
1. `python -m compileall .`
2. `python -m pytest -q`
3. `python -m scripts.run_backtest --symbols data/watchlist.csv --start <date> --end <date>`
4. `python -m scripts.optimize_strategy ...`
5. `python -m scripts.train_ml_models ...`

---

## Documentation Deliverables (must be kept current)
- `docs/INSTALLATION.md`: Windows setup steps + venv + requirements
- `docs/CONFIGURATION.md`: all env vars + config flags
- `docs/STRATEGIES.md`: list of strategies and inputs/outputs
- `docs/API_REFERENCE.md`: endpoints + auth
- `docs/AI_INTEGRATION.md`, `docs/OPENAI_SETUP.md`, `docs/COST_MANAGEMENT.md`
- `docs/TROUBLESHOOTING.md`: common errors and fixes

---

## CI Deliverables
- `.github/workflows/ci.yml` runs:
  - compileall
  - pytest
- `.github/workflows/ai-tests.yml` runs AI tests with secrets set in CI (optional)
