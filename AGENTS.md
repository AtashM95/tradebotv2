# Agent Rules (Ultimate Trading Bot v2.2)

- No placeholders (TODO/pass/stub). Everything must run.
- Always run:
  - python -m compileall .
  - python -m pytest -q
- AI is read-only; it must never place orders directly.
- All cross-module data uses src/core/contracts.py
- Secrets never committed; use .env.example
- Paper trading by default (alpaca-py TradingClient(paper=True))
- If something fails: fix root cause, add a regression test.
