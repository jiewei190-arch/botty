# Botty development handoff

Last updated: 2026-09-08 UTC

## User goal

Finish Botty as a fully automated, risk-first US-equity algorithmic trader. It
must wake for official market sessions, scan, size, paper-execute, supervise,
persist, recover after restarts, and notify the user only when useful or when
attention is required. Build and test the complete system with fakes/synthetic
data before asking for Alpaca or notification credentials.

## Non-negotiable safety rules

- Paper trading first. Unattended live orders remain blocked until the complete
  paper workflow is validated with real paper results.
- Never commit `.env`, API keys, webhook URLs, or account identifiers.
- Every entry is risk-approved and submitted with broker-hosted stop and target.
- Broker state is truth; SQLite is an idempotently reconciled audit mirror.
- Do not silently fabricate fills, exits, or prices. Alert on unexplained state.
- Work on a feature branch, run full tests and Ruff, then push the branch.

## Repository state

- Repository: `jiewei190-arch/botty`
- Base milestone on `main`: `f09b354` — market-open paper automation.
- Active branch: `codex/order-reconciliation`
- Core scanner, strategies, indicators, risk manager, backtester, market-wide
  universe discovery, dashboard, cache, logging, and SQLite repositories exist.
- `hunt --watch-market --paper-trade` uses Alpaca's market clock and scans once
  per regular market session.

## Work implemented on the active branch

- Normalized broker order queries, including nested bracket legs.
- Idempotent order import into SQLite.
- Filled Botty entries open trade/position records.
- Filled stop/target legs close trades and calculate realized P&L.
- Broker positions and equity are refreshed during market-hours supervision.
- Rejected orders and unexplained position disappearance trigger alerts.
- Reconciliation runs every open-market poll without repeating the daily scan.
- Completed scan session is stored in `bot_events`, preventing a restart from
  repeating the same successful session; failed scans remain retryable.
- Paper sizing uses the actual paper account and positions, not an empty model.
- Stale-signal and repeated-order-failure circuit breakers block unsafe entries.
- Docker/Compose packaging, heartbeat health check, operator `status` command,
  and a read-only Automation dashboard page are present.
- Simulated execution/reconciliation tests cover guarded bracket submission,
  stale and rejection circuit breakers, fills, exits, restart deduplication, and
  recovery of a complete round trip that occurred while Botty was offline.

## Verification commands

```bash
python -m pip install -e '.[dev]'
python -m ruff check trading_bot tests
python -m pytest -q
python main.py hunt --help
```

No credentials are required for the tests.

## Remaining work, in order

1. Run a paper soak-test once credentials are supplied; review actual fills,
   slippage, duplicate prevention, market holidays, restarts, and alerts.
2. Only after explicit user approval and paper evidence, design a separate live
   activation release. Do not enable it incidentally.

## Current scope boundaries

- US equities, not options. Options need separate contract selection, liquidity,
  Greeks, assignment/expiration, and risk handling.
- Regular-session market-open automation. Extended/overnight execution is not
  enabled and must be separately tested against feed/order limitations.
- Discord and Slack incoming webhooks are implemented; credentials come later.
