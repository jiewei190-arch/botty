# Botty development handoff

Last updated: 2026-09-08 UTC

## User goal

Finish Botty as a fully automated, risk-first swing-options trader. It scans
US equities for directional setups, selects liquid long calls/puts, paper
executes and supervises them, persists after restarts, and sends useful Slack
alerts. The intended holding period is roughly three to eight weeks.

## Non-negotiable safety rules

- Paper trading first. Unattended live orders remain blocked until the complete
  paper workflow is validated with real paper results.
- Never commit `.env`, API keys, webhook URLs, or account identifiers.
- Every entry is risk-approved. Long options cap loss at premium paid; Botty's
  market-hours supervisor manages premium stops, targets, time, and expiry exits.
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
- The primary automated instrument is now a long call for bullish setups or a
  long put for bearish setups; the equity scanner remains the signal engine.
- Normal contracts are 60–90 DTE. Only setups scoring at least 90 may use
  91–120 DTE contracts.
- Each swing must use $500–$1,000 of premium. At most two swings and four total
  contracts may be open. Same-day option exits are structurally blocked.
- Contract selection rejects weak delta, wide spreads, low volume, and low open
  interest. Limit orders are used instead of market orders.
- `Swing Tracker` stores and charts both the underlying and exact contract.
- `render.yaml` runs worker and dashboard in one always-on service with a shared
  persistent SQLite disk.
- Slack destination: `#general` in workspace `bottytrades`. The incoming webhook
  remains a private deployment secret named `AUTO_WEBHOOK_URL`.

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

- Long calls and long puts only. No naked short options, 0DTE, same-day trading,
  spreads, exercise, or unattended real-money orders.
- The process is online continuously, but scans, quotes, entries, and exits occur
  during regular US option-market sessions.
- A real paper soak test and private Render/Slack secrets remain required before
  the deployment can truthfully be called operational.
- Render Free has been validated for initial testing. An Oracle Always Free VM
  installer, systemd service, persistent data path, and authenticated dashboard
  are under `deploy/oracle-cloud/` for the continuous paper soak test. Stop the
  Render worker before starting Oracle to prevent duplicate automation.
