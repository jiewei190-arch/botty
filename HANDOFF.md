# Botty development handoff

Last updated: 2026-09-11 UTC

## User goal

Finish Botty as a risk-first swing-options trader. It scans US equities for
directional setups, names the exact contract to buy (ticker, strike, expiry,
call/put), paper executes and supervises positions, persists across restarts,
and sends useful alerts. The user trades manually on Robinhood and reads
Botty's output as a shortlist, so **an alert is a ranked opportunity, not an
instruction**. The intended holding period is roughly one to six weeks.

## Non-negotiable safety rules

- Paper trading first. Unattended live orders remain blocked until the complete
  paper workflow is validated with real paper results. The live path is behind a
  double lock (`ENABLE_LIVE_TRADING` + `LIVE_TRADING_CONFIRMATION`); never
  weaken either.
- Never commit `.env`, API keys, webhook URLs, or account identifiers.
- Every entry is risk-approved. Long options cap loss at premium paid; the
  market-hours supervisor manages premium stops, targets, time, and expiry exits.
- Broker state is truth; SQLite is an idempotently reconciled audit mirror.
- Do not silently fabricate fills, exits, or prices. Alert on unexplained state.
- Work on a feature branch, run full tests and Ruff, then push the branch.

## Repository state

- Repository: `jiewei190-arch/botty`
- `main` is at `ac073a9` — contract display on every scanned play plus the
  end-of-day market recap (PR #7). `codex/order-reconciliation` is merged into
  this history; do not merge it again.
- `fix/cloud-automation-status` (13 commits, head `a023da0`) is **still
  unmerged** and conflicts with `main`. Nobody has decided whether it lands.
  Read it before assuming it is dead work.
- SQLite schema is at `user_version` 4. `MIGRATIONS` is append-only:
  **never renumber a released migration.** v1/v2 shipped; a database in the wild
  is already at 2, so redefining an old slot leaves it permanently unmigrated.

## Verification commands

```bash
.venv/bin/python -m ruff check trading_bot tests
.venv/bin/python -m pytest
.venv/bin/python main.py --help
```

No credentials are required for the tests. Current baseline on `main`:
**1208 passed, 1 skipped, Ruff clean.**

## Architecture

Foundation → Access → Analytics → Decision → Orchestration → Interface. Each
layer depends only on the ones below it.

- `trading_bot/config/` — settings (pydantic) and the categorised symbol
  universe. `DEFAULT_STREAM_CATEGORIES` resolves to 27 symbols because the free
  Alpaca plan allows **30 per websocket subscription**.
- `trading_bot/data/` — Alpaca clients, bar cache, SQLite repositories, and
  `market_stream.StreamSupervisor` (REST credential pre-flight, worker thread,
  bounded queue, staleness watchdog).
- `trading_bot/detectors/` — five independent detectors (volume, momentum,
  breakout, gap, volatility expansion) emitting one `DetectorSignal` shape, plus
  the scoring engine. **Scoring ranks; it does not decide to trade.** A silent
  detector scores zero rather than "missing" — confluence is the point.
- `trading_bot/strategies/`, `indicators/`, `risk/`, `backtesting/` — signal
  generation, sizing, and simulation.
- `trading_bot/options/` — one contract selector shared by the CLI, the
  dashboard and the runner. Two implementations would drift and the contract on
  screen would stop matching the one the bot would buy.
- `trading_bot/alerts/`, `automation/`, `dashboard/` — delivery, the supervised
  runner, and the read-only UI.
- `trading_bot/utils/market_hours.py` — NYSE sessions, holidays and early closes
  computed offline, so no API call is needed to know whether the market is open.

CLI: `config check clock db-init status recap universe detect watch fetch
analyze signals scan calibrate hunt backtest dashboard cache`.

## Swing geometry — derived, not chosen

These numbers are coupled. Changing one without the others reintroduces bugs
that were measured on real bars, so change them together and re-measure.

| Setting | Value | Why |
|---|---|---|
| `max_holding_bars` | 21 | 21 trading days ≈ 30 calendar days |
| `atr_stop_multiplier` / `min_stop_atr` | 2.25 / 2.0 | survives normal noise |
| `atr_target_multiplier` | 4.5 | reward:risk 2.0, reachable inside the cap |
| `planned_max_hold_days` | 30 | matches the equity swing window |
| `exit_before_expiry_days` | 14 | theta and gamma accelerate inside 3 weeks |
| `min_dte` | 45 | **hold + exit buffer**, enforced by a validator |
| `target_dte` / `max_dte` | 60 / 90 | room for the thesis without paying for a year |

Under random-walk √t geometry, price covers *k* ATRs in roughly *k²* bars: a
4.5-ATR target needs ~20 bars, which is why the cap is 21 and not 10. The old
config allowed a 7-DTE contract while planning a 60-day hold with a 21-day exit
rule — it was past its own exit rule on the day it opened.

Three defects fixed here were found by **backtesting real bars, not by reading
config**: the holding cap was unreachable from the backtester, exits were too
tight to ever reach the target, and entries that gapped past the signal were
filled anyway. Median hold was 2 days against a 7–30 day target. Expectancy is
still indistinguishable from zero on 24 trades — that is a sample-size
statement, not a green light.

## Free Alpaca (Basic) plan limits — verified, work within them

- IEX feed only; **30 symbols per websocket subscription**; 200 REST req/min;
  historical data excludes the last 15 minutes.
- Options get the **indicative** feed, not OPRA (`ALPACA_OPTIONS_FEED`).
- An option snapshot carries `symbol`, `latest_trade`, `latest_quote`,
  `implied_volatility` and `greeks` — **there is no daily bar on it.** Reading
  one silently zeroed volume and rejected every chain. Spread and open interest
  are the real liquidity gates; `min_daily_volume` defaults to 0 on purpose.
- Greeks may be absent on the indicative feed, so the selector falls back to
  moneyness and records `delta_source` — an approximation must never read as a
  measurement.
- The SDK retries a rejected credential forever, so the stream supervisor does a
  REST pre-flight first. Do not remove it.

## Remaining work, in order

1. **Run the live data path at all.** Neither the live market data path nor the
   options path has ever executed — the account was under compliance review. The
   open question is whether the free indicative feed supplies delta, or whether
   every selection comes back flagged `moneyness`. Only a working key answers it.
2. Run a paper soak test: real fills, slippage, duplicate prevention, holidays,
   restarts, alerts.
3. Decide the fate of `fix/cloud-automation-status`.
4. A macro/news provider behind `MarketRecap.headlines`. The recap deliberately
   says **nothing about the economy**: CPI prints, Fed decisions and payroll
   numbers are not in a price feed, and inferring them from index moves is
   narration dressed as analysis. The field stays empty until a real source
   exists, and a test pins that.
5. Only after explicit user approval and paper evidence, design a separate live
   activation release. Do not enable it incidentally.

## Current scope boundaries

- Long calls and long puts only. No naked short options, 0DTE, same-day trading,
  spreads, exercise, or unattended real-money orders.
- Each swing uses $500–$1,000 of premium; at most two swings and four total
  contracts open. Same-day option exits are structurally blocked.
- The process is online continuously, but scans, quotes, entries and exits occur
  during regular US market sessions.
- A real paper soak test and private Render/Slack secrets remain required before
  the deployment can truthfully be called operational.
- Render Free is validated for initial testing; `deploy/oracle-cloud/` holds an
  Oracle Always Free VM installer for the continuous soak test. Stop the Render
  worker before starting Oracle to prevent duplicate automation.

## Working notes for the next agent

- **Measure, do not reason.** Every real defect in this repository was found by
  running code against real bars and reading the output, never by inspecting it.
  Tests passing is not evidence that the behaviour is right.
- Verify SDK surfaces against the installed package before calling them. A
  fabricated attribute fails silently here — the `daily_bar` read did.
- Do not overclaim results. If expectancy is noise, say so.
