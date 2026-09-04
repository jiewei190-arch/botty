# Algorithmic Trading Bot

A modular, risk-first trading bot for US equities. Built to **test ideas safely**:
backtest a strategy on historical data, then paper trade it against a live market
feed, with real-money trading locked behind two explicit switches.

> **Status: Phase 1 of 10 complete.** The foundation and data layers are built and
> tested. Strategies, risk management, backtesting and the dashboard follow in
> later phases (see the [roadmap](#roadmap)).

---

## Design principles

1. **Risk management is not optional.** No order can be placed without passing
   validation. Position size is derived from stop distance, never guessed.
2. **Backtests must be honest.** Fees, slippage, candle-by-candle execution and a
   hard guard against lookahead bias — a backtest that lies is worse than none.
3. **Paper before live.** `PAPER` is the default mode and real trading requires
   two independent switches, one of which is an exact confirmation phrase.
4. **Strategies are pluggable.** Every strategy implements one interface and is
   driven entirely by parameters, so ideas can be compared on equal footing.
5. **Everything is logged.** Signals, rejections, risk calculations and fills all
   land in both a human-readable log and a queryable database.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Interface        main.py (CLI)          dashboard/ (Phase 9) │
├──────────────────────────────────────────────────────────────┤
│  Orchestration    bot engine — scan → decide → size → execute │
├──────────────────────────────────────────────────────────────┤
│  Decision         scanner → strategies/ → risk/               │
├──────────────────────────────────────────────────────────────┤
│  Analytics        indicators/            backtesting/         │
├──────────────────────────────────────────────────────────────┤
│  Access           data/market_data.py  data/database.py       │
│                   execution/broker.py                         │
├──────────────────────────────────────────────────────────────┤
│  Foundation       config/  utils/ (logging, retry, timeframes)│
└──────────────────────────────────────────────────────────────┘
```

Each layer depends only on the layers below it. That is what makes the pieces
swappable: the backtester and the live bot both consume the same
`MarketDataProvider` interface, so a strategy cannot tell which one is running it.

### Project layout

```
botty/
├── trading_bot/
│   ├── config/settings.py        Typed settings + live-trading locks
│   ├── data/
│   │   ├── market_data.py        Provider interface, Alpaca client, normalization
│   │   ├── cache.py              Parquet bar cache
│   │   ├── database.py           SQLite schema + repositories
│   │   └── models.py             Quote, MarketClock, AccountSnapshot, AssetInfo
│   ├── execution/broker.py       Read-only broker access (orders: Phase 7)
│   ├── utils/
│   │   ├── logging_setup.py      Console + rotating file + JSON-lines sinks
│   │   ├── retry.py              Exponential backoff for API calls
│   │   └── timeframes.py         Bar-size parsing and calendar arithmetic
│   ├── indicators/               Phase 2
│   ├── strategies/               Phase 3
│   ├── risk/                     Phase 4
│   ├── backtesting/              Phase 6
│   ├── dashboard/                Phase 9
│   └── main.py                   CLI
├── tests/                        151 tests, no credentials required
├── logs/                         Runtime logs (gitignored)
├── storage/                      SQLite database + parquet cache (gitignored)
├── main.py                       Launcher
├── requirements.txt
└── .env.example
```

---

## Setup

**Requirements:** Python 3.10+ and a free Alpaca paper-trading account.

```bash
git clone <your-repo-url> && cd botty

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # then add your API keys
```

Get free paper-trading keys from the
[Alpaca paper dashboard](https://app.alpaca.markets/paper/dashboard/overview)
and paste them into `.env`:

```bash
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
```

Paper and live accounts have **separate key pairs**. Phase 1 only ever uses the
paper endpoint.

### Verify the installation

```bash
python main.py check
```

```
Health check results
------------------------------------------------------------------------
  [PASS] database       schema v1 at storage/trading_bot.db
  [PASS] credentials    ALPACA_API_KEY and ALPACA_SECRET_KEY present
  [PASS] broker         PAPER account 8f3c21a0… status=ACTIVE equity=$100,000.00
  [PASS] market clock   CLOSED (opens 2026-09-05T13:30:00+00:00)
  [PASS] market data    5 daily bars for AAPL, latest 2026-09-04 close $234.07
------------------------------------------------------------------------
```

The command exits non-zero if any check fails, so it works in CI too.

---

## Commands

| Command | What it does |
|---|---|
| `python main.py check` | Full health check: config, database, credentials, broker, data |
| `python main.py config` | Print resolved configuration (secrets masked) |
| `python main.py clock` | Market session state |
| `python main.py fetch` | Download and preview historical bars |
| `python main.py db-init` | Create or migrate the database |
| `python main.py cache` | Inspect (`--clear` to empty) the bar cache |

Global flags: `--mode {backtest,paper,live}` and `--log-level {DEBUG,INFO,WARNING,ERROR}`.

### Fetching data

```bash
# Watchlist at the configured timeframe
python main.py fetch

# Specific symbols, explicit date range, exported to CSV
python main.py fetch --symbols AAPL,NVDA --timeframe 1Day \
    --start 2024-01-01 --end 2024-12-31 --csv-dir ./exports

# Bypass the cache to force a fresh download
python main.py fetch --symbols SPY --timeframe 5Min --no-cache
```

---

## Configuration

All configuration lives in `.env` — see `.env.example` for the annotated list.

### Risk defaults

| Setting | Default | Meaning |
|---|---|---|
| `RISK_MAX_RISK_PER_TRADE_PCT` | `1.0` | $100 max loss per trade on a $10,000 account |
| `RISK_MAX_DAILY_LOSS_PCT` | `3.0` | Trading halts for the day at this loss |
| `RISK_MAX_OPEN_POSITIONS` | `5` | Concurrent position cap |
| `RISK_MAX_PORTFOLIO_EXPOSURE_PCT` | `60.0` | Combined position value cap |
| `RISK_MIN_RISK_REWARD` | `2.0` | Reject setups paying less than 2:1 |
| `RISK_CONSECUTIVE_LOSS_LIMIT` | `3` | Losses before a cooldown |
| `RISK_COOLDOWN_MINUTES` | `60` | Cooldown duration |

These are validated for coherence at startup — for example, risk-per-trade may
not exceed the daily loss limit, because a single trade should never be able to
end the trading day.

### The live-trading lock

`TRADING_MODE=live` alone does nothing. The bot refuses to start unless **all
three** of these agree:

```bash
TRADING_MODE=live
ENABLE_LIVE_TRADING=true
LIVE_TRADING_CONFIRMATION=I UNDERSTAND THE RISKS
```

Any other combination raises a configuration error before a client is
constructed. `AlpacaBroker` passes `paper=True` to the SDK unless all three
checks pass, so there is no code path from default configuration to the live
endpoint. Phase 1 additionally contains **no order-placing code at all**.

---

## Two correctness decisions worth knowing

**Bar timestamps are bar-*open* times.** A 15-minute bar labelled `14:00` is
still forming until `14:15`. Acting on it at `14:05` means trading on information
that does not exist yet — the single most common way a backtest produces returns
that evaporate in live trading. `drop_incomplete_bars()` removes any bar whose
period has not closed, and every live fetch passes through it.

**Normalization happens at the boundary.** Every provider returns a frame with a
UTC `DatetimeIndex`, sorted ascending, duplicates collapsed, `float64` columns,
and rows violating OHLC invariants (`high < low`, negative prices) dropped.
Strategies never handle vendor quirks, and a malformed bar cannot silently
produce a fake signal.

---

## Data storage

**`storage/trading_bot.db`** (SQLite, WAL mode) holds seven tables: `runs`,
`signals`, `orders`, `trades`, `positions`, `equity_snapshots` and `bot_events`.

Rejected signals are recorded alongside accepted ones with the reason for
rejection — when the bot is not trading, that table tells you why. Schema changes
go through numbered migrations tracked by `PRAGMA user_version`.

**`storage/cache/`** holds parquet bar files keyed by symbol, timeframe, feed and
adjustment. Completed bars never change, so caching them makes repeated backtests
fast and removes rate-limit pressure. Adjustment is part of the key because
split-adjusted and raw prices are genuinely different series.

---

## Logging

Three sinks, for three different readers:

| Sink | Purpose |
|---|---|
| Console | Concise operator view |
| `logs/trading_bot.log` | Rotating full history (10 MB × 5) |
| `logs/errors.log` | Errors only, for triage |
| `logs/events.jsonl` | One JSON object per record, for analysis |

Signals render as a readable block *and* as structured JSON in the same call:

```
[2026-09-04 09:32:12] AAPL SIGNAL DETECTED

  Strategy        : Momentum
  Direction       : LONG
  Confidence      : 82/100
  Entry           : $210.50
  Stop Loss       : $207.00
  Take Profit     : $218.00
  Risk/Reward     : 1:2.14

  ✓ Bullish EMA crossover
  ✓ Increasing volume
  ✓ MACD bullish
  ✓ RSI healthy

  Risk Validation : PASSED
  Position Size   : 25 shares
```

---

## Testing

```bash
pytest                    # full suite
pytest -v tests/test_settings.py
```

The suite runs against synthetic bars and an in-memory database — **no API
credentials or network access required**, so it is safe to run in CI. Tests
cover the live-trading locks, bar normalization, the lookahead guard, cache
coverage rules, P&L arithmetic, retry classification and CLI exit codes.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Project setup, Alpaca connection, market data | **Complete** |
| 2 | Technical indicator engine | Planned |
| 3 | Strategy engine (momentum, mean reversion, breakout) | Planned |
| 4 | Risk management and position sizing | Planned |
| 5 | Market scanner with confidence scoring | Planned |
| 6 | Backtesting engine | Planned |
| 7 | Alpaca paper trading execution | Planned |
| 8 | Position monitoring and automated exits | Planned |
| 9 | Streamlit dashboard | Planned |
| 10 | Performance optimisation | Planned |

---

## Disclaimer

This software is for research and education. It is not financial advice. Trading
involves substantial risk of loss. Backtested performance does not predict future
results. You are responsible for any orders this software places on your behalf —
test thoroughly in paper mode, and read the code before enabling live trading.
