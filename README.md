# Algorithmic Trading Bot

A modular, risk-first trading bot for US equities. Built to **test ideas safely**:
backtest a strategy on historical data, then paper trade it against a live market
feed, with real-money trading locked behind two explicit switches.

> **Status: Phases 1-3 of 10 complete, plus a read-only dashboard.** Foundation,
> data, analysis and the strategy engine are built and tested, and there is a
> Streamlit dashboard to look at them. Risk management, backtesting and paper
> trading follow, and the dashboard grows trade controls with them (see the
> [roadmap](#roadmap)).

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
│   ├── indicators/
│   │   ├── technical_indicators.py  SMA/EMA/RSI/MACD/Bollinger/ATR + signal helpers
│   │   ├── trend_analysis.py        Weighted trend direction, strength, confidence
│   │   ├── volume_analysis.py       Relative volume, participation, confirmation
│   │   └── price_action.py          Swing points, support/resistance
│   ├── strategies/
│   │   ├── base_strategy.py         Signal/Position contract, exits, confidence
│   │   ├── momentum_strategy.py     Trend continuation
│   │   ├── mean_reversion.py        Fade stretched moves, with a regime veto
│   │   └── breakout_strategy.py     Confirmed breaks out of consolidation
│   ├── risk/                     Phase 4
│   ├── backtesting/              Phase 6
│   ├── dashboard/
│   │   ├── app.py                   Streamlit app (read-only)
│   │   ├── charts.py                Plotly figure builders
│   │   ├── theme.py                 Validated palette, light and dark
│   │   └── data.py                  Cached data access for the UI
│   └── main.py                   CLI
├── tests/                        482 tests, no credentials required
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
| `python main.py analyze` | Full technical analysis of a symbol (Phase 2) |
| `python main.py signals` | Run strategies and report trade setups (Phase 3) |
| `python main.py dashboard` | Launch the Streamlit dashboard |

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

### Analysing a symbol

```bash
# Try it with no API keys at all — generated sample data
python main.py analyze --demo --symbols AAPL

# Real analysis of your watchlist
python main.py analyze

# One symbol on a daily timeframe
python main.py analyze --symbols NVDA --timeframe 1Day
```

```
================================================================
MARKET ANALYSIS — AAPL
================================================================

Bars analysed : 300  (2026-08-22 13:30 to 2026-09-04 19:45 UTC)
Price         : $234.07

TREND
  Direction   : BULLISH
  Strength    : 72/100   (50 = neutral)
  Confidence  : 84/100

MOVING AVERAGES
  EMA 9       : $233.11  (price above)
  EMA 20      : $231.40  (price above)
  EMA 50      : $228.85  (price above)
  EMA 200     : $221.02  (price above)

MOMENTUM
  RSI 14      : 62.4  — NEUTRAL
  MACD        : BULLISH  (increasing momentum)
  EMA 9/20    : NONE

VOLATILITY
  ATR 14      : $1.86  (0.79% of price)
  Bollinger   : NORMAL
  Band width  : 0.0312

VOLUME
  Relative    : 1.42x average
  Condition   : HIGH
  Trend       : RISING

KEY LEVELS
  Resistance  : $238.90  (+2.06%, 3 touches)
  Support     : $229.55  (-1.93%, 2 touches)

SIGNALS
  ✓ Bullish EMA alignment (EMA 9 > EMA 20 > EMA 50 > EMA 200)
  ✓ Price above EMA 50
  ✓ Higher highs and higher lows
  ✓ Positive MACD momentum and rising
  ✓ Above-average volume at 1.42x
================================================================
```

---

## The indicator engine

Strategies never compute indicator maths themselves. They call
`calculate_all_indicators()` once and read columns, so a backtest and the live
scanner can never disagree about what a value was.

```python
from trading_bot.indicators import (
    calculate_all_indicators, analyze_trend, analyze_volume, find_support_resistance,
)

enriched = calculate_all_indicators(bars)      # appends every indicator column
trend = analyze_trend(enriched)                # BULLISH, 72/100, with reasons
volume = analyze_volume(enriched)              # HIGH, 1.42x average
levels = find_support_resistance(enriched)     # clustered from confirmed pivots
```

**Columns produced:** `SMA_20/50/100/200`, `EMA_9/20/50/200`, `RSI_14`, `MACD`,
`MACD_SIGNAL`, `MACD_HISTOGRAM`, `BB_UPPER/MIDDLE/LOWER/WIDTH/PERCENT_B`,
`ATR_14`, `ATR_14_PCT`, `VOLUME_SMA_20`, `RELATIVE_VOLUME`.

Every period and threshold lives in `IndicatorConfig`, so a strategy or a
backtest parameter sweep can vary them without touching calculation code:

```python
from trading_bot.indicators import IndicatorConfig
fast = IndicatorConfig(rsi_period=7, ema_periods=(5, 13, 34), volume_spike_threshold=3.0)
```

### Conventions that determine whether values match your charts

| Choice | Why |
|---|---|
| EMA seeded with an SMA | TA-Lib / TradingView convention. Seeding with the first value instead visibly changes the first few dozen bars. |
| RSI and ATR use Wilder's smoothing | `alpha = 1/period`, not a simple average. This is what "RSI 14" means everywhere it is quoted. |
| Bollinger uses population σ (`ddof=0`) | pandas defaults to the sample σ, which makes the bands too wide. |
| True Range undefined on bar 1 | It needs a previous close, so ATR's first value lands at index `period`. |

RSI, EMA and ATR are verified in the test suite against values derived
independently — by hand from the indicator's definition, or by a separate plain
Python loop.

### Warm-up returns NaN, not a guess

An EMA-200 computed from 30 bars is not a rough EMA-200, it is a wrong number.
Every indicator returns `NaN` until it has enough history. `calculate_all_indicators`
logs which indicators were short; pass `strict=True` to raise instead.

### Two lookahead guards

**Indicator maths is causal.** Every rolling window is trailing, never centred.
A test proves it: computing on a truncated history reproduces the values
computed on the full history, bar for bar.

**Pivots carry a confirmation lag.** A swing high at bar `i` cannot be known
until bar `i + strength`, because the rule needs the bars after it. Every
`SwingPoint` records `confirmed_index`, and `find_support_resistance(as_of=N)`
uses only pivots confirmed by bar `N` — so a backtest asking "what levels were
visible here?" gets an honest answer.

---

## The dashboard

```bash
python main.py dashboard        # http://localhost:8501
```

Four pages: **Overview** (account, watchlist snapshot), **Market Scanner**
(setups and, when there are none, what blocked them), **Chart** (four-panel
interactive price chart), **Strategy Settings** (live indicator tuning).

It works with no API keys — the sidebar's *Demo data* toggle is on by default
when credentials are absent, and every generated price is labelled as such.

**It is read-only and cannot place an order.** There is no order-placing code
anywhere in the system yet, by design; a test asserts the dashboard package
contains none. Trade controls arrive with the execution layer in Phase 7,
defaulting to paper trading with manual approval.

### Chart design

Four stacked panels share one x-axis — price, volume, RSI, MACD. **Never a dual
axis:** two scales on one plot let any two lines be placed in any relationship
the author likes, which is the most common way a chart misleads.

Colour choices are validated, not eyeballed. The three EMA hues clear
colour-vision-deficiency separation on every pair in both light and dark modes.
Two rules follow from that:

- **Up and down never depend on colour alone.** Candles differ in *fill* as well
  as hue — hollow for up, solid for down — so direction survives colourblindness
  and greyscale printing. Direction labels elsewhere pair an arrow with the word.
- **Series carry both a legend and end-of-line labels.** Labels alone fail
  exactly when EMAs converge and their labels collide.

Green means bullish and red means bearish everywhere, including the MACD
histogram. Streamlit's default red accent is overridden in
`.streamlit/config.toml` — left alone, it painted a `STRONG_BULLISH` trend bar
red, contradicting the app's own colour meaning.

Light or dark is set by `theme.base` in `.streamlit/config.toml`, and the chart
palette reads that same value, so the chrome and the charts cannot disagree.

---

## The strategy engine

Three strategies with deliberately different edges, behind one interface:

| Strategy | Thesis | Works when |
|---|---|---|
| `momentum` | Trends continue | Markets are trending |
| `mean_reversion` | Stretched prices snap back | Markets are ranging |
| `breakout` | Compression precedes expansion | Ranges are resolving |

They are *meant* to disagree. Momentum and mean reversion looking at the same
oversold chart should reach opposite conclusions — that is why you run more than
one, and why the scanner (Phase 5) will rank their output rather than average it.

```bash
python main.py signals --demo                      # no API keys needed
python main.py signals --strategy momentum         # one strategy
python main.py signals --min-confidence 75         # only high-conviction setups
```

```python
from trading_bot.strategies import build_strategy

strategy = build_strategy("momentum", min_confidence=70)
prepared = strategy.prepare(bars)                  # enrich indicators once
signal = strategy.generate_signal("AAPL", prepared)
if signal:
    print(signal.direction, signal.confidence, signal.risk_reward_ratio)
```

### One evaluation rule

`generate_signal` reads **the last bar of the frame it is given, and nothing
else**. That single rule is what makes backtest and live behaviour identical: the
backtester passes `bars.iloc[:i+1]` for each `i`, the live bot passes everything
up to now, and the strategy cannot tell which is which. It also makes lookahead
bias structurally impossible — a strategy has no way to reach a bar it was not
handed.

### Signals are proposals, not orders

A `Signal` carries entry, stop and target, so the risk manager can size it
without re-deriving anything. Strategies never size positions, check account
limits, or place orders. Construction validates that the stop and target sit on
the correct sides of the entry — a long whose stop is above its entry would make
position sizing divide by a negative risk, so it is rejected at the source.

### Confidence

Each strategy declares its entry conditions. Required conditions are vetoes;
optional ones contribute weight, and confidence is the share of total weight that
passed. **A signal at 62/100 means "the setup is valid and about 62% of the
supporting evidence is present" — not a 62% chance of profit.**

### Why didn't it trade?

The most important question a trading bot has to answer. Every evaluation records
which conditions blocked it:

```
No setups met the entry criteria on the latest bar.

Most common blockers (condition, times it blocked an entry):
   10x  momentum.trigger
    9x  breakout.consolidation
    9x  mean_reversion.band_stretched
    8x  breakout.volume
```

An idle bot with no explanation is indistinguishable from a broken one.

### Strategies have different risk shapes

Mean reversion ships with a **tighter stop and a lower reward:risk floor** (1.5
rather than 2.0) than momentum. This is deliberate. Reverting to the mean pays
roughly one standard deviation, so demanding a trend-following 2:1 would reject
essentially every valid setup. Mean reversion earns its expectancy from a high
win rate at modest reward — the mirror image of momentum. Comparing the two on
reward:risk alone is a category error; compare them on expectancy, which Phase 6's
backtester measures.

### Adding your own

Subclass `BaseStrategy`, implement `evaluate`, call `register_strategy`. The
scanner, backtester and dashboard all work through the registry, so nothing else
changes.

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
cover the live-trading locks, bar normalization, the lookahead guards, cache
coverage rules, P&L arithmetic, retry classification, CLI exit codes, and every
indicator's maths against independently derived reference values.

| Area | File | Tests |
|---|---|---|
| Dashboard charts and palette | `tests/test_dashboard.py` | 55 |
| Strategy contract and registry | `tests/test_strategies.py` | 52 |
| Per-strategy behaviour | `tests/test_strategy_signals.py` | 36 |
| Indicator maths and validation | `tests/test_indicators.py` | 91 |
| Swing points and levels | `tests/test_price_action.py` | 33 |
| Trend classification | `tests/test_trend_analysis.py` | 17 |
| Volume analysis | `tests/test_volume_analysis.py` | 30 |
| Phase 1 foundation and CLI | 9 further files | 168 |

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Project setup, Alpaca connection, market data | **Complete** |
| 2 | Technical indicator engine | **Complete** |
| 3 | Strategy engine (momentum, mean reversion, breakout) | **Complete** |
| 4 | Risk management and position sizing | Planned |
| 5 | Market scanner with confidence scoring | Planned |
| 6 | Backtesting engine | Planned |
| 7 | Alpaca paper trading execution | Planned |
| 8 | Position monitoring and automated exits | Planned |
| 9 | Streamlit dashboard | **Read-only version shipped**; trade controls with Phase 7 |
| 10 | Performance optimisation | Planned |

---

## Disclaimer

This software is for research and education. It is not financial advice. Trading
involves substantial risk of loss. Backtested performance does not predict future
results. You are responsible for any orders this software places on your behalf —
test thoroughly in paper mode, and read the code before enabling live trading.
