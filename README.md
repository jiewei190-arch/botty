# Market Hunter

A risk-first **swing-trading scanner** for US equities. It analyses every liquid
listed stock, ranks the setups worth acting on, and gives you an entry, a stop,
a target and a share count sized to your account. You place the orders yourself,
wherever you trade.

```bash
python main.py hunt
```

> **It places no orders and connects to no broker for trading.** Market data is
> read from a data provider; execution is yours. That separation is deliberate,
> not a missing feature — see [Where you trade](#where-you-trade-is-not-where-you-get-data).

> **Status: the scanner, the analysis behind it and the backtester are complete
> and tested** — 850 tests, no credentials needed to run them. See the
> [roadmap](#roadmap).

---

## Design principles

1. **Risk management is not optional.** Nothing is presented as an entry without
   passing validation. Position size is derived from stop distance, never
   guessed, and the constraint that bound it is always reported.
2. **Backtests must be honest.** Fees, slippage, candle-by-candle execution and a
   hard guard against lookahead bias — a backtest that lies is worse than none.
3. **A quiet scan must explain itself.** Every filter stage reports what it
   dropped and why. A scan returning four names out of six hundred is either
   working perfectly or badly broken, and only the stage counts tell them apart.
4. **Strategies are pluggable.** Every strategy implements one interface and is
   driven entirely by parameters, so ideas can be compared on equal footing.
5. **A score is a ranking, not a probability.** The scanner orders candidates
   against each other. It does not estimate a chance of profit, and says so
   wherever a score appears.

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
│   ├── risk/
│   │   ├── risk_manager.py          The gate: nine limits, explicit verdicts
│   │   ├── position_sizing.py       Size from stop distance, four caps
│   │   └── portfolio.py             The state the limits are judged against
│   ├── scanner/
│   │   ├── scanner.py               Filter, find, score, size, rank
│   │   └── scoring.py               Seven direction-aware factors
│   ├── backtesting/              Phase 6
│   ├── dashboard/
│   │   ├── app.py                   Streamlit app (read-only)
│   │   ├── charts.py                Plotly figure builders
│   │   ├── theme.py                 Validated palette, light and dark
│   │   └── data.py                  Cached data access for the UI
│   └── main.py                   CLI
├── tests/                        603 tests, no credentials required
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

### Market data

Scanning the market means reading the market, so `hunt` needs a data key. Get a
free one from the
[Alpaca dashboard](https://app.alpaca.markets/paper/dashboard/overview) and
paste it into `.env`:

```bash
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
```

**No funding, no minimum, and no order is ever placed through it.** The account
exists purely as a data feed; you trade wherever you already do. Alpaca is used
because its API is built for this — one request returns bars for a hundred
symbols, and the full tradable-symbol list comes down in one call — which is what
makes a whole-market sweep finish in minutes instead of hours.

Then set the balance you actually trade, because every share count is derived
from it:

```bash
RISK_ACCOUNT_EQUITY=15000
```

### Verify the installation

```bash
python main.py check
```

```
Health check results
------------------------------------------------------------------------
  [PASS] database       schema v1 at storage/trading_bot.db
  [PASS] credentials    ALPACA_API_KEY and ALPACA_SECRET_KEY present
  [PASS] market clock   CLOSED (opens 2026-09-05T13:30:00+00:00)
  [PASS] market data    5 daily bars for AAPL, latest 2026-09-04 close $319.97
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
| `python main.py hunt` | **Scan the whole market and rank the best entries** |
| `python main.py scan` | Rank a watchlist by trade confidence |
| `python main.py backtest` | Simulate a strategy over historical bars (Phase 6) |
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

## The market scanner

```bash
python main.py scan --demo --bars 405 --min-dollar-volume 0
```

```
#1  AAPL
Direction: LONG
Confidence: 78/100

Score breakdown:
  risk_reward     100  ██████████  1:4.00 reward to risk
  trend            81  ████████    STRONG_BULLISH at 85/100 (agreement 88)
  momentum        100  ██████████  MACD bullish, histogram increasing
  conviction       85  ████████    momentum fired at 85/100
  rsi_headroom    100  ██████████  RSI 58.0 — room to run
  volume           67  ███████     1.08x average, confirming the move
  structure        29  ███         0.9 ATR of room to resistance at 118.38

Suggested Entry : $117.31
Stop Loss       : $116.37  (0.80% away)
Take Profit     : $121.08
Risk/Reward     : 1:4.00
Position Size   : 6 shares (risking $5.66)
Risk Validation : PASSED — limited by maximum position size
```

### Why re-score at all?

**A strategy's own confidence is not comparable across strategies.** Momentum
reporting 80 means "80% of momentum's evidence is present"; mean reversion
reporting 80 means 80% of a completely different checklist. Sorting a mixed list
by those numbers ranks the checklists, not the opportunities.

So every candidate is scored again on a **common yardstick** measured from the
market rather than from the strategy that found it. Seven factors, each 0-100,
each answering the same question: *how much does this support a trade in this
direction?*

| Factor | Weight | Reads |
|---|---|---|
| Trend | 0.20 | Direction and strength, damped by component agreement |
| Risk / reward | 0.17 | What the setup pays for what it risks (saturates at 4:1) |
| Momentum | 0.15 | MACD state and whether it is building |
| Conviction | 0.15 | The strategy's own confidence |
| Volume | 0.13 | Participation, and whether it confirms the move |
| Structure | 0.10 | Room to the nearest level in the way |
| RSI headroom | 0.10 | How far from exhaustion the move is |

Every factor is direction-aware — a strongly bullish trend scores near 100 for a
long and near 0 for a short. Factors without data are dropped and the remaining
weights renormalised, so a short history reduces the evidence rather than
scoring a missing input as zero.

**The score ranks opportunities against each other. It is not a probability of
profit.**

### Order of operations

1. **Filter** — drop symbols too illiquid to trade before spending analysis on
   them. A perfect setup in something trading 4,000 shares a day is not an
   opportunity.
2. **Find** — every strategy evaluates every surviving symbol.
3. **Score** — on the common yardstick.
4. **Size** — the risk manager approves or rejects, working down the *ranked*
   order so scarce position slots reach the best candidates.
5. **Rank** — best first.

Risk runs *after* scoring, so a rejected opportunity still appears with its score
and the reason it was refused. A scanner that silently dropped everything the
limits blocked would leave you unable to tell a quiet market from a mis-set limit.

---

## Risk management

**The gate every trade passes through.** `RiskManager` is the only thing in the
system that turns a signal into a quantity, and the execution layer will take a
`RiskDecision` rather than a `Signal` — so "forgetting" to check risk is not an
available mistake.

```bash
python main.py signals --demo --symbols AAPL,AMZN --bars 405
```

```
  Risk Validation : PASSED
  Position Size   : 17 shares

SYMBOL  STRATEGY       DIR    CONF     ENTRY   R:R   QTY   RISK $  STATUS
------------------------------------------------------------------------------
AAPL    momentum       LONG     85    117.31 11.57    17     5.54  APPROVED
AMZN    momentum       LONG     85    101.94  4.18    19    13.63  APPROVED
```

### Sizing is arithmetic, not preference

```
risk budget    = equity × max_risk_per_trade_pct / 100
risk per share = |entry − stop|
quantity       = risk budget ÷ risk per share
```

A wide stop buys fewer shares and a tight stop buys more, so **every trade risks
the same amount** regardless of how volatile the instrument is. Sizing by a fixed
dollar amount instead makes each trade risk a different, unknown quantity — which
is how accounts die from a run of "small" trades.

Four caps apply and the smallest wins: risk budget, position size, portfolio
exposure, buying power. **The result reports which one bound**, because "why is
my position so small?" is otherwise unanswerable:

| $10,000 account, entry $210.50, stop $207.00 | Shares |
|---|---|
| 1% risk budget alone | 28 |
| With the 20% position cap | **9** ← binds |

That is not a bug — knowing which cap it was is the difference between tuning the
right number and the wrong one.

### The nine checks

| Check | Blocks when |
|---|---|
| `account_tradable` | Broker blocked the account, or the kill switch is pulled |
| `daily_loss` | Today's loss has reached the daily limit |
| `cooldown` | A run of losses is still cooling off |
| `open_positions` | Every position slot is used |
| `duplicate` | A position in this symbol is already open |
| `confidence` | The signal is below the confidence floor |
| `risk_reward` | The setup pays too little for what it risks |
| `position_size` | The limits leave less than one share |
| `exposure` | The trade would breach total exposure |

Every check is reported whether it passed or failed, so a rejection explains
itself and an approval shows how much headroom was left. A rejected signal still
reports its *would-be* size — exactly what you need to decide which limit to tune.

### Details that matter

- **Unrealised losses count toward the daily limit.** A limit that ignored open
  positions could be satisfied while the account bled, because nothing had been
  closed yet.
- **Batches accumulate.** `evaluate_many` folds each approval into the portfolio
  before judging the next signal, so six signals cannot fill five slots. Highest
  confidence is considered first, so scarce slots go to the strongest setups.
- **An unknown cooldown fails closed.** A losing streak with no recorded
  timestamp refuses to trade rather than assuming the cooldown expired.
- **Money is `Decimal`.** Prices convert through `str`, so `0.1` stays `0.1`.
- **The manager does no I/O.** It is a pure function of a signal and a
  `PortfolioState`, so every limit is testable without a network or a database,
  and a backtest exercises the same code the live bot does.

---

## Hunting the market

```bash
python main.py hunt                                  # scan everything liquid
python main.py hunt --top 5 --min-risk-reward 3      # only the best setups
python main.py hunt --equity 25000 --risk-per-trade 0.5
python main.py hunt --symbols AAPL,MSFT,NVDA         # just these names
python main.py hunt --csv setups.csv
```

Each result is a plan you could work from:

```
------------------------------------------------------------------------------
#1  AAPL  ·  LONG  ·  momentum  ·  score 94/100
------------------------------------------------------------------------------
  ✓ Pullback to EMA 9 resumed higher
  ✓ STRONG_BULLISH trend at 86/100
  ✓ RSI 61.7 leaves room to run
  ✓ Volume 1.29x average confirms the move

  Buy              38 shares near $77.80   ($2,957)
  Stop                       $76.35   (1.86% away)
  Target                     $83.60   (7.45% away)

  Risking $55.10 to make $220.39 (4.00:1)
  Sized by: maximum position size
```

The same thing lives on the dashboard's **Hunt** page, with the funnel, the
filters and a CSV download.

### The funnel

Analysing eleven thousand symbols the way you analyse ten is not a thing that
finishes. Four stages, cheapest first:

| Stage | Cuts | Cost |
|---|---|---|
| **Metadata** | exchange, tradability, instrument type | no price data; ~half |
| **Liquidity** | price and turnover floors | the network stage |
| **Strategies** | no setup on this bar | most symbols, most days |
| **Score + risk** | fails the reward floor or sizing | the final few |

The bars fetched for the liquidity screen are kept and reused, so a market-wide
scan downloads the market **once**, not twice.

### What gets filtered out, and why

- **OTC and pink sheets.** Most retail brokers cannot trade them, so ranking
  them produces opportunities you cannot act on.
- **Leveraged and inverse funds.** Their daily reset decays a multi-day hold —
  a 3x fund does not return 3x over a week. Excluded from a swing scan by
  default; `--include-leveraged` if you want them anyway.
- **SPACs, closed-end funds, warrants, rights and units.** Their price action
  reflects deal news and flows rather than the behaviour these strategies read.
- **Thin turnover.** Measured as price × volume, never volume alone: a million
  shares of a $0.40 stock is $400,000 of liquidity, and volume alone hides that.
  This is the single most effective filter for making results actionable.
- **Anything under ~200 bars of history.** A recent listing has no 200-day
  average and no established structure; indicators computed on it are arithmetic
  without meaning.

### Setups go stale

A swing setup that triggered four sessions ago has already made its move. It
still *evaluates* as valid — the conditions that fired are still true — but
entering now means paying for the part you missed while carrying the same stop.
Every candidate is dated, and stale ones are dropped rather than presented as
fresh. `--max-age` controls the window; the default is one bar.

### Share counts are per-trade

Each setup is sized as though it were the only trade you take, because that is
the question a ranked list actually raises: *if I take this one, how many
shares?* How many fit **together** is reported separately.

The alternative — sizing down the list cumulatively, so each entry assumes the
ones above it were already bought — answers a different question and reads as a
bug. On a $15,000 account it made the fourth-best idea report four shares, and
rejected 92 of 101 otherwise-tradable setups outright, purely because earlier
ranks had consumed the account.

---

## Where you trade is not where you get data

The scanner reads market data from one place and assumes you execute somewhere
else entirely. Nothing in it connects to a broker for trading.

That has one consequence worth stating plainly: **position sizing uses
`RISK_ACCOUNT_EQUITY` from your `.env`, not a broker balance.** If you scan with
a free data-only key and trade elsewhere, the data account holds nothing, and
sizing against it would produce share counts unrelated to the money genuinely at
risk. Set it to the balance you actually trade:

```bash
RISK_ACCOUNT_EQUITY=15000
RISK_MAX_RISK_PER_TRADE_PCT=1.0     # $150 at risk per trade on that balance
```

A data-only account needs no funding and places no orders. The market data is
the only thing the scanner wants from it.

---

## Backtesting

```bash
python main.py backtest --demo --symbols AAPL,MSFT,NVDA   # no API keys needed
python main.py backtest --symbols AAPL,MSFT --strategy momentum \
    --start 2025-01-02 --end 2025-06-30 --capital 25000
python main.py backtest --symbols AAPL --trades --csv trades.csv
```

Output from the first of those, which needs no credentials and is reproducible:

```
====================================================================
BACKTEST — AAPL, MSFT, NVDA · momentum
15Min bars · 550 bars simulated
====================================================================

Starting equity   : $10,000.00
Ending equity     : $10,149.96
Net profit        : $149.96  (+1.50%)

Trades            : 20  (7 win / 13 loss)
Win rate          : 35.0%
Profit factor     : 2.08
Expectancy        : +0.412 R per trade

Max drawdown      : 0.67%  ($68.40 over 118 bars)
Sharpe ratio      : 5.31
Time in market    : 9.6%
Slippage cost     : $33.99

!! Only 20 trade(s). Win rate, profit factor and Sharpe are dominated by luck
   below about 30; treat them as anecdotes rather than measurements.
```

Those figures describe a **random walk**, not any real instrument — `--demo`
generates synthetic bars. The warning at the bottom is the engine's, not a note
added here for the README.

The same run is available on the dashboard's **Backtest** page, with an equity
curve, a drawdown panel, every trade in R, and entry/exit markers on price.

### The four assumptions that decide whether a backtest is honest

Most backtests flatter themselves in the same few places, so each is made
explicit here rather than buried in the engine.

**1. You cannot trade at a price you have only just observed.** A signal
generated from bar `i`'s close fills at bar `i + 1`'s **open**. Filling at the
signal bar's own close assumes you saw the close and traded at it, which is not
a thing that can happen; on a trending instrument it quietly awards every trade
a free bar of profit.

**2. Gaps blow through stops.** A stop is a trigger, not a guaranteed price.
When a bar opens beyond the stop, the fill is the *open*. A backtester that
always fills stops exactly at the stop understates the tail of the loss
distribution — the part that actually matters.

**3. Costs are paid on both sides.** Commission, plus slippage that always
moves *against* the trade. Slippage that sometimes helped would be modelling a
friendlier market than the one you trade in.

**4. An ambiguous bar resolves against you.** When a bar touches both the stop
and the target, the stop wins. The intrabar path is unknowable, and assuming
the good outcome is how a backtest manufactures an edge.

The one place the model is *conservative* rather than pessimistic: a target
fills exactly at its limit, never better, even when the bar traded well through
it — a resting limit order would not have captured the overshoot.

### The bar loop

Each bar is processed in a fixed order, and nothing generated on a bar can act
on that same bar:

1. Exits queued on the previous bar fill at this bar's open.
2. Entries queued on the previous bar fill at this bar's open.
3. Stops and targets the bar traded through are executed.
4. Equity is marked to market.
5. Strategies read this bar's close and queue intentions for the *next* bar.

Discretionary exits go through the same queue as entries. Deciding from a bar's
close and filling at that bar's open would be selling at a price that came
before the information behind the decision — a subtle lookahead that produces
trades opening and closing on the same bar at the same price.

### Metrics, and what they are worth

Return, win rate, profit factor, expectancy in R, drawdown depth *and*
duration, Sharpe, Sortino, exposure, cost totals and an exit breakdown.

Two details worth knowing:

- **Ratios annualise by the bar size**, not by a hardcoded 252. A Sharpe
  computed from 15-minute bars and scaled by `sqrt(252)` would be overstated
  about fivefold.
- **Runs under 30 trades carry a warning.** Win rate, profit factor and Sharpe
  are dominated by luck below that, so the result is labelled an anecdote
  rather than presented as a measurement.

Warm-up bars are excluded from the results. No position can exist during
warm-up, so including them would pad the bar count, understate exposure, and
drag Sharpe toward zero.

### Every backtest is a measurement, not a forecast

The engine tells you what a strategy *would have done* on that data under those
assumptions. It cannot tell you what it will do next. Vary the date range,
symbols and costs before believing any single number, and treat a strategy that
only works on one window as untested rather than proven.

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

**850+ tests, `ruff check` clean.** The suite runs against synthetic bars and an
in-memory database — **no API credentials or network access required**, so it is
safe to run in CI. Tests cover the universe filters, the scan funnel, per-trade
sizing, bar normalization, the lookahead guards, cache coverage rules, P&L
arithmetic, retry classification, CLI exit codes, and every indicator's maths
against independently derived reference values.

The synthetic bars are calibrated rather than arbitrary: a generated daily bar's
true range averages 2.54% of price, against 2.48% measured on real AAPL
sessions. That matters more than it sounds — an earlier fixed volatility was
right for 15-minute bars and five times too calm for daily ones, which made
ATR-derived stops come out near 1% and read as a strategy bug that did not
exist.

| Area | File | Tests |
|---|---|---|
| Risk gate and portfolio state | `tests/test_risk_manager.py` | 40 |
| Scanner scoring and ranking | `tests/test_scanner.py` | 34 |
| Position sizing arithmetic | `tests/test_position_sizing.py` | 32 |
| Dashboard charts, palette, pages | `tests/test_dashboard.py` | 92 |
| Strategy contract and registry | `tests/test_strategies.py` | 57 |
| Per-strategy behaviour | `tests/test_strategy_signals.py` | 36 |
| Backtest engine and lookahead guards | `tests/test_backtest_engine.py` | 36 |
| Performance metrics | `tests/test_backtest_metrics.py` | 28 |
| Fill and cost model | `tests/test_backtest_execution.py` | 24 |
| Indicator maths and validation | `tests/test_indicators.py` | 100 |
| Swing points and levels | `tests/test_price_action.py` | 49 |
| Trend classification | `tests/test_trend_analysis.py` | 17 |
| Volume analysis | `tests/test_volume_analysis.py` | 30 |
| Universe filters and discovery | `tests/test_universe.py` | 45 |
| Market-wide sweep | `tests/test_market_scan.py` | 26 |
| Foundation, data access and CLI | 10 further files | 196 |

---

## Roadmap

The project began as a paper-trading bot and was redirected into a
decision-support scanner: it finds and ranks entries, and a person executes
them. Order placement and automated exits were dropped from the plan rather
than deferred — they are not features this tool is missing.

| Scope | Status |
|---|---|
| Project setup, market data access | **Complete** |
| Technical indicator engine | **Complete** |
| Strategy engine (momentum, mean reversion, breakout) | **Complete** |
| Risk management and position sizing | **Complete** |
| Confidence scoring and watchlist scanner | **Complete** |
| Backtesting engine | **Complete** |
| Universe discovery and market-wide hunt | **Complete** |
| Dashboard (hunt, charts, backtests, settings) | **Complete** |
| Alerts when a setup appears (email/push, scheduled scans) | Planned |
| Tracking setups you took, to measure the scanner against reality | Planned |
| Broker order placement | **Not planned** — you execute |

---

## Disclaimer

This software is for research and education. It is not financial advice. Trading
involves substantial risk of loss. Backtested performance does not predict future
results. You are responsible for any orders this software places on your behalf —
test thoroughly in paper mode, and read the code before enabling live trading.
