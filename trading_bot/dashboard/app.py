"""Streamlit dashboard.

Read-only. It shows what the analysis and strategy layers see — charts,
indicators, and proposed setups — and it cannot place an order, because no
order-placing code exists in the system yet. Trade controls arrive with the
execution layer in Phase 7, defaulting to paper trading with manual approval.

Run it with::

    python main.py dashboard

or directly::

    streamlit run trading_bot/dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - streamlit runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from trading_bot import __version__
from trading_bot.backtesting import CostModel
from trading_bot.backtesting.runner import BacktestRequest
from trading_bot.dashboard import data as dashboard_data
from trading_bot.dashboard.charts import (
    confidence_bar,
    equity_chart,
    equity_placeholder,
    price_chart,
    r_multiple_chart,
    trade_chart,
)
from trading_bot.dashboard.theme import app_css, direction_marker, get_palette
from trading_bot.indicators import (
    IndicatorConfig,
    analyze_trend,
    analyze_volume,
    atr_column,
    detect_bollinger_condition,
    detect_macd_signal,
    detect_rsi_condition,
    find_support_resistance,
    rsi_column,
)
from trading_bot.strategies import available_strategies
from trading_bot.utils.timeframes import SUPPORTED_TIMEFRAMES

PAGES = (
    "Hunt",
    "Overview",
    "Market Scanner",
    "Chart",
    "Backtest",
    "Strategy Settings",
)


def main() -> None:
    st.set_page_config(
        page_title="Trading Bot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    settings = dashboard_data.get_settings_cached()
    controls = _sidebar(settings)
    palette = get_palette(controls["mode"])
    st.markdown(app_css(palette), unsafe_allow_html=True)

    if controls["demo"]:
        st.markdown(
            '<div class="demo-banner"><b>Demo mode</b> — every price on this page is '
            "generated sample data, not the market. Add Alpaca credentials and turn "
            "demo mode off for real analysis.</div>",
            unsafe_allow_html=True,
        )

    page = controls["page"]
    if page == "Hunt":
        _hunt(settings, controls, palette)
    elif page == "Overview":
        _overview(settings, controls, palette)
    elif page == "Market Scanner":
        _scanner(settings, controls, palette)
    elif page == "Chart":
        _chart(settings, controls, palette)
    elif page == "Backtest":
        _backtest(settings, controls, palette)
    else:
        _strategy_settings(controls, palette)

    _footer(palette)


# -- sidebar -----------------------------------------------------------------


def _sidebar(settings) -> dict:
    with st.sidebar:
        st.markdown("### 📈 Trading Bot")
        st.caption(f"v{__version__} · mode **{settings.trading_mode.value.upper()}**")

        page = st.radio("Page", PAGES, label_visibility="collapsed")
        st.divider()

        has_keys = settings.alpaca.has_credentials
        # Never defaults on. Silently substituting generated prices when a key
        # is missing means the app looks like it is working while showing
        # numbers that describe nothing — the one failure mode a person cannot
        # catch by looking. Missing credentials are reported as the error they
        # are, and demo data stays something you switch on deliberately.
        demo = st.toggle(
            "Demo data",
            value=False,
            help="Generated sample data for trying the interface without a key. "
            "Never enabled automatically, and every page says so while it is on.",
        )
        if not has_keys and not demo:
            st.error(
                "**No market-data credentials.** Add `ALPACA_API_KEY` and "
                "`ALPACA_SECRET_KEY` to your `.env` and restart. A free "
                "data-only account is enough — no funding, and no order is ever "
                "placed through it.",
                icon="🔑",
            )

        st.caption(
            f"{_active_theme().title()} theme — set `theme.base` in "
            "`.streamlit/config.toml` to switch."
        )
        st.divider()

        watchlist = settings.data.watchlist
        symbols = st.multiselect("Watchlist", watchlist, default=list(watchlist))
        extra = st.text_input("Add symbols", placeholder="e.g. COIN, PLTR")
        if extra:
            symbols = symbols + [
                item.strip().upper() for item in extra.split(",") if item.strip()
            ]

        timeframe = st.selectbox(
            "Timeframe",
            SUPPORTED_TIMEFRAMES,
            index=SUPPORTED_TIMEFRAMES.index(settings.data.timeframe)
            if settings.data.timeframe in SUPPORTED_TIMEFRAMES
            else 2,
        )
        bars = st.slider("Bars of history", 250, 1000, 400, step=50)
        st.divider()

        strategies = st.multiselect(
            "Strategies", available_strategies(), default=available_strategies()
        )
        min_confidence = st.slider("Minimum confidence", 0, 100, 55, step=5)
        allow_short = st.checkbox("Allow short signals", value=False)

        min_dollar_volume = st.number_input(
            "Minimum average turnover ($)",
            min_value=0.0,
            value=0.0,
            step=250_000.0,
            help="Skip symbols that trade less than this per bar on average.",
        )

        st.divider()
        st.markdown("**Risk sizing**")
        equity = st.number_input(
            "Account equity ($)",
            min_value=100.0,
            value=10_000.0,
            step=1_000.0,
            help="Used when no broker account is connected. A live account overrides this.",
        )

        if st.button("Refresh data", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    return {
        "page": page,
        "demo": demo,
        "mode": _active_theme(),
        "symbols": symbols or list(settings.data.watchlist),
        "timeframe": timeframe,
        "bars": bars,
        "strategies": strategies or available_strategies(),
        "min_confidence": float(min_confidence),
        "allow_short": allow_short,
        "equity": equity,
        "min_dollar_volume": min_dollar_volume,
    }


def _active_theme() -> str:
    """The theme Streamlit is configured to render in.

    Read from ``theme.base`` in ``.streamlit/config.toml`` — the same value
    Streamlit styles its own widgets from — so the chart palette and the app
    chrome always agree. Deriving it from the browser's preference instead was
    not stable across reruns and produced a light chart inside a dark app.
    """
    try:
        base = st.get_option("theme.base")
    except Exception:  # noqa: BLE001 - no config available
        return "dark"
    return "light" if str(base).lower() == "light" else "dark"


def _indicator_config(controls: dict) -> IndicatorConfig:
    overrides = controls.get("indicator_overrides") or {}
    return IndicatorConfig(**overrides) if overrides else IndicatorConfig()


def _metric(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>'
    )


# -- pages -------------------------------------------------------------------


def _hunt(settings, controls: dict, palette) -> None:
    """Scan the whole market and show the setups worth acting on.

    The centrepiece page. It reads live market data, ranks every liquid US
    equity, and prints entry, stop, target and a share count for each survivor.
    It places nothing — the orders are yours to work, wherever you trade.
    """
    st.subheader("Hunt")
    st.caption(
        "Scans every liquid US equity for swing setups and ranks what it finds. "
        "Each result is a proposal with the prices to work it at — nothing is "
        "ordered, and the score ranks candidates against each other rather than "
        "estimating a chance of profit."
    )

    if not settings.alpaca.has_credentials:
        st.error(
            "**Market data credentials are missing.** Add `ALPACA_API_KEY` and "
            "`ALPACA_SECRET_KEY` to your `.env` and restart. A free data-only "
            "account is enough — no funding, and no order is ever placed through "
            "it. You keep trading wherever you already do."
        )
        return

    with st.form("hunt"):
        first, second, third = st.columns(3)
        strategy_names = first.multiselect(
            "Strategies", available_strategies(), default=list(available_strategies())
        )
        equity = second.number_input(
            "Account equity ($)",
            min_value=100.0,
            # Seeded from the sidebar so the two controls never show different
            # balances for the same account.
            value=float(controls.get("equity") or settings.risk.account_equity),
            step=500.0,
            help="The balance you actually trade. Share counts are sized from this.",
        )
        top_n = third.number_input("Show top", min_value=1, max_value=50, value=10, step=1)

        fourth, fifth, sixth = st.columns(3)
        min_turnover = fourth.number_input(
            "Min turnover ($/day)",
            min_value=0.0,
            value=10_000_000.0,
            step=1_000_000.0,
            help="Price x volume. The single most effective filter for making "
            "results actionable — a perfect setup you cannot get filled in is not "
            "an opportunity.",
        )
        min_rr = fifth.number_input(
            "Min reward:risk", min_value=1.0, max_value=10.0, value=2.0, step=0.25
        )
        max_age = sixth.number_input(
            "Max signal age (bars)",
            min_value=1, max_value=10, value=1, step=1,
            help="A swing setup from several sessions ago has already made its "
            "move; entering now pays for the part you missed.",
        )

        seventh, eighth, ninth = st.columns(3)
        min_price = seventh.number_input("Min price ($)", min_value=0.0, value=5.0, step=1.0)
        max_price = eighth.number_input("Max price ($)", min_value=10.0, value=1000.0, step=50.0)
        risk_pct = ninth.number_input(
            "Risk per trade (%)",
            min_value=0.05, max_value=10.0,
            value=float(settings.risk.max_risk_per_trade_pct), step=0.25,
        )

        include_leveraged = st.checkbox(
            "Include leveraged and inverse ETFs",
            value=False,
            help="Their daily reset decays a multi-day hold, so a 3x fund does "
            "not return 3x over a week.",
        )
        submitted = st.form_submit_button("Run hunt", type="primary")

    if submitted:
        if not strategy_names:
            st.error("Choose at least one strategy.")
            return
        with st.spinner("Scanning the market — this takes a few minutes…"):
            try:
                sweep = dashboard_data.run_hunt(
                    settings,
                    strategies=tuple(strategy_names),
                    equity=float(equity),
                    risk_pct=float(risk_pct),
                    top_n=int(top_n),
                    min_turnover=float(min_turnover),
                    min_price=float(min_price),
                    max_price=float(max_price),
                    min_risk_reward=float(min_rr),
                    max_age=int(max_age),
                    include_leveraged=bool(include_leveraged),
                )
            except Exception as error:  # noqa: BLE001 - surfaced in the UI
                st.error(f"Hunt failed: {error}")
                return
        st.session_state["hunt_sweep"] = sweep
        st.session_state["hunt_equity"] = float(equity)

    sweep = st.session_state.get("hunt_sweep")
    if sweep is None:
        st.info(
            "Set your filters and run a hunt. The first run downloads the market's "
            "daily bars and takes a few minutes; later runs reuse the cache."
        )
        return

    _hunt_results(sweep, st.session_state.get("hunt_equity", 0.0), palette)


def _hunt_results(sweep, equity: float, palette) -> None:
    """Render a completed sweep."""
    columns = st.columns(4)
    columns[0].markdown(
        _metric("Scanned", f"{sweep.universe_size:,}", "liquid symbols"),
        unsafe_allow_html=True,
    )
    columns[1].markdown(
        _metric("Setups found", f"{len(sweep.opportunities)}", "after every filter"),
        unsafe_allow_html=True,
    )
    columns[2].markdown(
        _metric(
            "Fit at once",
            f"{sweep.concurrent_capacity}",
            f"on ${equity:,.0f}" if equity else "given your limits",
        ),
        unsafe_allow_html=True,
    )
    columns[3].markdown(
        _metric("Scan time", f"{sweep.elapsed_seconds:,.0f}s", "whole market"),
        unsafe_allow_html=True,
    )

    if sweep.halt_reason:
        st.error(f"Trading halted: {sweep.halt_reason}")

    with st.expander("Where the symbols went"):
        for line in sweep.summary_lines():
            st.text(line)

    if not sweep.opportunities:
        st.info(
            "No setups met the criteria. That is a normal outcome — most days, "
            "most stocks are not at an entry. Loosening the reward:risk floor or "
            "raising the maximum signal age will surface more, and worse, entries."
        )
        if sweep.blockers:
            st.markdown("**Most common blockers**")
            st.dataframe(
                pd.DataFrame(
                    sorted(sweep.blockers.items(), key=lambda item: -item[1])[:10],
                    columns=["Blocked by", "Symbols"],
                ),
                width="stretch",
                hide_index=True,
            )
        return

    st.markdown(
        f"**Each share count assumes this is the only trade you take.** Your "
        f"account supports about **{sweep.concurrent_capacity}** of these at "
        "once; taking more means sizing each one smaller."
    )

    for opportunity in sweep.opportunities:
        _entry_plan_card(opportunity, palette)

    st.divider()
    frame = sweep.as_frame()
    with st.expander("All setups as a table"):
        st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button(
        "Download setups as CSV",
        frame.to_csv(index=False),
        file_name="hunt_setups.csv",
        mime="text/csv",
    )
    st.caption(
        "Ranked against each other, not scored for probability of profit. "
        "Nothing on this page has been placed as an order."
    )


def _entry_plan_card(opportunity, palette) -> None:
    """One setup, as a plan you could work from."""
    signal = opportunity.signal
    decision = opportunity.decision
    shares = int(decision.shares) if decision else 0
    is_long = signal.direction.value == "LONG"
    stop_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
    target_pct = abs(signal.take_profit - signal.entry_price) / signal.entry_price * 100
    arrow = direction_marker(signal.direction.value)

    with st.container(border=True):
        header, badge = st.columns([4, 1])
        # direction_marker already pairs the arrow with the word, so the
        # direction is never carried by the glyph alone.
        header.markdown(f"### #{opportunity.rank} · {signal.symbol} {arrow}")
        badge.markdown(
            confidence_bar(opportunity.confidence, palette), unsafe_allow_html=True
        )
        st.caption(f"{signal.strategy} · score {opportunity.confidence:.0f}/100")

        plan = st.columns(4)
        plan[0].markdown(
            _metric(
                "Buy" if is_long else "Short",
                f"{shares:,}",
                f"shares ≈ ${shares * signal.entry_price:,.0f}",
            ),
            unsafe_allow_html=True,
        )
        plan[1].markdown(
            _metric("Entry near", f"${signal.entry_price:,.2f}", "limit order"),
            unsafe_allow_html=True,
        )
        plan[2].markdown(
            _metric("Stop", f"${signal.stop_loss:,.2f}", f"{stop_pct:.2f}% away"),
            unsafe_allow_html=True,
        )
        plan[3].markdown(
            _metric("Target", f"${signal.take_profit:,.2f}", f"{target_pct:.2f}% away"),
            unsafe_allow_html=True,
        )

        if decision is not None and decision.approved:
            risk = float(decision.risk_amount)
            reward = risk * signal.risk_reward_ratio
            st.markdown(
                f"Risking **{_dollars(risk)}** to make **{_dollars(reward)}** "
                f"({signal.risk_reward_ratio:.2f}:1) · sized by "
                f"{decision.sizing.binding_constraint.description}"
            )
        elif decision is not None:
            st.warning(f"Not sized — {decision.rejection_reason}")

        if signal.reasons:
            with st.expander("Why this fired"):
                for reason in signal.reasons:
                    st.markdown(f"- {reason}")


def _overview(settings, controls: dict, palette) -> None:
    st.subheader("Overview")

    account, account_error = dashboard_data.account_snapshot(settings)
    summary = dashboard_data.database_summary(settings)

    columns = st.columns(4)
    if account is not None:
        equity = f"${float(account.equity):,.2f}"
        daily = f"{account.daily_pnl_pct:+.2f}% today"
        kind = "PAPER" if account.is_paper else "LIVE"
        columns[0].markdown(_metric("Account equity", equity, f"{kind} · {daily}"),
                            unsafe_allow_html=True)
        columns[1].markdown(
            _metric("Buying power", f"${float(account.buying_power):,.2f}",
                    f"status {account.status}"),
            unsafe_allow_html=True,
        )
    else:
        columns[0].markdown(
            _metric("Account equity", "—", account_error or "unavailable"),
            unsafe_allow_html=True,
        )
        columns[1].markdown(
            _metric("Buying power", "—", "connect Alpaca to populate"),
            unsafe_allow_html=True,
        )

    columns[2].markdown(
        _metric("Closed trades", str(summary.get("total_trades", 0)),
                f"win rate {summary.get('win_rate', 0):.0f}%"),
        unsafe_allow_html=True,
    )
    columns[3].markdown(
        _metric("Open positions", str(summary.get("open_positions", 0)),
                "position tracking lands in Phase 8"),
        unsafe_allow_html=True,
    )

    st.markdown("")
    left, right = st.columns([3, 2])

    with left:
        st.markdown("##### Equity curve")
        st.plotly_chart(equity_placeholder(palette), width="stretch",
                        config={"displayModeBar": False})

    with right:
        st.markdown("##### What works today")
        st.markdown(
            """
| Capability | Status |
| --- | --- |
| Market data, caching, database | Ready |
| Indicators and trend analysis | Ready |
| Strategy signals | Ready |
| Position sizing and risk limits | Phase 4 |
| Ranked watchlist scoring | Phase 5 |
| Backtesting and equity curves | Phase 6 |
| **Placing trades** | **Phase 7** |
"""
        )
        st.caption(
            "This dashboard is read-only. There is no order-placing code in the "
            "system yet, by design."
        )

    st.markdown("##### Watchlist snapshot")
    _watchlist_table(settings, controls, palette)


def _watchlist_table(settings, controls: dict, palette) -> None:
    """A row per symbol: price, trend, momentum, volume."""
    indicators = _indicator_config(controls)
    rows = []
    problems = {}

    progress = st.progress(0.0, text="Loading watchlist…")
    symbols = controls["symbols"]
    for index, symbol in enumerate(symbols, start=1):
        loaded = dashboard_data.load_symbol(
            symbol, controls["timeframe"], controls["bars"],
            controls["demo"], settings, indicators,
        )
        progress.progress(index / max(len(symbols), 1), text=f"Loading {symbol}…")
        if not loaded.ok:
            problems[symbol] = loaded.error or "no data"
            continue

        frame = loaded.frame
        trend = analyze_trend(frame, indicators)
        volume = analyze_volume(frame, indicators)
        close = float(frame["close"].iloc[-1])
        change = (close / float(frame["close"].iloc[-2]) - 1) * 100 if len(frame) > 1 else 0.0
        rsi_value = frame[rsi_column(indicators.rsi_period)].iloc[-1]
        atr_value = frame[atr_column(indicators.atr_period)].iloc[-1]

        rows.append(
            {
                "Symbol": symbol,
                "Price": close,
                "Change %": change,
                "Trend": trend.direction.value,
                "Strength": trend.strength,
                "RSI": None if pd.isna(rsi_value) else float(rsi_value),
                "Momentum": detect_macd_signal(frame, indicators),
                "Volume": volume.condition.value,
                "Rel vol": volume.relative_volume,
                "ATR %": None if pd.isna(atr_value) else float(atr_value) / close * 100,
            }
        )
    progress.empty()

    if rows:
        table = pd.DataFrame(rows).sort_values("Strength", ascending=False)
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Change %": st.column_config.NumberColumn(format="%+.2f%%"),
                "Strength": st.column_config.ProgressColumn(
                    "Trend strength", min_value=0, max_value=100, format="%d"
                ),
                "RSI": st.column_config.NumberColumn(format="%.1f"),
                "Rel vol": st.column_config.NumberColumn(format="%.2fx"),
                "ATR %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
        st.caption(
            "Trend strength runs 0–100 with 50 neutral: it is a directional score, "
            "not a probability."
        )
    if problems:
        st.warning("Not loaded: " + ", ".join(f"{k} ({v})" for k, v in problems.items()))


def _scanner(settings, controls: dict, palette) -> None:
    st.subheader("Market Scanner")
    st.caption(
        f"Ranking {len(controls['symbols'])} symbol(s) with "
        f"{len(controls['strategies'])} strateg"
        f"{'ies' if len(controls['strategies']) != 1 else 'y'} on "
        f"{controls['timeframe']} bars."
    )

    overrides: dict = {"min_confidence": controls["min_confidence"]}
    if controls["allow_short"]:
        overrides["allow_short"] = True

    with st.spinner("Scanning…"):
        result, portfolio = dashboard_data.run_ranked_scan(
            controls["symbols"],
            controls["strategies"],
            controls["timeframe"],
            controls["bars"],
            controls["demo"],
            settings,
            _indicator_config(controls),
            equity=controls.get("equity"),
            overrides=overrides,
            min_dollar_volume=controls.get("min_dollar_volume", 0.0),
        )

    if result.halt_reason:
        st.error(f"**Trading halted** — {result.halt_reason}", icon="🛑")

    st.caption(
        f"{result.summary()} · sizing against ${float(portfolio.equity):,.2f} equity"
    )

    if not result.opportunities:
        st.info(
            "No opportunities scored above the threshold. That is the normal "
            "outcome — every strategy requires several conditions to align at once."
        )
        if result.blockers:
            st.markdown("##### Why not")
            st.caption(
                "An idle bot with no explanation is indistinguishable from a broken "
                "one. These are the conditions that blocked an entry."
            )
            st.dataframe(
                pd.DataFrame(
                    sorted(result.blockers.items(), key=lambda item: -item[1]),
                    columns=["Condition", "Times blocked"],
                ),
                width="stretch",
                hide_index=True,
            )
    else:
        st.markdown("##### Ranked opportunities")
        table = pd.DataFrame(
            [
                {
                    "#": item.rank,
                    "Symbol": item.symbol,
                    "Dir": item.signal.direction.value,
                    "Strategy": item.signal.strategy,
                    "Score": item.confidence,
                    "Entry": item.signal.entry_price,
                    "R:R": item.signal.risk_reward_ratio,
                    "Qty": item.quantity,
                    "Status": "Tradable" if item.tradable else (
                        item.rejection_reason or "not sized"
                    ),
                }
                for item in result.opportunities
            ]
        )
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Trade confidence", min_value=0, max_value=100, format="%.0f"
                ),
                "Entry": st.column_config.NumberColumn(format="$%.2f"),
                "R:R": st.column_config.NumberColumn(format="1:%.2f"),
            },
        )
        st.caption(
            "The score ranks these against each other on one common yardstick, so "
            "candidates from different strategies are comparable. It is **not** a "
            "probability of profit."
        )
        for opportunity in result.opportunities:
            _opportunity_card(opportunity, palette)

    if result.skipped:
        with st.expander(f"Filtered out before analysis ({len(result.skipped)})"):
            for symbol, reason in result.skipped.items():
                st.markdown(f"- **{symbol}** — {reason}")
    if result.failures:
        st.warning(
            "Not scanned: "
            + ", ".join(f"{key} ({reason})" for key, reason in result.failures.items())
        )

    st.divider()
    st.caption(
        "These are sized proposals, not orders. Nothing here can place a trade — "
        "order placement arrives in Phase 7."
    )


def _opportunity_card(opportunity, palette) -> None:
    """One ranked opportunity, with its score breakdown."""
    signal = opportunity.signal
    with st.container(border=True):
        header, meter = st.columns([3, 2])
        header.markdown(
            f"### #{opportunity.rank} &nbsp; {signal.symbol} &nbsp; "
            f'<span class="pill">{direction_marker(signal.direction.value)}</span> '
            f'<span class="pill">{signal.strategy}</span>',
            unsafe_allow_html=True,
        )
        meter.markdown("**Trade confidence**")
        meter.markdown(
            confidence_bar(opportunity.confidence, palette), unsafe_allow_html=True
        )

        columns = st.columns(4)
        columns[0].markdown(_metric("Entry", f"${signal.entry_price:,.2f}"),
                            unsafe_allow_html=True)
        columns[1].markdown(
            _metric("Stop", f"${signal.stop_loss:,.2f}",
                    f"{signal.stop_distance_pct:.2f}% away"),
            unsafe_allow_html=True,
        )
        columns[2].markdown(_metric("Target", f"${signal.take_profit:,.2f}"),
                            unsafe_allow_html=True)
        columns[3].markdown(
            _metric("Reward : risk", f"1:{signal.risk_reward_ratio:.2f}"),
            unsafe_allow_html=True,
        )

        decision = opportunity.decision
        if decision is not None and decision.approved:
            st.markdown(
                f"**Risk validation: PASSED** — {decision.shares} share(s), risking "
                f"${float(decision.risk_amount):,.2f} "
                f"(${float(decision.position_value):,.2f} position). "
                f"Limited by {decision.sizing.binding_constraint.description}."
            )
        elif decision is not None:
            st.warning(
                f"**Risk validation: REJECTED** — {decision.rejection_reason}",
                icon="⚠️",
            )

        weakest = opportunity.weakest_factor()
        if weakest is not None and weakest.score < 40:
            st.caption(f"⚠️ Weakest factor — **{weakest.name}**: {weakest.detail}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Score breakdown**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Factor": f.name, "Score": f.score, "Detail": f.detail}
                        for f in sorted(opportunity.factors, key=lambda x: -x.contribution)
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=100, format="%.0f"
                    )
                },
            )
        with right:
            if opportunity.reasons:
                st.markdown("**Why the setup**")
                for reason in opportunity.reasons:
                    st.markdown(f"- {reason}")
            if decision is not None:
                with st.expander("Risk checks"):
                    for check in decision.checks:
                        mark = "✅" if check.passed else "❌"
                        st.markdown(f"{mark} **{check.name}** — {check.detail}")


def _signal_card(signal, palette, decision=None) -> None:
    """One setup, with its reasons, levels and risk verdict."""
    with st.container(border=True):
        header, meter = st.columns([3, 2])
        header.markdown(
            f"### {signal.symbol} &nbsp; "
            f'<span class="pill">{direction_marker(signal.direction.value)}</span> '
            f'<span class="pill">{signal.strategy}</span>',
            unsafe_allow_html=True,
        )
        meter.markdown("**Confidence**", unsafe_allow_html=True)
        meter.markdown(confidence_bar(signal.confidence, palette), unsafe_allow_html=True)

        columns = st.columns(4)
        columns[0].markdown(_metric("Entry", f"${signal.entry_price:,.2f}"),
                            unsafe_allow_html=True)
        columns[1].markdown(
            _metric("Stop", f"${signal.stop_loss:,.2f}",
                    f"{signal.stop_distance_pct:.2f}% away"),
            unsafe_allow_html=True,
        )
        columns[2].markdown(_metric("Target", f"${signal.take_profit:,.2f}"),
                            unsafe_allow_html=True)
        columns[3].markdown(
            _metric("Reward : risk", f"1:{signal.risk_reward_ratio:.2f}"),
            unsafe_allow_html=True,
        )

        if decision is not None:
            if decision.approved:
                st.markdown(
                    f"**Risk validation: PASSED** — "
                    f"{decision.shares} share(s), risking "
                    f"${float(decision.risk_amount):,.2f} "
                    f"(${float(decision.position_value):,.2f} position). "
                    f"Limited by {decision.sizing.binding_constraint.description}."
                )
            else:
                st.warning(
                    f"**Risk validation: REJECTED** — {decision.rejection_reason}",
                    icon="⚠️",
                )

        columns = st.columns(2)
        if signal.reasons:
            with columns[0]:
                st.markdown("**Why the setup**")
                for reason in signal.reasons:
                    st.markdown(f"- {reason}")
        if decision is not None:
            with columns[1], st.expander("Risk checks"):
                for check in decision.checks:
                    mark = "✅" if check.passed else "❌"
                    st.markdown(f"{mark} **{check.name}** — {check.detail}")


def _chart(settings, controls: dict, palette) -> None:
    st.subheader("Chart")
    symbols = controls["symbols"]
    chosen = st.selectbox("Symbol", symbols, index=0)

    left, right, _ = st.columns([1, 1, 3])
    show_bands = left.checkbox("Bollinger Bands", value=True)
    show_levels = right.checkbox("Support / resistance", value=True)

    indicators = _indicator_config(controls)
    loaded = dashboard_data.load_symbol(
        chosen, controls["timeframe"], controls["bars"],
        controls["demo"], settings, indicators,
    )
    if not loaded.ok:
        st.error(f"Could not load {chosen}: {loaded.error}")
        return

    frame = loaded.frame
    trend = analyze_trend(frame, indicators)
    volume = analyze_volume(frame, indicators)
    levels = find_support_resistance(frame, indicators)
    close = float(frame["close"].iloc[-1])

    columns = st.columns(5)
    columns[0].markdown(_metric("Price", f"${close:,.2f}"), unsafe_allow_html=True)
    columns[1].markdown(
        _metric("Trend", trend.direction.value.replace("_", " ").title(),
                f"{trend.strength}/100 · confidence {trend.confidence}"),
        unsafe_allow_html=True,
    )
    rsi_value = frame[rsi_column(indicators.rsi_period)].iloc[-1]
    columns[2].markdown(
        _metric("RSI", "—" if pd.isna(rsi_value) else f"{float(rsi_value):.1f}",
                detect_rsi_condition(frame, indicators)),
        unsafe_allow_html=True,
    )
    columns[3].markdown(
        _metric("Volatility", detect_bollinger_condition(frame, indicators),
                "Bollinger position"),
        unsafe_allow_html=True,
    )
    columns[4].markdown(
        _metric("Volume", volume.condition.value,
                "—" if volume.relative_volume is None
                else f"{volume.relative_volume:.2f}x average"),
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        price_chart(
            frame, chosen, palette, levels=levels,
            rsi_period=indicators.rsi_period,
            volume_period=indicators.volume_sma_period,
            show_bollinger=show_bands, show_levels=show_levels,
        ),
        width="stretch",
        config={"scrollZoom": True, "displaylogo": False},
    )
    st.caption(
        "Up candles are hollow and down candles solid, so direction reads without "
        "relying on colour. Forming bars are dropped before charting."
    )

    with st.expander("Latest indicator values"):
        latest = frame.iloc[-1]
        wanted = [c for c in frame.columns if c.upper() == c and c not in ("Open",)]
        table = pd.DataFrame(
            {"Indicator": wanted, "Value": [latest[c] for c in wanted]}
        )
        st.dataframe(table, width="stretch", hide_index=True)


def _backtest(settings, controls: dict, palette) -> None:
    """Run a backtest from the browser and read the result.

    The run is deliberately explicit rather than automatic: backtests are slow
    enough that firing one on every widget change would make the page unusable,
    and cheap enough to re-run that a button is no hardship.
    """
    st.subheader("Backtest")
    st.caption(
        "Simulates a strategy candle by candle over historical bars. Entries fill "
        "at the next bar's open, stops honour gaps, and costs are charged both "
        "ways — so the result is a measurement under stated assumptions, not a "
        "forecast."
    )

    with st.form("backtest"):
        first, second, third = st.columns(3)
        symbols_raw = first.text_input(
            "Symbols", value=", ".join(controls["symbols"][:5]),
            help="Comma-separated. More symbols means a longer run.",
        )
        strategy_names = second.multiselect(
            "Strategies", available_strategies(), default=["momentum"],
        )
        capital = third.number_input(
            "Starting capital", min_value=100.0, value=10_000.0, step=1_000.0,
        )

        fourth, fifth, sixth = st.columns(3)
        today = date.today()
        start_date = fourth.date_input(
            "Start", value=today - timedelta(days=60), max_value=today,
        )
        end_date = fifth.date_input("End", value=today, max_value=today)
        slippage = sixth.number_input(
            "Slippage %", min_value=0.0, max_value=5.0, value=0.05, step=0.01,
            format="%.3f",
            help="Applied to every fill, always against the trade.",
        )

        seventh, eighth, ninth = st.columns(3)
        commission = seventh.number_input(
            "Commission per fill", min_value=0.0, value=0.0, step=0.5,
        )
        risk_pct = eighth.number_input(
            "Risk per trade %", min_value=0.05, max_value=10.0,
            value=float(settings.risk.max_risk_per_trade_pct), step=0.25,
        )
        max_positions = ninth.number_input(
            "Max open positions", min_value=1, max_value=20,
            value=int(settings.risk.max_open_positions), step=1,
        )
        submitted = st.form_submit_button("Run backtest", type="primary")

    if submitted:
        symbols = tuple(
            item.strip().upper() for item in symbols_raw.split(",") if item.strip()
        )
        if not symbols:
            st.error("Enter at least one symbol.")
            return
        if not strategy_names:
            st.error("Choose at least one strategy.")
            return
        if start_date >= end_date:
            st.error("The start date must be before the end date.")
            return

        request = BacktestRequest(
            symbols=symbols,
            strategies=tuple(strategy_names),
            timeframe=controls["timeframe"],
            start=datetime.combine(start_date, time.min, tzinfo=UTC),
            end=datetime.combine(end_date, time.min, tzinfo=UTC),
            starting_equity=float(capital),
            costs=CostModel(
                commission_per_trade=float(commission),
                slippage_pct=float(slippage),
            ),
            risk=settings.risk.model_copy(
                update={
                    "max_risk_per_trade_pct": float(risk_pct),
                    "max_open_positions": int(max_positions),
                }
            ),
            demo=controls["demo"],
        )
        with st.spinner(f"Simulating {len(symbols)} symbol(s)…"):
            try:
                result = dashboard_data.run_backtest_cached(request, settings)
            except Exception as error:  # noqa: BLE001 - surfaced in the UI
                st.error(f"Backtest failed: {error}")
                return
        st.session_state["backtest_result"] = result
        st.session_state["backtest_frames"] = dashboard_data.backtest_frames(
            request, settings
        )

    result = st.session_state.get("backtest_result")
    if result is None:
        st.info("Set the parameters above and run a backtest to see results here.")
        return

    _backtest_result(result, palette)


def _dollars(value: float) -> str:
    """Currency for a markdown context, with the dollar sign escaped.

    Streamlit renders markdown, where a pair of unescaped ``$`` delimits LaTeX
    math: "$55.10** to make **$220.39" loses both signs and italicises the words
    between them. Anything written with st.markdown needs this; the HTML metric
    cards do not, because they are not parsed as markdown.
    """
    return f"\\${abs(value):,.2f}" if value >= 0 else f"-\\${abs(value):,.2f}"


def _money(value: float) -> str:
    """Currency with the sign in front of the symbol, not after it."""
    return f"-${abs(value):,.2f}" if value < 0 else f"${value:,.2f}"


def _backtest_result(result, palette) -> None:
    """Render a completed backtest."""
    metrics = result.metrics
    if metrics.sample_warning:
        st.warning(metrics.sample_warning)

    columns = st.columns(5)
    columns[0].markdown(
        _metric(
            "Net profit",
            _money(metrics.net_profit),
            f"{metrics.total_return_pct:+.2f}% on ${metrics.starting_equity:,.0f}",
        ),
        unsafe_allow_html=True,
    )
    columns[1].markdown(
        _metric(
            "Trades", f"{metrics.total_trades}",
            f"{metrics.wins} win / {metrics.losses} loss · {metrics.win_rate:.0f}%",
        ),
        unsafe_allow_html=True,
    )
    columns[2].markdown(
        _metric(
            "Profit factor",
            "∞" if metrics.profit_factor == float("inf") else f"{metrics.profit_factor:.2f}",
            f"expectancy {metrics.expectancy_r:+.2f}R per trade",
        ),
        unsafe_allow_html=True,
    )
    columns[3].markdown(
        _metric(
            "Max drawdown", f"{metrics.max_drawdown_pct:.2f}%",
            f"${metrics.max_drawdown_value:,.0f} lost over "
            f"{metrics.max_drawdown_bars} bars",
        ),
        unsafe_allow_html=True,
    )
    columns[4].markdown(
        _metric(
            "Sharpe", f"{metrics.sharpe_ratio:.2f}",
            f"Sortino {metrics.sortino_ratio:.2f} · {metrics.exposure_pct:.0f}% in market",
        ),
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        equity_chart(
            result.equity_curve, result.drawdown, palette,
            starting_equity=metrics.starting_equity,
        ),
        width="stretch",
        config={"displaylogo": False},
    )

    trades = result.trade_frame
    if not trades.empty:
        st.plotly_chart(
            r_multiple_chart(trades, palette),
            width="stretch",
            config={"displaylogo": False},
        )

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Costs and exits**")
        st.markdown(
            f"- Commission paid: **${metrics.total_commission:,.2f}**\n"
            f"- Slippage cost: **${metrics.total_slippage:,.2f}**\n"
            f"- Stops that gapped: **{metrics.gapped_stops}**\n"
            f"- Signals generated: **{result.signals_generated}** "
            f"({result.signals_rejected} rejected by risk)"
        )
    with right:
        if metrics.exit_breakdown:
            st.markdown("**How trades ended**")
            st.dataframe(
                pd.DataFrame(
                    sorted(
                        metrics.exit_breakdown.items(),
                        key=lambda item: -item[1],
                    ),
                    columns=["Exit reason", "Count"],
                ),
                width="stretch",
                hide_index=True,
            )

    if not trades.empty:
        symbols = sorted(trades["symbol"].unique())
        chosen = st.selectbox("Trades on chart", symbols, key="backtest_symbol")
        frame = st.session_state.get("backtest_frames", {}).get(chosen)
        if frame is not None:
            st.plotly_chart(
                trade_chart(frame, trades, chosen, palette),
                width="stretch",
                config={"scrollZoom": True, "displaylogo": False},
            )

        with st.expander(f"All {len(trades)} trades"):
            st.dataframe(
                trades[
                    [
                        "symbol", "direction", "entry_time", "entry_price",
                        "exit_time", "exit_price", "quantity", "pnl",
                        "r_multiple", "bars_held", "exit_reason",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        st.download_button(
            "Download trades as CSV",
            trades.to_csv(index=False),
            file_name="backtest_trades.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No trades were taken. Widen the date range, lower the confidence "
            "floor on the Strategy Settings page, or try another strategy."
        )

    if result.rejection_reasons:
        with st.expander("Why signals were rejected by risk"):
            st.dataframe(
                pd.DataFrame(
                    sorted(result.rejection_reasons.items(), key=lambda i: -i[1]),
                    columns=["Failed check", "Count"],
                ),
                width="stretch",
                hide_index=True,
            )


def _strategy_settings(controls: dict, palette) -> None:
    st.subheader("Strategy Settings")
    st.caption(
        "Adjust indicator periods and thresholds. Changes apply to this session; "
        "edit `.env` to make them permanent."
    )

    defaults = IndicatorConfig()
    left, middle, right = st.columns(3)

    with left:
        st.markdown("##### Momentum")
        rsi_period = st.number_input("RSI period", 2, 50, defaults.rsi_period)
        rsi_oversold = st.slider("RSI oversold", 5, 45, int(defaults.rsi_oversold))
        rsi_overbought = st.slider("RSI overbought", 55, 95, int(defaults.rsi_overbought))
        macd_fast = st.number_input("MACD fast", 2, 50, defaults.macd_fast)
        macd_slow = st.number_input("MACD slow", 3, 100, defaults.macd_slow)

    with middle:
        st.markdown("##### Volatility")
        bollinger_period = st.number_input(
            "Bollinger period", 5, 100, defaults.bollinger_period
        )
        bollinger_std = st.slider("Bollinger σ", 1.0, 4.0, float(defaults.bollinger_std), 0.1)
        atr_period = st.number_input("ATR period", 2, 50, defaults.atr_period)

    with right:
        st.markdown("##### Volume & structure")
        volume_period = st.number_input(
            "Volume average", 5, 100, defaults.volume_sma_period
        )
        volume_high = st.slider(
            "High volume ×", 1.0, 3.0, float(defaults.volume_high_threshold), 0.1
        )
        volume_spike = st.slider(
            "Spike ×", 1.5, 6.0, float(defaults.volume_spike_threshold), 0.1
        )
        swing_strength = st.number_input("Swing strength", 1, 10, defaults.swing_strength)

    try:
        config = IndicatorConfig(
            rsi_period=int(rsi_period),
            rsi_oversold=float(rsi_oversold),
            rsi_overbought=float(rsi_overbought),
            macd_fast=int(macd_fast),
            macd_slow=int(macd_slow),
            bollinger_period=int(bollinger_period),
            bollinger_std=float(bollinger_std),
            atr_period=int(atr_period),
            volume_sma_period=int(volume_period),
            volume_high_threshold=float(volume_high),
            volume_spike_threshold=float(volume_spike),
            swing_strength=int(swing_strength),
        )
    except ValueError as error:
        st.error(f"Invalid combination: {error}")
        return

    controls["indicator_overrides"] = {
        "rsi_period": config.rsi_period,
        "rsi_oversold": config.rsi_oversold,
        "rsi_overbought": config.rsi_overbought,
        "macd_fast": config.macd_fast,
        "macd_slow": config.macd_slow,
        "bollinger_period": config.bollinger_period,
        "bollinger_std": config.bollinger_std,
        "atr_period": config.atr_period,
        "volume_sma_period": config.volume_sma_period,
        "volume_high_threshold": config.volume_high_threshold,
        "volume_spike_threshold": config.volume_spike_threshold,
        "swing_strength": config.swing_strength,
    }
    st.success(
        f"Configuration valid. Warm-up requires {config.max_lookback} bars before "
        "every indicator has a value."
    )
    st.caption(
        "Settings are validated the same way the engine validates them — an "
        "incoherent combination is rejected here rather than producing quiet nonsense."
    )


def _footer(palette) -> None:
    st.divider()
    st.caption(
        "Research tool, not financial advice. Trading involves substantial risk of "
        "loss. This dashboard is read-only and cannot place orders."
    )


# `streamlit run` executes this file with __name__ == "__main__", so this single
# guard covers both that and direct execution. Importing the module (in tests, or
# to reuse a helper) must never launch the app.
if __name__ == "__main__":
    main()
