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
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - streamlit runs this as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from trading_bot import __version__
from trading_bot.dashboard import data as dashboard_data
from trading_bot.dashboard.charts import (
    confidence_bar,
    equity_placeholder,
    price_chart,
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

PAGES = ("Overview", "Market Scanner", "Chart", "Strategy Settings")


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
    if page == "Overview":
        _overview(settings, controls, palette)
    elif page == "Market Scanner":
        _scanner(settings, controls, palette)
    elif page == "Chart":
        _chart(settings, controls, palette)
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
        demo = st.toggle(
            "Demo data",
            value=not has_keys,
            help="Generated sample data, so the dashboard works without API keys.",
        )
        if not has_keys and not demo:
            st.warning("No Alpaca credentials — live data will fail.", icon="⚠️")

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
