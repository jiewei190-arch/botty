"""Dashboard: palette validity, chart construction, and the safety guarantee.

Charts are built as Plotly figures and returned, so they can be asserted on
without a browser. The tests that matter most here are the ones covering
mistakes that are invisible in code review but obvious on screen — an
annotation placed on the wrong axis, a legend that names nothing, a colour that
contradicts its own meaning.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_bars
from trading_bot.dashboard.charts import (
    confidence_bar,
    equity_chart,
    equity_placeholder,
    price_chart,
    r_multiple_chart,
    trade_chart,
)
from trading_bot.dashboard.theme import (
    CONTEXT_EMA,
    DARK,
    EMA_SLOTS,
    LIGHT,
    app_css,
    direction_marker,
    ema_colour,
    get_palette,
)
from trading_bot.indicators import calculate_all_indicators, find_support_resistance


@pytest.fixture(scope="module")
def enriched() -> pd.DataFrame:
    return calculate_all_indicators(make_bars(400, seed=99))


@pytest.fixture(params=["light", "dark"])
def palette(request):
    return get_palette(request.param)


# ============================================================================
# Palette
# ============================================================================


def test_both_modes_are_defined():
    assert get_palette("light") is LIGHT
    assert get_palette("dark") is DARK


def test_unknown_mode_falls_back_to_dark():
    assert get_palette("solarized") is DARK
    assert get_palette("") is DARK


def test_each_ema_slot_has_a_distinct_colour(palette):
    colours = [ema_colour(palette, period) for period in EMA_SLOTS]
    assert len(set(colours)) == len(colours)


def test_the_context_ema_is_muted_not_a_series_colour(palette):
    """EMA 200 is background context, so it must not spend a categorical slot."""
    assert ema_colour(palette, CONTEXT_EMA) == palette.muted
    assert ema_colour(palette, CONTEXT_EMA) not in [
        ema_colour(palette, period) for period in EMA_SLOTS
    ]


def test_bullish_and_bearish_are_different_in_both_modes():
    for mode in (LIGHT, DARK):
        assert mode.bullish != mode.bearish


def test_status_colours_are_mode_invariant():
    """Polarity should not shift meaning between themes."""
    assert LIGHT.bullish == DARK.bullish
    assert LIGHT.bearish == DARK.bearish


def test_direction_never_depends_on_colour_alone():
    assert "LONG" in direction_marker("LONG")
    assert "▲" in direction_marker("LONG")
    assert "SHORT" in direction_marker("short")
    assert "▼" in direction_marker("short")


def test_up_candles_are_hollow():
    """Fill is the second channel that survives colourblindness and greyscale."""
    for mode in (LIGHT, DARK):
        assert "rgba" in mode.bullish_fill
        assert mode.bullish_fill.endswith("0)")


def test_css_carries_the_palette(palette):
    css = app_css(palette)
    assert palette.page in css
    assert palette.surface in css
    assert "stHeader" in css   # Streamlit's header is otherwise unthemed


# ============================================================================
# Price chart
# ============================================================================


def test_chart_builds_all_four_panels(enriched, palette):
    figure = price_chart(enriched, "AAPL", palette)
    assert figure.layout.height
    names = {trace.name for trace in figure.data}
    assert "Price" in names
    assert "Volume" in names
    assert any(name.startswith("RSI") for name in names)
    assert "MACD" in names


def test_chart_has_no_secondary_y_axis(enriched, palette):
    """A dual axis lets any two lines be placed in any relationship — never used."""
    figure = price_chart(enriched, "AAPL", palette)
    axes = [key for key in figure.layout.to_plotly_json() if key.startswith("yaxis")]
    assert len(axes) == 4          # one per stacked panel, none overlaid


def test_every_data_axis_annotation_sits_on_a_real_timestamp(enriched, palette):
    """The bug this guards against crushed 400 bars into one pixel column.

    A fixed slice over ``layout.annotations`` reached past the subplot titles
    into the series labels, and writing x=0.005 onto a datetime axis placed a
    label at the epoch — stretching the range across five decades.
    """
    figure = price_chart(enriched, "AAPL", palette)
    for annotation in figure.layout.annotations:
        if annotation.xref == "x":
            assert isinstance(annotation.x, pd.Timestamp), annotation.text


def test_series_labels_sit_at_the_final_bar(enriched, palette):
    figure = price_chart(enriched, "AAPL", palette)
    last = enriched.index[-1]
    labels = [a for a in figure.layout.annotations if a.xref == "x"]
    assert labels
    for annotation in labels:
        assert annotation.x == last


def test_legend_names_every_multi_series_panel(enriched, palette):
    """Direct labels alone fail when converging EMAs collide."""
    figure = price_chart(enriched, "AAPL", palette)
    legend = {t.name for t in figure.data if t.showlegend is not False}
    assert {f"EMA {period}" for period in EMA_SLOTS} <= legend
    assert {"MACD", "Signal"} <= legend


def test_single_series_panels_stay_out_of_the_legend(enriched, palette):
    """A lone series is named by its panel title; a legend row is noise."""
    figure = price_chart(enriched, "AAPL", palette)
    legend = {t.name for t in figure.data if t.showlegend is not False}
    assert not any(name.startswith("RSI") for name in legend)
    assert "Volume" not in legend
    assert "Price" not in legend


def test_candles_encode_direction_with_fill_as_well_as_colour(enriched, palette):
    figure = price_chart(enriched, "AAPL", palette)
    candles = next(t for t in figure.data if t.type == "candlestick")
    assert candles.increasing.fillcolor != candles.decreasing.fillcolor
    assert candles.increasing.line.color == palette.bullish
    assert candles.decreasing.line.color == palette.bearish


def test_levels_are_drawn_when_supplied(enriched, palette):
    levels = find_support_resistance(enriched)
    with_levels = price_chart(enriched, "AAPL", palette, levels=levels)
    without = price_chart(enriched, "AAPL", palette, show_levels=False)
    assert len(with_levels.layout.shapes) >= len(without.layout.shapes)


def test_bollinger_bands_can_be_hidden(enriched, palette):
    shown = price_chart(enriched, "AAPL", palette, show_bollinger=True)
    hidden = price_chart(enriched, "AAPL", palette, show_bollinger=False)
    assert len(hidden.data) < len(shown.data)


def test_rsi_panel_is_pinned_to_its_natural_range(enriched, palette):
    figure = price_chart(enriched, "AAPL", palette)
    assert tuple(figure.layout.yaxis3.range) == (0, 100)


def test_chart_survives_a_short_history(palette):
    """Warm-up NaNs must not break rendering."""
    short = calculate_all_indicators(make_bars(40, seed=101))
    figure = price_chart(short, "AAPL", palette)
    assert figure.data


def test_chart_uses_the_matching_template(enriched):
    assert price_chart(enriched, "A", DARK).layout.template.layout.paper_bgcolor is not None
    assert price_chart(enriched, "A", DARK).layout.plot_bgcolor == DARK.surface
    assert price_chart(enriched, "A", LIGHT).layout.plot_bgcolor == LIGHT.surface


# ============================================================================
# Small components
# ============================================================================


@pytest.mark.parametrize("value", [0, 25, 55, 60, 75, 100])
def test_confidence_bar_renders_within_bounds(value, palette):
    html = confidence_bar(value, palette)
    assert f"{value:.0f}" in html
    assert f"width:{float(value)}%" in html


def test_confidence_bar_clamps_out_of_range_values(palette):
    assert "width:100.0%" in confidence_bar(150, palette)
    assert "width:0.0%" in confidence_bar(-20, palette)


def test_confidence_bar_always_shows_the_number(palette):
    """The bar is an aid; the value is never conveyed by length alone."""
    assert "82" in confidence_bar(82, palette)


def test_equity_placeholder_explains_itself(palette):
    figure = equity_placeholder(palette)
    assert figure.layout.annotations
    assert "Phase 6" in figure.layout.annotations[0].text


# ============================================================================
# Safety
# ============================================================================


def test_importing_the_app_does_not_launch_it():
    """`streamlit run` sets __name__ to __main__; a plain import must be inert."""
    import trading_bot.dashboard.app as app

    assert callable(app.main)
    # Hunt leads: it is what the app is for.
    assert app.PAGES[0] == "Hunt"


def test_the_dashboard_contains_no_order_placing_code():
    """The read-only guarantee, asserted rather than assumed."""
    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "trading_bot" / "dashboard"
    forbidden = ("submit_order", "close_position", "cancel_order", "MarketOrderRequest")
    for path in package.glob("*.py"):
        source = path.read_text()
        for token in forbidden:
            assert token not in source, f"{path.name} references {token}"


# ============================================================================
# End-to-end app runs
# ============================================================================
#
# Streamlit's AppTest executes the real app headlessly, so these catch the
# failures that only appear when the pages actually render — a broken widget
# chain, an exception inside a page, a deprecated API. They are slower than the
# unit tests above, so there are few of them and each covers a whole page.


@pytest.fixture(scope="module")
def app_path() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[1] / "trading_bot" / "dashboard" / "app.py")


def _enable_demo(app):
    """Switch on demo data. It no longer enables itself when a key is absent."""
    for toggle in app.get("toggle"):
        if "Demo" in toggle.label:
            toggle.set_value(True)
            return app
    raise AssertionError("the demo toggle is missing")


def _run(app_path, *, demo=False, **widgets):
    """Run the app headlessly, optionally setting sidebar widgets first."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(app_path, default_timeout=300)
    app.run()
    if demo:
        _enable_demo(app).run()
    if widgets:
        if "bars" in widgets:
            app.slider[0].set_value(widgets["bars"])
        if "symbols" in widgets:
            app.multiselect[0].set_value(widgets["symbols"])
        if "strategies" in widgets:
            app.multiselect[1].set_value(widgets["strategies"])
        if "page" in widgets:
            app.radio[0].set_value(widgets["page"])
        app.run()
    return app


def test_the_app_starts_without_error(app_path):
    app = _run(app_path)
    assert not app.exception


def test_every_page_renders(app_path):
    from trading_bot.dashboard.app import PAGES

    for page in PAGES:
        app = _run(
            app_path, demo=True, page=page, symbols=["AAPL"],
            strategies=["momentum"],
        )
        assert not app.exception, f"{page} raised"


def test_the_scanner_page_ranks_and_sizes_opportunities(app_path):
    """850 bars of AAPL demo data yields a momentum signal, which must be
    ranked, scored and sized."""
    app = _run(
        app_path, demo=True, page="Market Scanner", bars=850, symbols=["AAPL"],
        strategies=["momentum"],
    )
    assert not app.exception
    rendered = " ".join(str(item.value) for item in app.markdown)
    rendered += " ".join(str(item.value) for item in app.caption)
    assert "Ranked opportunities" in rendered
    assert "Score breakdown" in rendered
    assert "Risk validation" in rendered
    # The score's meaning is stated wherever it is shown.
    assert "probability of profit" in rendered


def _click(app, label: str):
    """Press a button by label. Form submit buttons appear in ``app.button``."""
    for button in app.button:
        if button.label == label:
            return button.click()
    raise AssertionError(f"no button labelled {label!r}: {[b.label for b in app.button]}")


def test_the_backtest_page_runs_and_renders_a_result(app_path):
    """Submitting the form is what exercises the result charts.

    Loading the page only builds the form; every result chart is built after
    the run, so a figure that raises at render time (addressing a subplot axis
    on a plain figure, say) is invisible until the button is pressed.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(app_path, default_timeout=900)
    app.run()
    _enable_demo(app).run()
    app.radio[0].set_value("Backtest").run()
    _click(app, "Run backtest").run()

    assert not app.exception, f"backtest page raised: {app.exception}"
    rendered = " ".join(str(item.value) for item in app.markdown)
    assert "Net profit" in rendered
    assert "Max drawdown" in rendered
    # A result is never presented as a forecast.
    captions = " ".join(str(item.value) for item in app.caption)
    assert "not a forecast" in captions


def test_the_backtest_page_rejects_a_backwards_date_range(app_path):
    from datetime import date, timedelta

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(app_path, default_timeout=900)
    app.run()
    app.radio[0].set_value("Backtest").run()
    # Push the start past the end.
    app.date_input[0].set_value(date.today())
    app.date_input[1].set_value(date.today() - timedelta(days=30))
    _click(app, "Run backtest").run()

    assert not app.exception
    assert any("before the end date" in str(item.value) for item in app.error)


def _fake_sweep():
    """A completed sweep, so the results path can be rendered without a network."""
    from decimal import Decimal

    from tests.conftest import make_bars
    from trading_bot.risk import PortfolioState, RiskManager
    from trading_bot.scanner.market_scan import HuntConfig, sweep_market
    from trading_bot.strategies import build_strategy
    from trading_bot.universe import Universe
    from trading_bot.universe.filters import profile_liquidity

    frames, profiles = {}, {}
    for index in range(160):
        symbol = f"S{index:03d}"
        bars = make_bars(
            300, seed=index, freq="1D", start_price=float(20 + (index * 17) % 300)
        )
        frames[symbol] = bars
        profiles[symbol] = profile_liquidity(symbol, bars)

    equity = Decimal("15000")
    return sweep_market(
        Universe(symbols=tuple(frames), profiles=profiles, frames=frames),
        [build_strategy("momentum")],
        portfolio=PortfolioState(
            equity=equity, cash=equity, buying_power=equity
        ),
        risk_manager=RiskManager(),
        config=HuntConfig(top_n=5),
    )


def test_the_hunt_page_asks_for_credentials_when_missing(app_path):
    """It reads live data by design, so it must say so rather than fail oddly."""
    app = _run(app_path, page="Hunt")
    assert not app.exception
    messages = " ".join(str(item.value) for item in app.error)
    assert "ALPACA_API_KEY" in messages
    # And it must be clear the key is for data, not for trading.
    assert "no order is ever placed" in messages


def test_the_hunt_page_renders_results(app_path, monkeypatch):
    """The result cards only exist after a sweep, so rendering needs one."""
    from streamlit.testing.v1 import AppTest

    import trading_bot.dashboard.data as dashboard_data

    sweep = _fake_sweep()
    monkeypatch.setattr(
        dashboard_data, "run_hunt", lambda *args, **kwargs: sweep, raising=False
    )
    monkeypatch.setattr(
        "trading_bot.config.settings.AlpacaSettings.has_credentials",
        property(lambda self: True),
    )

    app = AppTest.from_file(app_path, default_timeout=600)
    app.run()
    app.radio[0].set_value("Hunt").run()
    assert not app.exception, f"hunt page raised: {app.exception}"

    _click(app, "Run hunt").run()
    assert not app.exception, f"running the hunt raised: {app.exception}"

    rendered = " ".join(str(item.value) for item in app.markdown)
    assert "Scanned" in rendered
    assert "Setups found" in rendered
    # Per-trade sizing must never imply every setup can be taken at once.
    if sweep.opportunities:
        assert "only trade you take" in rendered
        assert "Entry near" in rendered
        assert "Stop" in rendered
        assert "Target" in rendered


def test_the_hunt_page_never_claims_a_probability(app_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import trading_bot.dashboard.data as dashboard_data

    sweep = _fake_sweep()
    monkeypatch.setattr(
        dashboard_data, "run_hunt", lambda *args, **kwargs: sweep, raising=False
    )
    monkeypatch.setattr(
        "trading_bot.config.settings.AlpacaSettings.has_credentials",
        property(lambda self: True),
    )
    app = AppTest.from_file(app_path, default_timeout=600)
    app.run()
    app.radio[0].set_value("Hunt").run()
    _click(app, "Run hunt").run()

    text = " ".join(str(item.value) for item in app.caption)
    text += " ".join(str(item.value) for item in app.markdown)
    if sweep.opportunities:
        assert "not scored for probability" in text or "rather than estimating" in text


def test_currency_in_markdown_escapes_the_dollar_sign():
    """Streamlit parses markdown, where a pair of $ delimits LaTeX math.

    "Risking $55.10 to make $220.39" rendered as "Risking 55.10 *to make* 220.39":
    both signs consumed as delimiters and the words between them italicised.
    """
    from trading_bot.dashboard.app import _dollars

    assert _dollars(55.10) == r"\$55.10"
    assert _dollars(-220.39) == r"-\$220.39"
    assert _dollars(1234.5) == r"\$1,234.50"


def test_the_direction_marker_is_not_doubled_up(app_path):
    """direction_marker already pairs the arrow with the word."""
    from trading_bot.dashboard.theme import direction_marker

    source = Path(app_path).read_text()
    assert "{arrow} {signal.direction.value}" not in source
    assert direction_marker("LONG") == "\u25b2 LONG"


def test_demo_data_never_enables_itself(app_path):
    """Missing credentials must read as an error, not as working software.

    Auto-substituting generated prices when a key is absent is the one failure
    mode a person cannot catch by looking: the app appears to work, and every
    number on it describes nothing.
    """
    app = _run(app_path)
    assert not app.exception
    toggles = [t for t in app.get("toggle") if "Demo" in t.label]
    assert toggles, "the demo toggle is missing"
    assert toggles[0].value is False

    # And the missing key is surfaced as an error.
    messages = " ".join(str(item.value) for item in app.error)
    assert "ALPACA_API_KEY" in messages


def test_the_hunt_command_has_no_demo_mode():
    """A scan of generated noise looks exactly like a scan of the market."""
    from trading_bot.main import build_parser

    parser = build_parser()
    actions = parser.parse_args(["hunt"])
    assert not hasattr(actions, "demo")


def test_only_one_place_asks_for_the_account_balance(app_path):
    """Two inputs for one balance is two chances to size against the wrong one."""
    app = _run(app_path, page="Hunt")
    assert not app.exception
    equity_inputs = [
        widget for widget in app.number_input if "equity" in widget.label.lower()
    ]
    assert len(equity_inputs) == 1, (
        f"expected one equity input, found {[w.label for w in equity_inputs]}"
    )


def test_the_balance_is_never_read_from_a_broker(app_path):
    """The data provider's account is not the account you trade."""
    source = Path(app_path).read_text()
    hunt = source.split("def _hunt(")[1].split("\ndef ")[0]
    assert "get_account" not in hunt
    assert "build_broker" not in hunt


def test_the_app_uses_no_deprecated_streamlit_apis(app_path):
    """`use_container_width` was removed after 2025-12-31."""
    from pathlib import Path

    source = Path(app_path).read_text()
    assert "use_container_width" not in source


# ============================================================================
# Backtest result charts
#
# These figures are built from a backtest's output, which is a shape the price
# chart never sees: an equity Series, a drawdown Series, and a trade table. The
# render-time failures they can hit — addressing a subplot axis on a plain
# figure, an empty trade list, a symbol with no trades — do not show up at
# import time, so each has a test.
# ============================================================================


@pytest.fixture(scope="module")
def backtest_result():
    """A real backtest over generated bars, so the charts see real output."""
    from datetime import UTC, datetime

    from trading_bot.backtesting import BacktestConfig, Backtester
    from trading_bot.strategies import build_strategy

    strategy = build_strategy("momentum")
    frames = {
        symbol: strategy.prepare(
            make_bars(400, start=datetime(2025, 6, 2, tzinfo=UTC), seed=seed)
        )
        for symbol, seed in (("AAA", 11), ("BBB", 23))
    }
    result = Backtester([strategy], BacktestConfig(starting_equity=25_000.0)).run(frames)
    return result, frames


def test_equity_chart_stacks_equity_over_drawdown(backtest_result, palette):
    result, _ = backtest_result
    figure = equity_chart(result.equity_curve, result.drawdown, palette)
    axes = {trace.yaxis or "y" for trace in figure.data}
    assert len(axes) == 2, "equity and drawdown must be separate panels"


def test_equity_chart_has_no_secondary_y_axis(backtest_result, palette):
    """Two scales on one plot is the most common way a chart misleads."""
    result, _ = backtest_result
    figure = equity_chart(result.equity_curve, result.drawdown, palette)
    for name in dir(figure.layout):
        if name.startswith("yaxis"):
            axis = getattr(figure.layout, name, None)
            if axis is not None:
                assert getattr(axis, "overlaying", None) is None


def test_equity_chart_marks_the_starting_equity(backtest_result, palette):
    result, _ = backtest_result
    figure = equity_chart(
        result.equity_curve, result.drawdown, palette, starting_equity=25_000.0
    )
    labels = [a.text for a in figure.layout.annotations if a.text]
    assert any("25,000" in text for text in labels)


def test_equity_chart_handles_an_empty_curve(palette):
    figure = equity_chart(pd.Series(dtype="float64"), pd.Series(dtype="float64"), palette)
    assert figure.layout.annotations
    assert "No equity" in figure.layout.annotations[0].text


def test_trade_chart_marks_entries_by_shape_not_only_colour(backtest_result, palette):
    result, frames = backtest_result
    figure = trade_chart(frames["AAA"], result.trade_frame, "AAA", palette)
    markers = {
        trace.marker.symbol
        for trace in figure.data
        if trace.mode == "markers" and trace.marker.symbol
    }
    assert markers, "no trade markers were drawn"
    # Entry and exit must differ in shape, not just in colour.
    assert len(markers) >= 2


def test_trade_chart_names_every_marker_series(backtest_result, palette):
    result, frames = backtest_result
    figure = trade_chart(frames["AAA"], result.trade_frame, "AAA", palette)
    for trace in figure.data:
        assert trace.name, "an unnamed trace cannot appear in the legend"


def test_trade_chart_only_draws_the_chosen_symbol(backtest_result, palette):
    result, frames = backtest_result
    trades = result.trade_frame
    if trades.empty:
        pytest.skip("this run produced no trades")
    figure = trade_chart(frames["AAA"], trades, "AAA", palette)
    plotted = sum(
        len(trace.x) for trace in figure.data if trace.mode == "markers"
    )
    expected = len(trades[trades["symbol"] == "AAA"]) * 2  # an entry and an exit
    assert plotted == expected


def test_trade_chart_handles_a_symbol_with_no_trades(backtest_result, palette):
    result, frames = backtest_result
    figure = trade_chart(frames["AAA"], result.trade_frame.iloc[0:0], "AAA", palette)
    assert len(figure.data) == 1  # just the price line


def test_trade_chart_handles_an_empty_frame(palette):
    figure = trade_chart(pd.DataFrame(), pd.DataFrame(), "AAA", palette)
    assert "No bars" in figure.layout.annotations[0].text


def test_r_multiple_chart_builds_without_a_subplot_grid(backtest_result, palette):
    """A plain figure raises if an axis is addressed by row and column."""
    result, _ = backtest_result
    figure = r_multiple_chart(result.trade_frame, palette)
    assert figure.data


def test_r_multiple_chart_handles_no_trades(palette):
    figure = r_multiple_chart(pd.DataFrame(), palette)
    assert "No trades" in figure.layout.annotations[0].text


def test_r_multiple_chart_handles_missing_r_values(palette):
    trades = pd.DataFrame({"r_multiple": [None, None], "symbol": ["A", "A"],
                           "exit_reason": ["STOP_LOSS", "STOP_LOSS"]})
    figure = r_multiple_chart(trades, palette)
    assert "No R multiples" in figure.layout.annotations[0].text


def test_result_charts_paint_the_palette_background(backtest_result, palette):
    """A transparent background would borrow the host's theme."""
    result, frames = backtest_result
    for figure in (
        equity_chart(result.equity_curve, result.drawdown, palette),
        trade_chart(frames["AAA"], result.trade_frame, "AAA", palette),
        r_multiple_chart(result.trade_frame, palette),
    ):
        assert figure.layout.paper_bgcolor == palette.page
        assert figure.layout.plot_bgcolor == palette.surface


def test_streamlit_secrets_populate_the_environment(monkeypatch):
    """A hosted deploy has no .env — credentials arrive as Streamlit secrets."""
    import trading_bot.dashboard.data as dashboard_data

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.setattr(
        dashboard_data.st, "secrets", {"ALPACA_API_KEY": "PKFROMSECRETS"}
    )
    dashboard_data._adopt_streamlit_secrets()
    assert os.environ["ALPACA_API_KEY"] == "PKFROMSECRETS"


def test_an_existing_environment_variable_wins(monkeypatch):
    """A value set on the host must not be overridden by a stale secret."""
    import trading_bot.dashboard.data as dashboard_data

    monkeypatch.setenv("ALPACA_API_KEY", "PKFROMENV")
    monkeypatch.setattr(
        dashboard_data.st, "secrets", {"ALPACA_API_KEY": "PKFROMSECRETS"}
    )
    dashboard_data._adopt_streamlit_secrets()
    assert os.environ["ALPACA_API_KEY"] == "PKFROMENV"


def test_missing_secrets_are_not_an_error(monkeypatch):
    """Running locally with no secrets file is the normal case, not a failure."""
    import trading_bot.dashboard.data as dashboard_data

    class Exploding:
        def __getitem__(self, key):
            raise FileNotFoundError("no secrets.toml")

    monkeypatch.setattr(dashboard_data.st, "secrets", Exploding())
    dashboard_data._adopt_streamlit_secrets()  # must not raise
