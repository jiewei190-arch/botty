"""Dashboard: palette validity, chart construction, and the safety guarantee.

Charts are built as Plotly figures and returned, so they can be asserted on
without a browser. The tests that matter most here are the ones covering
mistakes that are invisible in code review but obvious on screen — an
annotation placed on the wrong axis, a legend that names nothing, a colour that
contradicts its own meaning.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import make_bars
from trading_bot.dashboard.charts import (
    confidence_bar,
    equity_placeholder,
    price_chart,
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
    assert app.PAGES[0] == "Overview"


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


def _run(app_path, **widgets):
    """Run the app headlessly, optionally setting sidebar widgets first."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(app_path, default_timeout=300)
    app.run()
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
        app = _run(app_path, page=page, symbols=["AAPL"], strategies=["momentum"])
        assert not app.exception, f"{page} raised"


def test_the_scanner_page_ranks_and_sizes_opportunities(app_path):
    """850 bars of AAPL demo data yields a momentum signal, which must be
    ranked, scored and sized."""
    app = _run(
        app_path, page="Market Scanner", bars=850, symbols=["AAPL"],
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


def test_the_app_uses_no_deprecated_streamlit_apis(app_path):
    """`use_container_width` was removed after 2025-12-31."""
    from pathlib import Path

    source = Path(app_path).read_text()
    assert "use_container_width" not in source
