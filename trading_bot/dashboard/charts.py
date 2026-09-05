"""Chart construction.

Every chart is a plain Plotly figure, built from an indicator-enriched frame and
returned rather than drawn — so they can be unit-tested without a browser.

Layout rules held throughout:

* **One measure per axis.** Price, volume, RSI and MACD are four stacked panels
  sharing an x-axis, never two scales on one plot. A dual axis lets the author
  place any two lines in any relationship they like, which is why it is the most
  common way a chart misleads.
* **Recessive chrome.** Hairline grid, muted axes, no chart junk. The data is the
  only thing drawn at full strength.
* **Direct labels.** EMAs are named at their right-hand end, so identity never
  depends on matching a colour to a legend swatch.
* **Shape carries polarity.** Up candles are hollow and down candles solid, so
  direction survives colourblindness and greyscale printing.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_bot.dashboard.theme import CONTEXT_EMA, EMA_SLOTS, Palette, ema_colour
from trading_bot.indicators import (
    BB_LOWER_COL,
    BB_MIDDLE_COL,
    BB_UPPER_COL,
    MACD_COL,
    MACD_HISTOGRAM_COL,
    MACD_SIGNAL_COL,
    SupportResistance,
    ema_column,
    rsi_column,
    volume_sma_column,
)

#: Relative heights of the four panels. Price dominates; the rest are context.
PANEL_HEIGHTS = (0.52, 0.14, 0.17, 0.17)


def _axis_style(palette: Palette) -> dict:
    return {
        "showgrid": True,
        "gridcolor": palette.grid,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": palette.axis,
        "tickfont": {"color": palette.muted, "size": 11},
        "title_font": {"color": palette.text_secondary, "size": 12},
    }


def price_chart(
    data: pd.DataFrame,
    symbol: str,
    palette: Palette,
    *,
    levels: SupportResistance | None = None,
    rsi_period: int = 14,
    volume_period: int = 20,
    height: int = 780,
    show_bollinger: bool = True,
    show_levels: bool = True,
) -> go.Figure:
    """Build the four-panel price chart.

    Parameters
    ----------
    data:
        Indicator-enriched OHLCV frame.
    symbol:
        Used in the title.
    palette:
        Colours for the current mode.
    levels:
        Support and resistance to draw as reference lines.
    rsi_period, volume_period:
        Which indicator columns to read.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=list(PANEL_HEIGHTS),
        subplot_titles=("", "Volume", f"RSI {rsi_period}", "MACD"),
    )

    _add_bollinger(figure, data, palette, enabled=show_bollinger)
    _add_candles(figure, data, palette)
    _add_emas(figure, data, palette)
    if show_levels and levels is not None:
        _add_levels(figure, data, levels, palette)
    _add_volume(figure, data, palette, volume_period)
    _add_rsi(figure, data, palette, rsi_period)
    _add_macd(figure, data, palette)

    axis = _axis_style(palette)
    figure.update_layout(
        template=palette.plotly_template,
        height=height,
        margin={"l": 8, "r": 74, "t": 54, "b": 8},   # right margin holds direct labels
        paper_bgcolor=palette.page,
        plot_bgcolor=palette.surface,
        font={"color": palette.text_secondary, "size": 12},
        title={
            "text": f"{symbol}",
            "font": {"color": palette.text_primary, "size": 17},
            "x": 0.005,
            "xanchor": "left",
        },
        # Legend *and* direct labels. Direct labels alone are not enough: EMAs
        # converge, and when their end labels collide the series lose their
        # identity exactly when the chart is hardest to read.
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.005,
            "xanchor": "right",
            "x": 1,
            "font": {"color": palette.text_secondary, "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        bargap=0.15,
    )
    for row in range(1, 5):
        figure.update_xaxes(**axis, row=row, col=1, showspikes=True,
                            spikemode="across", spikethickness=1,
                            spikecolor=palette.muted, spikedash="dot")
        figure.update_yaxes(**axis, row=row, col=1)
    figure.update_yaxes(range=[0, 100], row=3, col=1)

    # Restyle the subplot titles only. Selecting them by position would be a
    # trap: an empty title creates no annotation, so a fixed slice reaches past
    # the titles into the series labels added later — and setting x=0.005 on one
    # of those puts it at the epoch on a datetime axis, stretching the range
    # across decades and crushing every bar into the right-hand edge.
    for annotation in figure.layout.annotations:
        if annotation.xref == "paper":
            annotation.font.color = palette.text_secondary
            annotation.font.size = 12
            annotation.x = 0.005
            annotation.xanchor = "left"
    return figure


def _add_candles(figure: go.Figure, data: pd.DataFrame, palette: Palette) -> None:
    """Candles, with fill as a second channel beside colour.

    Hollow bodies mean up and solid bodies mean down, so a reader who cannot
    separate the hues still reads direction correctly.
    """
    figure.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price",
            showlegend=False,
            increasing={
                "line": {"color": palette.bullish, "width": 1},
                "fillcolor": palette.bullish_fill,
            },
            decreasing={
                "line": {"color": palette.bearish, "width": 1},
                "fillcolor": palette.bearish,
            },
            hoverlabel={"namelength": 0},
        ),
        row=1,
        col=1,
    )


def _add_emas(figure: go.Figure, data: pd.DataFrame, palette: Palette) -> None:
    """EMA overlays, each labelled at its right-hand end."""
    for period in EMA_SLOTS:
        column = ema_column(period)
        if column not in data.columns or data[column].dropna().empty:
            continue
        colour = ema_colour(palette, period)
        figure.add_trace(
            go.Scatter(
                x=data.index,
                y=data[column],
                mode="lines",
                name=f"EMA {period}",
                line={"color": colour, "width": 2},
                hovertemplate=f"EMA {period}: %{{y:,.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        _label_series_end(figure, data, column, f"EMA {period}", colour)

    context = ema_column(CONTEXT_EMA)
    if context in data.columns and not data[context].dropna().empty:
        figure.add_trace(
            go.Scatter(
                x=data.index,
                y=data[context],
                mode="lines",
                name=f"EMA {CONTEXT_EMA}",
                line={"color": palette.muted, "width": 1.5, "dash": "dash"},
                hovertemplate=f"EMA {CONTEXT_EMA}: %{{y:,.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        _label_series_end(figure, data, context, f"EMA {CONTEXT_EMA}", palette.muted)


def _label_series_end(
    figure: go.Figure, data: pd.DataFrame, column: str, text: str, colour: str
) -> None:
    """Write a series' name at its final point, in the chart margin."""
    series = data[column].dropna()
    if series.empty:
        return
    figure.add_annotation(
        x=series.index[-1],
        y=float(series.iloc[-1]),
        text=f" {text}",
        showarrow=False,
        xanchor="left",
        font={"color": colour, "size": 11},
        row=1,
        col=1,
    )


def _add_bollinger(
    figure: go.Figure, data: pd.DataFrame, palette: Palette, *, enabled: bool
) -> None:
    """Bollinger Bands as a filled range — context, drawn under everything else."""
    if not enabled or BB_UPPER_COL not in data.columns:
        return
    if data[BB_UPPER_COL].dropna().empty:
        return

    figure.add_trace(
        go.Scatter(
            x=data.index, y=data[BB_UPPER_COL], mode="lines", name="BB upper", showlegend=False,
            line={"color": palette.axis, "width": 1}, hoverinfo="skip",
        ),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=data.index, y=data[BB_LOWER_COL], mode="lines", name="BB lower", showlegend=False,
            line={"color": palette.axis, "width": 1},
            fill="tonexty", fillcolor=palette.band_fill, hoverinfo="skip",
        ),
        row=1, col=1,
    )
    if BB_MIDDLE_COL in data.columns:
        figure.add_trace(
            go.Scatter(
                x=data.index, y=data[BB_MIDDLE_COL], mode="lines", name="BB middle",
                showlegend=False,
                line={"color": palette.axis, "width": 1, "dash": "dot"},
                hovertemplate="BB mid: %{y:,.2f}<extra></extra>",
            ),
            row=1, col=1,
        )


def _add_levels(
    figure: go.Figure,
    data: pd.DataFrame,
    levels: SupportResistance,
    palette: Palette,
) -> None:
    """Support and resistance as labelled reference lines.

    Drawn in muted ink rather than a new hue: they are reference marks, and
    spending a categorical colour on them would compete with the series.
    """
    for level, label in (
        (levels.nearest_resistance, "Resistance"),
        (levels.nearest_support, "Support"),
    ):
        if level is None:
            continue
        figure.add_hline(
            y=level.price,
            line={"color": palette.muted, "width": 1, "dash": "dot"},
            annotation_text=f"{label} {level.price:,.2f} · {level.touches}×",
            # Inside the plot: "left" places the label outside the axes, where the
            # margin clips it and a price reads as a different number.
            annotation_position="top left",
            annotation_font={"color": palette.text_secondary, "size": 10},
            row=1,
            col=1,
        )


def _add_volume(
    figure: go.Figure, data: pd.DataFrame, palette: Palette, period: int
) -> None:
    """Volume bars tinted by the bar's own direction, plus its moving average."""
    rising = data["close"] >= data["open"]
    colours = [
        palette.bullish if up else palette.bearish for up in rising
    ]
    figure.add_trace(
        go.Bar(
            x=data.index,
            y=data["volume"],
            name="Volume",
            showlegend=False,
            marker={"color": colours, "opacity": 0.45, "line": {"width": 0}},
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    column = volume_sma_column(period)
    if column in data.columns and not data[column].dropna().empty:
        figure.add_trace(
            go.Scatter(
                x=data.index, y=data[column], mode="lines", name=f"Avg {period}",
                showlegend=False,
                line={"color": palette.text_secondary, "width": 1.5},
                hovertemplate=f"Avg volume ({period}): %{{y:,.0f}}<extra></extra>",
            ),
            row=2, col=1,
        )


def _add_rsi(
    figure: go.Figure, data: pd.DataFrame, palette: Palette, period: int
) -> None:
    """RSI with its reference bands drawn as chrome, not as data."""
    column = rsi_column(period)
    if column not in data.columns or data[column].dropna().empty:
        return

    for level in (30, 70):
        figure.add_hline(
            y=level,
            line={"color": palette.axis, "width": 1, "dash": "dot"},
            row=3, col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=data.index, y=data[column], mode="lines", name=f"RSI {period}",
            showlegend=False,   # single series; the panel title names it
            line={"color": palette.series_1, "width": 2},
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ),
        row=3, col=1,
    )


def _add_macd(figure: go.Figure, data: pd.DataFrame, palette: Palette) -> None:
    """MACD line, signal line, and a histogram coloured by sign.

    The histogram reuses the bullish/bearish pair rather than a third identity
    colour, because its meaning is polarity — the same meaning the candles carry.
    """
    if MACD_COL not in data.columns or data[MACD_COL].dropna().empty:
        return

    histogram = data[MACD_HISTOGRAM_COL]
    figure.add_trace(
        go.Bar(
            x=data.index,
            y=histogram,
            name="Histogram",
            showlegend=False,
            marker={
                "color": [
                    palette.bullish if value >= 0 else palette.bearish
                    for value in histogram.fillna(0)
                ],
                "opacity": 0.45,
                "line": {"width": 0},
            },
            hovertemplate="Histogram: %{y:.4f}<extra></extra>",
        ),
        row=4, col=1,
    )
    for column, colour, label in (
        (MACD_COL, palette.series_1, "MACD"),
        (MACD_SIGNAL_COL, palette.series_2, "Signal"),
    ):
        figure.add_trace(
            go.Scatter(
                x=data.index, y=data[column], mode="lines", name=label,
                line={"color": colour, "width": 2},
                hovertemplate=f"{label}: %{{y:.4f}}<extra></extra>",
            ),
            row=4, col=1,
        )


def confidence_bar(confidence: float, palette: Palette, width: int = 120) -> str:
    """A small HTML meter for a confidence score.

    The number is always shown beside it — the bar is a visual aid, not the only
    way to read the value.
    """
    clamped = max(0.0, min(100.0, float(confidence)))
    colour = (
        palette.bullish if clamped >= 75 else
        palette.series_1 if clamped >= 60 else
        palette.warning
    )
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="width:{width}px;height:6px;border-radius:3px;'
        f'background:{palette.grid}">'
        f'<div style="width:{clamped}%;height:6px;border-radius:3px;'
        f'background:{colour}"></div></div>'
        f'<span style="font-variant-numeric:tabular-nums;'
        f'color:{palette.text_primary}">{clamped:.0f}</span></div>'
    )


def equity_placeholder(palette: Palette) -> go.Figure:
    """An empty equity curve, shown until the backtester exists in Phase 6."""
    figure = go.Figure()
    figure.update_layout(
        template=palette.plotly_template,
        height=220,
        paper_bgcolor=palette.page,
        plot_bgcolor=palette.surface,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        annotations=[
            {
                "text": "Equity curve appears once the backtester lands (Phase 6)",
                "showarrow": False,
                "font": {"color": palette.muted, "size": 13},
                "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
            }
        ],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure
