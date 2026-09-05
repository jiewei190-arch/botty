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


# ============================================================================
# Backtest results
#
# Equity and drawdown are two measures of different scale, so they are two
# stacked panels rather than one plot with two y-axes. Drawdown is drawn as a
# filled area below zero because it only ever goes one way: filling it makes the
# depth and the duration readable at a glance, which the equity line alone hides.
# ============================================================================


def equity_chart(
    equity_curve: pd.Series,
    drawdown: pd.Series,
    palette: Palette,
    *,
    starting_equity: float | None = None,
    height: int = 420,
) -> go.Figure:
    """Equity over the run, with its drawdown beneath.

    Parameters
    ----------
    equity_curve:
        Account equity marked to market on every bar.
    drawdown:
        Percentage below the running peak, as produced by ``drawdown_curve``.
    starting_equity:
        Drawn as a reference line, so profit and loss are readable without
        arithmetic. Defaults to the curve's first value.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.66, 0.34],
        subplot_titles=("Equity", "Drawdown from peak (%)"),
    )
    if equity_curve.empty:
        return _empty_figure(palette, "No equity to plot", height=height)

    baseline = (
        float(starting_equity)
        if starting_equity is not None
        else float(equity_curve.iloc[0])
    )

    figure.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.to_numpy(),
            name="Equity",
            mode="lines",
            line={"color": palette.series_1, "width": 2},
            hovertemplate="%{x|%d %b %H:%M}<br>$%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # The starting line turns the curve into a profit-or-loss reading rather
    # than a shape whose baseline the reader has to infer.
    figure.add_hline(
        y=baseline,
        line={"color": palette.muted, "width": 1, "dash": "dot"},
        row=1,
        col=1,
    )
    figure.add_annotation(
        x=equity_curve.index[-1],
        y=baseline,
        text=f" start ${baseline:,.0f}",
        showarrow=False,
        xanchor="left",
        font={"color": palette.muted, "size": 11},
        row=1,
        col=1,
    )
    figure.add_annotation(
        x=equity_curve.index[-1],
        y=float(equity_curve.iloc[-1]),
        text=f" ${float(equity_curve.iloc[-1]):,.0f}",
        showarrow=False,
        xanchor="left",
        font={"color": palette.series_1, "size": 11},
        row=1,
        col=1,
    )

    if not drawdown.empty:
        figure.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.to_numpy(),
                name="Drawdown",
                mode="lines",
                line={"color": palette.bearish, "width": 1.5},
                fill="tozeroy",
                fillcolor=_translucent(palette.bearish, 0.18),
                hovertemplate="%{x|%d %b %H:%M}<br>%{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )
        worst = float(drawdown.min())
        if worst < 0:
            trough = drawdown.idxmin()
            figure.add_annotation(
                x=trough,
                y=worst,
                text=f"worst {abs(worst):.2f}%",
                showarrow=True,
                arrowhead=0,
                arrowcolor=palette.muted,
                ax=0,
                ay=18,
                font={"color": palette.text_secondary, "size": 11},
                row=2,
                col=1,
            )

    _apply_result_layout(figure, palette, height=height, rows=2)
    figure.update_yaxes(tickprefix="$", row=1, col=1)
    figure.update_yaxes(ticksuffix="%", row=2, col=1)
    return figure


def trade_chart(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str,
    palette: Palette,
    *,
    height: int = 420,
) -> go.Figure:
    """Price with entry and exit markers for one symbol.

    Entries and exits are distinguished by *shape* as well as colour —
    triangle-up for a long entry, triangle-down for a short — so the chart
    survives colourblindness and greyscale. Exits are hollow squares, which
    reads as "closing" without competing with the entry markers.
    """
    if data.empty:
        return _empty_figure(palette, f"No bars for {symbol}", height=height)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["close"].to_numpy(),
            name="Close",
            mode="lines",
            line={"color": palette.muted, "width": 1.4},
            hovertemplate="%{x|%d %b %H:%M}<br>$%{y:,.2f}<extra></extra>",
        )
    )

    rows = (
        trades[trades["symbol"] == symbol]
        if not trades.empty and "symbol" in trades.columns
        else trades.iloc[0:0]
    )
    if not rows.empty:
        longs = rows[rows["direction"] == "LONG"]
        shorts = rows[rows["direction"] != "LONG"]
        for subset, marker, colour, label in (
            (longs, "triangle-up", palette.bullish, "Long entry"),
            (shorts, "triangle-down", palette.bearish, "Short entry"),
        ):
            if subset.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=subset["entry_time"],
                    y=subset["entry_price"],
                    name=label,
                    mode="markers",
                    marker={
                        "symbol": marker,
                        "size": 11,
                        "color": colour,
                        "line": {"color": palette.surface, "width": 1.5},
                    },
                    hovertemplate=(
                        f"{label}<br>%{{x|%d %b %H:%M}}<br>$%{{y:,.2f}}<extra></extra>"
                    ),
                )
            )

        # Exits are coloured by outcome, but the win/loss split is also written
        # into the hover text — colour alone never carries the meaning.
        wins = rows[rows["pnl"] > 0]
        losses = rows[rows["pnl"] <= 0]
        for subset, colour, label in (
            (wins, palette.bullish, "Exit (profit)"),
            (losses, palette.bearish, "Exit (loss)"),
        ):
            if subset.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=subset["exit_time"],
                    y=subset["exit_price"],
                    name=label,
                    mode="markers",
                    marker={
                        "symbol": "square-open",
                        "size": 10,
                        "color": colour,
                        "line": {"width": 2},
                    },
                    customdata=subset[["pnl", "exit_reason"]].to_numpy(),
                    hovertemplate=(
                        "%{x|%d %b %H:%M}<br>$%{y:,.2f}"
                        "<br>P&L $%{customdata[0]:,.2f}"
                        "<br>%{customdata[1]}<extra></extra>"
                    ),
                )
            )

    _apply_result_layout(figure, palette, height=height, rows=1)
    figure.update_layout(
        title={
            "text": f"{symbol} — trades",
            "font": {"color": palette.text_primary, "size": 15},
            "x": 0.005,
            "xanchor": "left",
        },
    )
    figure.update_yaxes(tickprefix="$")
    return figure


def r_multiple_chart(
    trades: pd.DataFrame, palette: Palette, *, height: int = 260
) -> go.Figure:
    """Each trade's outcome in R, ordered as they happened.

    R is the unit that makes trades comparable: a 2R win on a small position and
    on a large one are the same result. Zero is drawn as the reference because
    that, not the average, is the line a trade has to clear.
    """
    if trades.empty or "r_multiple" not in trades.columns:
        return _empty_figure(palette, "No trades to plot", height=height)

    values = trades["r_multiple"].astype("float64")
    valid = values.notna()
    if not valid.any():
        return _empty_figure(palette, "No R multiples recorded", height=height)

    ordered = trades.loc[valid].reset_index(drop=True)
    outcomes = ordered["r_multiple"].astype("float64")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=list(range(1, len(outcomes) + 1)),
            y=outcomes.to_numpy(),
            name="Result (R)",
            marker={
                "color": [
                    palette.bullish if value > 0 else palette.bearish
                    for value in outcomes
                ],
                "line": {"color": palette.surface, "width": 1},
            },
            customdata=ordered[["symbol", "exit_reason"]].to_numpy(),
            hovertemplate=(
                "Trade %{x}<br>%{customdata[0]}<br>%{y:.2f}R"
                "<br>%{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line={"color": palette.axis, "width": 1})
    _apply_result_layout(figure, palette, height=height, rows=1, legend=False)
    figure.update_layout(
        title={
            "text": "Every trade, in R",
            "font": {"color": palette.text_primary, "size": 15},
            "x": 0.005,
            "xanchor": "left",
        },
        bargap=0.25,
    )
    figure.update_xaxes(title_text="Trade number")
    figure.update_yaxes(ticksuffix="R")
    return figure


def _translucent(colour: str, alpha: float) -> str:
    """A hex colour as rgba, for fills that must not compete with lines."""
    value = colour.lstrip("#")
    if len(value) != 6:
        return colour
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _empty_figure(palette: Palette, message: str, *, height: int) -> go.Figure:
    """A labelled blank, so an empty result reads as empty rather than broken."""
    figure = go.Figure()
    figure.update_layout(
        template=palette.plotly_template,
        height=height,
        paper_bgcolor=palette.page,
        plot_bgcolor=palette.surface,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        annotations=[
            {
                "text": message,
                "showarrow": False,
                "font": {"color": palette.muted, "size": 13},
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
            }
        ],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def _apply_result_layout(
    figure: go.Figure,
    palette: Palette,
    *,
    height: int,
    rows: int,
    legend: bool = True,
) -> None:
    """Shared chrome for the result charts."""
    axis = _axis_style(palette)
    figure.update_layout(
        template=palette.plotly_template,
        height=height,
        margin={"l": 8, "r": 112, "t": 44, "b": 34},  # right margin holds the end labels
        paper_bgcolor=palette.page,
        plot_bgcolor=palette.surface,
        font={"color": palette.text_secondary, "size": 12},
        showlegend=legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.005,
            "xanchor": "right",
            "x": 1,
            "font": {"color": palette.text_secondary, "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="x unified" if rows > 1 else "closest",
        dragmode="pan",
    )
    if rows == 1:
        # A plain go.Figure has no subplot grid, and addressing an axis by
        # row/col on one raises rather than being ignored.
        figure.update_xaxes(**axis)
        figure.update_yaxes(**axis)
    else:
        for row in range(1, rows + 1):
            figure.update_xaxes(**axis, row=row, col=1)
            figure.update_yaxes(**axis, row=row, col=1)

    for annotation in figure.layout.annotations:
        if annotation.xref == "paper":
            annotation.font.color = palette.text_secondary
            annotation.font.size = 12
            annotation.x = 0.005
            annotation.xanchor = "left"
