"""Colour and styling for the dashboard.

The palette is validated, not chosen by eye. Every categorical colour below
cleared the six checks in both light and dark mode — lightness band, chroma
floor, colour-vision-deficiency separation on all pairs, normal-vision
separation, and contrast against the surface it is drawn on.

Two rules the charts follow from that validation:

* **Green and red never carry meaning alone.** Up and down candles differ in
  *fill* as well as hue — hollow for up, solid for down — so the direction is
  readable without colour vision. Direction labels elsewhere pair an arrow with
  the word.
* **EMAs are direct-labelled at the right edge.** In light mode the aqua slot
  measures 2.74:1 against the surface, below the 3:1 bar, and a visible label is
  what makes that legal.

Semantics are consistent everywhere: green means bullish, red means bearish,
and the numbered series colours mean identity only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Palette:
    """Every colour the dashboard draws with, for one mode."""

    name: str
    surface: str
    page: str
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    border: str

    # Categorical slots — identity only, never magnitude or polarity.
    series_1: str
    series_2: str
    series_3: str

    # Status — polarity only, always paired with a shape or a label.
    bullish: str
    bearish: str
    warning: str

    @property
    def plotly_template(self) -> str:
        return "plotly_dark" if self.name == "dark" else "plotly_white"

    @property
    def band_fill(self) -> str:
        """Recessive fill for Bollinger Bands — a range, not a series."""
        return (
            "rgba(255,255,255,0.045)" if self.name == "dark" else "rgba(11,11,11,0.04)"
        )

    @property
    def bullish_fill(self) -> str:
        """Hollow: up candles are outlined, not filled."""
        return "rgba(0,0,0,0)"


LIGHT = Palette(
    name="light",
    surface="#fcfcfb",
    page="#f9f9f7",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    border="rgba(11,11,11,0.10)",
    series_1="#2a78d6",
    series_2="#eb6834",
    series_3="#1baf7a",
    bullish="#0ca30c",
    bearish="#d03b3b",
    warning="#fab219",
)

DARK = Palette(
    name="dark",
    surface="#1a1a19",
    page="#0d0d0d",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    border="rgba(255,255,255,0.10)",
    series_1="#3987e5",
    series_2="#d95926",
    series_3="#199e70",
    bullish="#0ca30c",
    bearish="#d03b3b",
    warning="#fab219",
)

PALETTES: dict[str, Palette] = {"light": LIGHT, "dark": DARK}

#: EMA period to categorical slot. Only three periods get an identity colour;
#: the longest is drawn as muted context rather than a fourth compared series,
#: which also keeps the palette inside its validated three-slot all-pairs range.
EMA_SLOTS: tuple[int, ...] = (9, 20, 50)
CONTEXT_EMA: int = 200


def get_palette(mode: str) -> Palette:
    """Look up a palette by mode name, defaulting to dark."""
    return PALETTES.get(mode.strip().lower(), DARK)


def ema_colour(palette: Palette, period: int) -> str:
    """Colour for an EMA line — identity, assigned in fixed slot order."""
    slots = {
        EMA_SLOTS[0]: palette.series_1,
        EMA_SLOTS[1]: palette.series_2,
        EMA_SLOTS[2]: palette.series_3,
    }
    return slots.get(period, palette.muted)


def direction_marker(direction: str) -> str:
    """Arrow plus word, so direction never depends on colour alone."""
    return "▲ LONG" if direction.upper() == "LONG" else "▼ SHORT"


def app_css(palette: Palette) -> str:
    """Page styling that matches the chart surface."""
    return f"""
<style>
  .stApp {{ background: {palette.page}; color: {palette.text_primary}; }}
  /* Streamlit's header ships unthemed and reads as a white band above a dark
     app; paint it with the page colour so the chrome is continuous. */
  header[data-testid="stHeader"] {{
      background: {palette.page};
      border-bottom: 1px solid {palette.border};
  }}
  section[data-testid="stSidebar"] {{ background: {palette.surface}; }}
  section[data-testid="stSidebar"] > div {{ background: {palette.surface}; }}
  .metric-card {{
      background: {palette.surface};
      border: 1px solid {palette.border};
      border-radius: 10px;
      padding: 14px 16px;
  }}
  .metric-label {{
      color: {palette.text_secondary};
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
  }}
  .metric-value {{
      color: {palette.text_primary};
      font-size: 1.5rem;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
  }}
  .metric-note {{ color: {palette.text_secondary}; font-size: 0.8rem; }}
  .pill {{
      display: inline-block; padding: 2px 10px; border-radius: 999px;
      font-size: 0.75rem; font-weight: 600; border: 1px solid {palette.border};
  }}
  .demo-banner {{
      background: {palette.warning}22;
      border: 1px solid {palette.warning};
      border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;
      color: {palette.text_primary};
  }}
  table {{ font-variant-numeric: tabular-nums; }}
</style>
"""
