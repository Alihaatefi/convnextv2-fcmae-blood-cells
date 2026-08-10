"""Shared plotting tokens.

Categorical slots and chrome come from a validated design-system palette; the
three-slot subsets used here were checked with an all-pairs CVD/contrast
validator and pass in both light and dark mode. Figures are rendered twice so
that GitHub's dark theme gets a version stepped for a dark surface rather than
an inverted light one.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        # single-hue sequential ramp (blue 100 -> 700) for magnitude encodings
        "sequential": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "sequential": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
    },
}

SUFFIX = {"light": "", "dark": "_dark"}


def sequential_cmap(theme):
    return LinearSegmentedColormap.from_list("seq", theme["sequential"])


def style_axes(ax, theme, grid_axis="both"):
    """Recessive hairline chrome: no top/right spines, solid hairline grid, muted ticks."""
    ax.set_facecolor(theme["surface"])
    if grid_axis != "none":
        ax.grid(True, which="major", axis=grid_axis, color=theme["grid"],
                linewidth=0.8, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    return ax
