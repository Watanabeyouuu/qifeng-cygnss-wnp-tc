#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared fonts, colours and panel labels for the study figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

# Paul Tol's bright palette: https://personal.sron.nl/~pault/
PALETTE = {
    "blue":      "#4477AA",
    "cyan":      "#66CCEE",
    "green":     "#228833",
    "yellow":    "#CCBB44",
    "red":       "#EE6677",
    "vermilion": "#EE6677",   # alias: scripts that asked for "vermilion"
    "orange":    "#EE8866",
    "purple":    "#AA3377",
    "skyblue":   "#66CCEE",
    "teal":      "#44AA99",
    "grey":      "#BBBBBB",
    "black":     "#000000",
}

# Default colour cycle.
CYCLE = [PALETTE[k] for k in
         ("blue", "red", "green", "yellow", "purple", "cyan", "orange", "grey")]

# Colormaps.
SEQ_CMAP = "viridis"       # perceptually-uniform sequential (fields)
DIV_CMAP = "RdBu_r"        # balanced diverging (difference maps)
SCATTER_CMAP = "RdBu_r"    # value-coloured scatter points (per request)

# Reference column widths (mm) for Nature: single 89, 1.5-col 120, double 183.
MM = 1.0 / 25.4
COL1 = 89 * MM      # ~3.50 in
COL15 = 120 * MM    # ~4.72 in
COL2 = 183 * MM     # ~7.20 in

_SANS = ["Nimbus Sans", "Helvetica", "Arial", "TeX Gyre Heros", "DejaVu Sans"]


def apply():
    """Apply the shared figure settings."""
    mpl.rcParams.update({
        # ---- typography ----
        "font.family": "sans-serif",
        "font.sans-serif": _SANS,
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        # ---- axes / lines ----
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#222222",
        "axes.titlepad": 4.0,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",   # title reads "a  Title" next to the panel label
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "axes.autolimit_mode": "round_numbers",
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "patch.linewidth": 0.6,
        "grid.linewidth": 0.5,
        "grid.color": "#d9d9d9",
        # ---- ticks ----
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        # ---- legend ----
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.4,
        # ---- text/colour ----
        "text.color": "#222222",
        # ---- figure / output ----
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,   # embed editable TrueType (no Type-3) text
        "ps.fonttype": 42,
    })


def despine(ax, which=("top", "right")):
    """Hide the requested spines (defaults to the Nature top+right removal)."""
    for s in which:
        if s in ax.spines:
            ax.spines[s].set_visible(False)


def panel_label(ax, letter, dx_pt=-16.0, dy_pt=4.0, fontsize=11, weight="bold",
                color="black"):
    """Draw a **bold** panel label just outside the top-left corner of *ax*.

    The offset is specified in *points* from the axes' top-left corner, so the
    placement is stable regardless of the panel's physical size. With
    left-located titles the label sits just left of the title (``a  Title``).
    ``bbox_inches='tight'`` keeps the label inside the saved canvas.
    """
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(dx_pt, dy_pt), textcoords="offset points",
                fontsize=fontsize, fontweight=weight, color=color,
                va="bottom", ha="left", annotation_clip=False, zorder=10)


def label_axes(axes, letters="abcdefghijklmnop", **kw):
    """Convenience: stamp a/b/c... on a flat iterable of axes in order."""
    for ax, ltr in zip(list(axes), letters):
        panel_label(ax, ltr, **kw)


def finalize(fig, out, tight=True, **savefig_kw):
    """tight_layout (optional) + savefig with the style defaults."""
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(out, **savefig_kw)
    plt.close(fig)
