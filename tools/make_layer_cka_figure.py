#!/usr/bin/env python3
"""Render the 32x32 raw-representation correspondence maps for the paper.

Reads the canonical layer-correspondence table and writes a two-panel figure
(linear CKA, per-pair best affine fit R^2) so the figure is always derived from
the same data the prose statistics are checked against.

The papers place this figure as a full-width figure* at \\textwidth (503 pt =
6.96 in) with \\includegraphics[width=\\textwidth], i.e. at scale 1:1, so the
canvas is laid out at exactly that width and every text element is set at no
less than MIN_TEXT_PT (one step below the body text), apart from the math
superscript in R^2.
"""

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

FIG_W_IN = 503.0 / 72.27  # \textwidth of the IPSJ technical-report layout
FIG_H_IN = 3.0
TICK_PT = 10.0
LABEL_PT = 10.0
TITLE_PT = 10.5
MIN_TEXT_PT = 9.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", default="artifacts/metrics/tables")
    parser.add_argument("--out", default="paper/fig_layer_cka.pdf")
    parser.add_argument("--lang", default="ja", choices=["ja", "en"])
    args = parser.parse_args()

    for candidate in font_manager.findSystemFonts(fontpaths=None):
        if "NotoSansCJK" in candidate:
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = "Noto Sans CJK JP"
            break

    cka = np.zeros((32, 32))
    fit = np.zeros((32, 32))
    with open(pathlib.Path(args.table_dir) / "paper_layer_correspondence.csv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            s, t = int(row["source_layer"]), int(row["target_layer"])
            cka[s, t] = float(row["linear_cka"])
            fit[s, t] = float(row["best_linear_r2"])

    ink = "#404040"
    plt.rcParams.update({
        "font.size": TICK_PT,
        "text.color": ink,
        "axes.edgecolor": "#b0b0b0",
        "xtick.color": ink,
        "ytick.color": ink,
    })

    # Fixed geometry in inches: two square panels, each followed by its own
    # colour bar, centred on the full-width canvas.
    left_margin = 0.50  # room for the y label and the y tick labels
    bottom_margin = 0.50  # x tick labels and x label
    top_margin = 0.30  # panel titles
    cbar_gap, cbar_w = 0.08, 0.13
    cbar_labels = 0.40  # colour-bar tick labels such as 0.90
    side = FIG_H_IN - bottom_margin - top_margin
    used = left_margin + 2 * (side + cbar_gap + cbar_w + cbar_labels)
    spare = (FIG_W_IN - used) / 3.0  # split over the left edge, the middle and the right edge
    if spare < 0:
        raise SystemExit(f"layout does not fit: needs {used:.2f} in of {FIG_W_IN:.2f} in")

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)

    def rect(x_in, y_in, w_in, h_in):
        return [x_in / FIG_W_IN, y_in / FIG_H_IN, w_in / FIG_W_IN, h_in / FIG_H_IN]

    x0 = spare + left_margin
    x1 = x0 + side + cbar_gap + cbar_w + cbar_labels + spare
    axes = []
    caxes = []
    for x in (x0, x1):
        axes.append(fig.add_axes(rect(x, bottom_margin, side, side)))
        caxes.append(fig.add_axes(rect(x + side + cbar_gap, bottom_margin, cbar_w, side)))

    labels = {
        "ja": ("線形 CKA", "適合 $R^2$", "Pythia 層 $k$", "RWKV 層 $j$"),
        "en": ("Linear CKA", "Fit $R^2$", "Pythia layer $k$", "RWKV layer $j$"),
    }[args.lang]
    panels = (
        (axes[0], caxes[0], cka, labels[0], 0.0, float(cka.max())),
        (axes[1], caxes[1], fit, labels[1], float(np.floor(fit.min() * 10) / 10), float(fit.max())),
    )
    ticks = [0, 8, 16, 24, 31]
    for ax, cax, mat, title, vmin, vmax in panels:
        im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, origin="upper",
                       interpolation="nearest", aspect="auto")
        ax.plot([0, 31], [0, 31], linestyle=(0, (2, 2)), linewidth=0.6, color="#909090")
        ax.set_title(title, fontsize=TITLE_PT, pad=4)
        ax.set_xlabel(labels[2], fontsize=LABEL_PT, labelpad=2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(length=2.5, pad=2, labelsize=TICK_PT)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(length=2.5, pad=2, labelsize=TICK_PT)
        cbar.outline.set_linewidth(0.6)
    axes[0].set_ylabel(labels[3], fontsize=LABEL_PT, labelpad=2)
    axes[1].set_yticklabels([])

    # Guard the fixed layout: every text must be legible and everything drawn must
    # stay on the canvas (the tight bbox only counts artists that are drawn).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text() and text.get_fontsize() < MIN_TEXT_PT:
            raise SystemExit(f"text {text.get_text()!r} is {text.get_fontsize()} pt < {MIN_TEXT_PT} pt")
    drawn = fig.get_tightbbox(renderer)  # inches
    if drawn.x0 < 0 or drawn.y0 < 0 or drawn.x1 > FIG_W_IN or drawn.y1 > FIG_H_IN:
        raise SystemExit(f"drawn content {drawn.bounds} exceeds the {FIG_W_IN:.2f}x{FIG_H_IN:.2f} in canvas")

    fig.savefig(args.out)
    print(f"wrote {args.out} ({FIG_W_IN:.2f}x{FIG_H_IN:.2f} in; "
          f"CKA max {cka.max():.3f}, fit range {fit.min():.3f}-{fit.max():.3f})")


if __name__ == "__main__":
    main()
