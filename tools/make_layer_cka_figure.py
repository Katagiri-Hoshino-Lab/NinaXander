#!/usr/bin/env python3
"""Render the 32x32 raw-representation correspondence maps for the paper.

Reads the canonical layer-correspondence table and writes a two-panel figure
(linear CKA, per-pair best affine fit R^2) so the figure is always derived from
the same data the prose statistics are checked against.
"""

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


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
        "font.size": 7,
        "text.color": ink,
        "axes.edgecolor": "#b0b0b0",
        "xtick.color": ink,
        "ytick.color": ink,
    })

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), dpi=300)
    labels = {
        "ja": ("線形 CKA", "適合 $R^2$", "Pythia 層 $k$", "RWKV 層 $j$"),
        "en": ("Linear CKA", "Fit $R^2$", "Pythia layer $k$", "RWKV layer $j$"),
    }[args.lang]
    panels = (
        (axes[0], cka, labels[0], 0.0, float(cka.max())),
        (axes[1], fit, labels[1], float(np.floor(fit.min() * 10) / 10), float(fit.max())),
    )
    ticks = [0, 8, 16, 24, 31]
    for ax, mat, title, vmin, vmax in panels:
        im = ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
        ax.plot([0, 31], [0, 31], linestyle=(0, (2, 2)), linewidth=0.6, color="#909090")
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xlabel(labels[2], fontsize=7, labelpad=1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(length=1.5, pad=1, labelsize=6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(length=1.5, pad=1, labelsize=6)
        cbar.outline.set_linewidth(0.5)
    axes[0].set_ylabel(labels[3], fontsize=7, labelpad=1)
    axes[1].set_yticklabels([])

    fig.tight_layout(pad=0.4)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} (CKA max {cka.max():.3f}, fit range {fit.min():.3f}-{fit.max():.3f})")


if __name__ == "__main__":
    main()
