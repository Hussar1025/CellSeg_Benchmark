# -*- coding: utf-8 -*-
"""
Supplementary figure — dataset features vs segmentation accuracy (two figures, one per panel: supp_fig_features_a / supp_fig_features_b)
  a  Spearman correlation of F1 with four dataset features per method (cell density, transcripts per cell, gene panel size, region cell count);
     each cell shows ρ, P < 0.05 adds * (per-cell n is written to _spearman.csv); empty cells show a thin black dashed line
  b  One cell per method: F1 vs cell density (cells/mm²), colored by platform, with Spearman ρ over all datasets and within MERFISH
Transcripts per cell is taken from the whole sample (scale info in Table S1); density = region reference cell count / area, datasets without area are excluded from the density analysis.
Local run: python supp_fig_features.py (same directory as supp_common.py, robustness_bars.py)
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import supplement_6 as C

OUT_NAME = "supp_fig_features"
MIN_N = 5
FONT = 6
C.style_rc(mpl, FONT)
CMAP = LinearSegmentedColormap.from_list("rb", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"])
FEATS = [("density", "Cell\ndensity"), ("tx_per_cell", "Transcripts\nper cell"),
         ("genes", "Gene panel\nsize"), ("roi_cells", "Cells in\nregion")]


def rho(x, y):
    q = np.isfinite(x) & np.isfinite(y)
    if q.sum() < MIN_N:
        return np.nan, np.nan, int(q.sum())
    r, p = spearmanr(x[q], y[q])
    return r, p, int(q.sum())


def compute():
    acc = C.load_accuracy()
    meta = C.dataset_table(acc)
    d = acc.join(meta, on="ds")
    rows = []
    for m in C.ORDER:
        g = d[d.method == m]
        for col, lab in FEATS:
            r, p, n = rho(g[col].to_numpy(float), g.f1.to_numpy(float))
            rows.append(dict(method=m, feature=lab.replace("\n", " "), rho=r, p=p, n=n))
    R = pd.DataFrame(rows)
    print(R.pivot_table(index="method", columns="feature", values="rho").reindex(C.ORDER).round(2).to_string())
    return R, d


def figure_a(R):
    """Spearman correlation heatmap between F1 and dataset features (single figure)."""
    fig = plt.figure(figsize=(5.4, 4.4))
    axA = fig.add_axes([0.20, 0.12, 0.50, 0.78])
    labs = [f.replace("\n", " ") for _, f in FEATS]
    get = lambda m, f, k: R[(R.method == m) & (R.feature == f)][k].iloc[0]
    M = np.array([[get(m, f, "rho") for f in labs] for m in C.ORDER])
    P = np.array([[get(m, f, "p") for f in labs] for m in C.ORDER])
    Nn = np.array([[get(m, f, "n") for f in labs] for m in C.ORDER])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cm = CMAP.copy(); cm.set_bad("white")
    axA.imshow(np.ma.masked_invalid(M), cmap=cm, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                sig = np.isfinite(P[i, j]) and P[i, j] < 0.05
                axA.text(j, i, f"{M[i, j]:.2f}{'*' if sig else ''}", ha="center", va="center",
                         fontsize=FONT - 1.2, color="white" if abs(M[i, j]) > 0.65 else "black")
            else:
                C.blank_dash(axA, i, j)
    axA.set_xticks(range(len(FEATS))); axA.set_xticklabels([f for _, f in FEATS], fontsize=FONT - 0.5)
    axA.set_yticks(range(len(C.ORDER))); axA.set_yticklabels(C.ORDER)
    axA.tick_params(length=0)
    C.category_strip(fig, axA, C.ORDER)
    C.full_frame(axA)
    axA.set_title("Spearman $\\rho$ between F1 and dataset features", fontsize=FONT + 0.5, pad=4)
    cax = fig.add_axes([0.74, 0.12, 0.022, 0.35])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("Spearman $\\rho$", fontsize=FONT); cb.outline.set_linewidth(0.4); cax.tick_params(labelsize=FONT - 1)
    fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=5, mfc=v, mec="none", label=k) for k, v in C.CAT_COLOR.items()],
               loc="upper left", bbox_to_anchor=(0.735, 0.91), frameon=False, fontsize=FONT, title="Category",
               title_fontsize=FONT)
    C.save(fig, OUT_NAME + "_a")
    plt.close(fig)


def figure_b(d, gap_x=0.08, gap_y=0.11):
    """One cell per method: F1 vs cell density (single figure). Spacing is auto-enlarged until panels no longer overlap."""
    nc = 4
    for attempt in range(8):
        fig = plt.figure(figsize=(7.2, 7.6))
        gw = (0.92 - 3 * gap_x) / nc
        gh = (0.86 - 3 * gap_y) / 4
        gx0, gy0 = 0.07, 0.08
        axes = []
        for idx, m in enumerate(C.ORDER):
            r, c = divmod(idx, nc)
            ax = fig.add_axes([gx0 + c * (gw + gap_x), gy0 + (3 - r) * (gh + gap_y), gw, gh])
            axes.append(ax)
            g = d[(d.method == m)].dropna(subset=["density", "f1"])
            ax.scatter(g.density, g.f1, s=8, c=[C.PLAT_COLOR.get(p, "#9E9E9E") for p in g.platform],
                       edgecolor="black", linewidth=0.25, zorder=3)
            r_all, _, n_all = rho(g.density.to_numpy(float), g.f1.to_numpy(float))
            gm = g[g.platform == "merfish"]
            r_mf, _, n_mf = rho(gm.density.to_numpy(float), gm.f1.to_numpy(float))
            line1 = f"$\\rho$ = {r_all:.2f} (n = {n_all})" if np.isfinite(r_all) else f"n = {n_all}"
            line2 = f"MERFISH $\\rho$ = {r_mf:.2f} (n = {n_mf})" if np.isfinite(r_mf) else ""
            ax.set_xscale("log"); ax.set_ylim(-0.03, 1.05)
            ax.set_xticks([1e3, 3e3, 1e4, 3e4]); ax.set_xticklabels(["1k", "3k", "10k", "30k"])
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.set_title(m, fontsize=FONT, pad=16 if line2 else 10)
            ax.text(0.5, 1.03, line1 + ("\n" + line2 if line2 else ""), transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=FONT - 1.6, linespacing=1.15)
            ax.tick_params(labelsize=FONT - 1.5, which="both")
            ax.set_ylabel("F1", fontsize=FONT - 1)
            ax.set_xlabel("Cell density (cells/mm$^2$)", fontsize=FONT - 1)
            C.full_frame(ax)
        hits = C.check_no_overlap(fig, axes)
        if hits == 0:
            break
        print(f"  [b] layout attempt {attempt + 1} has {hits} overlaps, enlarging spacing and retrying")
        plt.close(fig)
        gap_x, gap_y = gap_x + 0.015, gap_y + 0.015
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, mfc=C.PLAT_COLOR[p], mec="black", mew=0.3,
                               label=C.PLAT_NAME[p]) for p in C.PLAT_ORDER],
               loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=5, frameon=False, fontsize=FONT)
    C.save(fig, OUT_NAME + "_b")
    plt.close(fig)


def main():
    R, d = compute()
    figure_a(R)
    figure_b(d)
    R.round(4).to_csv(f"{C.OUT_DIR}/{OUT_NAME}_spearman.csv", index=False)
    print("Saved:", f"{C.OUT_DIR}/{OUT_NAME}_spearman.csv")


if __name__ == "__main__":
    main()