# -*- coding: utf-8 -*-
"""
Supplementary figure — computing resource vs data scale (two figures, one per panel: supp_fig_scaling_a / supp_fig_scaling_b)
  a  Scaling exponent heatmap: log-log fit slope k of runtime / peak CPU memory vs
     region cell count, transcript count (est.), and area for each method (R² shown in each cell)
  b  One cell per method: log-log scatter of runtime vs region transcript count (est.) with fit line, colored by platform
Transcript count = region reference cell count × whole-sample transcripts per cell (scale info in Table S1); an estimate.
Local run: python supp_fig_scaling.py (same directory as supp_common.py, robustness_bars.py)
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import supplement_6 as C

OUT_NAME = "supp_fig_scaling"
MIN_N = 5                        # fits with fewer than 5 datasets are not reported
FONT = 6
C.style_rc(mpl, FONT)
CMAP = LinearSegmentedColormap.from_list("k", ["#F7F7F7", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
XVARS = [("roi_cells", "cells"), ("roi_tx_est", "transcripts"), ("area_mm2", "area")]
YVARS = [("runtime_min", "Runtime"), ("cpu_gib", "Peak CPU memory")]


def fit(x, y):
    q = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if q.sum() < MIN_N:
        return np.nan, np.nan, int(q.sum())
    lx, ly = np.log10(x[q]), np.log10(y[q])
    k, b = np.polyfit(lx, ly, 1)
    r2 = 1 - np.sum((ly - (k * lx + b)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return k, r2, int(q.sum())


def compute():
    acc = C.load_accuracy()
    meta = C.dataset_table(acc)
    res = C.load_resources().join(meta, on="ds")
    rows = []
    for m in C.ORDER:
        d = res[res.method == m]
        for ycol, ylab in YVARS:
            for xcol, xlab in XVARS:
                k, r2, n = fit(d[xcol].to_numpy(float), d[ycol].to_numpy(float))
                rows.append(dict(method=m, y=ylab, x=xlab, k=k, r2=r2, n=n))
    K = pd.DataFrame(rows)
    print(K.pivot_table(index="method", columns=["y", "x"], values="k").reindex(C.ORDER).round(2).to_string())
    return K, res


def figure_a(K):
    """Scaling exponent heatmap (single figure)."""
    fig = plt.figure(figsize=(6.2, 4.4))
    axA = fig.add_axes([0.17, 0.08, 0.55, 0.80])
    cols = [(y, x) for _, y in YVARS for _, x in XVARS]
    M = np.array([[K[(K.method == m) & (K.y == y) & (K.x == x)].k.iloc[0] for (y, x) in cols] for m in C.ORDER])
    R = np.array([[K[(K.method == m) & (K.y == y) & (K.x == x)].r2.iloc[0] for (y, x) in cols] for m in C.ORDER])
    norm = Normalize(vmin=0, vmax=max(1.5, np.nanmax(M)))
    cm = CMAP.copy(); cm.set_bad("white")
    axA.imshow(np.ma.masked_invalid(M), cmap=cm, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                dark = norm(M[i, j]) > 0.6
                axA.text(j, i, f"{M[i, j]:.2f}\n$R^2$ {R[i, j]:.2f}", ha="center", va="center",
                         fontsize=FONT - 1.8, color="white" if dark else "black", linespacing=0.9)
            else:
                C.blank_dash(axA, i, j)
    axA.set_xticks(range(len(cols)))
    axA.set_xticklabels([f"vs {x}" for (_, x) in cols], fontsize=FONT - 0.5)
    axA.set_yticks(range(len(C.ORDER))); axA.set_yticklabels(C.ORDER)
    axA.axvline(2.5, color="black", lw=0.8)
    for xc, lab in [(1, "Runtime"), (4, "Peak CPU memory")]:
        axA.text(xc, -1.0, lab, ha="center", va="bottom", fontsize=FONT + 0.5)
    axA.tick_params(length=0)
    C.full_frame(axA)
    C.category_strip(fig, axA, C.ORDER)
    cax = fig.add_axes([0.75, 0.08, 0.018, 0.35])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("Scaling exponent $k$ (log–log slope)", fontsize=FONT)
    cb.outline.set_linewidth(0.4); cax.tick_params(labelsize=FONT - 1)
    fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=5, mfc=v, mec="none", label=k) for k, v in C.CAT_COLOR.items()],
               loc="upper left", bbox_to_anchor=(0.745, 0.89), frameon=False, fontsize=FONT, title="Category",
               title_fontsize=FONT)
    C.save(fig, OUT_NAME + "_a")
    plt.close(fig)


def figure_b(res, gap_x=0.08, gap_y=0.11):
    """One cell per method: runtime vs transcript count (single figure). Spacing is auto-enlarged until panels no longer overlap."""
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
            d = res[(res.method == m)].dropna(subset=["roi_tx_est", "runtime_min"])
            ax.scatter(d.roi_tx_est, d.runtime_min, s=8, c=[C.PLAT_COLOR.get(p, "#9E9E9E") for p in d.platform],
                       edgecolor="black", linewidth=0.25, zorder=3)
            k, r2, n = fit(d.roi_tx_est.to_numpy(float), d.runtime_min.to_numpy(float))
            ax.set_xscale("log"); ax.set_yscale("log")
            if np.isfinite(k):
                xs = np.logspace(np.log10(d.roi_tx_est.min()), np.log10(d.roi_tx_est.max()), 20)
                lx, ly = np.log10(d.roi_tx_est), np.log10(d.runtime_min)
                b = ly.mean() - k * lx.mean()
                ax.plot(xs, 10 ** (b + k * np.log10(xs)), color="black", lw=0.7, zorder=2)
                ax.text(0.5, 1.03, f"$k$ = {k:.2f}, $R^2$ = {r2:.2f}", transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=FONT - 1.4)
            ax.set_title(m, fontsize=FONT, pad=10)
            ax.tick_params(labelsize=FONT - 1.5, which="both")
            ax.set_ylabel("Runtime (min)", fontsize=FONT - 1)
            ax.set_xlabel("Transcripts in region (est.)", fontsize=FONT - 1)
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
    K, res = compute()
    figure_a(K)
    figure_b(res)
    K.round(3).to_csv(f"{C.OUT_DIR}/{OUT_NAME}_slopes.csv", index=False)
    print("Saved:", f"{C.OUT_DIR}/{OUT_NAME}_slopes.csv")


if __name__ == "__main__":
    main()