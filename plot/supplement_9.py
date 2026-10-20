# -*- coding: utf-8 -*-
"""
Supplementary figure — segmentation performance stratified by data category (strata laid out as side-by-side panels in one figure supp_fig_stratified)
  Platform (Xenium / MERFISH / CosMx / Stereo-seq / STARmap), gene panel (targeted ≤600 / high-throughput 1,000–6,600 / whole-transcriptome),
  image input (DAPI / nuclear + membrane / H&E / ssDNA), species (human / mouse); to drop a stratum, delete the corresponding row in GROUPS
Each cell is the median F1 of that method on that category of datasets (per-cell dataset count n is written to _data.csv); cells with a single dataset get † after the value;
cells with no dataset are left blank and get a thin black dashed diagonal (solid 2 pt, empty 1 pt);
a column empty across the whole stratum is dropped; a row empty across all strata is dropped (all panels share one set of rows to stay aligned).
Dataset species and image input are corrected per Table S1 (see supp_common.py).
Local run: python supp_fig_stratified.py (same directory as supp_common.py, robustness_bars.py)
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import supplement_6 as C

OUT_NAME = "supp_fig_stratified"
FONT = 6
C.style_rc(mpl, FONT)
CMAP = LinearSegmentedColormap.from_list("f1", ["#FFF5EB", "#FDD0A2", "#FD8D3C", "#D94801", "#7F2704"])
GROUPS = [   # (file suffix, title, column name, value order, display name)
    ("platform", "Platform", "platform", ["xenium", "merfish", "cosmx", "stereo", "starmap"], C.PLAT_NAME),
    ("panel", "Gene panel", "panel", ["Targeted (≤600)", "High-plex (1,000–6,600)", "Whole transcriptome"], None),
    ("image", "Image input", "stain", ["DAPI", "Nuclear + membrane", "H&E", "ssDNA"], None),
    ("species", "Species", "species", ["Human", "Mouse"], None),
]
CELL_W, CELL_H = 0.40, 0.205      # cell width, height (inch)
GAP_IN = 0.28                      # gap between panels (inch)
DIAG_LW = 0.4                      # diagonal line width of empty cells (pt)
DIAG_DASH = (2.0, 1.0)             # dashed line: solid 2 pt, empty 1 pt


def blank_diag(ax, i, j):
    """Draw a thin black dashed diagonal in empty cells; matplotlib scales dashes by line width, so divide by line width to get actual 2 pt / 1 pt lengths."""
    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i - 0.5], color="black", lw=DIAG_LW, zorder=4,
            ls=(0, (DIAG_DASH[0] / DIAG_LW, DIAG_DASH[1] / DIAG_LW)), solid_capstyle="butt",
            dash_capstyle="butt")


def stats(d, key, vals):
    med = np.full((len(C.ORDER), len(vals)), np.nan)
    num = np.zeros_like(med, dtype=int)
    for i, m in enumerate(C.ORDER):
        g = d[d.method == m]
        for j, v in enumerate(vals):
            x = g[g[key] == v].f1.dropna()
            num[i, j] = len(x)
            if len(x):
                med[i, j] = x.median()
    return med, num


def main():
    acc = C.load_accuracy()
    meta = C.dataset_table(acc)
    d = acc.join(meta, on="ds")

    blocks, rows_any, csv = [], np.zeros(len(C.ORDER), bool), []
    for suffix, title, key, vals, names in GROUPS:
        labels = [(names or {}).get(v, v) for v in vals]
        med, num = stats(d, key, vals)
        for i, m in enumerate(C.ORDER):
            for j, lab in enumerate(labels):
                csv.append(dict(group=title, level=lab, method=m, n=int(num[i, j]),
                                median_f1=med[i, j]))
        keep_c = num.sum(axis=0) > 0
        if (~keep_c).any():
            print(f"[{title}] removed empty columns:", [l for l, k in zip(labels, keep_c) if not k])
        rows_any |= num[:, keep_c].sum(axis=1) > 0
        blocks.append((title, [l for l, k in zip(labels, keep_c) if k], med[:, keep_c], num[:, keep_c]))
    if (~rows_any).any():
        print("removed empty rows:", [m for m, k in zip(C.ORDER, rows_any) if not k])
    methods = [m for m, k in zip(C.ORDER, rows_any) if k]
    blocks = [(t, l, me[rows_any], nu[rows_any]) for t, l, me, nu in blocks if len(l)]

    L, B, Rm, Tm = 1.0, 1.05, 1.6, 0.45
    grid_w = sum(len(l) for _, l, _, _ in blocks) * CELL_W + GAP_IN * (len(blocks) - 1)
    fig_w, fig_h = L + grid_w + Rm, B + len(methods) * CELL_H + Tm
    fig = plt.figure(figsize=(fig_w, fig_h))
    norm = Normalize(0, 1)
    cm = CMAP.copy(); cm.set_bad("white")
    x = L
    axes = []
    for k, (title, labels, med, num) in enumerate(blocks):
        w = len(labels) * CELL_W
        ax = fig.add_axes([x / fig_w, B / fig_h, w / fig_w, len(methods) * CELL_H / fig_h])
        axes.append(ax)
        ax.imshow(np.ma.masked_invalid(med), cmap=cm, norm=norm, aspect="auto")
        for i in range(med.shape[0]):
            for j in range(med.shape[1]):
                if np.isfinite(med[i, j]):
                    ax.text(j, i, f"{med[i, j]:.2f}" + ("$^{\\dagger}$" if num[i, j] == 1 else ""),
                            ha="center", va="center", fontsize=FONT - 1.2,
                            color="white" if med[i, j] > 0.6 else "black")
                else:
                    blank_diag(ax, i, j)
        ax.set_xlim(-0.5, len(labels) - 0.5); ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=FONT - 0.5)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods if k == 0 else [])
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=FONT + 0.5, pad=4)
        n_tx = sum(C.CAT_OF[m] != "Image-based" for m in methods)
        if 0 < n_tx < len(methods):
            ax.axhline(n_tx - 0.5, color="black", lw=0.6)
        C.full_frame(ax)
        x += w + GAP_IN
    C.category_strip(fig, axes[0], methods)
    x_r = (L + grid_w + 0.18) / fig_w
    cax = fig.add_axes([x_r, B / fig_h, 0.08 / fig_w, len(methods) * CELL_H * 0.40 / fig_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.set_label("Median F1", fontsize=FONT); cb.outline.set_linewidth(0.4); cax.tick_params(labelsize=FONT - 1)
    fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=5, mfc=v, mec="none", label=k) for k, v in C.CAT_COLOR.items()],
               loc="upper left", bbox_to_anchor=(x_r - 0.01, (B + len(methods) * CELL_H) / fig_h), frameon=False,
               fontsize=FONT, title="Category", title_fontsize=FONT)
    C.save(fig, OUT_NAME)
    plt.close(fig)
    pd.DataFrame(csv).round(4).to_csv(f"{C.OUT_DIR}/{OUT_NAME}_data.csv", index=False)
    print("Saved:", f"{C.OUT_DIR}/{OUT_NAME}_data.csv")


if __name__ == "__main__":
    main()