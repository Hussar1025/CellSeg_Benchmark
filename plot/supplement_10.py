# -*- coding: utf-8 -*-
"""
Supplementary figure — all heatmaps stacked vertically in one figure (columns = methods, small gap between heatmaps)
  1  Runtime scaling          log-log slope k of runtime vs cell count / transcript count (est.) / area, R² in parentheses
  2  Memory scaling           same slope for peak CPU memory
  3  Correlation with F1      Spearman ρ of F1 vs cell density / transcripts per cell / gene panel size / region cell count (P < 0.05 adds *)
  4–7 Platform / Gene panel / Image input / Species   median F1 on datasets of each category († when only one dataset)
Empty cells get a thin black dashed diagonal (solid 2 pt, empty 1 pt); a column empty across all methods is dropped, a row empty across all heatmaps is dropped.
All text is Arial Regular. After drawing, auto-check for pairwise text overlaps and cell text overflowing its cell; on problems enlarge cell width and spacing and redraw.
Values come from the same three standalone scripts (their compute functions are called directly).
Local run: python supp_fig_heatmaps_combined.py
(same directory as supp_common.py, supp_fig_scaling.py, supp_fig_features.py, supp_fig_stratified.py, robustness_bars.py)
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D

import supplement_6 as C
import supplement_7 as S
import supplement_8 as F
import supplement_9 as T

OUT_NAME = "supp_fig_heatmaps_combined"
FONT = 7.5                         # base font size (pt)
CELL_FONT = 6.5                    # cell text font size
CELL_W, CELL_H = 0.34, 0.24        # cell width, height (inch); auto-enlarged on overlap
GAP_IN = 0.30                      # gap between stacked heatmaps (inch, includes block title); auto-enlarged on overlap
DIAG_LW, DIAG_DASH = 0.4, (2.0, 1.0)

mpl.rcParams.update({
    "font.family": "Arial", "font.sans-serif": ["Arial"], "font.size": FONT, "font.weight": "normal",
    "font.style": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial", "mathtext.bf": "Arial",
    "mathtext.sf": "Arial", "mathtext.default": "regular",
    "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

CM_K = LinearSegmentedColormap.from_list("k", ["#F7F7F7", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
CM_R = LinearSegmentedColormap.from_list("rb", ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"])
CM_F = LinearSegmentedColormap.from_list("f1", ["#FFF5EB", "#FDD0A2", "#FD8D3C", "#D94801", "#7F2704"])


def blank_diag(ax, i, j):
    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i - 0.5], color="black", lw=DIAG_LW, zorder=4,
            ls=(0, (DIAG_DASH[0] / DIAG_LW, DIAG_DASH[1] / DIAG_LW)), dash_capstyle="butt")


# ======================== assemble data for each heatmap ========================
def build_blocks():
    K, _ = S.compute()
    R, _ = F.compute()
    acc = C.load_accuracy()
    meta = C.dataset_table(acc)
    d = acc.join(meta, on="ds")
    blocks = []          # (title, column name, value matrix, cell text matrix, colormap key)
    for ylab, title in [("Runtime", "Runtime scaling\nk (R²)"), ("Peak CPU memory", "Memory scaling\nk (R²)")]:
        cols = ["cells", "transcripts", "area"]
        V = np.array([[K[(K.method == m) & (K.y == ylab) & (K.x == x)].k.iloc[0] for x in cols] for m in C.ORDER])
        Q = np.array([[K[(K.method == m) & (K.y == ylab) & (K.x == x)].r2.iloc[0] for x in cols] for m in C.ORDER])
        txt = [[f"{V[i, j]:.2f}\n({Q[i, j]:.2f})" if np.isfinite(V[i, j]) else "" for j in range(3)]
               for i in range(len(C.ORDER))]
        blocks.append((title, ["vs cells", "vs transcripts", "vs area"], V, txt, "k"))
    feats = [f.replace("\n", " ") for _, f in F.FEATS]
    V = np.array([[R[(R.method == m) & (R.feature == f)].rho.iloc[0] for f in feats] for m in C.ORDER])
    P = np.array([[R[(R.method == m) & (R.feature == f)].p.iloc[0] for f in feats] for m in C.ORDER])
    txt = [[(f"{V[i, j]:.2f}" + ("*" if np.isfinite(P[i, j]) and P[i, j] < 0.05 else "")) if np.isfinite(V[i, j])
            else "" for j in range(len(feats))] for i in range(len(C.ORDER))]
    blocks.append(("Correlation with F1\nSpearman ρ", feats, V, txt, "rho"))
    for suffix, title, key, vals, names in T.GROUPS:
        labels = [(names or {}).get(v, v) for v in vals]
        med, num = T.stats(d, key, vals)
        txt = [[(f"{med[i, j]:.2f}" + ("†" if num[i, j] == 1 else "")) if np.isfinite(med[i, j]) else ""
                for j in range(len(vals))] for i in range(len(C.ORDER))]
        blocks.append((f"{title}\nmedian F1", labels, med, txt, "f1"))
    # drop all-empty columns and rows
    out = []
    rows_any = np.zeros(len(C.ORDER), bool)
    for title, labels, V, txt, key in blocks:
        keep_c = np.isfinite(V).any(axis=0)
        if (~keep_c).any():
            print(f"[{title}] removed empty columns:", [l for l, k in zip(labels, keep_c) if not k])
        V = V[:, keep_c]
        txt = [[t for t, k in zip(row, keep_c) if k] for row in txt]
        labels = [l for l, k in zip(labels, keep_c) if k]
        rows_any |= np.isfinite(V).any(axis=1)
        if labels:
            out.append([title, labels, V, txt, key])
    methods = [m for m, k in zip(C.ORDER, rows_any) if k]
    if (~rows_any).any():
        print("removed empty rows:", [m for m, k in zip(C.ORDER, rows_any) if not k])
    for b in out:
        b[2] = b[2][rows_any]
        b[3] = [row for row, k in zip(b[3], rows_any) if k]
    return methods, out


NORMS = {"k": (CM_K, None), "rho": (CM_R, TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)), "f1": (CM_F, Normalize(0, 1))}
CB_LABEL = {"k": "Scaling exponent k", "rho": "Spearman ρ", "f1": "Median F1"}


# ======================== plotting (vertical layout: rows = metrics or categories, columns = methods) ========================
def method_strip(fig, ax, methods, height_in=0.07, gap_in=0.03):
    """Draw a category color strip aligned with the columns below the last heatmap, and push method names below the strip."""
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    sx = fig.add_axes([pos.x0, pos.y0 - (gap_in + height_in) / H, pos.width, height_in / H])
    for j, m in enumerate(methods):
        sx.add_patch(plt.Rectangle((j - 0.5, 0), 1, 1, color=C.CAT_COLOR[C.CAT_OF[m]], lw=0))
    sx.set_xlim(-0.5, len(methods) - 0.5); sx.set_ylim(0, 1); sx.axis("off")
    ax.tick_params(axis="x", pad=(height_in + gap_in) * 72 + 3)
    return sx


def draw(methods, blocks, cell_w, cell_h, gap):
    kmax = max(1.5, np.nanmax(np.concatenate([b[2].ravel() for b in blocks if b[4] == "k"])))
    norms = {"k": Normalize(0, kmax), "rho": NORMS["rho"][1], "f1": NORMS["f1"][1]}
    nrow = sum(len(b[1]) for b in blocks)
    grid_w = len(methods) * cell_w
    grid_h = nrow * cell_h + gap * (len(blocks) - 1)
    L, Rm, Tm, Bm = 1.75, 1.6, 0.35, 1.45             # left (row names), right (color bars and legend), top (first block title), bottom (method names)
    fig_w, fig_h = L + grid_w + Rm, Tm + grid_h + Bm
    fig = plt.figure(figsize=(fig_w, fig_h))
    texts_cells, axes, spans = [], [], {}
    y_top = fig_h - Tm                                # top edge of current block (inch)
    n_tx = sum(C.CAT_OF[m] != "Image-based" for m in methods)
    for k, (title, labels, V, txt, key) in enumerate(blocks):
        h = len(labels) * cell_h
        ax = fig.add_axes([L / fig_w, (y_top - h) / fig_h, grid_w / fig_w, h / fig_h])
        axes.append(ax)
        spans.setdefault(key, []).append((y_top - h, h))
        Vt = V.T
        cm = NORMS[key][0].copy(); cm.set_bad("white")
        ax.imshow(np.ma.masked_invalid(Vt), cmap=cm, norm=norms[key], aspect="auto")
        for i in range(Vt.shape[0]):              # row = metric or category
            for j in range(Vt.shape[1]):          # column = method
                if np.isfinite(Vt[i, j]):
                    dark = norms[key](Vt[i, j]) > 0.62 if key != "rho" else abs(Vt[i, j]) > 0.65
                    t = ax.text(j, i, txt[j][i], ha="center", va="center", fontsize=CELL_FONT,
                                color="white" if dark else "black", linespacing=0.95)
                    texts_cells.append((t, ax, i, j))
                else:
                    blank_diag(ax, i, j)
        ax.set_xlim(-0.5, len(methods) - 0.5); ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=FONT - 0.5)
        ax.set_xticks(range(len(methods)))
        if k == len(blocks) - 1:
            ax.set_xticklabels(methods, rotation=90, fontsize=FONT)
        else:
            ax.set_xticklabels([])
        ax.tick_params(length=0, pad=2)
        ax.set_title(title.replace("\n", "  ·  "), fontsize=FONT, loc="left", pad=3)
        if 0 < n_tx < len(methods):
            ax.axvline(n_tx - 0.5, color="black", lw=0.6)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        y_top -= h + gap
    strip = method_strip(fig, axes[-1], methods)
    # color bars: one vertical color bar per colormap, to the right of its heatmap group
    cbars = []
    x_cb = (L + grid_w + 0.18) / fig_w
    for key, sp in spans.items():
        y0 = min(a for a, _ in sp); y1 = max(a + b for a, b in sp)
        hcb = min(1.6, y1 - y0)
        cy = y0 + (y1 - y0 - hcb) / 2
        cax = fig.add_axes([x_cb, cy / fig_h, 0.08 / fig_w, hcb / fig_h])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norms[key], cmap=NORMS[key][0]), cax=cax)
        cb.set_label(CB_LABEL[key], fontsize=FONT - 0.5, labelpad=3)
        cb.outline.set_linewidth(0.4); cax.tick_params(labelsize=FONT - 1.5, length=2, width=0.4)
        cbars.append(cax)
    leg = fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=6, mfc=v, mec="none", label=k)
                              for k, v in C.CAT_COLOR.items()],
                     loc="upper left", bbox_to_anchor=(x_cb - 0.01, (Bm - 0.1) / fig_h),
                     frameon=False, fontsize=FONT, title="Category", title_fontsize=FONT)
    leg._strip = strip
    return fig, axes, cbars, leg, texts_cells


def problems(fig, axes, cbars, leg, texts_cells):
    """Return (number of pairwise text overlaps, number of cell texts overflowing their cell)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    items = []
    for k, ax in enumerate(axes):
        if ax.title.get_text():
            items.append(ax.title.get_window_extent(rend))
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text():
                items.append(t.get_window_extent(rend))
    for cax in cbars:
        items.append(cax.get_tightbbox(rend))
    items.append(leg.get_window_extent(rend))
    items.append(leg._strip.get_window_extent(rend))        # category strip is also checked
    n_ov = sum(items[a].overlaps(items[b]) for a in range(len(items)) for b in range(a + 1, len(items)))
    n_out = 0
    for t, ax, i, j in texts_cells:
        bb = t.get_window_extent(rend)
        (x0, y0), (x1, y1) = ax.transData.transform([(j - 0.5, i + 0.5), (j + 0.5, i - 0.5)])
        if bb.x0 < x0 + 1 or bb.x1 > x1 - 1 or bb.y0 < y0 + 1 or bb.y1 > y1 - 1:
            n_out += 1
    return n_ov, n_out


def main():
    methods, blocks = build_blocks()
    cw, ch, gap = CELL_W, CELL_H, GAP_IN
    for attempt in range(12):
        fig, axes, cbars, leg, texts_cells = draw(methods, blocks, cw, ch, gap)
        n_ov, n_out = problems(fig, axes, cbars, leg, texts_cells)
        print(f"layout {attempt + 1}: cell width {cw:.2f} in, cell height {ch:.2f} in, gap {gap:.2f} in → "
              f"text overlaps {n_ov}, cell text overflows {n_out}")
        if n_ov == 0 and n_out == 0:
            break
        plt.close(fig)
        if n_out:
            cw *= 1.08; ch *= 1.05
        if n_ov:
            gap += 0.05
    C.save(fig, OUT_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()