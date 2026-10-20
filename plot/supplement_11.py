# -*- coding: utf-8 -*-
"""
Main figure -- robustness and challenge scenarios (six panels, each output as one PDF and one PNG)
  a  platform effect vs tissue effect (one point per method, y = x diagonal, shading = platform effect dominates)       <- supplementary table S6 "Platform vs tissue"
  b  rank consistency of methods between datasets: same-platform vs across-platform dataset pairs (transcript / image methods)  <- same computation as supplementary Fig. S3a,b
  c  F1 vs cell density: BOMS, GeneSegNet, ComSeg, ProSeg on the same axes, points colored by method and shapes distinguishing methods,
     one linear trend line of F1 ~ log10(density) per method, Spearman ρ shown in the legend                          <- same data as supplementary Fig. S14
  d  gene panel: median F1 on the Xenium standard panel vs the Prime 5K panel (one dumbbell row per method);
     image methods exclude the H&E-input xenium_1                                                         <- same data as Fig. 3c,d
  e  runtime vs transcripts in the analysis region (estimated): Cellist, Baysor, ComSeg, Cellpose-SAM on the same log-log axes,
     one fitted line per method, slope k shown in the legend                                                     <- same data as supplementary Fig. S15
  f  CPU peak memory per method: median over all datasets (hollow), whole-slide Xenium pancreatic cancer (filled),
     maximum on any dataset (x, drawn only when above the whole slide); vertical dashed line = 256 GiB typical workstation limit      <- resource table
All text Arial Regular; after each panel, check whether text overlaps and enlarge the canvas to redraw if so.
Run locally: python main_fig_robustness.py
(place in the same directory as supp_common.py, supp_figB_rank_consistency.py, robustness_bars.py)
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

import supplement_6 as C
import supp_figB_rank_consistency as RC

# ======================== Parameters ========================
S6_FILES = ["yourpath/Supplementary_Table_S6.xlsx", "yourpath/Supplementary_Table_S5.xlsx", "yourpath/Supplementary_Table_S7.xlsx"]
OUT_DIR = C.OUT_DIR
OUT_PREFIX = "main_fig_robustness"
FONT = 7
PANEL_W, PANEL_H = 2.35, 2.25          # default panel size (inch); auto-enlarged when text overlaps

C_METHODS = ["BOMS", "GeneSegNet", "ComSeg", "ProSeg"]           # panel c
E_METHODS = ["Cellist", "Baysor", "ComSeg", "Cellpose-SAM"]      # panel e
METHOD_STYLE = {   # color and shape per method in panels c/e (colors from the four method categories; two same-category methods use different shapes)
    "ProSeg": ("#4490A9", "o"), "BOMS": ("#1F5F73", "s"), "Baysor": ("#7FB8CC", "D"),
    "ComSeg": ("#5B9BD5", "^"), "GeneSegNet": ("#E0A800", "v"), "Cellist": ("#FFC000", "P"),
    "Cellpose-SAM": ("#ED7D31", "X"),
}
PRIME_5K = sorted(C.PRIME_5K)                                     # xenium_3, 5, 7, 9, 10
STANDARD = ["xenium_1", "xenium_2", "xenium_4", "xenium_6", "xenium_8"]
HE_DATASET = "xenium_1"
WHOLE_SLIDE = "xenium_1"
WORKSTATION_GIB = 256

mpl.rcParams.update({
    "font.family": "Arial", "font.sans-serif": ["Arial"], "font.size": FONT, "font.weight": "normal",
    "font.style": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial", "mathtext.bf": "Arial",
    "mathtext.sf": "Arial", "mathtext.default": "regular",
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ======================== Utilities ========================
def frame(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_color("black")


def save(fig, letter):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, kw in [(".pdf", {}), (".png", {"dpi": 600})]:
        p = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{letter}{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print("Saved:", p)
    plt.close(fig)


def text_overlaps(fig):
    """Count overlapping pairs among all visible text (ticks, titles, legend)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_alpha() != 0:
            try:
                boxes.append(t.get_window_extent(rend))
            except Exception:
                pass
    return sum(boxes[i].overlaps(boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes)))


def build(draw_fn, letter, w=PANEL_W, h=PANEL_H, **kw):
    """Draw and check text overlap; enlarge the canvas and redraw if any (up to 10 tries)."""
    n = None
    for attempt in range(10):
        fig = plt.figure(figsize=(w, h))
        draw_fn(fig, **kw)
        n = text_overlaps(fig)
        if n == 0:
            break
        print(f"  [{letter}] layout attempt {attempt + 1}: {n} text overlaps, enlarging canvas and retrying")
        plt.close(fig)
        w, h = w * 1.08, h * 1.08
    if n:
        print(f"  [{letter}] [Warning] {n} text overlaps remain after 10 enlargements, please check")
    save(fig, letter)


def place_labels(ax, xs, ys, names, fs=FONT - 1.2):
    """Place method names beside points: no overlap with other labels, any point, or existing text, and fully inside the axes; draw a leader line when far from the point."""
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ppt = fig.dpi / 72
    P = ax.transData.transform(np.column_stack([xs, ys]))
    axbb = ax.get_window_extent(rend)
    inner = Bbox.from_extents(axbb.x0 + 2, axbb.y0 + 2, axbb.x1 - 2, axbb.y1 - 2)
    pts = [Bbox.from_extents(x - 3 * ppt, y - 3 * ppt, x + 3 * ppt, y + 3 * ppt) for x, y in P]
    placed = [t.get_window_extent(rend).padded(2 * ppt) for t in ax.texts if t.get_text()]
    angles = np.deg2rad([45, 135, 315, 225, 0, 180, 90, 270, 22.5, 157.5, 337.5, 202.5])
    for i in np.argsort([-sum(np.hypot(*(P[j] - P[k])) < 30 * ppt for k in range(len(P))) for j in range(len(P))]):
        best = None
        for r in [5, 8, 11, 15, 20, 26, 33, 41, 50, 60]:
            for a in angles:
                dx, dy = r * np.cos(a), r * np.sin(a)
                t = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", fontsize=fs,
                                ha="left" if dx > 1 else ("right" if dx < -1 else "center"),
                                va="bottom" if dy > 1 else ("top" if dy < -1 else "center"), zorder=5)
                bb = t.get_window_extent(rend).padded(1.5 * ppt)
                inside = inner.x0 <= bb.x0 and bb.x1 <= inner.x1 and inner.y0 <= bb.y0 and bb.y1 <= inner.y1
                hits = (0 if inside else 100) + sum(bb.overlaps(b) for b in placed) + sum(bb.overlaps(b) for b in pts)
                if hits == 0:
                    placed.append(bb)
                    if r >= 11:
                        qx, qy = min(max(P[i][0], bb.x0), bb.x1), min(max(P[i][1], bb.y0), bb.y1)
                        (x0, y0), (x1, y1) = ax.transData.inverted().transform([P[i], (qx, qy)])
                        ax.plot([x0, x1], [y0, y1], color="#8C8C8C", lw=0.4, zorder=2)
                    if best is not None:
                        best[0].remove()
                    best = None
                    break
                if best is None or hits < best[1]:
                    if best is not None:
                        best[0].remove()
                    best = (t, hits, bb)
                else:
                    t.remove()
            else:
                continue
            break
        if best is not None:
            placed.append(best[2])
            print(f"    [Hint] label '{names[i]}' has no fully non-overlapping position")


# ======================== Data ========================
def load_s6():
    for f in S6_FILES:
        if os.path.exists(f) and "Platform vs tissue" in pd.ExcelFile(f).sheet_names:
            d = pd.read_excel(f, sheet_name="Platform vs tissue", header=3).dropna(subset=["Method"])
            print("Robustness supplementary table:", f)
            return d
    raise FileNotFoundError(f"S6_FILES has no supplementary table with 'Platform vs tissue': {S6_FILES}")


# ======================== a ========================
def draw_a(fig, pvt):
    ax = fig.add_axes([0.20, 0.18, 0.72, 0.72])
    d = pvt.dropna(subset=["Platform effect", "Tissue effect"])
    hi = float(max(d["Platform effect"].max(), d["Tissue effect"].max())) * 1.15
    lo = -0.08 * hi
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color="#F0F0F0", lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], ls=(0, (3, 2)), lw=0.6, color="#7F7F7F", zorder=1)
    ax.text(hi - 0.03 * (hi - lo), lo + 0.03 * (hi - lo), "Platform effect\ndominates", ha="right", va="bottom",
            fontsize=FONT - 1, color="#6B6B6B")
    ax.text(lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo), "Tissue effect\ndominates", ha="left", va="top",
            fontsize=FONT - 1, color="#6B6B6B")
    for _, r in d.iterrows():
        ax.scatter(r["Platform effect"], r["Tissue effect"], s=16, color=C.CAT_COLOR[C.CAT_OF[r["Method"]]],
                   edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_xlabel("Platform effect (median |ΔF1|)")
    ax.set_ylabel("Tissue effect (median |ΔF1|)")
    frame(ax)
    xs, ys, ms = d["Platform effect"].to_numpy(float), d["Tissue effect"].to_numpy(float), list(d["Method"])
    # merge nearly overlapping points into one label
    P = ax.transData.transform(np.column_stack([xs, ys])) / (fig.dpi / 72)
    left, gx, gy, gn = list(range(len(xs))), [], [], []
    while left:
        i = left.pop(0)
        g = [i] + [j for j in left if np.hypot(*(P[j] - P[i])) < 6]
        left = [j for j in left if j not in g]
        gx.append(np.mean(xs[g])); gy.append(np.mean(ys[g])); gn.append(", ".join(ms[k] for k in g))
    place_labels(ax, np.array(gx), np.array(gy), gn)


# ======================== b ========================
def draw_b(fig, pairs):
    ax = fig.add_axes([0.20, 0.20, 0.75, 0.70])
    data, pos, cols = [], [], []
    for k, kind in enumerate(["rna", "image"]):
        pt = pairs[kind]
        for j, same in enumerate([True, False]):
            data.append(pt[pt.same == same].rho.to_numpy()); pos.append(k * 2.6 + j)
            cols.append("#7F7F7F" if same else "#D9D9D9")
    bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=0.9), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    boxprops=dict(lw=0.6))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
    rng = np.random.default_rng(0)
    for d, p in zip(data, pos):
        ax.scatter(p + rng.uniform(-0.22, 0.22, len(d)), d, s=2.5, color="black", alpha=0.35, lw=0, zorder=3)
        ax.text(p, 1.08, f"{np.median(d):.2f}", ha="center", va="bottom", fontsize=FONT - 1)
    ax.axhline(0, color="#9E9E9E", lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks([0.5, 3.1]); ax.set_xticklabels(["Transcript-based", "Image-based"])
    ax.set_ylabel("Spearman ρ of method ranks\nbetween two datasets")
    ax.set_ylim(-1.05, 1.28); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlim(-0.6, 4.2)
    ax.legend(handles=[Patch(fc="#7F7F7F", ec="black", lw=0.5, label="Same platform"),
                       Patch(fc="#D9D9D9", ec="black", lw=0.5, label="Different platforms")],
              loc="lower left", frameon=False, fontsize=FONT - 1, handlelength=1.2)
    frame(ax)


# ======================== c ========================
def draw_c(fig, d):
    ax = fig.add_axes([0.18, 0.18, 0.78, 0.62])
    handles = []
    for m in C_METHODS:
        g = d[(d.method == m)].dropna(subset=["density", "f1"])
        col, mk = METHOD_STYLE[m]
        ax.scatter(g.density, g.f1, s=14, marker=mk, facecolor=col, edgecolor="black", linewidth=0.3, zorder=3,
                   alpha=0.9)
        r, _ = spearmanr(g.density, g.f1)
        lx = np.log10(g.density.to_numpy(float))
        k, b = np.polyfit(lx, g.f1.to_numpy(float), 1)
        xs = np.linspace(lx.min(), lx.max(), 20)
        ax.plot(10 ** xs, k * xs + b, color=col, lw=1.0, zorder=2)
        handles.append(Line2D([], [], color=col, marker=mk, mfc=col, mec="black", mew=0.3, ms=4, lw=1.0,
                              label=f"{m}  (ρ = {r:.2f})"))
    ax.set_xscale("log")
    ax.set_xticks([1e3, 3e3, 1e4, 3e4]); ax.set_xticklabels(["1k", "3k", "10k", "30k"])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Cell density (cells/mm²)")
    ax.set_ylabel("F1")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.57, 0.81), ncol=2, frameon=False,
               fontsize=FONT - 1, handlelength=1.6, columnspacing=1.0)
    frame(ax)


# ======================== d ========================
def draw_d(fig, acc):
    ax = fig.add_axes([0.30, 0.14, 0.64, 0.74])
    n = len(C.ORDER)
    y = np.arange(n)[::-1]
    for i, m in enumerate(C.ORDER):
        g = acc[acc.method == m]
        std = STANDARD if C.CAT_OF[m] != "Image-based" else [s for s in STANDARD if s != HE_DATASET]
        a = g[g.ds.isin(std)].f1.median()
        b = g[g.ds.isin(PRIME_5K)].f1.median()
        col = C.CAT_COLOR[C.CAT_OF[m]]
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([a, b], [y[i], y[i]], color=col, lw=1.2, zorder=2)
        if np.isfinite(a):
            ax.scatter(a, y[i], s=16, color=col, edgecolor="black", linewidth=0.4, zorder=3)
        if np.isfinite(b):
            ax.scatter(b, y[i], s=16, facecolor="white", edgecolor=col, linewidth=1.0, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(C.ORDER)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Median F1 on Xenium")
    ax.grid(axis="x", color="#E5E5E5", lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Line2D([], [], ls="", marker="o", mfc="#9E9E9E", mec="black", mew=0.4, ms=4.5,
                               label="Standard panel"),
                        Line2D([], [], ls="", marker="o", mfc="white", mec="#7F7F7F", mew=1.0, ms=4.5,
                               label="Prime 5K panel")],
               loc="lower center", bbox_to_anchor=(0.62, 0.885), ncol=2, frameon=False, fontsize=FONT - 1,
               handletextpad=0.3, columnspacing=1.0)
    frame(ax)


# ======================== e ========================
def draw_e(fig, res):
    ax = fig.add_axes([0.18, 0.18, 0.78, 0.62])
    handles = []
    for m in E_METHODS:
        g = res[res.method == m].dropna(subset=["roi_tx_est", "runtime_min"])
        g = g[(g.roi_tx_est > 0) & (g.runtime_min > 0)]
        col, mk = METHOD_STYLE[m]
        ax.scatter(g.roi_tx_est, g.runtime_min, s=14, marker=mk, facecolor=col, edgecolor="black", linewidth=0.3,
                   zorder=3, alpha=0.9)
        lx, ly = np.log10(g.roi_tx_est.to_numpy(float)), np.log10(g.runtime_min.to_numpy(float))
        k, b = np.polyfit(lx, ly, 1)
        xs = np.linspace(lx.min(), lx.max(), 20)
        ax.plot(10 ** xs, 10 ** (k * xs + b), color=col, lw=1.0, zorder=2)
        handles.append(Line2D([], [], color=col, marker=mk, mfc=col, mec="black", mew=0.3, ms=4, lw=1.0,
                              label=f"{m}  (k = {k:.2f})"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Transcripts in region (estimated)")
    ax.set_ylabel("Runtime (min)")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.57, 0.81), ncol=2, frameon=False,
               fontsize=FONT - 1, handlelength=1.6, columnspacing=1.0)
    frame(ax)


# ======================== f ========================
def draw_f(fig, res):
    ax = fig.add_axes([0.30, 0.14, 0.64, 0.74])
    n = len(C.ORDER)
    y = np.arange(n)[::-1]
    for i, m in enumerate(C.ORDER):
        g = res[res.method == m]
        med = g.cpu_gib.median()
        ws = g[g.ds == WHOLE_SLIDE].cpu_gib
        col = C.CAT_COLOR[C.CAT_OF[m]]
        w = float(ws.iloc[0]) if len(ws) and np.isfinite(ws.iloc[0]) else np.nan
        if np.isfinite(med) and np.isfinite(w):
            ax.plot([med, w], [y[i], y[i]], color=col, lw=1.2, zorder=2)
        if np.isfinite(med):
            ax.scatter(med, y[i], s=16, facecolor="white", edgecolor=col, linewidth=1.0, zorder=3)
        if np.isfinite(w):
            ax.scatter(w, y[i], s=16, color=col, edgecolor="black", linewidth=0.4, zorder=3)
        mx = g.cpu_gib.max()
        if np.isfinite(mx) and (not np.isfinite(w) or mx > w * 1.02):
            ax.scatter(mx, y[i], s=12, marker="x", color="black", linewidth=0.7, zorder=4)
    ax.axvline(WORKSTATION_GIB, color="black", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(WORKSTATION_GIB * 0.92, -0.35, f"{WORKSTATION_GIB} GiB", ha="right", va="center", fontsize=FONT - 1.5,
            color="#4D4D4D")
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(C.ORDER)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("Peak CPU memory (GiB)")
    ax.grid(axis="x", color="#E5E5E5", lw=0.5, zorder=0, which="major")
    ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Line2D([], [], ls="", marker="o", mfc="white", mec="#7F7F7F", mew=1.0, ms=4.5,
                               label="Median, all datasets"),
                        Line2D([], [], ls="", marker="o", mfc="#9E9E9E", mec="black", mew=0.4, ms=4.5,
                               label="Whole slide (Xenium pancreatic cancer)"),
                        Line2D([], [], ls="", marker="x", color="black", mew=0.7, ms=4,
                               label="Maximum, any dataset")],
               loc="lower center", bbox_to_anchor=(0.55, 0.885), ncol=1, frameon=False, fontsize=FONT - 1,
               handletextpad=0.3)
    frame(ax)


# ======================== Main ========================
def main():
    acc = C.load_accuracy()
    meta = C.dataset_table(acc)
    d = acc.join(meta, on="ds")
    res = C.load_resources().join(meta, on="ds")

    # b data: identical computation to supplementary Fig. S3a,b
    agg, _ = C.RB.load_both(C.PATH_OMICS, C.PATH_IMAGE)
    pairs = {}
    for kind in ["rna", "image"]:
        Cm, plat, _ = RC.corr_matrix(agg, kind)
        Cm, plat, _ = RC.drop_incomplete(Cm, plat)
        pairs[kind] = RC.pair_table(Cm, plat)
        print(f"[b] {kind}: within-platform median ρ = {pairs[kind][pairs[kind].same].rho.median():.2f}, "
              f"across-platform median ρ = {pairs[kind][~pairs[kind].same].rho.median():.2f}")

    build(draw_a, "a", w=3.1, h=3.1, pvt=load_s6())
    build(draw_b, "b", pairs=pairs)
    build(draw_c, "c", d=d)
    build(draw_d, "d", w=PANEL_W, h=PANEL_H * 1.25, acc=acc)
    build(draw_e, "e", res=res)
    build(draw_f, "f", w=PANEL_W, h=PANEL_H * 1.25, res=res)


if __name__ == "__main__":
    main()