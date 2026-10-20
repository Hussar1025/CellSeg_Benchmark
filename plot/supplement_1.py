# -*- coding: utf-8 -*-
"""
Supplementary figure A - robustness (data: the Excel robustness supplementary table, with 4 sheets including "Platform vs tissue")
  a  Platform effect vs tissue effect: one point per method, y = x diagonal; color = algorithm category, point size = number of cross-platform sample pairs
  b  Same tissue across platforms: breast / liver / mouse brain / pancreas columns, one row per method, F1 of the two platforms with different markers, connected
  c  Within-platform range: MERFISH (13 datasets) and Xenium (10 datasets), one row per method, min-max line + median point
Full frame for all subplots; content (method names, annotations) inside the frame; in a, method names never overlap each other or the points,
      gray region aligned exactly with the frame; c drops n; platform marker legend and category legend unified at the bottom.
Local run: python supp_figA_robustness.py
The Excel listed in FILES is auto-detected by content (the one with the "Platform vs tissue" sheet is the robustness table),
regardless of its S6 or S7 numbering.
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ======================== Parameters ========================
FILES = ["yourpath/Supplementary_Table_S5.xlsx", "yourpath/Supplementary_Table_S6.xlsx", "yourpath/Supplementary_Table_S7.xlsx"]
OUT_DIR = "yourpath/supp_figures"
OUT_NAME = "supp_fig_robustness"

FIG_W = 7.2                    # inch, double-column width
FONT = 6.5
ROW_H = 0.155                  # height per method row in b, c (inch)
CAT_COLOR = {                  # display names and colors of algorithm categories (consistent with supplementary figure C)
    "Statistical point-cloud": "#4490A9",
    "Graph-based": "#5B9BD5",
    "Multimodal": "#FFC000",
    "Image-based": "#ED7D31",
}
CAT_RENAME = {"Image-guided multimodal": "Multimodal"}   # category name in the table -> display name in the figure
TISSUES = [                    # b: (display name, Xenium columns, other-platform columns, other-platform name)
    ("Breast", ["xenium_2"], ["merfish_breast"], "MERFISH"),
    ("Liver", ["xenium_4"], ["merfish_2", "merfish_3"], "MERFISH"),
    ("Mouse brain", ["xenium_5"], ["merfish_1"], "MERFISH"),
    ("Pancreas", ["xenium_1"], ["cosmx_1"], "CosMx"),
]
HE_DATASET = "xenium_1"        # image-based methods are H&E-segmented on this dataset: hollow markers in b
PLAT_MARK = {"Xenium": "o", "MERFISH": "s", "CosMx": "D"}
# manual positions of a few labels in a (in points, relative to the data point; merged labels use the comma-joined full name). Other labels are placed automatically
LABEL_NUDGE = {"ComSeg": (5, 5), "Baysor, Bering": (7, -8), "UCS, ProSeg": (7, 5)}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FONT, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ======================== Load ========================
def find_book(sheet):
    for f in FILES:
        if os.path.exists(f) and sheet in pd.ExcelFile(f).sheet_names:
            return f
    raise FileNotFoundError(f"No Excel in FILES contains the '{sheet}' sheet: {FILES}")


BOOK = find_book("Platform vs tissue")
print(f"Robustness supplementary table: {BOOK}")
read = lambda s: pd.read_excel(BOOK, sheet_name=s, header=3).dropna(subset=["Method"])
PVT = read("Platform vs tissue")
SAME = read("Same tissue across platforms")
RNG = read("Within-platform range")
SAME.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in SAME.columns]
ds_col = {c.split()[-1]: c for c in SAME.columns if "_" in c.split()[-1]}   # column names like "Breast xenium_2"
ORDER = list(RNG["Method"])                                                 # method order = table order (by category)
CAT = {m: CAT_RENAME.get(c, c) for m, c in zip(RNG["Method"], RNG["Category"])}


def color(m):
    return CAT_COLOR.get(CAT.get(m), "#7F7F7F")


def style(ax):
    """Full frame."""
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_color("black")


# ======================== Label placement (non-overlapping, all inside frame) ========================
def place_labels(ax, xs, ys, names, nudge=None, fs=None, point_px=4.5, lead_from=9):
    """Place a label for each point: labels must not overlap each other or any data point, and must lie fully inside the axes.
    Candidate positions circle the point at radius 5-60 pt in 16 directions; pick the first that satisfies all constraints from near to far;
    draw a thin leader line when the label is farther (>= lead_from pt). nudge may set (dx, dy) for specific labels (in points).
    If no conflict-free position is found, a hint is printed; then widen the axis range or use nudge."""
    from matplotlib.transforms import Bbox
    nudge = nudge or {}
    fs = fs or FONT - 0.8
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72.0
    P = ax.transData.transform(np.column_stack([xs, ys]))
    axbb = ax.get_window_extent(rend)
    inner = Bbox.from_extents(axbb.x0 + 2, axbb.y0 + 2, axbb.x1 - 2, axbb.y1 - 2)
    pboxes = [Bbox.from_extents(x - point_px * px_per_pt / 1.4, y - point_px * px_per_pt / 1.4,
                                x + point_px * px_per_pt / 1.4, y + point_px * px_per_pt / 1.4) for x, y in P]
    dens = [sum(np.hypot(*(P[j] - P[i])) < 40 * px_per_pt for j in range(len(P))) for i in range(len(P))]
    order = sorted(range(len(P)), key=lambda i: (names[i] not in nudge, -dens[i]))   # place manually specified ones first
    angles = np.deg2rad(np.r_[45, 135, 315, 225, 0, 180, 90, 270, 22.5, 157.5, 337.5, 202.5, 67.5, 112.5, 292.5, 247.5])
    radii = [5, 8, 12, 17, 23, 30, 38, 47, 58]
    # existing text inside the axes (e.g. corner annotations) also counts as obstacles; labels must not overlap them
    placed = [t.get_window_extent(rend).padded(2.0 * px_per_pt) for t in ax.texts if t.get_visible() and t.get_text()]

    def make(i, dx, dy):
        ha = "left" if dx > 1 else ("right" if dx < -1 else "center")
        va = "bottom" if dy > 1 else ("top" if dy < -1 else "center")
        t = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=fs, ha=ha, va=va, zorder=5, annotation_clip=False)
        return t, t.get_window_extent(rend).padded(2.0 * px_per_pt)

    def leader(i, bb):
        """Draw a thin line from the data point to the nearest edge of the text box (display coordinates)."""
        px, py = P[i]
        qx, qy = min(max(px, bb.x0), bb.x1), min(max(py, bb.y0), bb.y1)
        v = np.array([qx - px, qy - py]); L = np.hypot(*v)
        if L < 1e-6:
            return
        p0 = np.array([px, py]) + v / L * point_px * px_per_pt * 0.75
        inv = ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform([p0, [qx, qy]])
        ax.plot([x0, x1], [y0, y1], color="#8C8C8C", lw=0.4, zorder=2, clip_on=False)

    def bdist(bb, p):
        dx = max(bb.x0 - p[0], 0, p[0] - bb.x1)
        dy = max(bb.y0 - p[1], 0, p[1] - bb.y1)
        return np.hypot(dx, dy)

    def ok(bb, i, has_leader=False):
        if not (bb.x0 >= inner.x0 and bb.x1 <= inner.x1 and bb.y0 >= inner.y0 and bb.y1 <= inner.y1):
            return False, 1000
        hits = sum(bb.overlaps(b) for b in placed) + sum(bb.overlaps(pb) for pb in pboxes)
        # clear ownership: the label must be closer to its own point than to any other (otherwise readers may assign it to the wrong point)
        # with a leader line ownership is already shown, so nearest is not required
        d_own = bdist(bb, P[i])
        if not has_leader and any(bdist(bb, P[j]) <= d_own + 1.5 * px_per_pt for j in range(len(P)) if j != i):
            hits += 1
        return hits == 0, hits

    for i in order:
        if names[i] in nudge:
            t, bb = make(i, *nudge[names[i]])
            placed.append(bb)
            if np.hypot(*nudge[names[i]]) >= lead_from:
                leader(i, bb)
            continue
        best = None
        done = False
        for r in radii:
            for a in angles:
                dx, dy = r * np.cos(a), r * np.sin(a)
                t, bb = make(i, dx, dy)
                good, hits = ok(bb, i, has_leader=r >= lead_from)
                if good:
                    placed.append(bb); done = True
                    if r >= lead_from:
                        leader(i, bb)
                    if best is not None:
                        best[0].remove()
                    break
                if best is None or hits < best[2]:
                    if best is not None:
                        best[0].remove()
                    best = (t, bb, hits)
                else:
                    t.remove()
            if done:
                break
        if not done:
            placed.append(best[1])
            leader(i, best[1])
            print(f"    [Hint] label '{names[i]}' found no fully non-overlapping position; widen the axis range or set it in LABEL_NUDGE")


GROUP_PX = 7                   # in a, points within this screen distance (pt) merge into one label (e.g. ProSeg and UCS almost coincide)


def group_points(ax, xs, ys, names):
    P = ax.transData.transform(np.column_stack([xs, ys])) / (ax.figure.dpi / 72.0)
    left = list(range(len(xs)))
    gx, gy, gn = [], [], []
    while left:
        i = left.pop(0)
        grp = [i] + [j for j in left if np.hypot(*(P[j] - P[i])) < GROUP_PX]
        left = [j for j in left if j not in grp]
        grp.sort(key=lambda k: xs[k])
        gx.append(float(np.mean([xs[k] for k in grp]))); gy.append(float(np.mean([ys[k] for k in grp])))
        gn.append(", ".join(names[k] for k in grp))
    return np.array(gx), np.array(gy), gn


# ======================== a: platform effect vs tissue effect ========================
def panel_a(ax):
    d = PVT.dropna(subset=["Platform effect", "Tissue effect"])
    hi = float(max(d["Platform effect"].max(), d["Tissue effect"].max())) * 1.15
    lo = -0.08 * hi                                   # leave a little space at the origin so points near (0, 0) do not sit on the frame
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    # gray region = below the diagonal (platform effect > tissue effect), from the lower-left of the frame, aligned exactly with it
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color="#F0F0F0", zorder=0, lw=0)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color="#7F7F7F", zorder=1)
    ax.text(hi - 0.03 * (hi - lo), lo + 0.03 * (hi - lo), "platform effect\ndominates", ha="right", va="bottom",
            fontsize=FONT - 0.5, color="#6B6B6B", zorder=2)
    ax.text(lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo), "tissue effect\ndominates", ha="left", va="top",
            fontsize=FONT - 0.5, color="#6B6B6B", zorder=2)
    npair = d["Cross-platform pairs (n)"].astype(float)
    size = 14 + 10 * (npair - npair.min()) / max(npair.max() - npair.min(), 1)
    for (_, r), s in zip(d.iterrows(), size):
        ax.scatter(r["Platform effect"], r["Tissue effect"], s=s, color=color(r["Method"]),
                   edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_xlabel("Platform effect\n(median |ΔF1|, same tissue, different platforms)")
    ax.set_ylabel("Tissue effect\n(median |ΔF1|, different tissues, same platform)")
    style(ax)
    xs, ys, ms = d["Platform effect"].to_numpy(float), d["Tissue effect"].to_numpy(float), list(d["Method"])
    gx, gy, gn = group_points(ax, xs, ys, ms)
    place_labels(ax, gx, gy, gn, nudge=LABEL_NUDGE, fs=FONT - 0.8)
    miss = [m for m in PVT["Method"] if m not in set(d["Method"])]
    if miss:
        print("  [caption] not shown in a (no cross-platform sample pairs): " + ", ".join(miss))


# ======================== b: same tissue across platforms ========================
def panel_b(axes):
    n = len(ORDER)
    y = np.arange(n)[::-1]
    for k, (ax, (tname, xcols, ocols, oplat)) in enumerate(zip(axes, TISSUES)):
        for i, m in enumerate(ORDER):
            row = SAME[SAME["Method"] == m]
            if row.empty:
                continue
            row = row.iloc[0]
            xs = [row[ds_col[c]] for c in xcols if c in ds_col]
            os_ = [row[ds_col[c]] for c in ocols if c in ds_col]
            vals = [v for v in xs + os_ if pd.notna(v)]
            he = (HE_DATASET in xcols and CAT.get(m) == "Image-based")
            if len(vals) >= 2:
                ax.plot([min(vals), max(vals)], [y[i]] * 2, color=color(m), lw=0.9 if he else 1.0,
                        ls=(0, (2, 1.5)) if he else "-", alpha=0.6 if he else 0.8, zorder=2)
            for v, c in zip(xs, xcols):
                if pd.notna(v):
                    hollow = (c == HE_DATASET and CAT.get(m) == "Image-based")
                    ax.scatter(v, y[i], marker=PLAT_MARK["Xenium"], s=13, zorder=3,
                               facecolor="white" if hollow else color(m), edgecolor=color(m), linewidth=0.7)
            for v in os_:
                if pd.notna(v):
                    ax.scatter(v, y[i], marker=PLAT_MARK[oplat], s=11, zorder=3,
                               facecolor=color(m), edgecolor="black", linewidth=0.4)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.7, n - 0.3)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.grid(axis="x", color="#E5E5E5", lw=0.5, zorder=0)
        ax.set_title(f"{tname}\nXenium vs {oplat}", fontsize=FONT)
        ax.set_yticks(y)
        ax.set_yticklabels(ORDER if k == 0 else [])
        if k == 0:
            for t, m in zip(ax.get_yticklabels(), ORDER):
                t.set_color(color(m))
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("F1")
        style(ax)


# ======================== c: within-platform range ========================
def panel_c(axes):
    n = len(ORDER)
    y = np.arange(n)[::-1]
    for k, (ax, plat) in enumerate(zip(axes, ["MERFISH", "Xenium"])):
        for i, m in enumerate(ORDER):
            r = RNG[RNG["Method"] == m].iloc[0]
            lo, med, hi_, nn = r[f"{plat} min"], r[f"{plat} median"], r[f"{plat} max"], r[f"{plat} n"]
            if pd.isna(med) or nn == 0:
                ax.text(0.5, y[i], "n/a", ha="center", va="center", fontsize=FONT - 1, color="#9E9E9E")
                continue
            ax.plot([lo, hi_], [y[i]] * 2, color=color(m), lw=1.6, solid_capstyle="round", alpha=0.85, zorder=2)
            ax.scatter(med, y[i], s=12, color="white", edgecolor=color(m), linewidth=0.9, zorder=3)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.7, n - 0.3)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.grid(axis="x", color="#E5E5E5", lw=0.5, zorder=0)
        ax.set_title(f"{plat}\n(range across tissues)", fontsize=FONT)
        ax.set_yticks(y); ax.set_yticklabels(ORDER if k == 0 else [])
        if k == 0:
            for t, m in zip(ax.get_yticklabels(), ORDER):
                t.set_color(color(m))
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("F1")
        style(ax)


# ======================== Layout ========================
n = len(ORDER)
H_ROW = n * ROW_H + 0.75                    # height of one b/c row (with title and x-axis)
FIG_H = max(2.9, H_ROW) + H_ROW + 0.35 + 0.45      # trailing 0.45 inch for the category legend at the bottom
fig = plt.figure(figsize=(FIG_W, FIG_H))
top = FIG_H - 0.15


def axes_at(x, y, w, h):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


# first row: a (left, square) + c (right, two columns)
a_w = 2.55
ax_a = axes_at(0.62, top - 0.35 - a_w, a_w, a_w)
c_x0, c_w, c_gap = 4.05, 1.25, 0.32
c_h = n * ROW_H
ax_c = [axes_at(c_x0, top - 0.35 - c_h, c_w, c_h), axes_at(c_x0 + c_w + c_gap, top - 0.35 - c_h, c_w, c_h)]
# second row: b (four columns)
b_top = top - max(a_w, c_h) - 0.35 - 0.62
b_x0, b_w, b_gap = 0.95, 1.05, 0.12
ax_b = [axes_at(b_x0 + i * (b_w + b_gap), b_top - 0.35 - c_h, b_w, c_h) for i in range(4)]

panel_a(ax_a); panel_c(ax_c); panel_b(ax_b)

# category legend (shared across the figure)
cats = [c for c in CAT_COLOR if c in set(CAT.values())]
b_bottom = b_top - 0.35 - c_h
shape_h = [Line2D([], [], ls="", marker=PLAT_MARK["Xenium"], mfc="#BFBFBF", mec="#7F7F7F", ms=4, label="Xenium"),
           Line2D([], [], ls="", marker=PLAT_MARK["MERFISH"], mfc="#BFBFBF", mec="black", mew=0.4, ms=3.6, label="MERFISH"),
           Line2D([], [], ls="", marker=PLAT_MARK["CosMx"], mfc="#BFBFBF", mec="black", mew=0.4, ms=3.4, label="CosMx"),
           Line2D([], [], ls="", marker="o", mfc="white", mec="#7F7F7F", ms=4, label="Xenium, H&E input (image-based)")]
fig.legend(handles=[Line2D([], [], ls="", marker="o", mfc=CAT_COLOR[c], mec="black", mew=0.4, ms=4.5, label=c)
                    for c in cats] + shape_h, loc="upper center", ncol=4, frameon=False, fontsize=FONT,
           bbox_to_anchor=(0.5, (b_bottom - 0.40) / FIG_H))
for ax, lab in [(ax_a, "a"), (ax_c[0], "c"), (ax_b[0], "b")]:
    ax.text(-0.38 if ax is not ax_a else -0.30, 1.12, lab, transform=ax.transAxes, fontsize=FONT + 3,
            fontweight="bold", va="bottom")

os.makedirs(OUT_DIR, exist_ok=True)
for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
    p = os.path.join(OUT_DIR, OUT_NAME + ext)
    fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
    print("Saved:", p)
plt.close(fig)