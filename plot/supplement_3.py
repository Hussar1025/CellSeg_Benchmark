# -*- coding: utf-8 -*-
"""
Supplementary figure C - downstream contamination (data: the Excel supplementary table for "downstream contamination", with the "Table S5" sheet)
  b  Cell recovery vs contamination: one point per method, x-axis = evaluable protein-defined cell count, y-axis = median foreign/own transcript ratio;
     T cell and myeloid cell panels; thin black solid step line = Pareto front (no method with more cells and lower contamination)
  c  Contamination source composition: T cell and myeloid cell panels, one 100% stacked bar per method
Full frame, both b panels enlarged; method names never overlap each other or the data points, and all stay inside the frame
      (near-coincident points merge into one label, e.g. "Cellpose-3, Cellpose-SAM, Mesmer"); c keeps only T and myeloid
The Excel in FILES is auto-detected by content (the one with a "Table S5" sheet and a "Share of foreign transcripts" column), regardless of numbering.
Local run: python supp_figC_contamination.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ======================== Parameters ========================
FILES = ["yourpath/Supplementary_Table_S5.xlsx", "yourpath/Supplementary_Table_S6.xlsx", "yourpath/Supplementary_Table_S7.xlsx"]
OUT_DIR = "yourpath/supp_figures"
OUT_NAME = "supp_fig_contamination"
FIG_W, FONT = 7.2, 6.5
SMALL_N = 30                   # evaluable cells below this: drawn as hollow markers in b
GROUP_PX = 7                   # in b, points within this screen distance (pt) merge into one label
PANEL_B = (3.15, 3.0)          # width and height of each b panel (inch)
CAT_OF = {
    "ProSeg": "Statistical point-cloud", "Baysor": "Statistical point-cloud", "BOMS": "Statistical point-cloud",
    "Bering": "Graph-based", "ComSeg": "Graph-based",
    "UCS": "Multimodal", "Cellist": "Multimodal",
    "GeneSegNet": "Multimodal", "DISSECT": "Multimodal",
    "Cellpose-SAM": "Image-based", "Cellpose-3": "Image-based", "CellSAM": "Image-based",
    "CelloType": "Image-based", "Mesmer": "Image-based", "StarDist": "Image-based", "UNSEG": "Image-based",
}
CAT_COLOR = {"Statistical point-cloud": "#4490A9", "Graph-based": "#5B9BD5",
             "Multimodal": "#FFC000", "Image-based": "#ED7D31"}
ORDER = ["ProSeg", "Baysor", "BOMS", "Bering", "ComSeg", "UCS", "Cellist", "GeneSegNet", "DISSECT",
         "Cellpose-SAM", "Cellpose-3", "CellSAM", "CelloType", "Mesmer", "StarDist", "UNSEG"]
LINEAGES = ["T", "B", "Plasma", "Myeloid", "Epi/Tumor", "NormalEpi", "Endo", "Stromal"]
LIN_COLOR = {"T": "#ED7D31", "B": "#F4B183", "Plasma": "#FFD966", "Myeloid": "#5B9BD5",
             "Epi/Tumor": "#C55A5A", "NormalEpi": "#A9D18E", "Endo": "#8E7CC3", "Stromal": "#A5A5A5"}
CASES = [("T", "T cells"), ("Myeloid", "Myeloid cells")]
LABEL_NUDGE = {"T": {}, "Myeloid": {}}   # manual positions of specific labels (in points), e.g. {"T": {"BOMS": (6, -4)}}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": FONT,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def find_book():
    for f in FILES:
        if os.path.exists(f) and "Table S5" in pd.ExcelFile(f).sheet_names:
            d = pd.read_excel(f, sheet_name="Table S5", header=3)
            if any(str(c).startswith("Share of foreign transcripts") for c in d.columns):
                return f, d
    raise FileNotFoundError(f"No Excel in FILES contains 'Table S5' (contamination supplementary table): {FILES}")


BOOK, S5 = find_book()
print(f"Contamination supplementary table: {BOOK}")
S5 = S5.dropna(subset=["Method"])
S5["case"] = S5["Protein-defined cell type"].astype(str).str.split(" ").str[0]
RATIO = "Median foreign/own transcript ratio"
NCELL = "Evaluable cells"


def full_frame(ax):
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_color("black")


def open_frame(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def group_points(ax, xs, ys, names):
    """Merge near-coincident points into one label (position at the centroid)."""
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


# ======================== b: cell recovery vs contamination ========================
def panel_b(axes):
    for ax, (case, title) in zip(axes, CASES):
        d = S5[(S5.case == case) & S5[RATIO].notna()]
        x, y = d[NCELL].to_numpy(float), d[RATIO].to_numpy(float)
        m = list(d["Method"])
        # axis ranges: leave enough room on all sides for labels
        xr, yr = x.max() - x.min(), y.max() - y.min()
        ax.set_xlim(x.min() - 0.16 * xr, x.max() + 0.30 * xr)
        ax.set_ylim(y.min() - 0.14 * yr, y.max() + 0.14 * yr)
        o = np.argsort(-x)
        front, best = [], np.inf
        for k in o:
            if y[k] < best:
                front.append(k); best = y[k]
        fx, fy = x[front], y[front]
        ax.step(np.r_[fx[0], fx], np.r_[fy[0], fy], where="post", ls="-", lw=0.45, color="black", zorder=1)
        for xi, yi, mi in zip(x, y, m):
            small = xi < SMALL_N
            c = CAT_COLOR[CAT_OF.get(mi, "Image-based")]
            ax.scatter(xi, yi, s=22, facecolor="white" if small else c, edgecolor=c if small else "black",
                       linewidth=0.9 if small else 0.4, zorder=3)
        ax.set_xlabel(f"Evaluable protein-defined {'T' if case == 'T' else 'myeloid'} cells")
        ax.set_ylabel("Median foreign/own transcript ratio")
        ax.set_title(title, fontsize=FONT + 0.5)
        ax.grid(color="#EDEDED", lw=0.5, zorder=0)
        full_frame(ax)
        gx, gy, gn = group_points(ax, x, y, m)
        place_labels(ax, gx, gy, gn, nudge=LABEL_NUDGE.get(case), fs=FONT - 0.6)


# ======================== c: contamination source composition ========================
def panel_c(axes):
    n = len(ORDER)
    ypos = np.arange(n)[::-1]
    for k, (ax, (case, title)) in enumerate(zip(axes, CASES)):
        d = S5[S5.case == case].set_index("Method")
        for i, m in enumerate(ORDER):
            if m not in d.index or pd.isna(d.at[m, RATIO]):
                ax.text(0.5, ypos[i], "n/a", ha="center", va="center", fontsize=FONT - 1, color="#9E9E9E")
                continue
            left = 0.0
            for lin in LINEAGES:
                v = d.at[m, f"Share of foreign transcripts: {lin}"] if lin != case else np.nan
                if pd.notna(v) and v > 0:
                    ax.barh(ypos[i], v, left=left, height=0.78, color=LIN_COLOR[lin], edgecolor="white", lw=0.3)
                    left += v
        ax.set_xlim(0, 1); ax.set_ylim(-0.6, n - 0.4)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
        ax.set_yticks(ypos); ax.set_yticklabels(ORDER if k == 0 else [])
        if k == 0:
            for t, m in zip(ax.get_yticklabels(), ORDER):
                t.set_color(CAT_COLOR[CAT_OF[m]])
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{title}\n(own lineage excluded)", fontsize=FONT)
        ax.set_xlabel("Share of foreign transcripts")
        open_frame(ax)
    axes[-1].legend(handles=[Patch(fc=LIN_COLOR[l], label=l) for l in LINEAGES], title="Source lineage",
                    title_fontsize=FONT, loc="upper left", bbox_to_anchor=(1.04, 1.0), frameon=False,
                    fontsize=FONT - 0.5)


# ======================== Layout ========================
ROW_H = 0.16
c_h = len(ORDER) * ROW_H
bw, bh = PANEL_B
LEG_H = 0.35                                   # height of the category legend below b
FIG_H = 0.35 + bh + 0.55 + LEG_H + 0.40 + c_h + 0.6
fig = plt.figure(figsize=(FIG_W, FIG_H))
top = FIG_H - 0.35


def axes_at(x, y, w, h):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


bx0, bgap = 0.45, (FIG_W - 0.45 - 2 * bw - 0.05)
ax_b = [axes_at(bx0, top - bh, bw, bh), axes_at(bx0 + bw + bgap, top - bh, bw, bh)]
leg_y = top - bh - 0.55
c_top = leg_y - LEG_H - 0.40
c_w, c_gap = 2.05, 0.22
ax_c = [axes_at(1.0 + i * (c_w + c_gap), c_top - c_h, c_w, c_h) for i in range(2)]
panel_b(ax_b)
fig.legend(handles=[Line2D([], [], ls="", marker="o", mfc=CAT_COLOR[c], mec="black", mew=0.4, ms=4.5, label=c)
                    for c in CAT_COLOR] +
           [Line2D([], [], ls="", marker="o", mfc="white", mec="#7F7F7F", ms=4.5, label=f"< {SMALL_N} evaluable cells"),
            Line2D([], [], ls="-", lw=0.45, color="black", label="Pareto front")],
           loc="upper center", bbox_to_anchor=(0.5, leg_y / FIG_H), ncol=3, frameon=False, fontsize=FONT - 0.3)
panel_c(ax_c)
for ax, lab, dx in [(ax_b[0], "b", -0.13), (ax_c[0], "c", -0.36)]:
    ax.text(dx, 1.04, lab, transform=ax.transAxes, fontsize=FONT + 3, fontweight="bold", va="bottom")

os.makedirs(OUT_DIR, exist_ok=True)
for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
    p = os.path.join(OUT_DIR, OUT_NAME + ext)
    fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
    print("Saved:", p)
plt.close(fig)