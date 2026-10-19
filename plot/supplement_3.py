# -*- coding: utf-8 -*-
"""
补充图 C —— 下游污染（数据：附表"下游污染"那一份 Excel，含 "Table S5" 工作表）
  b  细胞回收 vs 污染：每个方法一个点，横轴 = 可评估的蛋白定义细胞数，纵轴 = 外来/自身转录本比值中位数；
     T 细胞、髓系细胞各一格；细黑实线阶梯 = 帕累托前沿（不存在细胞更多且污染更低的方法）
  c  污染来源构成：T 细胞、髓系细胞两格，每个方法一条 100% 堆叠条
本版：b 为全框线、两格放大；方法名与方法名、方法名与数据点之间无任何重叠，且全部落在框内
      （几乎重合的点合并为一个标签，如 "Cellpose-3, Cellpose-SAM, Mesmer"）；c 只保留 T 与髓系
FILES 中的 Excel 按内容自动识别（含 "Table S5" 且有 "Share of foreign transcripts" 列的那份），与编号无关。
本地运行：python supp_figC_contamination.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ======================== 参数 ========================
FILES = [r"E:\Supplementary_Table_S5.xlsx", r"E:\Supplementary_Table_S6.xlsx", r"E:\Supplementary_Table_S7.xlsx"]
OUT_DIR = r"E:\supp_figures"
OUT_NAME = "supp_fig_contamination"
FIG_W, FONT = 7.2, 6.5
SMALL_N = 30                   # 可评估细胞少于此数：b 中画空心点
GROUP_PX = 7                   # b 中屏幕距离小于此值（pt）的点合并为一个标签
PANEL_B = (3.15, 3.0)          # b 每一格的宽、高（inch）
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
LABEL_NUDGE = {"T": {}, "Myeloid": {}}   # 个别标签手动位置（点为单位），如 {"T": {"BOMS": (6, -4)}}

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
    raise FileNotFoundError(f"FILES 中没有含 'Table S5'（污染附表）的 Excel：{FILES}")


BOOK, S5 = find_book()
print(f"污染附表：{BOOK}")
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
    """几乎重合的点合并为一个标签（位置取重心）。"""
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


# ======================== 标签放置（无重叠、全部在框内） ========================
def place_labels(ax, xs, ys, names, nudge=None, fs=None, point_px=4.5, lead_from=9):
    """为每个点放一个标签：标签之间、标签与任何数据点之间都不重叠，且标签完全落在坐标框内。
    候选位置为以点为中心、半径 5–60 pt、16 个方向；从近到远选第一个满足全部约束的位置；
    离点较远（>= lead_from pt）时画一条细引线。nudge 可为个别标签指定 (dx, dy)（点为单位）。
    若某个标签找不到无冲突位置，会打印提示，此时可把坐标轴范围放宽或用 nudge 指定。"""
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
    order = sorted(range(len(P)), key=lambda i: (names[i] not in nudge, -dens[i]))   # 手动指定的先放
    angles = np.deg2rad(np.r_[45, 135, 315, 225, 0, 180, 90, 270, 22.5, 157.5, 337.5, 202.5, 67.5, 112.5, 292.5, 247.5])
    radii = [5, 8, 12, 17, 23, 30, 38, 47, 58]
    # 坐标框内已有的文字（如角落注释）也作为障碍，标签不得与之重叠
    placed = [t.get_window_extent(rend).padded(2.0 * px_per_pt) for t in ax.texts if t.get_visible() and t.get_text()]

    def make(i, dx, dy):
        ha = "left" if dx > 1 else ("right" if dx < -1 else "center")
        va = "bottom" if dy > 1 else ("top" if dy < -1 else "center")
        t = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=fs, ha=ha, va=va, zorder=5, annotation_clip=False)
        return t, t.get_window_extent(rend).padded(2.0 * px_per_pt)

    def leader(i, bb):
        """从数据点画一条细线到文字框最近的边（文字框为显示坐标）。"""
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
        # 归属清楚：标签离自己的点必须比离任何其他点都近（否则读者会把它认成别的点的标签）
        # 有引线时归属已由引线标明，不再要求最近
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
            print(f"    [提示] 标签 '{names[i]}' 未找到完全无重叠的位置，请放宽坐标范围或在 LABEL_NUDGE 中指定")


# ======================== b：细胞回收 vs 污染 ========================
def panel_b(axes):
    for ax, (case, title) in zip(axes, CASES):
        d = S5[(S5.case == case) & S5[RATIO].notna()]
        x, y = d[NCELL].to_numpy(float), d[RATIO].to_numpy(float)
        m = list(d["Method"])
        # 坐标范围：四周各留出足够空间放标签
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


# ======================== c：污染来源构成 ========================
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


# ======================== 版式 ========================
ROW_H = 0.16
c_h = len(ORDER) * ROW_H
bw, bh = PANEL_B
LEG_H = 0.35                                   # b 下方类别图例的高度
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
    print("已保存:", p)
plt.close(fig)