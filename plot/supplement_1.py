# -*- coding: utf-8 -*-
"""
补充图 A —— 稳健性（数据：附表"稳健性"那一份 Excel，含 Platform vs tissue 等 4 个工作表）
  a  平台效应 vs 组织效应：每个方法一个点，y = x 对角线；颜色 = 算法类别，点大小 = 跨平台样本对数
  b  同组织跨平台：乳腺 / 肝 / 小鼠脑 / 胰腺 四列，每个方法一行，两种平台的 F1 用不同形状并连线
  c  平台内范围：MERFISH（13 个数据集）与 Xenium（10 个数据集），每个方法一行，min–max 横线 + 中位数点
本版：所有子图为全框线，内容（方法名、注释）全部在框内；a 中方法名之间、方法名与点之间无重叠，
      灰色区域与坐标框严格对齐；c 去掉了 n；平台形状图例与类别图例统一放在图底部。
本地运行：python supp_figA_robustness.py
FILES 中列出的 Excel 会按内容自动识别（含 "Platform vs tissue" 工作表的那份即为稳健性附表），
不必关心它被编号为 S6 还是 S7。
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ======================== 参数 ========================
FILES = [r"E:\Supplementary_Table_S5.xlsx", r"E:\Supplementary_Table_S6.xlsx", r"E:\Supplementary_Table_S7.xlsx"]
OUT_DIR = r"E:\supp_figures"
OUT_NAME = "supp_fig_robustness"

FIG_W = 7.2                    # inch，双栏宽
FONT = 6.5
ROW_H = 0.155                  # b、c 中每个方法一行的高度（inch）
CAT_COLOR = {                  # 算法类别的显示名与配色（与补充图 C 一致）
    "Statistical point-cloud": "#4490A9",
    "Graph-based": "#5B9BD5",
    "Multimodal": "#FFC000",
    "Image-based": "#ED7D31",
}
CAT_RENAME = {"Image-guided multimodal": "Multimodal"}   # 附表中的类别名 -> 图中显示名
TISSUES = [                    # b：（显示名, Xenium 列, 另一平台的列, 另一平台名）
    ("Breast", ["xenium_2"], ["merfish_breast"], "MERFISH"),
    ("Liver", ["xenium_4"], ["merfish_2", "merfish_3"], "MERFISH"),
    ("Mouse brain", ["xenium_5"], ["merfish_1"], "MERFISH"),
    ("Pancreas", ["xenium_1"], ["cosmx_1"], "CosMx"),
]
HE_DATASET = "xenium_1"        # 图像方法在该数据集上用 H&E 分割：b 中画空心点
PLAT_MARK = {"Xenium": "o", "MERFISH": "s", "CosMx": "D"}
# a 中个别标签的手动位置（点为单位，相对数据点；合并标签用逗号连接的全名）。其余标签自动放置
LABEL_NUDGE = {"ComSeg": (5, 5), "Baysor, Bering": (7, -8), "UCS, ProSeg": (7, 5)}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FONT, "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ======================== 读取 ========================
def find_book(sheet):
    for f in FILES:
        if os.path.exists(f) and sheet in pd.ExcelFile(f).sheet_names:
            return f
    raise FileNotFoundError(f"FILES 中没有含工作表 '{sheet}' 的 Excel：{FILES}")


BOOK = find_book("Platform vs tissue")
print(f"稳健性附表：{BOOK}")
read = lambda s: pd.read_excel(BOOK, sheet_name=s, header=3).dropna(subset=["Method"])
PVT = read("Platform vs tissue")
SAME = read("Same tissue across platforms")
RNG = read("Within-platform range")
SAME.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in SAME.columns]
ds_col = {c.split()[-1]: c for c in SAME.columns if "_" in c.split()[-1]}   # 列名形如 "Breast xenium_2"
ORDER = list(RNG["Method"])                                                 # 方法顺序 = 附表顺序（按类别）
CAT = {m: CAT_RENAME.get(c, c) for m, c in zip(RNG["Method"], RNG["Category"])}


def color(m):
    return CAT_COLOR.get(CAT.get(m), "#7F7F7F")


def style(ax):
    """全框线。"""
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_color("black")


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


GROUP_PX = 7                   # a 中屏幕距离小于此值（pt）的点合并为一个标签（如 ProSeg 与 UCS 几乎重合）


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


# ======================== a：平台效应 vs 组织效应 ========================
def panel_a(ax):
    d = PVT.dropna(subset=["Platform effect", "Tissue effect"])
    hi = float(max(d["Platform effect"].max(), d["Tissue effect"].max())) * 1.15
    lo = -0.08 * hi                                   # 原点处留出一点空间，使 (0, 0) 附近的点不压在框线上
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    # 灰色区域 = 对角线以下（平台效应 > 组织效应），从坐标框的左下角起算，与框严格对齐
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
        print("  [图注] a 中未显示（没有跨平台样本对）：" + ", ".join(miss))


# ======================== b：同组织跨平台 ========================
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


# ======================== c：平台内范围 ========================
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


# ======================== 版式 ========================
n = len(ORDER)
H_ROW = n * ROW_H + 0.75                    # b、c 一行（含标题与 x 轴）高度
FIG_H = max(2.9, H_ROW) + H_ROW + 0.35 + 0.45      # 末尾 0.45 inch 留给底部的类别图例
fig = plt.figure(figsize=(FIG_W, FIG_H))
top = FIG_H - 0.15


def axes_at(x, y, w, h):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


# 第一行：a（左，方形）+ c（右，两列）
a_w = 2.55
ax_a = axes_at(0.62, top - 0.35 - a_w, a_w, a_w)
c_x0, c_w, c_gap = 4.05, 1.25, 0.32
c_h = n * ROW_H
ax_c = [axes_at(c_x0, top - 0.35 - c_h, c_w, c_h), axes_at(c_x0 + c_w + c_gap, top - 0.35 - c_h, c_w, c_h)]
# 第二行：b（四列）
b_top = top - max(a_w, c_h) - 0.35 - 0.62
b_x0, b_w, b_gap = 0.95, 1.05, 0.12
ax_b = [axes_at(b_x0 + i * (b_w + b_gap), b_top - 0.35 - c_h, b_w, c_h) for i in range(4)]

panel_a(ax_a); panel_c(ax_c); panel_b(ax_b)

# 类别图例（全图共用）
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
    print("已保存:", p)
plt.close(fig)