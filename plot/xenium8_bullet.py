#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xenium_8 污染来源构成 —— 独立气泡矩阵脚本
==========================================
从 contamination_xenium8.py 中拆出的气泡图，可单独反复调样式，不必重跑整个分析。

数据：OUTDIR/bubble_source_data.csv（contamination_xenium8.py 运行时自动写出）
      若该文件不存在，本脚本会调用 contamination_xenium8.compute() 现算一次并写出。
      --recompute 强制重算。

每格 = 某方法、某 case 中，外来转录本来自某谱系的占比；按列归一化为"洁净度分数"
（占比越小 -> 分数越高 -> 气泡越大、越方、越浅）。计算逻辑与主脚本一致。

样式（相对主脚本的修改）：
  · case 配色：T = 橙 #ED7D31、髓系 = 蓝 #5B9BD5（v3 原配色），Epi/Tumor = #FFC000
  · 气泡描边 0.8 -> 1.6（更粗）
  · 气泡间距更小：GAP_EPS 0.018 -> 0.004（相邻最大气泡之间只留描边宽度 + 0.004 英寸）
运行： python bubble_xenium8.py            输出 OUTDIR/bubble_source.pdf / .png
       python bubble_xenium8.py --recompute
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon, Rectangle

from panel import OUTDIR, LINEAGES, CASES, METHODS, DISPLAY

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

DATA_CSV = os.path.join(OUTDIR, "bubble_source_data.csv")
OUT = os.path.join(OUTDIR, "bubble_source.pdf")

# ---- 样式 ----
CASE_COLORS = {"T": "#ED7D31", "Myeloid": "#5B9BD5", "Epi/Tumor": "#FFC000"}
ROW_BG_A, ROW_BG_B = "#EDEDED", "#FFFFFF"
EDGE_C, EDGE_W = "#1A1A1A", 1.6          # 气泡描边（更粗）
GAP_EPS = 0.004                          # 相邻最大气泡间额外留白（英寸），越小越紧
MIN_INDENT = EDGE_W / 72.0 / 2.0 + GAP_EPS
LIGHT_MAX, LIGHT_GAMMA = 0.97, 0.65
HALF_MIN, HALF_MAX = 0.077, 0.300
SQ_RANGE, SQ_POWER = (2.0, 8.0), 6.0
CAT_GAP = 0.5                            # case 之间的横向间隔（格子边长的倍数）
LABEL_ROT, LEADER_IN = 45, 0.13
FS = 12
SCORE_LO, SCORE_SPAN = 0.05, 0.90
_SCRATCH = None


def text_w(t, fs):
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = plt.figure(figsize=(1, 1), dpi=100)
    o = _SCRATCH.text(0, 0, str(t), fontsize=fs)
    _SCRATCH.canvas.draw()
    bb = o.get_window_extent(renderer=_SCRATCH.canvas.get_renderer())
    o.remove()
    return bb.width / _SCRATCH.dpi


def superellipse(cx, cy, half, n, k=180):
    t = np.linspace(0, 2 * np.pi, k)
    ct, st = np.cos(t), np.sin(t)
    return np.column_stack([cx + half * np.sign(ct) * np.abs(ct) ** (2.0 / n),
                            cy + half * np.sign(st) * np.abs(st) ** (2.0 / n)])


def shade(base, v):
    rgb = np.asarray(to_rgb(base), float)
    t = float(np.clip(v, 0, 1)) ** LIGHT_GAMMA * LIGHT_MAX
    return tuple(rgb + (1.0 - rgb) * t)


def shape_of(v):
    v = float(np.clip(v, 0, 1))
    lo, hi = SQ_RANGE
    return HALF_MIN + v * (HALF_MAX - HALF_MIN), lo + (v ** SQ_POWER) * (hi - lo)


def load_data():
    if os.path.exists(DATA_CSV) and "--recompute" not in sys.argv:
        print(f"读取 {DATA_CSV}")
        return pd.read_csv(DATA_CSV)
    print("没有现成的气泡数据（或指定了 --recompute），调用 contamination_xenium8.compute() 现算")
    import contamination_xenium8 as C
    methods, store = C.compute()
    C.save_bubble_data(methods, store)
    return pd.read_csv(DATA_CSV)


def build_blocks(df):
    """-> [(case, 列(外来谱系), 占比矩阵, 行(方法))]，行为在所有 case 中都有结果的方法。"""
    cases = [c for c in CASES if c in set(df["case"])]
    order = [m for m in METHODS if m in set(df["method"])] + \
            sorted(set(df["method"]) - set(METHODS))
    rows = [m for m in order if all(((df["method"] == m) & (df["case"] == c)).any() for c in cases)]
    dropped = [m for m in order if m not in rows]
    if dropped:
        print(f"[注意] 这些方法不是每个 case 都有结果，不进入气泡矩阵：{dropped}")
    blocks = []
    for c in cases:
        self_lin = CASES[c][1]
        cols = [a for a in LINEAGES if a != self_lin]
        sub = df[df["case"] == c].pivot_table(index="method", columns="source_lineage",
                                              values="share", aggfunc="first")
        Mx = sub.reindex(index=rows, columns=cols).to_numpy(float)
        blocks.append((c, cols, np.nan_to_num(Mx, nan=0.0), rows))
    return blocks, rows


def plot(blocks, rows, out=OUT):
    scores = []
    for cname, cols, Mx, _ in blocks:
        S = np.zeros_like(Mx)
        for j in range(Mx.shape[1]):
            col = Mx[:, j]
            lo, hi = np.nanmin(col), np.nanmax(col)
            S[:, j] = (SCORE_LO + SCORE_SPAN * 0.5 if hi - lo < 1e-12
                       else SCORE_LO + SCORE_SPAN * (1.0 - (col - lo) / (hi - lo)))
        scores.append(S)

    vmax_s = max(s.max() for s in scores)
    half_top = HALF_MIN + float(np.clip(vmax_s, 0, 1)) * (HALF_MAX - HALF_MIN)
    cell = 2 * (half_top + MIN_INDENT)
    col_x, xlab, span, cur = [], [], [], 0.0
    for bi, (cname, cols, _M, _h) in enumerate(blocks):
        if bi:
            cur += cell * CAT_GAP
        s0 = cur
        for c in cols:
            col_x.append(cur); xlab.append(c); cur += cell
        span.append((cname, s0, cur - cell))
    grid_w, nR = cur, len(rows)
    grid_h = nR * cell
    rot = np.deg2rad(LABEL_ROT)
    bar_gap, bar_h = 0.06, 0.14
    cat_h = FS * 1.35 / 72.0
    y_bar = grid_h + bar_gap
    y_cat = y_bar + bar_h + 0.05
    y_lead = y_cat + cat_h + 0.06
    lab_len = max(text_w(a, FS) for a in xlab)
    head_h = y_lead + LEADER_IN * np.sin(rot) + lab_len * np.sin(rot) + 0.10 - grid_h
    left_in = max(text_w(DISPLAY.get(m, m), FS) for m in rows) + 0.22
    plot_w = grid_w + lab_len * np.cos(rot) * 0.6 + 0.3
    fig_w = left_in + plot_w + 0.3
    fig_h = 0.35 + grid_h + head_h + FS * 1.6 / 72.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([left_in / fig_w, 0.35 / fig_h, plot_w / fig_w, (grid_h + head_h) / fig_h])
    ax.set_xlim(0, plot_w); ax.set_ylim(0, grid_h + head_h)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(nR):
        ax.add_patch(Rectangle((0, (nR - 1 - i) * cell), grid_w, cell, zorder=0, edgecolor="none",
                               facecolor=ROW_BG_A if i % 2 == 0 else ROW_BG_B))
    k = 0
    for bi, (cname, cols, Mx, _) in enumerate(blocks):
        S = scores[bi]
        for j in range(len(cols)):
            X = col_x[k] + cell / 2
            k += 1
            for i in range(nR):
                v = S[i, j]
                half, nexp = shape_of(v)
                ax.add_patch(Polygon(superellipse(X, (nR - 1 - i) * cell + cell / 2, half, nexp),
                                     closed=True, facecolor=shade(CASE_COLORS.get(cname, "#8C8C8C"), v),
                                     edgecolor=EDGE_C, linewidth=EDGE_W, zorder=3))
    for i, m in enumerate(rows):
        ax.text(-0.10, (nR - 1 - i) * cell + cell / 2, DISPLAY.get(m, m),
                ha="right", va="center", fontsize=FS, color="#1A1A1A")
    for cname, a, b in span:
        ax.add_patch(Rectangle((a + 0.006, y_bar), (b + cell - a) - 0.012, bar_h,
                               facecolor=CASE_COLORS.get(cname, "#8C8C8C"), edgecolor="none", zorder=2))
        ax.text((a + b + cell) / 2, y_cat, f"{cname} cells", ha="center", va="bottom",
                fontsize=FS, color="#1A1A1A")
    for j, a in enumerate(xlab):
        xc = col_x[j] + cell / 2
        xe = xc + LEADER_IN * np.cos(rot)
        ye = y_lead + LEADER_IN * np.sin(rot)
        ax.plot([xc, xe], [y_lead, ye], color="#6B7280", lw=0.7, zorder=2)
        ax.text(xe + 0.015 * np.cos(rot), ye + 0.015 * np.sin(rot), a, rotation=LABEL_ROT,
                rotation_mode="anchor", ha="left", va="center", fontsize=FS, color="#1A1A1A")
    fig.text(left_in / fig_w - 0.004, 1 - 0.06 / fig_h,
             "Contamination source composition   (larger & squarer & lighter = smaller share)",
             ha="left", va="top", fontsize=FS, color="#1A1A1A")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", facecolor="white", dpi=400)
    plt.close(fig)
    print(f">>> {out}")


def main():
    df = load_data()
    print("各方法、各 case 的可评估细胞数：")
    print(df.drop_duplicates(["method", "case"]).pivot(index="display", columns="case", values="n_cell")
          .to_string())
    blocks, rows = build_blocks(df)
    plot(blocks, rows)


if __name__ == "__main__":
    main()