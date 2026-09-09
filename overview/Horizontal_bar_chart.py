# -*- coding: utf-8 -*-
"""
方法稳健性合成柱状图
====================
纵轴 = 13 个方法，每个方法四根柱子代表四个指标，四种颜色与气泡矩阵的大类配色一致。
横轴刻度在上方，图例在下方。

数据只用两张汇总表：
  benchmark_full_omics.xlsx  第一个 sheet          -> 8 个 RNA 类方法
  benchmark_result_image.xlsx 图像分割方法汇总 sheet -> 5 个图像类方法
两表真正共有的指标只有 cell_ratio / precision / recall / f1 四个
（图像表没有 loc / count / vec 任何一列），正好四根柱子。

计算链路与气泡矩阵一致：
  1. 同一平台内逐个数据集求秩（按指标方向校正，秩 1 = 最好）
  2. 秩归一化到 0~1，消除各数据集参评方法数不等的偏差
  3. 平台内按 GT 细胞数的对数加权平均
  4. 各平台等权平均
再逐指标在方法之间做 min-max 归一化（该指标最差的方法 = 0，最好 = 1），
绘图长度取 0.05 + 归一化值 × 0.9，最差的柱子仍有一小段可见，最长的也不顶到右边框。

这张图回答的是：一个方法在四项关键指标上是全面平稳，还是长短板明显。
柱长参差不齐 = 有明显的强项弱项；四根齐平 = 表现均衡。
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42       # TrueType 嵌入，便于 Illustrator 编辑
plt.rcParams["ps.fonttype"] = 42


# ==========================================================================
# 路径（已写死）
# ==========================================================================
PATH_OMICS = r"C:\Users\阿飞\Desktop\benchmark_full_omics.xlsx"
PATH_IMAGE = r"C:\Users\阿飞\Desktop\benchmark_result_image.xlsx"
OUT_DIR = r"C:\Users\阿飞\Desktop"
OUT_NAME = "robustness_bars"
FILE_FORMAT = "pdf"

# 四个共有指标：(统一列名, 图例显示名, 方向)
# direction: high=越大越好, low=越小越好, target1=越接近 1 越好
METRICS = [
    ("cell_ratio", "Cell count ratio", "target1"),
    ("precision",  "Precision",        "high"),
    ("recall",     "Recall",           "high"),
    ("f1",         "F1",               "high"),
]

# 与气泡矩阵的四个大类同色号
BAR_COLORS = ["#4490A9", "#ED7D31", "#5B9BD5", "#FFC000"]

# 秩计算
RANK_MODE = "percentile"        # "percentile" / "raw"
WEIGHT_MODE = "log"             # "log" / "linear" / "sqrt" / "equal"

# 绘图长度 = BAR_FLOOR + 归一化值 × BAR_SPAN
# 取 0.05 + v×0.9，最短的柱子有 0.05 看得见，最长的到 0.95 也不会顶到右边框
BAR_FLOOR, BAR_SPAN = 0.05, 0.90

SORT_MODE = "alpha"             # "alpha" 字典序 / "mean" 按四项均值从高到低

# 外观
# 柱体描边：宽度设为 0 就完全没有框线，调大则加粗
BAR_EDGE_COLOR, BAR_EDGE_WIDTH = "#1A1A1A", 0.0
FRAME_COLOR, FRAME_WIDTH = "#1A1A1A", 0.8
# 绘图区边框：空元组 = 完全不要框线；想留某几条就填，如 ("left", "top")
FRAME_SIDES = ()
GROUP_H = 1.0                   # 每个方法占的纵向高度（数据单位）
BAR_FRAC = 0.82                 # 四根柱子合计占 GROUP_H 的比例，其余是组间空隙
BAR_PACK = 1.0                  # 组内每根柱子占自己槽位的比例，1.0 = 四根完全贴紧
ROW_BG_A, ROW_BG_B = "#F2F2F2", "#FFFFFF"      # 行底色灰白交替

PLOT_W_IN = 3.2                 # 画布区宽度，调小整张图更窄
GROUP_H_IN = 0.62               # 每个方法占的高度（英寸）
LEFT_PAD_IN = 0.26
RIGHT_IN = 0.30
TOP_IN = 0.62                   # 上方留给横轴刻度与轴标题
BOTTOM_IN = 0.86                # 下方留给图例

FS = 12                         # 全图统一字号
LEG_FS = 11
N_XTICKS = 5
LEGEND_COLSPACING = 3.2         # 图例四个色块之间的间距，太大会超出图宽
LEGEND_HANDLE_LEN = 1.4

DISPLAY_NAMES = {
    "proseg": "ProSeg", "ucs": "UCS", "boms": "BOMS", "baysor": "Baysor",
    "bering": "Bering", "cellist": "Cellist", "comseg": "ComSeg",
    "genesegnet": "GeneSegNet",
    "cellpose": "Cellpose", "stardist": "StarDist", "cellsam": "CellSAM",
    "mesmer": "Mesmer", "cellotype": "CelloType",
}


# ==========================================================================
# 读两张表并合并
# ==========================================================================
def normalize_method(name):
    """去掉 (supervised) 之类的括号后缀；大小写不敏感查规范名。
    图像表里 'Cellpose' 和 'cellpose' 各存了一份，不合并会变成两个方法。"""
    base = re.sub(r"\s*[（(].*?[)）]\s*$", "", str(name)).strip()
    return DISPLAY_NAMES.get(base.lower(), base)


def load_both(path_omics, path_image):
    xl = pd.ExcelFile(path_omics)
    a = pd.read_excel(xl, sheet_name=xl.sheet_names[0])
    a = a[a["method"].notna() & a["dataset"].notna()].copy()
    a = a.rename(columns={"gt_cells": "_gt"})
    a["_type"] = "rna"

    # 图像表必须指定 sheet，第一个 sheet 装的仍是 RNA 方法
    b = pd.read_excel(path_image, sheet_name="图像分割方法汇总")
    b = b[b["method"].notna() & b["dataset"].notna()].copy()
    if "Dice" in b.columns:                       # 表里有大量占位空行
        b = b[pd.to_numeric(b["Dice"], errors="coerce").notna()]
    b = b.rename(columns={"Precision": "precision", "Recall": "recall",
                          "F1": "f1", "GT细胞数": "_gt"})
    # 图像表没有现成的 cell_ratio，用 预测细胞数 / GT细胞数 现算
    b["cell_ratio"] = (pd.to_numeric(b["预测细胞数"], errors="coerce")
                       / pd.to_numeric(b["_gt"], errors="coerce").replace(0, np.nan))
    b["_type"] = "image"

    cols = [m[0] for m in METRICS] + ["_gt"]
    keep = ["method", "dataset", "_type"] + cols
    d = pd.concat([a.reindex(columns=keep), b.reindex(columns=keep)],
                  ignore_index=True)
    d["method"] = d["method"].map(normalize_method)
    # 'merfish_2 肝1' 和 'merfish_2' 是同一个数据集
    d["ds"] = d["dataset"].astype(str).str.split().str[0]
    d["platform"] = d["ds"].str.split("_").str[0]
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    agg = d.groupby(["platform", "ds", "method", "_type"], as_index=False)[cols].mean()
    gt = agg.groupby("ds")["_gt"].median()
    n_img = agg.loc[agg["_type"] == "image", "method"].nunique()
    print(f"  合并后 方法 {agg['method'].nunique()} 个"
          f"（图像类 {n_img}，RNA 类 {agg['method'].nunique() - n_img}）| "
          f"数据集 {agg['ds'].nunique()} 个 | 平台 {sorted(agg['platform'].unique())}")
    return agg, gt


def to_score(s, direction):
    if direction == "high":
        return s
    if direction == "low":
        return -s
    if direction == "target1":          # cell_ratio 越接近 1 越好
        return -(s - 1.0).abs()
    raise ValueError(direction)


def dataset_weight(gt):
    gt = float(gt) if gt is not None else np.nan
    if not np.isfinite(gt) or gt <= 0:
        return None
    if WEIGHT_MODE == "log":
        return float(np.log10(max(gt, 10.0)))   # 下限 10 防止 log 出负数
    if WEIGHT_MODE == "sqrt":
        return float(np.sqrt(gt))
    if WEIGHT_MODE == "equal":
        return 1.0
    return gt


def rank_matrix(agg, gt_cells):
    """平台内加权 -> 平台间等权。同一数据集上 RNA 与图像方法一起排名。"""
    methods = sorted(agg["method"].unique())
    out = pd.DataFrame(index=methods, columns=[m[0] for m in METRICS], dtype=float)
    for col, _disp, direction in METRICS:
        plat = {}
        for platform, pdf in agg.groupby("platform"):
            num, den = {}, {}
            for ds, ddf in pdf.groupby("ds"):
                vals = to_score(ddf.set_index("method")[col], direction).dropna()
                n = len(vals)
                if n < 2:                # 只有一个方法，无从比较
                    continue
                rank = vals.rank(ascending=False, method="average")   # 秩 1 = 最好
                norm = ((n - rank + 0.5) / n if RANK_MODE == "percentile"
                        else (n - rank) / max(n - 1, 1))
                w = dataset_weight(gt_cells.get(ds, np.nan))
                if w is None:
                    continue
                for m, v in norm.items():
                    num[m] = num.get(m, 0.0) + w * v
                    den[m] = den.get(m, 0.0) + w
            if den:
                plat[platform] = {m: num[m] / den[m] for m in num}
        for m in methods:
            vs = [p[m] for p in plat.values() if m in p]
            out.loc[m, col] = float(np.mean(vs)) if vs else np.nan
    return out


def normalize_columns(mat):
    """逐指标在方法之间做 min-max 归一化：该指标最差的方法 = 0，最好 = 1。"""
    out = mat.copy().astype(float)
    for c in out.columns:
        lo, hi = out[c].min(skipna=True), out[c].max(skipna=True)
        out[c] = 0.5 if not np.isfinite(hi - lo) or hi - lo < 1e-12 \
            else (out[c] - lo) / (hi - lo)
    return out


# ==========================================================================
# 作图
# ==========================================================================
def text_width_in(text, fs):
    if not hasattr(text_width_in, "_fig"):
        text_width_in._fig = plt.figure(figsize=(1, 1), dpi=100)
    f = text_width_in._fig
    t = f.text(0, 0, str(text), fontsize=fs)
    f.canvas.draw()
    w = t.get_window_extent(renderer=f.canvas.get_renderer()).width / f.dpi
    t.remove()
    return w


def draw(norm_mat, out_path):
    rows = (sorted(norm_mat.index, key=str.lower) if SORT_MODE == "alpha"
            else list(norm_mat.mean(axis=1).sort_values(ascending=False).index))
    n_row, n_met = len(rows), len(METRICS)

    left_in = max(text_width_in(s, FS) for s in rows) + LEFT_PAD_IN
    fig_w = left_in + PLOT_W_IN + RIGHT_IN
    fig_h = TOP_IN + n_row * GROUP_H_IN + BOTTOM_IN
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([left_in / fig_w, BOTTOM_IN / fig_h,
                       PLOT_W_IN / fig_w, n_row * GROUP_H_IN / fig_h])

    bar_h = GROUP_H * BAR_FRAC / n_met
    # 第一个方法在最上面；组内第一个指标也在上面
    centers = np.arange(n_row)[::-1] * GROUP_H

    # 行底色灰白交替，铺满整个绘图区
    for i in range(n_row):
        y0 = centers[i] - GROUP_H / 2
        ax.add_patch(plt.Rectangle((0, y0), 1.0, GROUP_H, zorder=0, edgecolor="none",
                                   facecolor=ROW_BG_A if i % 2 == 0 else ROW_BG_B))

    edge_kw = (dict(edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
               if BAR_EDGE_WIDTH > 0 else dict(edgecolor="none", linewidth=0))
    for j, (col, _disp, _d) in enumerate(METRICS):
        # 组内自上而下排列：j=0 在最上
        offs = GROUP_H * BAR_FRAC / 2 - bar_h * (j + 0.5)
        vals = norm_mat.loc[rows, col].to_numpy(dtype=float)
        lens = BAR_FLOOR + np.nan_to_num(vals, nan=0.0) * BAR_SPAN
        ax.barh(centers + offs, lens, height=bar_h * BAR_PACK,
                color=BAR_COLORS[j % len(BAR_COLORS)], zorder=3, **edge_kw)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(centers.min() - GROUP_H / 2, centers.max() + GROUP_H / 2)
    ax.set_yticks(centers)
    ax.set_yticklabels(rows, fontsize=FS)
    ax.tick_params(axis="y", length=0)

    # 横轴刻度放到上方，0 到 1 均匀分布
    ticks = np.linspace(0, 1, N_XTICKS)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=FS)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", labelsize=FS, length=3.5, width=FRAME_WIDTH,
                   direction="out", color=FRAME_COLOR)
    ax.set_xlabel("Normalized rank within each metric   "
                  "(0 = worst method, 1 = best)", fontsize=FS, labelpad=6)
    # 注：柱长 = 0.05 + 归一化值 × 0.9，所以最差的柱子仍有一小段、最长的不顶右框

    ax.grid(True, axis="x", linestyle="--", alpha=0.28, zorder=1)
    ax.set_axisbelow(False)
    for side in ("top", "right", "bottom", "left"):
        keep = side in FRAME_SIDES
        ax.spines[side].set_visible(keep)
        if keep:
            ax.spines[side].set_color(FRAME_COLOR)
            ax.spines[side].set_linewidth(FRAME_WIDTH)

    # 图例放在图的下方，四个色块间距加大但仍在图宽以内
    handles = [Patch(facecolor=BAR_COLORS[j % len(BAR_COLORS)],
                     label=METRICS[j][1], **edge_kw) for j in range(n_met)]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(left_in / fig_w + PLOT_W_IN / fig_w / 2, 0.012),
               ncol=n_met, frameon=False, fontsize=LEG_FS,
               columnspacing=LEGEND_COLSPACING, handlelength=LEGEND_HANDLE_LEN,
               handletextpad=0.6)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  已保存：{out_path}")


def main():
    p_o, p_i, outdir = PATH_OMICS, PATH_IMAGE, OUT_DIR
    if len(sys.argv) == 4:      # 仅用于换路径调试
        p_o, p_i, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)

    agg, gt = load_both(p_o, p_i)
    mat = rank_matrix(agg, gt)
    print("\n秩矩阵（0~1，越大越好）：")
    print(mat.round(3).to_string())

    nm = normalize_columns(mat)
    print("\n逐指标归一化后（0 = 该指标最差的方法，1 = 最好）：")
    print(nm.round(3).to_string())
    print("\n四项均值（越高越均衡地好）与极差（越小越平稳）：")
    summary = pd.DataFrame({"mean": nm.mean(axis=1), "range": nm.max(axis=1) - nm.min(axis=1)})
    print(summary.round(3).sort_values("mean", ascending=False).to_string())

    draw(nm, os.path.join(outdir, f"{OUT_NAME}.{FILE_FORMAT}"))


if __name__ == "__main__":
    main()