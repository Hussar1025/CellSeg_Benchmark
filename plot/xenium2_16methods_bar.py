# -*- coding: utf-8 -*-
"""
Fig. 3b（作图）—— xenium_2 上 16 个方法的转录本划分一致性柱状图
================================================================
读 fig3d_metrics_xenium2.py 产出的 metrics_transcript_xenium_2.csv，
以基线方法（Cellpose-3）为零点画双向柱状图，每个指标一张独立 PDF。
方法按字典序排列。超长的柱体在末端挖一道斜向留白表示已截断。

相对上一版的修改：
  · METHOD_TYPE 补全到 16 个方法；基线为 Cellpose-3；路径改到 SpaSegBench/result
  · 柱体加粗、行距收紧（BAR_HEIGHT 0.64 -> 0.86，ROW_H_IN 0.46 -> 0.36），柱间空隙更小
  · 所有方法现在都在同一批 qv>=20 转录本上评估，图像与转录本方法可以直接比较
画图逻辑不变。assign_rate 那一张要和前四张一起看——
归属率低的方法低 ARI 主要来自覆盖率而非归属错误。
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ==========================================================================
SPASEG = "/data/qiuyijia/SpaSegBench/result"
IN_CSV = f"{SPASEG}/metrics_transcript_xenium_2.csv"
OUT_DIR = f"{SPASEG}/figs_xenium_2"
FILE_FORMAT = "pdf"

REF_MODE = "baseline"
BASELINE = "Cellpose-3"
METRICS = ["ARI", "NMI", "HOM", "COM", "assign_rate"]
DATASET = None

METHOD_TYPE = {
    "Cellpose-SAM": "image", "Cellpose-3": "image", "CellSAM": "image", "Mesmer": "image",
    "StarDist": "image", "CelloType": "image", "UNSEG": "image",
    "ProSeg": "rna", "UCS": "rna", "ComSeg": "rna", "BOMS": "rna", "Cellist": "rna",
    "GeneSegNet": "rna", "Baysor": "rna", "Bering": "rna", "DISSECT": "rna",
}
SHOW_TYPE_MARK = False
TYPE_MARK = " †"

SORT_MODE = "alpha"
KEEP_BASELINE_ROW = True
SHOW_MISSING = True            # 没有结果的方法也占一行，标 "n/a"，保证 16 个方法都出现

# 柱长压缩：缩小长短柱之间的差距。柱长按 sign(Δ)·|Δ|^GAMMA 绘制，刻度仍标原始差值 Δ，
# 所以读数不变，只是长柱被压短、短柱被拉长。GAMMA=1 为线性（不压缩），0.5 为平方根压缩。
# 压缩后长柱不再一枝独秀，默认不再截断（CLIP=None）；如需线性 + 截断，设 GAMMA=1.0, CLIP="max_ratio"。
# 注意：压缩后柱长与差值不再成正比，图注须写明"横轴为平方根尺度"。
GAMMA = 1.0                 # 1.0 = 原始线性尺度（当前）；<1 为压缩尺度（已弃用，保留开关）
SYMMETRIC_X = True
PAD_FRAC = 0.10
CLIP = "max_ratio"          # 超长柱截断并加断口标记
CLIP_RATIO = 0.80
MIN_SPAN = 0.02
BRK_GAP_FRAC = 0.022
BRK_SLANT_FRAC = 0.016
BRK_POS_FRAC = 0.055

POS_COLOR, NEG_COLOR = "#ED7D31", "#4490A9"
FRAME_COLOR, FRAME_WIDTH = "#1A1A1A", 0.8
BAR_EDGE_WIDTH = 0.4
BAR_HEIGHT = 0.74            # 柱体占行高的比例；越大柱间空隙越小（1.0 = 无空隙）

PLOT_W_IN = 2.0
ROW_H_IN = 0.14             # 每行高度（英寸）：柱更细、行距更小、整张图更矮
LEFT_PAD_IN = 0.08
RIGHT_IN, TOP_IN, BOTTOM_IN = 0.08, 0.06, 0.34
TICK_FS, ROW_FS, LABEL_FS = 7, 8, 8
N_XTICKS = 5

METRIC_LABEL = {"assign_rate": "Assignment rate"}


def text_width_in(text, fs):
    if not hasattr(text_width_in, "_fig"):
        text_width_in._fig = plt.figure(figsize=(1, 1), dpi=100)
    f = text_width_in._fig
    t = f.text(0, 0, str(text), fontsize=fs)
    f.canvas.draw()
    w = t.get_window_extent(renderer=f.canvas.get_renderer()).width / f.dpi
    t.remove()
    return w


def label_of(m):
    return m + TYPE_MARK if (SHOW_TYPE_MARK and METHOD_TYPE.get(m) == "image") else m


def load(path):
    df = pd.read_csv(path)
    if DATASET is not None:
        df = df[df["dataset"].astype(str) == DATASET]
    ds = df["dataset"].astype(str).unique()
    if len(ds) != 1:
        raise SystemExit(f"csv 里有 {len(ds)} 个数据集 {list(ds)}，请设置 DATASET")
    have = [m for m in METRICS if m in df.columns]
    if set(METRICS) - set(have):
        print(f"[warn] csv 缺这些指标列：{sorted(set(METRICS) - set(have))}")
    if not have:
        raise SystemExit("一个指标列都没有")
    df = df.drop_duplicates("method").set_index("method")
    unk = [m for m in df.index if m not in METHOD_TYPE]
    if unk:
        print(f"[warn] 这些方法没登记类别：{unk}")
    miss = [m for m in METHOD_TYPE if m not in df.index]
    if miss:
        print(f"[warn] csv 里缺这些方法：{miss}" +
              ("（图中保留空行并标 n/a）" if SHOW_MISSING else "（不会出现在图中）"))
    return df, have, ds[0]


def reference_of(df, metric):
    col = df[metric].astype(float)
    if REF_MODE == "median":
        return float(col.median()), "vs median"
    if REF_MODE == "mean":
        return float(col.mean()), "vs mean"
    if REF_MODE == "baseline":
        return float(col.loc[BASELINE]), f"vs {BASELINE}"
    if REF_MODE == "absolute":
        return 0.0, ""
    raise ValueError(REF_MODE)


def clip_threshold(v):
    a = np.abs(np.asarray(v, dtype=float))
    a = a[np.isfinite(a) & (a > 0)]
    if CLIP is None or len(a) == 0:
        return None
    if CLIP == "max_ratio":
        t = float(a.max()) * CLIP_RATIO
        return t if t > 0 else None
    return float(CLIP)


def xlim_of(v, clip_at):
    if REF_MODE == "absolute":
        return 0.0, float(np.nanmax(v)) * (1 + PAD_FRAC)
    m = max(clip_at if clip_at else float(np.nanmax(np.abs(v))), MIN_SPAN / 2)
    if not SYMMETRIC_X:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        span = max(hi - lo, MIN_SPAN)
        return min(lo, 0) - span * PAD_FRAC, max(hi, 0) + span * PAD_FRAC
    return -m * (1 + PAD_FRAC), m * (1 + PAD_FRAC)


def draw_break_gap(ax, x_tip, y, bar_h, span):
    from matplotlib.patches import Polygon
    sign = 1.0 if x_tip > 0 else -1.0
    gap, slant = span * BRK_GAP_FRAC, span * BRK_SLANT_FRAC
    xc = x_tip - sign * span * BRK_POS_FRAC
    h = bar_h / 2 * 1.10
    x1, x2 = xc - sign * gap / 2, xc + sign * gap / 2
    ax.add_patch(Polygon([(x1 - slant / 2, y - h), (x1 + slant / 2, y + h),
                          (x2 + slant / 2, y + h), (x2 - slant / 2, y - h)],
                         closed=True, facecolor="white", edgecolor="none", zorder=5))
    for x in (x1, x2):
        ax.plot([x - slant / 2, x + slant / 2], [y - h, y + h], color=FRAME_COLOR,
                linewidth=BAR_EDGE_WIDTH, solid_capstyle="butt", zorder=6)


def fwd(d):
    return np.sign(d) * np.abs(d) ** GAMMA


def nice_ticks(vmax):
    """原始单位下的刻度（正负对称）：每侧 2 个 —— 不超过最大值的最大整数刻度，
    以及在绘图尺度上大约位于它一半处的刻度，避免 0 附近挤在一起。"""
    if vmax <= 0:
        return [0.0]
    cand = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0]
    pos = [c for c in cand if c <= vmax * 1.0001] or [vmax]
    big = pos[-1]
    half = min(pos[:-1] or [big / 2], key=lambda c: abs(fwd(c) - fwd(big) / 2))
    ticks = sorted({half, big})
    return [-t for t in reversed(ticks)] + [0.0] + ticks


def draw_one(df, metric, ds, out_path):
    base, ref_txt = reference_of(df, metric)
    rows = list(df.index) + ([m for m in METHOD_TYPE if m not in df.index] if SHOW_MISSING else [])
    rows = [m for m in rows if KEEP_BASELINE_ROW or REF_MODE != "baseline" or m != BASELINE]
    delta = df[metric].astype(float).reindex(rows) - base
    rows = (sorted(rows, key=str.lower) if SORT_MODE == "alpha"
            else list(delta.sort_values(ascending=False).index))
    v = delta.loc[rows].to_numpy()
    missing = ~np.isfinite(v)
    labels = [label_of(m) for m in rows]
    n_row = len(rows)

    left_in = max(text_width_in(s, ROW_FS) for s in labels) + LEFT_PAD_IN
    fig_w = left_in + PLOT_W_IN + RIGHT_IN
    fig_h = BOTTOM_IN + n_row * ROW_H_IN + TOP_IN
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([left_in / fig_w, BOTTOM_IN / fig_h,
                       PLOT_W_IN / fig_w, n_row * ROW_H_IN / fig_h])

    ypos = np.arange(n_row)[::-1]
    clip_at = clip_threshold(v)
    v0 = np.where(missing, 0.0, v)
    v_lin = np.clip(v0, -clip_at, clip_at) if clip_at else v0.copy()
    v_draw = fwd(v_lin)
    clipped = (np.abs(v0) >= (clip_at - 1e-12)) & ~missing if clip_at else np.zeros(n_row, bool)

    ax.barh(ypos[~missing], v_draw[~missing], height=BAR_HEIGHT,
            color=[POS_COLOR if t > 0 else NEG_COLOR for t in v[~missing]],
            edgecolor=FRAME_COLOR, linewidth=BAR_EDGE_WIDTH, zorder=3)
    if REF_MODE != "absolute":
        ax.axvline(0, color=FRAME_COLOR, linewidth=FRAME_WIDTH, zorder=4)

    m_ = max(float(np.nanmax(np.abs(v_lin))) if len(v_lin) else 0.0, MIN_SPAN / 2)
    lo, hi = -fwd(m_) * (1 + PAD_FRAC), fwd(m_) * (1 + PAD_FRAC)
    for i in np.nonzero(missing)[0]:
        ax.text(0, ypos[i], "  n/a", ha="left", va="center", fontsize=TICK_FS - 1,
                color="#8C8C8C", zorder=5)
    ax.set_xlim(lo, hi)
    if clipped.any():
        for i in np.nonzero(clipped)[0]:
            draw_break_gap(ax, v_draw[i], ypos[i], BAR_HEIGHT, hi - lo)
        print(f"    {metric}: 截断阈值 {clip_at:.4f}，截断 "
              f"{', '.join(rows[i] for i in np.nonzero(clipped)[0])}")

    ax.set_ylim(-0.6, n_row - 0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=ROW_FS)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=TICK_FS, length=3.5,
                   width=FRAME_WIDTH, direction="out", color=FRAME_COLOR)
    ticks = nice_ticks(m_)
    ax.set_xticks([fwd(t) for t in ticks])
    ax.set_xticklabels([("0" if t == 0 else f"{t:+g}") for t in ticks])
    ax.grid(True, axis="x", linestyle="--", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(FRAME_COLOR)
        ax.spines[side].set_linewidth(FRAME_WIDTH)

    lab = METRIC_LABEL.get(metric, metric)
    scale_txt = "" if GAMMA == 1 else ("  [√ scale]" if abs(GAMMA - 0.5) < 1e-9 else f"  [|Δ|^{GAMMA:g} scale]")
    ax.set_xlabel((f"Δ{lab} ({ref_txt})" if ref_txt else lab) + scale_txt, fontsize=LABEL_FS)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  已保存：{out_path}")


def main():
    src, outdir = IN_CSV, OUT_DIR
    if len(sys.argv) == 3:
        src, outdir = sys.argv[1:3]
    os.makedirs(outdir, exist_ok=True)
    df, metrics, ds = load(src)
    if REF_MODE == "baseline" and BASELINE not in df.index:
        raise SystemExit(f"基线 '{BASELINE}' 不在数据里。方法：{sorted(df.index)}")
    n_img = sum(1 for m in df.index if METHOD_TYPE.get(m) == "image")
    print(f"数据集 {ds} | 方法 {len(df)} 个（图像 {n_img}，RNA {len(df)-n_img}）"
          f" | 参照 {REF_MODE} | 排序 {SORT_MODE} | 输出 {FILE_FORMAT}")
    show = pd.DataFrame({m: df[m].astype(float) - reference_of(df, m)[0] for m in metrics})
    print("零点：" + ", ".join(f"{m}={reference_of(df,m)[0]:.4f}" for m in metrics))
    print(show.round(4).to_string())
    for met in metrics:
        draw_one(df, met, ds, os.path.join(outdir, f"bar_{ds}_{met}.{FILE_FORMAT}"))


if __name__ == "__main__":
    main()