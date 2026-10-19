#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像分割方法 benchmark 归一化热图
==========================================================
数据来源:工作簿里的 "图像分割方法汇总" sheet
行  = 方法 × 数据集(每个 method-dataset 组合一行)
列  = 各项指标(核心六指标:Dice / IoU / PQ / AJI / F1 / 边界F1)
颜色 = 该指标按列归一化后的值(浅红=更好,浅蓝=更差)
格内数字 = 原始指标值(颜色才是归一化结果)
无边框、无图例。缺失格用浅灰,不标注。

用法(不带参数即可,读写死的桌面文件,输出到桌面):
    python plot_seg_heatmap.py
    python plot_seg_heatmap.py --norm minmax
    python plot_seg_heatmap.py --row-order dataset
    python plot_seg_heatmap.py --metrics Dice,IoU,PQ,AJI,F1,边界F1
    python plot_seg_heatmap.py --add-counts          # 额外把计数列(TP/FP/FN/过分割…)也画进去
    python plot_seg_heatmap.py --annot-normalized    # 格内显示归一化值
    python plot_seg_heatmap.py --colorbar            # 需要色标时再加
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# 0. 写死的输入/输出路径 + sheet
# ============================================================
DEFAULT_XLSX = r"C:\Users\阿飞\Desktop\benchmark_full_all.xlsx"
DEFAULT_OUTDIR = os.path.dirname(DEFAULT_XLSX)     # 输出到桌面(与输入同目录)
DEFAULT_SHEET = "图像分割方法汇总"

# 指标展示顺序(存在且至少有一个非空的才会画)。核心六指标,都是越大越好的质量指标。
QUALITY_METRICS = [
    "Dice", "IoU", "PQ", "AJI", "F1", "Boundary F1",
]
# 计数类列(随数据集规模变化,默认不进热图;--add-counts 时追加)。这些里 FP/FN/过/欠 越小越好。
COUNT_METRICS = ["过分割(FP)", "欠分割(FN)", "TP", "FP", "FN"]

# 越小越好 → 归一化后翻转,列名后加 ↓
LOWER_IS_BETTER = {"过分割(FP)", "欠分割(FN)", "FP", "FN"}
# 无绝对好坏方向(cell_ratio 越接近 1 越好,这里只按原值上色,不翻转)
NEUTRAL = {"cell_ratio"}

# 非指标列(不会当成可画指标)
NON_METRIC_COLS = {"method", "dataset", "reference", "mDA_note",
                   "GT细胞数", "预测细胞数", "gt_cells", "pred_cells"}

# 方法名规范化:统一大小写,保留 (+15px) 之类的后缀
METHOD_CANON = {
    "cellpose": "Cellpose", "stardist": "StarDist", "cellsam": "CellSAM",
    "mesmer": "Mesmer", "cellotype": "CelloType", "celltype": "CelloType",
}

PLATFORM_ORDER = ["xenium", "merfish", "cosmx", "stereo"]


# ============================================================
# 1. 中文字体
# ============================================================
def setup_font(user_font: str | None = None) -> str:
    prefer = [user_font] if user_font else []
    prefer += ["Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC",
               "Source Han Sans SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
               "Arial Unicode MS", "DejaVu Sans"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((f for f in prefer if f and f in installed), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen] + [f for f in prefer if f and f in installed]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# ============================================================
# 2. 读数据 / 规范名字
# ============================================================
def load_sheet(xlsx: str, sheet: str) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx)
    if sheet not in xls.sheet_names:
        raise SystemExit(f"[错误] 没有 sheet {sheet!r}。可用:{xls.sheet_names}")
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def canon_method(m: str) -> str:
    s = str(m).strip()
    mm = re.match(r"^([A-Za-z]+)(.*)$", s)
    if not mm:
        return s
    stem, rest = mm.group(1), mm.group(2)
    return (METHOD_CANON.get(stem.lower(), stem) + rest).strip()


def dataset_base(name: str) -> str:
    m = re.match(r"^([a-zA-Z]+_\d+)", str(name).strip())
    return m.group(1) if m else str(name).strip()


def platform_of(base: str) -> str:
    return base.split("_")[0].lower()


def ds_sort_key(base: str):
    plat = platform_of(base)
    pi = PLATFORM_ORDER.index(plat) if plat in PLATFORM_ORDER else len(PLATFORM_ORDER)
    m = re.search(r"_(\d+)", base)
    return (pi, plat, int(m.group(1)) if m else 99, base)


# ============================================================
# 3. 清洗 / 去重
# ============================================================
def tidy(df: pd.DataFrame, dedup: str) -> pd.DataFrame:
    df = df.copy()
    for need in ("method", "dataset"):
        if need not in df.columns:
            raise SystemExit(f"[错误] 缺列 {need!r}。实际:{list(df.columns)}")
    df["method"] = df["method"].map(canon_method)
    df["ds_base"] = df["dataset"].map(dataset_base)
    df["platform"] = df["ds_base"].map(platform_of)

    # 标签:同一 base 取最长(信息最全)的数据集写法
    var_label: dict[str, str] = {}
    for raw in df["dataset"].astype(str):
        b = dataset_base(raw)
        if len(raw.strip()) > len(var_label.get(b, "")):
            var_label[b] = raw.strip()
    df["ds_label"] = df["ds_base"].map(lambda b: var_label.get(b, b))

    # cell_ratio = 预测/GT
    gt = pd.to_numeric(df.get("GT细胞数"), errors="coerce") if "GT细胞数" in df else pd.Series(np.nan, index=df.index)
    pred = pd.to_numeric(df.get("预测细胞数"), errors="coerce") if "预测细胞数" in df else pd.Series(np.nan, index=df.index)
    df["cell_ratio"] = pred / gt.replace(0, np.nan)

    # 全部候选指标转数值
    for c in QUALITY_METRICS + COUNT_METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if dedup != "none":
        present = [c for c in (QUALITY_METRICS + COUNT_METRICS + ["cell_ratio"]) if c in df.columns]
        df["__nc__"] = df[present].notna().sum(axis=1)
        df = (df.sort_values("__nc__", ascending=False)
                .drop_duplicates(["method", "ds_base"], keep="first")
                .drop(columns="__nc__"))
    return df


# ============================================================
# 4. 归一化(每列独立)
# ============================================================
def normalize_col(s: pd.Series, how: str, lower_better: bool) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    valid = x.dropna()
    out = pd.Series(np.nan, index=x.index, dtype=float)
    if len(valid) == 0:
        return out
    if len(valid.unique()) == 1:
        out.loc[valid.index] = 0.5
    elif how == "rank":
        out.loc[valid.index] = valid.rank(pct=True)          # 百分位秩 (0,1]
    elif how == "minmax":
        lo, hi = valid.min(), valid.max()
        out.loc[valid.index] = (valid - lo) / (hi - lo)
    elif how == "robust":
        lo, hi = valid.quantile(0.02), valid.quantile(0.98)
        if hi == lo:
            lo, hi = valid.min(), valid.max()
        out.loc[valid.index] = ((valid - lo) / (hi - lo)).clip(0, 1)
    elif how == "zscore":
        mu, sd = valid.mean(), valid.std(ddof=0)
        z = (valid - mu) / (sd if sd else 1.0)
        out.loc[valid.index] = (z.clip(-2, 2) + 2) / 4
    else:
        raise SystemExit(f"[错误] 未知归一化方式 {how!r}")
    if lower_better:
        out = 1.0 - out
    return out


# ============================================================
# 5. 画热图
# ============================================================
def make_cmap(name: str | None):
    if name:
        return plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        "light_rb",
        ["#a7c7e0", "#cfe0ee", "#f2f1ee", "#f6ded9", "#e8a6a6"])


def fmt_val(v: float) -> str:
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def draw(df: pd.DataFrame, metrics: list[str], args) -> plt.Figure:
    if args.row_order == "dataset":
        df = df.sort_values(by=["ds_base", "method"],
                            key=lambda col: col.map(ds_sort_key) if col.name == "ds_base" else col)
    else:
        df = df.sort_values(by=["method", "ds_base"],
                            key=lambda col: col.map(ds_sort_key) if col.name == "ds_base" else col)
    df = df.reset_index(drop=True)

    row_labels = [f"{m} × {d}" for m, d in zip(df["method"], df["ds_label"])]
    col_labels = [c + (" ↓" if c in LOWER_IS_BETTER else "") for c in metrics]

    norm_mat = np.full((len(df), len(metrics)), np.nan)
    raw_mat = np.full((len(df), len(metrics)), np.nan)
    for j, c in enumerate(metrics):
        raw = pd.to_numeric(df[c], errors="coerce")
        nrm = normalize_col(raw, args.norm, c in LOWER_IS_BETTER and c not in NEUTRAL)
        raw_mat[:, j] = raw.values
        norm_mat[:, j] = nrm.values

    masked = np.ma.masked_invalid(norm_mat)
    cmap = make_cmap(args.cmap).copy()
    cmap.set_bad(color="#e6e6e6")
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig_w = max(args.fig_w, 3.4 + args.cell_w * len(metrics))
    fig_h = max(args.fig_h, 1.6 + args.cell_h * len(df))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)

    ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto",
              interpolation="nearest", origin="upper")

    ax.set_xticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(col_labels, rotation=42, ha="left", fontsize=args.col_fs)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for t, c in zip(ax.get_xticklabels(), metrics):
        if c in LOWER_IS_BETTER:
            t.set_color("#7a4a4a")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(row_labels, fontsize=args.row_fs)

    for s in ax.spines.values():
        s.set_visible(False)

    if args.annot:
        for i in range(len(df)):
            for j in range(len(metrics)):
                v_raw, v_nrm = raw_mat[i, j], norm_mat[i, j]
                if np.isnan(v_nrm):
                    continue
                txt = f"{v_nrm:.2f}" if args.annot_normalized else fmt_val(v_raw)
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=args.annot_fs, color="#242424")

    if args.colorbar:
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          fraction=0.018, pad=0.01)
        cb.set_label("按列归一化(浅红=更好 / 浅蓝=更差)", fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=8)

    fig.tight_layout()
    return fig


# ============================================================
# 6. main
# ============================================================
def main(argv=None):
    p = argparse.ArgumentParser(
        description="图像分割 benchmark 归一化热图:行=方法×数据集,列=核心六指标,浅红/浅蓝",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--xlsx", default=DEFAULT_XLSX)
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--prefix", default="heatmap_seg")
    p.add_argument("--metrics", default=None,
                   help="要展示的指标,逗号分隔;默认核心六指标")
    p.add_argument("--add-counts", action="store_true",
                   help="把计数类列(TP/FP/FN/过分割/欠分割)也画进去")
    p.add_argument("--norm", default="rank",
                   choices=["rank", "minmax", "robust", "zscore"],
                   help="每列归一化方式(默认 rank,抗极端值)")
    p.add_argument("--dedup", default="complete",
                   choices=["complete", "none"], help="同组合多行时是否只留最全的一条")
    p.add_argument("--methods", default=None, help="只画这些方法,逗号分隔")
    p.add_argument("--datasets", default=None, help="只画这些数据集(基础 id),逗号分隔")
    p.add_argument("--row-order", default="method", choices=["method", "dataset"],
                   help="行排序:method=按方法聚在一起,dataset=按数据集聚在一起")
    p.add_argument("--cmap", default=None, help="覆盖默认浅红/浅蓝配色(如 RdBu_r)")
    p.add_argument("--no-annot", dest="annot", action="store_false", help="不写格内数字")
    p.add_argument("--annot-normalized", action="store_true",
                   help="格内显示归一化值而非原始值")
    p.add_argument("--colorbar", action="store_true", help="显示色标(默认不显示)")
    p.add_argument("--cell-w", type=float, default=0.72, help="每列宽(英寸)")
    p.add_argument("--cell-h", type=float, default=0.32, help="每行高(英寸)")
    p.add_argument("--annot-fs", type=float, default=6.8, help="格内字号")
    p.add_argument("--row-fs", type=float, default=8.0, help="行标签字号")
    p.add_argument("--col-fs", type=float, default=9.5, help="列标签字号")
    p.add_argument("--fig-w", type=float, default=11.0)
    p.add_argument("--fig-h", type=float, default=7.0)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--font", default=None)
    p.add_argument("--no-pdf", dest="save_pdf", action="store_false")
    p.set_defaults(annot=True, save_pdf=True)
    args = p.parse_args(argv)

    if not os.path.exists(args.xlsx):
        raise SystemExit(f"[错误] 找不到文件:{args.xlsx}")
    os.makedirs(args.outdir, exist_ok=True)

    font_used = setup_font(args.font)
    df = tidy(load_sheet(args.xlsx, args.sheet), args.dedup)

    if args.methods:
        df = df[df["method"].isin({canon_method(m.strip()) for m in args.methods.split(",")})]
    if args.datasets:
        df = df[df["ds_base"].isin({d.strip() for d in args.datasets.split(",")})]
    if df.empty:
        raise SystemExit("[错误] 过滤后无数据,检查 --methods / --datasets。")

    if args.metrics:
        want = [m.strip() for m in args.metrics.split(",")]
        metrics = [m for m in want if m in df.columns]
        missing = [m for m in want if m not in df.columns]
        if missing:
            print(f"[提示] 忽略数据里不存在的指标:{missing}")
    else:
        metrics = [m for m in QUALITY_METRICS if m in df.columns]
        if args.add_counts:
            metrics += [m for m in COUNT_METRICS if m in df.columns]
    metrics = [m for m in metrics if pd.to_numeric(df[m], errors="coerce").notna().any()]
    if not metrics:
        raise SystemExit("[错误] 没有任何可用指标列。")

    fig = draw(df, metrics, args)

    base = os.path.join(args.outdir, args.prefix)
    png = base + ".png"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    outs = [png]
    if args.save_pdf:
        pdf = base + ".pdf"
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        outs.append(pdf)
    plt.close(fig)

    long = df.copy()
    long["row"] = long["method"] + " × " + long["ds_label"]
    keep = ["row", "method", "ds_label"] + metrics
    long_csv = base + "_data.csv"
    long[keep].to_csv(long_csv, index=False, encoding="utf-8-sig")
    outs.append(long_csv)

    print(f"字体:{font_used}")
    print(f"行(方法×数据集):{len(df)}   列(指标):{len(metrics)}")
    print(f"归一化方式:{args.norm}    格内显示:{'归一化值' if args.annot_normalized else '原始值'}")
    print("输出:")
    for o in outs:
        print("  " + o)


if __name__ == "__main__":
    sys.exit(main())