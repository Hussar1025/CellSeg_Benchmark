# -*- coding: utf-8 -*-
"""
方法 × 指标 秩气泡矩阵（两套指标体系）
=====================================

图 1  RNA 类方法   <- benchmark_full_omics.xlsx / sheet '汇总'
图 2  图像分割方法 <- benchmark_result_image.xlsx / sheet '图像分割方法汇总'

注意：benchmark_result_image.xlsx 的第一个 sheet「全部结果汇总(19指标)」装的仍是
RNA 类方法，和 full_omics 的汇总页高度重复；图像分割方法（Cellpose / StarDist /
CellSAM / Mesmer / CelloType）在「图像分割方法汇总」这个 sheet 里，所以图 2 读的是它。

两套指标体系不同（RNA 方法没有像素掩膜，算不了 Dice/IoU/PQ），但都是 17 个指标、
四个大类、每类数量都是 4/4/5/4，两张图可以上下拼在一起对齐。

每格数值的计算链路：
  1. 同一平台内，逐个数据集对各方法在该指标上求秩（按指标方向校正，秩 1 = 最好）
  2. 秩归一化到 0~1，消除各数据集参评方法数不等带来的偏差
  3. 平台内按各数据集 GT 细胞数的对数加权平均
  4. 各平台等权平均

排版：指标名倾斜排在大类名上方，用短斜线指回对应列；
      格子边长按数据里实际最大的图形推算，缝隙只留描边不相碰的物理下限；
      四个大类之间横向空出半个格子；图例各档单独用更松的行距；
      四种色相 = 四个大类；每列各自归一化取色，该列最差的格子直接用色号原色、最好的接近白
      （COLOR_SCOPE="global" 可切回绝对分数着色）；
      方圆变化走高次幂，只有最接近满分的一小段才呈方形，其余都是圆；
      行底色灰白交替；每个图形描黑边；最右侧单独一列图例。
      全部文字统一 Arial 12 号。

直接 python 运行即可，路径已写死。
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle

# 全图用 Arial。Liberation Sans 与 Arial 度量完全一致，作为 Linux/无 Arial 环境的兜底，
# 排版结果不会有差别。
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42       # 字体按 TrueType 嵌入，便于 Illustrator 编辑
plt.rcParams["ps.fonttype"] = 42


# ==========================================================================
# 路径（已写死）
# ==========================================================================
PATH_OMICS = r"C:\Users\阿飞\Desktop\benchmark_full_omics.xlsx"
PATH_IMAGE = r"C:\Users\阿飞\Desktop\benchmark_result_image.xlsx"
OUT_DIR = r"C:\Users\阿飞\Desktop"

FILE_FORMAT = "pdf"     # "pdf" / "png" / "svg"


# ==========================================================================
# 关键方法学开关
# ==========================================================================
# 各数据集参评方法数不等（RNA 表里 cosmx_1 有 8 个方法，有的数据集只有 2 个）。
# 直接平均原始秩会出问题：2 个方法里排第 2（垫底）和 8 个方法里排第 2（接近最好）
# 都是数字 2，结论完全反过来。所以默认把秩归一化到 0~1 再平均。
#   "percentile" : 归一化秩 = (n - rank + 0.5) / n   —— 推荐，默认
#   "raw"        : 原始秩线性反转，严格按字面的"求秩"
RANK_MODE = "percentile"

# 平台内按 GT 细胞数加权。xenium_1 有 190,965 个细胞，是 xenium_3(2,442) 的 78 倍，
# 线性加权下它一家占 xenium 平台 73% 权重；取对数后降到约 21%（等权是 16.7%）。
#   "log" / "linear" / "sqrt" / "equal"
WEIGHT_MODE = "log"

SORT_ROWS_BY_SCORE = True   # 方法按总分从高到低排；False 则按字典序
SHOW_VALUES = False         # 在图形中央标数值

# 两个表的四个大类要不要强行用同一组名字。
# 两张图现在是按【每类的指标数量和配色位置】对齐的，不是按语义——
# 第一位 RNA 是 Detection accuracy 而图像是 Boundary quality，两者并不对应，
# 所以这个开关必须保持 False，否则会张冠李戴。
UNIFIED_CATEGORY_NAMES = False
UNIFIED_NAMES = ["Detection", "Spatial agreement", "Per-cell fidelity", "Overall quality"]


# ==========================================================================
# 外观
# ==========================================================================
# 四个大类的色相。注意原始需求里第二个写的是 (237,12,549)，549 超出 RGB 范围，
# 按配套的 Office 橙 (237,125,49) 处理。
CAT_COLORS = ["#4490A9", "#ED7D31", "#5B9BD5", "#FFC000"]
ROW_BG_A, ROW_BG_B = "#EDEDED", "#FFFFFF"                   # 行底色灰白交替
SHAPE_EDGE_COLOR = "#1A1A1A"    # 图形描边
SHAPE_EDGE_WIDTH = 1.4

# 颜色按数值渐变：CAT_COLORS 里的四个色号【就是最深的那一端】，直接用原色不做压暗；
# 配合 COLOR_SCOPE="column"，每列最差的格子拿到的正是该大类指定的色号，
# 分数升高则向白色过渡。
GRADIENT = True
# 颜色的取值范围：
#   "column" 每个指标列各自归一化——该列最差的格子取色号原色，最好的取最浅色。
#            对比度拉满，但颜色变成【列内相对】，跨列不可比：
#            某列分数挤在 0.4~0.6，另一列跨 0.1~0.9，两列最深的格子看上去一样深。
#            大小和方圆仍是绝对值，所以跨列可能出现"颜色更深但图形更大"。
#   "global" 用绝对分数 0~1，跨列可比，但数据里最低分若是 0.14，
#            最深的那一档就永远用不上，颜色跨度被浪费。
COLOR_SCOPE = "column"
# LIGHT_MAX：最高分时向白色靠拢的程度。1.0 = 纯白；0.90 表示仍留约 10% 的色，
# 也就是"非常接近白色但带一点点很浅的颜色"。
LIGHT_MAX = 0.90
# LIGHT_GAMMA：渐变曲线。1.0 线性；小于 1 让中段更快变浅，
# 使高分区集中在接近白色的一小段，低分之间的差别相对更醒目。
LIGHT_GAMMA = 0.80

# 图形半径范围（英寸）。注意：整体等比缩放不会让视觉变紧——缝隙和图形会一起缩。
# 真正决定疏密的是【缝隙 / 图形直径】这个比值，而缝隙的下限被描边宽度卡死，
# 所以把图形调大、描边不变，相对缝隙就变小，看起来才更紧凑。
HALF_MIN_IN, HALF_MAX_IN = 0.077, 0.300

# 间距压到物理下限：描边以路径为中线绘制，向外还要占半个线宽，
# 两个相邻图形之间至少要留 "半个线宽 × 2" 才不会碰上。
GAP_EPS_IN = 0.0                                    # 余量，0 = 描边正好相切
EDGE_HALF_IN = SHAPE_EDGE_WIDTH / 72.0 / 2.0        # 描边向外扩出的一半
MIN_INDENT_IN = EDGE_HALF_IN + GAP_EPS_IN

# 格子边长按【数据里实际出现的最大值】推算，而不是按理论满分。
# 满分格子很少见，按 1.0 留空等于白白浪费一圈空白。
FIT_TO_DATA = True
CELL_IN = 2 * (HALF_MAX_IN + MIN_INDENT_IN)    # 回退值，FIT_TO_DATA=False 时用
ROW_H_IN = CELL_IN

# 超椭圆指数：2 = 正圆，越大越接近正方形。
# SQUARE_POWER 越大，方形只出现在最接近满分的那一小段，其余都是圆。
SQUARE_RANGE = (2.0, 8.0)
SQUARE_POWER = 6.0

# 四个大类之间空出的横向间距，单位是格子边长的倍数
CAT_GAP_FRAC = 0.5

# 指标名倾斜排在大类名上方，用短斜线指到对应列，这样列间距可以压到最小
LABEL_ROT = 45                  # 指标名倾斜角度
LEADER_IN = 0.13                # 指引斜线长度
LEADER_COLOR, LEADER_WIDTH = "#6B7280", 0.7

# 全图统一 Arial 12 号
FS = 12
LABEL_FS = ROW_FS = CAT_FS = TITLE_FS = LEG_FS = FS

RIGHT_IN, BOTTOM_IN = 0.30, 0.35
LEGEND_GAP_IN = 0.34            # 指标区与图例列之间的留白
LEGEND_TITLE = "Rank"
LEGEND_TICKS = [1.00, 0.75, 0.50, 0.25, 0.00]   # 图例里展示的参考分值
LEGEND_SPACING = 1.5            # 图例各档之间的行距，单位是格子边长的倍数
LEGEND_COLOR = "#595959"   # 图例用中性深灰，同样向白色渐变


# ==========================================================================
# 指标体系
# ==========================================================================
# (列名, 图上显示的缩写, 大类, 方向, 是否派生)
# direction: high=越大越好, low=越小越好, target1=越接近 1 越好
# 派生指标在图上缩写后带 *

METRICS_RNA = [
    ("cell_ratio",       "Ratio",     "Detection accuracy", "target1", False),
    ("precision",        "Prec",      "Detection accuracy", "high",    False),
    ("recall",           "Rec",       "Detection accuracy", "high",    False),
    ("f1",               "F1",        "Detection accuracy", "high",    False),

    ("loc_mean_um",      "Loc.mean",  "Spatial localization", "low",   False),
    ("loc_median_um",    "Loc.med",   "Spatial localization", "low",   False),
    ("loc_p95_um",       "Loc.p95",   "Spatial localization", "low",   False),
    ("loc_tail_ratio",   "Loc.tail",  "Spatial localization", "low",   True),

    ("count_pearson",    "Cnt.r",     "Count fidelity", "high",  False),
    ("count_spearman",   "Cnt.rho",   "Count fidelity", "high",  False),
    ("count_mae",        "Cnt.MAE",   "Count fidelity", "low",   False),
    ("count_rmse",       "Cnt.RMSE",  "Count fidelity", "low",   False),
    ("count_tail_ratio", "Cnt.tail",  "Count fidelity", "low",   True),

    ("vec_cosine",       "Exp.cos",   "Expression & assignment", "high", False),
    ("vec_js_dist",      "Exp.JS",    "Expression & assignment", "low",  False),
    ("vec_pearson",      "Exp.r",     "Expression & assignment", "high", False),
    ("assign_accuracy",  "Assign",    "Expression & assignment", "high", False),
]

# 顺序：把 Boundary quality 提到第一类、Instance detection 挪到第三类，
# 于是每类数量变成 4/4/5/4，与 RNA 表完全一致，两张图可以上下拼在一起对齐。
# 大类颜色是按出现顺序取 CAT_COLORS 的，所以位置一换颜色自然跟着换，
# 第 1~4 位的配色与 RNA 表逐位对应。
# 注意：对齐的是【每类的指标数量和配色位置】，不是语义——
# 第一位 RNA 是 Detection accuracy 而图像是 Boundary quality，两者并不对应，
# 所以 UNIFIED_CATEGORY_NAMES 必须保持 False，否则会张冠李戴。
METRICS_IMG = [
    ("Boundary F1",        "B.F1",    "Boundary quality", "high", False),
    ("Boundary Precision", "B.Prec",  "Boundary quality", "high", False),
    ("Boundary Recall",    "B.Rec",   "Boundary quality", "high", False),
    ("SQ",                 "SQ",      "Boundary quality", "high", False),

    ("Dice",               "Dice",    "Region overlap", "high", False),
    ("IoU",                "IoU",     "Region overlap", "high", False),
    ("Pixel Accuracy",     "PixAcc",  "Region overlap", "high", False),
    ("AJI",                "AJI",     "Region overlap", "high", False),

    ("F1",                 "F1",      "Instance detection", "high",    False),
    ("Precision",          "Prec",    "Instance detection", "high",    False),
    ("Recall",             "Rec",     "Instance detection", "high",    False),
    ("DQ",                 "DQ",      "Instance detection", "high",    False),
    ("cell_ratio",         "Ratio",   "Instance detection", "target1", True),

    ("PQ",                 "PQ",      "Overall quality", "high", False),
    ("mDA(AP0.5:0.95)",    "mDA",     "Overall quality", "high", False),
    ("over_seg_rate",      "Over",    "Overall quality", "low",  True),
    ("under_seg_rate",     "Under",   "Overall quality", "low",  True),
]


# ==========================================================================
# 数据源
# ==========================================================================
SOURCES = [
    dict(
        label="RNA-based methods",
        path=PATH_OMICS,
        sheet=None,                 # None = 第一个 sheet
        gt_col="gt_cells",
        metrics=METRICS_RNA,
        out="bubble_full_omics",
        # 精确 0 视为缺失：0 µm 的定位误差物理上不可能（ComSeg 在 xenium_3 上
        # loc_mean/median/p95 全是 0），不清洗它会被排成定位最准的方法
        zero_is_missing=["loc_mean_um", "loc_median_um", "loc_p95_um"],
        check_f1=True,
    ),
    dict(
        label="Image-based segmentation methods",
        path=PATH_IMAGE,
        sheet="图像分割方法汇总",     # 第一个 sheet 装的是 RNA 方法，这里必须指定
        gt_col="GT细胞数",
        metrics=METRICS_IMG,
        out="bubble_result_image",
        zero_is_missing=[],
        check_f1=False,
    ),
]

# 方法名统一：大小写不敏感去重 + 规范显示名。
# 图像表里 'Cellpose' 和 'cellpose' 是同一个方法各存了一份，不合并会变成两行；
# RNA 表里 'ProSeg(supervised)' 和 'ProSeg' 同理。
DISPLAY_NAMES = {
    "proseg": "ProSeg", "ucs": "UCS", "boms": "BOMS", "baysor": "Baysor",
    "bering": "Bering", "cellist": "Cellist", "comseg": "ComSeg",
    "genesegnet": "GeneSegNet",
    "cellpose": "Cellpose", "stardist": "StarDist", "cellsam": "CellSAM",
    "mesmer": "Mesmer", "cellotype": "CelloType",
}


# ==========================================================================
# 文字宽度测量（表头排版的基础）
# ==========================================================================
_SCRATCH = None


def text_width_in(text, fs):
    """用真实渲染器量出这段文字有多宽（英寸）。"""
    global _SCRATCH
    if _SCRATCH is None:
        _SCRATCH = plt.figure(figsize=(1, 1), dpi=100)
    t = _SCRATCH.text(0, 0, str(text), fontsize=fs)
    _SCRATCH.canvas.draw()
    bb = t.get_window_extent(renderer=_SCRATCH.canvas.get_renderer())
    t.remove()
    return bb.width / _SCRATCH.dpi


# ==========================================================================
# 数据准备
# ==========================================================================
def normalize_method(name):
    """去掉 (supervised) 之类的括号后缀，再按小写查规范显示名。"""
    base = re.sub(r"\s*[（(].*?[)）]\s*$", "", str(name)).strip()
    return DISPLAY_NAMES.get(base.lower(), base)


def load_source(src):
    xl = pd.ExcelFile(src["path"])
    sheet = src["sheet"] or xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet)

    metrics = src["metrics"]
    native = [m[0] for m in metrics if not m[4] and m[0] in df.columns]
    # 图像表有大量占位空行，要求至少有一个原生指标有值
    df = df[df["method"].notna() & df["dataset"].notna()].copy()
    if native:
        df = df[df[native].notna().any(axis=1)]

    df["method"] = df["method"].map(normalize_method)
    # 'merfish_2 肝1' 和 'merfish_2' 是同一个数据集，取第一段做标准名
    df["ds"] = df["dataset"].astype(str).str.split().str[0]
    df["platform"] = df["ds"].str.split("_").str[0]

    for col in src["zero_is_missing"]:
        if col in df.columns:
            n = int((df[col] == 0).sum())
            if n:
                who = df.loc[df[col] == 0, "method"].value_counts().to_dict()
                print(f"  [清洗] {col}: {n} 个 0 视为缺失 -> {who}")
                df.loc[df[col] == 0, col] = np.nan

    # f1=0 但 precision/recall 都 >0 在算术上不可能，是没填而不是真的 0
    if src["check_f1"] and {"f1", "precision", "recall"} <= set(df.columns):
        bad = (df["f1"] == 0) & (df["precision"] > 0) & (df["recall"] > 0)
        if bad.any():
            print(f"  [清洗] f1: {int(bad.sum())} 个与 precision/recall 矛盾的 0 视为缺失")
            df.loc[bad, "f1"] = np.nan

    # 派生指标要用到的辅助列
    extra = [c for c in ["预测细胞数", "TP", "FP", "FN", "loc_p95_um", "loc_median_um",
                         "count_rmse", "count_mae"] if c in df.columns]
    keep = sorted(set(native + extra + [src["gt_col"]]))
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")   # 'SQ' 等列是 object 类型

    agg = df.groupby(["platform", "ds", "method"], as_index=False)[keep].mean()
    agg = derive_metrics(agg, src)
    agg = agg.replace([np.inf, -np.inf], np.nan)
    agg = agg.rename(columns={src["gt_col"]: "_gt"})

    gt = agg.groupby("ds")["_gt"].median()
    return agg, gt, sheet


def derive_metrics(agg, src):
    """按需现算派生指标，只依赖表里已有的列。"""
    gt = src["gt_col"]
    cols = set(agg.columns)

    # RNA：尾部稳健性
    if {"loc_p95_um", "loc_median_um"} <= cols:
        agg["loc_tail_ratio"] = agg["loc_p95_um"] / agg["loc_median_um"].replace(0, np.nan)
    if {"count_rmse", "count_mae"} <= cols:
        agg["count_tail_ratio"] = agg["count_rmse"] / agg["count_mae"].replace(0, np.nan)

    # 图像：预测/GT 细胞数比，以及过分割、欠分割率
    # 表里原有的「过分割(FP)」「欠分割(FN)」两列基本是 FP/FN 的原始计数，跨数据集
    # 量级差几个数量级，这里用干净的 TP/FP/FN 整数列换算成率，含义明确。
    if {"预测细胞数", gt} <= cols:
        agg["cell_ratio"] = agg["预测细胞数"] / agg[gt].replace(0, np.nan)
    if {"FP", gt} <= cols:
        agg["over_seg_rate"] = agg["FP"] / agg[gt].replace(0, np.nan)
    if {"FN", gt} <= cols:
        agg["under_seg_rate"] = agg["FN"] / agg[gt].replace(0, np.nan)

    return agg


def to_score(series, direction):
    """转成"越大越好"的可比量，供求秩使用。"""
    if direction == "high":
        return series
    if direction == "low":
        return -series
    if direction == "target1":       # cell_ratio 越接近 1 越好
        return -(series - 1.0).abs()
    raise ValueError(direction)


def dataset_weight(gt):
    """按 WEIGHT_MODE 把 GT 细胞数换算成权重，非法值返回 None。"""
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


def report_weights(agg, gt_cells):
    """打印各数据集在其平台内占多少权重，确认没有一家独大。"""
    print(f"\n  权重分配（WEIGHT_MODE={WEIGHT_MODE}）：")
    for platform, pdf in agg.groupby("platform"):
        items = []
        for ds in sorted(pdf["ds"].unique()):
            w = dataset_weight(gt_cells.get(ds, np.nan))
            if w is not None:
                items.append((ds, w, float(gt_cells.get(ds))))
        total = sum(w for _, w, _ in items)
        if not total:
            continue
        txt = "  ".join(f"{ds}({g:,.0f}) {w / total:5.1%}" for ds, w, g in items)
        print(f"    {platform:<8}{txt}")


def rank_matrix(agg, gt_cells, metrics):
    """平台内加权 -> 平台间等权，得到 method × metric 矩阵。"""
    methods = sorted(agg["method"].unique())
    out = pd.DataFrame(index=methods, columns=[m[0] for m in metrics], dtype=float)

    for col, _disp, _cat, direction, _der in metrics:
        if col not in agg.columns:
            continue
        plat_scores = {}

        for platform, pdf in agg.groupby("platform"):
            num, den = {}, {}
            for ds, ddf in pdf.groupby("ds"):
                vals = to_score(ddf.set_index("method")[col], direction).dropna()
                n = len(vals)
                if n < 2:            # 只有一个方法，无从比较
                    continue
                rank = vals.rank(ascending=False, method="average")   # 秩 1 = 最好
                if RANK_MODE == "percentile":
                    norm = (n - rank + 0.5) / n
                else:
                    norm = (n - rank) / max(n - 1, 1)
                w = dataset_weight(gt_cells.get(ds, np.nan))
                if w is None:
                    continue
                for m, v in norm.items():
                    num[m] = num.get(m, 0.0) + w * v
                    den[m] = den.get(m, 0.0) + w
            if den:
                plat_scores[platform] = {m: num[m] / den[m] for m in num}

        for m in methods:
            vals = [ps[m] for ps in plat_scores.values() if m in ps]
            out.loc[m, col] = float(np.mean(vals)) if vals else np.nan

    return out


# ==========================================================================
# 作图
# ==========================================================================
def superellipse(cx, cy, half, n_exp, k=180):
    """n_exp=2 是正圆，越大越接近正方形。"""
    t = np.linspace(0, 2 * np.pi, k)
    ct, st = np.cos(t), np.sin(t)
    x = np.sign(ct) * np.abs(ct) ** (2.0 / n_exp)
    y = np.sign(st) * np.abs(st) ** (2.0 / n_exp)
    return np.column_stack([cx + half * x, cy + half * y])


def shade(base_hex, v):
    """v=0 直接就是 CAT_COLORS 里的色号本身（不再额外压暗），
    v=1 渐变到接近白色。配合 COLOR_SCOPE="column"，
    每列最差的格子拿到的就是你指定的那个色号原色。"""
    from matplotlib.colors import to_rgb
    rgb = np.asarray(to_rgb(base_hex), dtype=float)
    if not GRADIENT:
        return tuple(rgb)
    t = float(np.clip(v, 0, 1)) ** LIGHT_GAMMA * LIGHT_MAX
    return tuple(rgb + (1.0 - rgb) * t)


def shape_of(v):
    """分数 -> (半径英寸, 超椭圆指数)。高次幂让方形只出现在接近满分的一小段。"""
    v = float(np.clip(v, 0, 1))
    s_lo, s_hi = SQUARE_RANGE
    return (HALF_MIN_IN + v * (HALF_MAX_IN - HALF_MIN_IN),
            s_lo + (v ** SQUARE_POWER) * (s_hi - s_lo))


def draw(matrix, metrics, title, out_path):
    """坐标系以英寸为单位（aspect=equal），表头也画在同一坐标系里，便于精确定位。"""
    cats = list(dict.fromkeys(m[2] for m in metrics))
    cat_color = {c: CAT_COLORS[i % len(CAT_COLORS)] for i, c in enumerate(cats)}
    rows = list(matrix.index)
    n_col, n_row = len(metrics), len(rows)

    # --- 格子边长：按数据里实际最大的图形来定，缝隙只留描边的物理下限 ---
    if FIT_TO_DATA:
        vmax = float(np.nanmax(matrix.to_numpy(dtype=float))) if matrix.notna().any().any() else 1.0
        vmax = float(np.clip(vmax, 0.0, 1.0))
        half_top = HALF_MIN_IN + vmax * (HALF_MAX_IN - HALF_MIN_IN)
        cell = 2 * (half_top + MIN_INDENT_IN)
    else:
        cell = CELL_IN
    col_w = row_h = cell

    # 逐列算左边界，大类切换处插入 CAT_GAP_FRAC 个格子的空隙
    col_x, x_cur, prev_cat = [], 0.0, None
    for m in metrics:
        if prev_cat is not None and m[2] != prev_cat:
            x_cur += cell * CAT_GAP_FRAC
        col_x.append(x_cur)
        x_cur += cell
        prev_cat = m[2]
    grid_w = x_cur
    grid_h = n_row * row_h
    print(f"  格子 {cell:.4f} 英寸 | 最大图形直径 {2*(cell/2-MIN_INDENT_IN):.4f} "
          f"| 相对缝隙 {2*MIN_INDENT_IN/(cell-2*MIN_INDENT_IN):.1%}")

    # --- 表头各层的高度（自下而上：色条 -> 大类名 -> 指引线 -> 倾斜的指标名）---
    rot = np.deg2rad(LABEL_ROT)
    bar_gap, bar_h = 0.06, 0.14
    cat_h = CAT_FS * 1.35 / 72.0
    y_bar0 = grid_h + bar_gap
    y_cat = y_bar0 + bar_h + 0.05                     # 大类名基线
    y_lead0 = y_cat + cat_h + 0.06                    # 指引线起点
    lab_len = max(text_width_in(m[1] + (" *" if m[4] else ""), LABEL_FS)
                  for m in metrics)
    head_h = (y_lead0 + LEADER_IN * np.sin(rot)
              + lab_len * np.sin(rot) + 0.10) - grid_h

    # --- 图例列 ---
    leg_num_w = max(text_width_in(f"{t:.2f}", LEG_FS) for t in LEGEND_TICKS)
    leg_w = 2 * HALF_MAX_IN + 0.10 + leg_num_w + 0.06
    # 最右侧指标名倾斜后会向右伸出，图例要让开这段距离
    leg_x0 = grid_w + max(LEGEND_GAP_IN, lab_len * np.cos(rot) * 0.55)

    left_in = max(text_width_in(m, ROW_FS) for m in rows) + 0.22
    title_h = TITLE_FS * 1.6 / 72.0
    plot_w = leg_x0 + leg_w
    fig_w = left_in + plot_w + RIGHT_IN
    fig_h = BOTTOM_IN + grid_h + head_h + title_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([left_in / fig_w, BOTTOM_IN / fig_h,
                       plot_w / fig_w, (grid_h + head_h) / fig_h])
    ax.set_xlim(0, plot_w)
    ax.set_ylim(0, grid_h + head_h)
    ax.set_aspect("equal")          # 1 数据单位 = 1 英寸，图形不会被拉扁
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # --- 着色用的取值范围：按列归一化时，先记下每列的最小/最大 ---
    if COLOR_SCOPE == "column":
        cmin = matrix.min(axis=0, skipna=True)
        cmax = matrix.max(axis=0, skipna=True)

    def color_v(col, v):
        """返回用于取色的 0~1 值。0 -> 色号原色，1 -> 接近白。"""
        if COLOR_SCOPE != "column" or col not in matrix.columns:
            return v
        lo, hi = float(cmin[col]), float(cmax[col])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return 0.5
        return (v - lo) / (hi - lo)

    def row_y(i):
        return (n_row - 1 - i) * row_h + row_h / 2

    # --- 行底色灰白交替，只铺指标区 ---
    for i in range(n_row):
        ax.add_patch(Rectangle((0, (n_row - 1 - i) * row_h), grid_w, row_h,
                               zorder=0, edgecolor="none",
                               facecolor=ROW_BG_A if i % 2 == 0 else ROW_BG_B))

    # --- 每格图形，列内居中；最大的图形正好撑满格子只留描边的物理下限 ---
    for i, method in enumerate(rows):
        y = row_y(i)
        for j, (col, _disp, cat, _d, _der) in enumerate(metrics):
            x = col_x[j] + col_w / 2
            v = matrix.loc[method, col] if col in matrix.columns else np.nan
            if not np.isfinite(v):
                ax.plot([x - 0.07, x + 0.07], [y, y], color="#9E9E9E",
                        linewidth=1.1, zorder=2)
                continue
            half, n_exp = shape_of(v)          # 大小与方圆始终用绝对分数
            ax.add_patch(Polygon(superellipse(x, y, half, n_exp), closed=True,
                                 facecolor=shade(cat_color[cat], color_v(col, v)),
                                 edgecolor=SHAPE_EDGE_COLOR,
                                 linewidth=SHAPE_EDGE_WIDTH, zorder=2))
            if SHOW_VALUES:
                ax.text(x, y, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.2, color="#333333", zorder=3)

    # --- 方法名 ---
    for i, method in enumerate(rows):
        ax.text(-0.10, row_y(i), method, ha="right", va="center",
                fontsize=ROW_FS, color="#1A1A1A")

    # --- 大类色条 + 大类名，紧贴网格上方 ---
    j = 0
    while j < n_col:
        cat = metrics[j][2]
        k = j
        while k + 1 < n_col and metrics[k + 1][2] == cat:
            k += 1
        x0, x1 = col_x[j], col_x[k] + col_w
        ax.add_patch(Rectangle((x0 + 0.006, y_bar0), (x1 - x0) - 0.012, bar_h,
                               facecolor=cat_color[cat], edgecolor="none", zorder=2))
        ax.text((x0 + x1) / 2, y_cat, cat, ha="center", va="bottom",
                fontsize=CAT_FS, color="#1A1A1A")
        j = k + 1

    # --- 指标名：倾斜排在大类名上方，短斜线指回各自的列 ---
    for j, (_c, disp, _cat, _d, derived) in enumerate(metrics):
        xc = col_x[j] + col_w / 2
        x_end = xc + LEADER_IN * np.cos(rot)
        y_end = y_lead0 + LEADER_IN * np.sin(rot)
        ax.plot([xc, x_end], [y_lead0, y_end], color=LEADER_COLOR,
                linewidth=LEADER_WIDTH, solid_capstyle="round", zorder=2)
        ax.text(x_end + 0.015 * np.cos(rot), y_end + 0.015 * np.sin(rot),
                disp + (" *" if derived else ""), rotation=LABEL_ROT,
                rotation_mode="anchor", ha="left", va="center",
                fontsize=LABEL_FS, color="#1A1A1A")

    # --- 图例列 ---
    ax.text(leg_x0 + leg_w / 2, y_bar0, LEGEND_TITLE, ha="center", va="bottom",
            fontsize=LABEL_FS, color="#1A1A1A")
    # 图例各档单独用更松的行距，并整体在网格高度内垂直居中
    n_leg = len(LEGEND_TICKS)
    room = (grid_h - row_h) / max(n_leg - 1, 1)      # 保证首尾不越出网格
    leg_h = min(row_h * LEGEND_SPACING, room)
    y_leg_top = grid_h / 2 + (n_leg - 1) * leg_h / 2
    for k2, tv in enumerate(LEGEND_TICKS):
        y = y_leg_top - k2 * leg_h
        half, n_exp = shape_of(tv)
        ax.add_patch(Polygon(superellipse(leg_x0 + HALF_MAX_IN, y, half, n_exp),
                             closed=True, facecolor=shade(LEGEND_COLOR, tv),
                             edgecolor=SHAPE_EDGE_COLOR,
                             linewidth=SHAPE_EDGE_WIDTH, zorder=2))
        ax.text(leg_x0 + 2 * HALF_MAX_IN + 0.07, y, f"{tv:.2f}",
                ha="left", va="center", fontsize=LEG_FS, color="#4B5563")

    fig.text(left_in / fig_w - 0.004, 1 - 0.06 / fig_h, title,
             ha="left", va="top", fontsize=TITLE_FS, color="#1A1A1A")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已保存：{out_path}")


# ==========================================================================
def apply_unified_names(metrics):
    """把各表自己的大类名按出现顺序替换成 UNIFIED_NAMES。"""
    if not UNIFIED_CATEGORY_NAMES:
        return metrics
    order = list(dict.fromkeys(m[2] for m in metrics))
    mapping = {c: UNIFIED_NAMES[i] for i, c in enumerate(order) if i < len(UNIFIED_NAMES)}
    return [(c, d, mapping.get(cat, cat), dr, dv) for c, d, cat, dr, dv in metrics]


def run(src, outdir):
    print(f"\n===== {src['label']} =====")
    agg, gt, sheet = load_source(src)
    print(f"  sheet: {sheet} | 方法 {agg['method'].nunique()} 个 | "
          f"数据集 {agg['ds'].nunique()} 个 | 平台 {sorted(agg['platform'].unique())}")
    print(f"  方法列表: {sorted(agg['method'].unique())}")

    metrics = apply_unified_names(src["metrics"])
    report_weights(agg, gt)
    mat = rank_matrix(agg, gt, metrics)
    if SORT_ROWS_BY_SCORE:
        mat = mat.loc[mat.mean(axis=1, skipna=True).sort_values(ascending=False).index]
    else:
        mat = mat.loc[sorted(mat.index, key=str.lower)]

    print("\n  秩矩阵（0~1，越大越好）：")
    print(mat.round(3).to_string())
    n_na = int(mat.isna().sum().sum())
    if n_na:
        print(f"  缺失格 {n_na} 个，图上画成灰色短横线")

    cov = agg.groupby("method")["ds"].nunique()
    print("  各方法覆盖的数据集数：", cov.to_dict())

    draw(mat, metrics,
         f"{src['label']}  —  composite rank across platforms"
         f"   (larger & squarer = better rank)",
         os.path.join(outdir, f"{src['out']}.{FILE_FORMAT}"))
    return mat


def main():
    outdir = OUT_DIR
    if len(sys.argv) == 4:      # 仅用于换路径调试，正常直接跑不用带参数
        SOURCES[0]["path"], SOURCES[1]["path"], outdir = sys.argv[1:4]
    for src in SOURCES:
        run(src, outdir)


if __name__ == "__main__":
    main()