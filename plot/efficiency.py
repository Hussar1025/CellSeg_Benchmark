# -*- coding: utf-8 -*-
"""
Resource benchmark matrix: runtime / CPU peak memory / GPU peak memory
Rows = datasets (grouped by platform); columns = methods (ordered by median runtime)
Colour: log scale, lower value = darker (P5 -> base colour, P95 -> light tint; clipped beyond)

【适配新版 CSV】表头为 方法名 / 数据集名称 / 运行时间(s) / CPU内存峰值(GiB) / GPU内存峰值(GiB)，数值列不带单位；
方法名与数据集名有多种写法（Proseg / CellposeSAM / Cellpose3-cyto3；xenium10 / xenium_10；
cosmx_pancreas / Pancreas-CosMx-WTx-FlatFiles 等），统一成与结果表一致的名字后再作图。旧版"数值 + 单位"格式仍兼容。
数据集按平台分组、组内按自然序（xenium_2 在 xenium_10 之前）排列。
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle

# ======================== 参数 ========================
CSV = Path(r"C:\Users\阿飞\Desktop\segmentation_resources_all_methods_summary.csv")
OUT_STEM = CSV.with_name("fig_resources_matrix")

FIG_W = 7.2            # inch，约 183 mm（Nature 双栏宽）
FONT = 6               # pt

ROBUST_PCT = (5, 95)   # 色标范围对应的百分位数
BASE_RGB = {"time": (68, 144, 169), "cpu": (237, 125, 49), "gpu": (91, 155, 213)}
LIGHT = 0.85           # 最浅色：基色与白色的混合比例（越大越浅）
INVERT_SIZE = False    # False：数值越大方块越大；True：数值越小方块越大

PLATFORM = {
    "xenium": "Xenium",
    "starmap": "STARmap",
    "stereo": "Stereo-seq",
    "merfish": "MERFISH",
    "cosmx": "CosMx",
}
PLATFORM_COLOR = {
    "xenium": "#8FCFC4",
    "starmap": "#F3DF7A",
    "stereo": "#E8837A",
    "merfish": "#8DB3D6",
    "cosmx": "#F2B26B",
}
DATASET_LABEL = {}     # 如需美化数据集显示名，在此添加（键为统一后的名字，如 "xenium_2"）

# 方法名、数据集名统一（键为小写原名）
METHOD_NAME = {"proseg": "ProSeg", "cellposesam": "Cellpose-SAM", "cellpose-sam": "Cellpose-SAM",
               "cellpose3-cyto3": "Cellpose-3", "cellpose3": "Cellpose-3", "cellpose-3": "Cellpose-3",
               "baysor": "Baysor", "bering": "Bering", "boms": "BOMS", "comseg": "ComSeg", "dissect": "DISSECT",
               "genesegnet": "GeneSegNet", "ucs": "UCS", "cellist": "Cellist", "cellotype": "CelloType",
               "cellsam": "CellSAM", "mesmer": "Mesmer", "stardist": "StarDist", "unseg": "UNSEG"}
DATASET_ALIAS = {
    "cosmx_pancreas": "cosmx_1", "pancreas-cosmx-wtx-flatfiles": "cosmx_1", "cosmx_lymph_node": "cosmx_2",
    "cosmx_human_brain": "cosmx_3", "merfish_mouse_brain": "merfish_1", "merfish_liver1": "merfish_2",
    "merfish_liver2": "merfish_3", "merfish_lung": "merfish_4", "merfish_lung_cancer": "merfish_4",
    "merfish_breast_cancer": "merfish_breast", "stereo_e14.5_e1s3": "stereo_1", "stereo_e16.5_e1s3": "stereo_2",
    "stereo_e16.5_e2s6": "stereo_3", "stereo_mouse_brain": "stereo_4", "xenium7": "xenium_7", "xenium9": "xenium_9",
    "xenium10": "xenium_10", "xenium_breast": "xenium_2", "xenium_roi_crop": "xenium_3", "xenium_liver": "xenium_4",
    "xenium_mouse_brain": "xenium_5", "xenium_colon": "xenium_6", "xenium_pancreas_cancer": "xenium_1",
    "starmap_by1": "starmap_BY1",
}

MISS_C = "#9E9E9E"
HOLLOW_C = "#9E9E9E"
EDGE_LW = 0.25         # 方块黑色边框线宽（pt）
SQ_MAX = 0.88          # 最大方块边长（单元格为 1）
SQ_MIN = 0.36          # 最小方块边长

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FONT,
    "axes.linewidth": 0.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ======================== 读取与解析 ========================
df = pd.read_csv(CSV, encoding="utf-8-sig").iloc[:, :5]
df.columns = ["method", "dataset", "time", "cpu", "gpu"]
df["method"] = df["method"].astype(str).str.strip().map(lambda m: METHOD_NAME.get(m.lower(), m))
df["dataset"] = df["dataset"].astype(str).str.strip().map(lambda s: DATASET_ALIAS.get(s.lower(), s.lower()))

num_re = r"^\s*(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-z]*)\s*$"
for col, unit in [("time", "s"), ("cpu", "GiB"), ("gpu", "GiB")]:
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):          # 新版：纯数值
        df[col] = s.astype(float)
        continue
    ext = s.astype(str).str.extract(num_re)       # 旧版：数值 + 单位
    bad = s.notna() & ext[0].isna()
    if bad.any():
        raise ValueError(f"[{col}] 无法解析的条目: {s[bad].unique()[:10]}")
    units = {u for u in ext[1].dropna() if u}
    if units - {unit}:
        raise ValueError(f"[{col}] 单位不一致: {units}")
    df[col] = ext[0].astype(float)

dup = df.duplicated(["method", "dataset"], keep=False)
if dup.any():
    print("[警告] 名字统一后存在重复的 方法-数据集 组合，仅保留第一条：")
    print(df[dup].sort_values(["method", "dataset"]))
df = df.drop_duplicates(["method", "dataset"], keep="first")
print(f"读取 {len(df)} 条记录：{df.method.nunique()} 个方法，{df.dataset.nunique()} 个数据集")


def plat_key(d):
    return d.split("_")[0].lower()


unknown = {plat_key(d) for d in df["dataset"]} - set(PLATFORM)
if unknown:
    raise ValueError(f"未在 PLATFORM 中定义的数据集前缀: {unknown}")


def nat_key(d):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", d)]


plat_order = list(PLATFORM)
datasets = sorted(pd.unique(df["dataset"]), key=lambda d: (plat_order.index(plat_key(d)), nat_key(d)))


def matrix(col):
    return df.pivot(index="dataset", columns="method", values=col)


T, C, G = matrix("time"), matrix("cpu"), matrix("gpu")
methods = T.median(axis=0, skipna=True).sort_values().index.tolist()
T, C, G = [M.reindex(index=datasets, columns=methods) for M in (T, C, G)]
n, m = T.shape

gpu_methods = [mm for mm in methods if (G[mm] > 0).any()]
Gp = G[gpu_methods]
print(f"矩阵大小: {n} datasets x {m} methods")
print("GPU panel 中不显示的方法（从未使用 GPU）:", [mm for mm in methods if mm not in gpu_methods])

# ======================== 汇总表 ========================
summary = pd.DataFrame({
    "median_runtime_s": T.median(),
    "median_runtime_min": T.median() / 60,
    "median_cpu_GiB": C.median(),
    "median_gpu_GiB_nonzero": G.where(G > 0).median(),
    "n_completed": T.notna().sum(),
    "n_missing": T.isna().sum(),
    "n_gpu_used": (G > 0).sum(),
}).loc[methods]
summary.to_csv(OUT_STEM.with_name(OUT_STEM.name + "_summary.csv"), encoding="utf-8-sig")
print(summary.round(2).to_string())


# ======================== 工具函数 ========================
def seq_cmap(rgb):
    """低值 = 基色（最深），高值 = 浅色；不在基色基础上加深"""
    base = np.array(rgb, dtype=float) / 255
    light = base + (1 - base) * LIGHT
    return LinearSegmentedColormap.from_list("seq_r", [base, light], N=256)


def make_norm(M, label):
    v = M.values[np.isfinite(M.values) & (M.values > 0)]
    lo, hi = np.percentile(v, ROBUST_PCT)
    ext = ("min" if v.min() < lo else "") + ("max" if v.max() > hi else "")
    ext = {"minmax": "both", "min": "min", "max": "max", "": "neither"}[ext]
    print(f"[{label}] 色标范围: P{ROBUST_PCT[0]} = {lo:.4g}, P{ROBUST_PCT[1]} = {hi:.4g}"
          f"（超出上限 {np.sum(v > hi)} 个，低于下限 {np.sum(v < lo)} 个）")
    return LogNorm(lo, hi, clip=True), ext


def pick_ticks(norm, cands, labels, min_gap=0.13):
    sel, last = [], -np.inf
    for t, l in zip(cands, labels):
        if norm.vmin <= t <= norm.vmax:
            p = float(norm(t))
            if p - last >= min_gap:
                sel.append((t, l))
                last = p
    if len(sel) < 2:
        return None, None
    t, l = zip(*sel)
    return list(t), list(l)


def draw_cell(ax, x, y, v, norm, cmap, zero_hollow):
    if not np.isfinite(v):
        d = 0.20
        ax.plot([x - d, x + d], [y - d, y + d], color=MISS_C, lw=0.6, solid_capstyle="round")
        ax.plot([x - d, x + d], [y + d, y - d], color=MISS_C, lw=0.6, solid_capstyle="round")
    elif zero_hollow and v == 0:
        s = 0.45
        ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s,
                               facecolor="none", edgecolor=HOLLOW_C, lw=0.5))
    else:
        f = float(np.clip(norm(v), 0, 1))
        fs = 1 - f if INVERT_SIZE else f
        s = SQ_MIN + (SQ_MAX - SQ_MIN) * fs
        ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, facecolor=cmap(f),
                               edgecolor="black", lw=EDGE_LW))


def draw_panel(ax, M, norm, cmap, title, zero_hollow=False):
    nr, nc = M.shape
    for i in range(nr):
        for j in range(nc):
            draw_cell(ax, j, i, M.iat[i, j], norm, cmap, zero_hollow)
    ax.set_xlim(-0.6, nc - 0.4)
    ax.set_ylim(nr - 0.4, -0.6)
    ax.set_xticks(range(nc))
    ax.set_xticklabels(list(M.columns), rotation=90)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, pad=2)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=FONT + 1, fontweight="bold", pad=4)


# ======================== 版式（单位：inch） ========================
L, SW, SG, GAP, R = 1.20, 0.08, 0.03, 0.20, 0.08
T_M, B_M = 0.80, 0.55

TIME_C = [1, 10, 30, 60, 300, 600, 1800, 3600, 3 * 3600, 10 * 3600, 86400, 3 * 86400, 604800]
TIME_L = ["1 s", "10 s", "30 s", "1 min", "5 min", "10 min", "30 min", "1 h", "3 h", "10 h",
          "1 d", "3 d", "1 wk"]
MEM_C = [0.1, 0.3, 1, 2, 3, 5, 10, 20, 30, 50, 100, 300]
MEM_L = ["0.1", "0.3", "1", "2", "3", "5", "10", "20", "30", "50", "100", "300"]

nT, eT = make_norm(T, "Runtime")
nC, eC = make_norm(C, "CPU")
nG, eG = make_norm(Gp, "GPU")

panels = [
    (T, nT, eT, seq_cmap(BASE_RGB["time"]), "Runtime", "Runtime", False,
     pick_ticks(nT, TIME_C, TIME_L)),
    (C, nC, eC, seq_cmap(BASE_RGB["cpu"]), "CPU peak memory", "Peak memory (GiB)", False,
     pick_ticks(nC, MEM_C, MEM_L)),
    (Gp, nG, eG, seq_cmap(BASE_RGB["gpu"]), "GPU peak memory", "Peak memory (GiB)", True,
     pick_ticks(nG, MEM_C, MEM_L)),
]

units = [p[0].shape[1] + 0.2 for p in panels]
cell = (FIG_W - L - SW - SG - 2 * GAP - R) / sum(units)
PH = (n + 0.2) * cell
FIG_H = T_M + PH + B_M

fig = plt.figure(figsize=(FIG_W, FIG_H))


def frac(x, y, w, h):
    return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]


# 左侧平台色条 + 数据集标签
sax = fig.add_axes(frac(L, B_M, SW, PH))
for i, d in enumerate(datasets):
    sax.add_patch(Rectangle((0, i - 0.5), 1, 1, facecolor=PLATFORM_COLOR[plat_key(d)], lw=0))
sax.set_xlim(0, 1)
sax.set_ylim(n - 0.4, -0.6)
sax.set_xticks([])
sax.set_yticks(range(n))
sax.set_yticklabels([DATASET_LABEL.get(d, d) for d in datasets])
sax.tick_params(pad=2)
for sp in sax.spines.values():
    sp.set_visible(False)

# 三个 panel + colorbar
xk = L + SW + SG
for (M, norm, extend, cmap, title, cb_title, zero_hollow, (ticks, labels)), u in zip(panels, units):
    PW = u * cell
    ax = fig.add_axes(frac(xk, B_M, PW, PH))
    draw_panel(ax, M, norm, cmap, title, zero_hollow)

    cb_w = min(PW, max(0.75 * PW, 1.0))
    cax = fig.add_axes(frac(xk + (PW - cb_w) / 2, B_M - 0.36, cb_w, 0.06))
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", extend=extend)
    cb.outline.set_visible(False)
    cb.minorticks_off()
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
    cax.tick_params(length=2, width=0.4, pad=1)
    cax.set_title(cb_title, fontsize=FONT, pad=2)

    xk += PW + GAP

# ======================== 导出 ========================
for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
    out = OUT_STEM.with_suffix(ext)
    fig.savefig(out, **kw)
    print("已保存:", out)
plt.close(fig)