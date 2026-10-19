# -*- coding: utf-8 -*-
"""
补充图 —— 运行状态矩阵（31 个数据集 × 16 个方法 = 496 项任务）
颜色 = 平台（五色：Xenium 浅绿，其余四个平台用约定的四色系），同一平台的数据集格子同色；
形状 = 任务状态（由三份本地表格判定）：
判定顺序（先满足者为准）：
  – 浅灰短横线  不适用：7 个图像方法 × 4 个 Stereo-seq 数据集、9 个转录本方法 × STARmap 数据集中，
                结果表没有记录的格子（此条优先于一切，不看资源表）
  ◩ 半填充方块  在子区域上完成：仅 SUBREGION_TASKS 中列出的任务
  ■ 实心方块    完成并纳入评估：结果表中有记录；FORCE_SUCCESS 中的方法（GeneSegNet）的其余格子也记为成功
  □ 空心方块    完成但未纳入评估：资源表中有该次运行，结果表中没有
  × 深灰色叉    失败：以上都不是
行 = 数据集（按平台分组，左侧平台色条）；列 = 方法（方法名黑色）。
顶部条形 = 每个方法完成的任务数（按平台堆叠，同色）；右侧条形 = 每个数据集完成的方法数（平台色），
条旁数字 = 完成数 / 适用数。
读取与名字统一沿用 robustness_bars.py（与 Fig. 2d 一致），请把本脚本与 robustness_bars.py 放在同一目录。
本地运行：python supp_fig_run_status.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supplement_4 as RB          # noqa: E402

# ======================== 参数 ========================
PATH_OMICS = RB.PATH_OMICS
PATH_IMAGE = RB.PATH_IMAGE
PATH_RES = r"C:\Users\阿飞\Desktop\segmentation_resources_all_methods_summary.csv"
OUT_DIR = r"E:\supp_figures"
OUT_NAME = "supp_fig_run_status"
FIG_W, FONT = 7.2, 6
CELL = 0.21                           # 每格边长（inch）
GLYPH = 0.34                          # 方块半边长（格子为 1）：越小格子间距越大
EDGE_LW = 0.15                        # 方块黑色描边线宽（pt）
FRAME_LW = 0.4                        # 外框与平台分隔线线宽（pt）
SHOW_LEGEND = False                   # 只要矩阵本身：图例默认不画（状态与颜色的含义写在图注里）
# 只有这几项是在子区域上运行的（其余参考细胞数对不上的属于 GT 口径差异，按完整运行处理）：
# ComSeg 在 xenium_1（中心 6000 px ROI）、cosmx_1、cosmx_2 上，以及 Cellist 在 xenium_6 上
SUBREGION_TASKS = {("ComSeg", "xenium_1"), ("ComSeg", "cosmx_1"), ("ComSeg", "cosmx_2"), ("Cellist", "xenium_6")}
# 这些方法除"不适用"的格子外一律记为运行成功（无论资源表 / 结果表是否有记录）
FORCE_SUCCESS = {"GeneSegNet"}

IMAGE_METHODS = ["Cellpose-SAM", "Cellpose-3", "CellSAM", "CelloType", "Mesmer", "StarDist", "UNSEG"]
OMICS_METHODS = ["ProSeg", "Baysor", "BOMS", "Bering", "ComSeg", "UCS", "Cellist", "GeneSegNet", "DISSECT"]
ORDER = OMICS_METHODS + IMAGE_METHODS
NA_RULES = [(IMAGE_METHODS, "stereo"), (OMICS_METHODS, "starmap")]     # (方法集合, 平台前缀) -> 不适用

STATUS = [   # (键, 图例文字)
    ("eval", "Completed"),
    ("sub", "Completed on a sub-region"),
    ("noeval", "Completed, not evaluated"),
    ("fail", "Failed"),
]
PLAT_COLOR = {               # 与资源矩阵图（fig_resources_matrix.py）相同的五个平台色
    "xenium": "#8FCFC4",
    "starmap": "#F3DF7A",
    "stereo": "#E8837A",
    "merfish": "#8DB3D6",
    "cosmx": "#F2B26B",
}
FAIL_C = "#595959"
NA_C = "#BDBDBD"
PLAT_ORDER = ["xenium", "starmap", "stereo", "merfish", "cosmx"]
PLAT_NAME = {"xenium": "Xenium", "starmap": "STARmap", "stereo": "Stereo-seq", "merfish": "MERFISH", "cosmx": "CosMx"}

# 资源表里的名字 -> 统一名（与结果表一致，小写）
RES_METHOD = {"proseg": "ProSeg", "cellposesam": "Cellpose-SAM", "cellpose-sam": "Cellpose-SAM",
              "cellpose3-cyto3": "Cellpose-3", "cellpose3": "Cellpose-3", "cellpose-3": "Cellpose-3",
              "baysor": "Baysor", "bering": "Bering", "boms": "BOMS", "comseg": "ComSeg", "dissect": "DISSECT",
              "genesegnet": "GeneSegNet", "ucs": "UCS", "cellist": "Cellist", "cellotype": "CelloType",
              "cellsam": "CellSAM", "mesmer": "Mesmer", "stardist": "StarDist", "unseg": "UNSEG"}
RES_DATASET = {
    "cosmx_pancreas": "cosmx_1", "pancreas-cosmx-wtx-flatfiles": "cosmx_1", "cosmx_lymph_node": "cosmx_2",
    "cosmx_human_brain": "cosmx_3", "merfish_mouse_brain": "merfish_1", "merfish_liver1": "merfish_2",
    "merfish_liver2": "merfish_3", "merfish_lung": "merfish_4", "merfish_lung_cancer": "merfish_4",
    "merfish_breast_cancer": "merfish_breast", "merfish_uterine": "merfish_uterian",
    "stereo_e14.5_e1s3": "stereo_1", "stereo_e16.5_e1s3": "stereo_2", "stereo_e16.5_e2s6": "stereo_3",
    "stereo_e16.5_e2s7": "stereo_3", "stereo_mouse_brain": "stereo_4",
    "xenium_pancreas_cancer": "xenium_1", "xenium_breast": "xenium_2", "xenium_roi_crop": "xenium_3",
    "xenium_liver": "xenium_4", "xenium_mouse_brain": "xenium_5", "xenium_colon": "xenium_6",
}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": FONT,
    "axes.linewidth": 0.6, "xtick.major.size": 0, "ytick.major.size": 0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})


def nat_key(d):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", d)]


def norm_res_ds(s):
    k = str(s).strip().split()[0].lower()
    k = RES_DATASET.get(k, k)
    m = re.fullmatch(r"xenium(\d+)", k)
    return f"xenium_{m.group(1)}" if m else k


def load_status():
    agg, gt = RB.load_both(PATH_OMICS, PATH_IMAGE)
    ev = agg.groupby(["method", "ds", "_type"], as_index=False)["_gt"].mean().rename(columns={"_gt": "gtc"})
    # 该数据集的标准参考细胞数（众数），仅用于在终端打印子区域任务的对照
    ref = ev.groupby(["ds", "_type"])["gtc"].agg(lambda v: v.round().mode().iloc[0])
    ev["ref"] = [ref[(d, t)] for d, t in zip(ev["ds"], ev["_type"])]
    res = pd.read_csv(PATH_RES, encoding="utf-8-sig").iloc[:, :2]
    res.columns = ["method", "dataset"]
    res["method"] = res["method"].astype(str).str.strip().map(lambda m: RES_METHOD.get(m.lower(), m))
    res["ds"] = res["dataset"].map(norm_res_ds)
    datasets = sorted(set(ev["ds"]) | set(res["ds"]),
                      key=lambda d: (PLAT_ORDER.index(d.split("_")[0]) if d.split("_")[0] in PLAT_ORDER else 99,
                                     nat_key(d)))
    unknown_m = sorted((set(ev["method"]) | set(res["method"])) - set(ORDER))
    if unknown_m:
        print("[警告] 不在 ORDER 中的方法名（未作图）：", unknown_m)
    ran = set(zip(res["method"], res["ds"]))
    evaluated = set(zip(ev["method"], ev["ds"]))
    S = pd.DataFrame(index=datasets, columns=ORDER, dtype=object)
    for d in datasets:
        plat = d.split("_")[0]
        for m in ORDER:
            na_cell = any(m in ms and plat == p for ms, p in NA_RULES)
            if na_cell and (m, d) not in evaluated:
                S.at[d, m] = "na"                      # 规则优先于一切
            elif (m, d) in SUBREGION_TASKS:
                S.at[d, m] = "sub"
            elif (m, d) in evaluated or m in FORCE_SUCCESS:
                S.at[d, m] = "eval"
            elif (m, d) in ran:
                S.at[d, m] = "noeval"
            else:
                S.at[d, m] = "fail"
    miss_sub = [t for t in SUBREGION_TASKS if t[1] in S.index and t[0] in S.columns and S.at[t[1], t[0]] != "sub"]
    if miss_sub:
        print("[警告] SUBREGION_TASKS 中这些任务未标为子区域：", miss_sub)
    sub_info = ev[[(m, d) in SUBREGION_TASKS for m, d in zip(ev["method"], ev["ds"])]]
    return S, sub_info


def draw_glyph(ax, x, y, st, pc, h=None):
    """在 (x, y) 处画一个任务的符号；pc = 平台色。"""
    from matplotlib.patches import Polygon
    h = GLYPH if h is None else h
    if st == "na":
        ax.plot([x - 0.22, x + 0.22], [y, y], color=NA_C, lw=0.9, solid_capstyle="round")
    elif st == "eval":
        ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, facecolor=pc, edgecolor="black", lw=EDGE_LW))
    elif st == "sub":                                   # 左下半填充
        ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, facecolor="white", edgecolor="black", lw=EDGE_LW))
        ax.add_patch(Polygon([(x - h, y - h), (x - h, y + h), (x + h, y + h)], closed=True, facecolor=pc,
                             edgecolor="none"))
        ax.add_patch(Rectangle((x - h, y - h), 2 * h, 2 * h, facecolor="none", edgecolor="black", lw=EDGE_LW))
    elif st == "noeval":                                # 空心：平台色粗描边
        ax.add_patch(Rectangle((x - h * 0.86, y - h * 0.86), 2 * h * 0.86, 2 * h * 0.86, facecolor="white",
                               edgecolor=pc, lw=0.8))
    elif st == "fail":
        d_ = h * 0.62
        ax.plot([x - d_, x + d_], [y - d_, y + d_], color=FAIL_C, lw=0.7, solid_capstyle="round")
        ax.plot([x - d_, x + d_], [y + d_, y - d_], color=FAIL_C, lw=0.7, solid_capstyle="round")


def main():
    S, subs = load_status()
    nD, nM = S.shape
    keys = [k for k, _ in STATUS]
    cnt = {k: int((S == k).sum().sum()) for k in keys + ["na"]}
    applicable = nD * nM - cnt["na"]
    done = cnt["eval"] + cnt["sub"] + cnt["noeval"]
    print(f"矩阵 {nD} 个数据集 × {nM} 个方法 = {nD * nM} 项；不适用 {cnt['na']}，适用 {applicable}")
    for k, lab in STATUS:
        print(f"  {lab:<62} {cnt[k]:>4}")
    print(f"  完成（①+②+③）{done} / {applicable} 项适用任务")
    if len(subs):
        print("  在子区域上运行的任务：")
        for r in subs.itertuples():
            print(f"      {r.method:<12} {r.ds:<16} 参考细胞 {r.gtc:,.0f}（该数据集标准 {r.ref:,.0f}）")

    # ---------- 版式（inch）：只画矩阵 ----------
    L, T_, R_, B_ = 1.05, 0.75, 0.1, 0.1
    grid_w, grid_h = nM * CELL, nD * CELL
    fig_w, fig_h = L + grid_w + R_, T_ + grid_h + B_
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([L / fig_w, B_ / fig_h, grid_w / fig_w, grid_h / fig_h])
    for i, d in enumerate(S.index):
        pc = PLAT_COLOR.get(d.split("_")[0], "#9E9E9E")
        for j, m in enumerate(S.columns):
            draw_glyph(ax, j, i, S.at[d, m], pc)
    ax.set_xlim(-0.5, nM - 0.5); ax.set_ylim(nD - 0.5, -0.5)
    ax.set_xticks(range(nM)); ax.set_xticklabels(S.columns, rotation=90, color="black")
    ax.xaxis.tick_top(); ax.tick_params(axis="x", pad=2)
    ax.set_yticks(range(nD)); ax.set_yticklabels(S.index, color="black")
    ax.tick_params(axis="y", pad=2)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(FRAME_LW)
    plats = [d.split("_")[0] for d in S.index]
    for k in range(1, nD):                                  # 平台之间的分隔线
        if plats[k] != plats[k - 1]:
            ax.axhline(k - 0.5, color="black", lw=FRAME_LW)
    ax.axvline(len(OMICS_METHODS) - 0.5, color="black", lw=FRAME_LW)   # 转录本 / 图像方法之间

    if SHOW_LEGEND:
        from matplotlib.patches import Patch as _P
        fig.legend(handles=[_P(facecolor=PLAT_COLOR[p], edgecolor="black", lw=EDGE_LW, label=PLAT_NAME[p])
                            for p in PLAT_ORDER if p in plats], loc="upper left",
                   bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=FONT - 0.3)

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
        p = os.path.join(OUT_DIR, OUT_NAME + ext)
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print("已保存:", p)
    plt.close(fig)
    lab = {k: l for k, l in STATUS}; lab["na"] = "Not applicable"
    S.replace(lab).to_excel(os.path.join(OUT_DIR, OUT_NAME + "_status.xlsx"))
    print("已保存:", os.path.join(OUT_DIR, OUT_NAME + "_status.xlsx"))


if __name__ == "__main__":
    main()