# -*- coding: utf-8 -*-
"""
三张新附图（计算规模、数据特征、分层表现）共用的读取模块。
  · 准确性结果：沿用 robustness_bars.py 的读取与名字统一（与 Fig. 2d 口径一致）
  · 资源：segmentation_resources_all_methods_summary.csv（运行时间 s、CPU / GPU 内存峰值 GiB）
  · 数据集元信息：与资源矩阵图 fig_resources_matrix.py 的 DATA 表一致，另加分析区域面积与染色类型
请与 robustness_bars.py 放在同一目录。
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supplement_4 as RB          # noqa: E402

# ======================== 路径 ========================
PATH_OMICS = RB.PATH_OMICS
PATH_IMAGE = RB.PATH_IMAGE
PATH_RES = r"C:\Users\阿飞\Desktop\segmentation_resources_all_methods_summary.csv"
OUT_DIR = r"E:\supp_figures"

# ======================== 方法与配色 ========================
ORDER = ["ProSeg", "Baysor", "BOMS", "Bering", "ComSeg", "UCS", "Cellist", "GeneSegNet", "DISSECT",
         "Cellpose-SAM", "Cellpose-3", "CellSAM", "CelloType", "Mesmer", "StarDist", "UNSEG"]
CAT_OF = {"ProSeg": "Statistical point-cloud", "Baysor": "Statistical point-cloud", "BOMS": "Statistical point-cloud",
          "Bering": "Graph-based", "ComSeg": "Graph-based",
          "UCS": "Multimodal", "Cellist": "Multimodal", "GeneSegNet": "Multimodal", "DISSECT": "Multimodal",
          "Cellpose-SAM": "Image-based", "Cellpose-3": "Image-based", "CellSAM": "Image-based",
          "CelloType": "Image-based", "Mesmer": "Image-based", "StarDist": "Image-based", "UNSEG": "Image-based"}
CAT_COLOR = {"Statistical point-cloud": "#4490A9", "Graph-based": "#5B9BD5",
             "Multimodal": "#FFC000", "Image-based": "#ED7D31"}
PLAT_ORDER = ["xenium", "starmap", "stereo", "merfish", "cosmx"]
PLAT_NAME = {"xenium": "Xenium", "starmap": "STARmap", "stereo": "Stereo-seq", "merfish": "MERFISH", "cosmx": "CosMx"}
PLAT_COLOR = {"xenium": "#8FCFC4", "starmap": "#F3DF7A", "stereo": "#E8837A", "merfish": "#8DB3D6", "cosmx": "#F2B26B"}

# ======================== 数据集元信息 ========================
# (物种, 组织, 全样本细胞数, 基因数, 全样本转录本数)，与 fig_resources_matrix.py 的 DATA 表一致
META = {
    "xenium_1": ("Human", "Pancreatic cancer", 190965, 538, 28_455_659),
    "xenium_2": ("Human", "Breast cancer", 209467, 541, 64_581_006),
    "xenium_3": ("Human", "Lymph node", 2442, 4509, 838_883),
    "xenium_4": ("Human", "Liver", 239271, 541, 25_424_111),
    "xenium_5": ("Mouse", "Brain", 63173, 13780, 188_540_362),
    "xenium_6": ("Mouse", "Colon", 219797, 541, 82_549_008),          # 附表 S1：Mouse colon
    "xenium_7": ("Human", "Prostate cancer", 193000, 10017, 59_727_355),
    "xenium_8": ("Human", "Kidney cancer", 465534, 543, 77_056_936),
    "xenium_9": ("Human", "Skin melanoma", 112551, 10017, 74_011_892),
    "xenium_10": ("Human", "Multi-tissue (Prime 5K)", 405927, 5001, 349_884_289),
    "cosmx_1": ("Human", "Pancreas (WTx)", 48944, 18946, 65_794_237),
    "cosmx_2": ("Human", "Lymph node", 1852946, 6175, 1_413_412_378),
    "cosmx_3": ("Human", "Brain", 188686, 6622, 163_339_286),
    "merfish_1": ("Mouse", "Brain", 78329, 483, 54_712_414),
    "merfish_2": ("Human", "Liver (cancer)", 568355, 500, 272_021_991),
    "merfish_3": ("Human", "Liver", 598141, 500, 283_068_068),
    "merfish_4": ("Human", "Lung cancer", 353762, 500, 144_388_044),
    "merfish_breast": ("Human", "Breast cancer", 713121, 500, 490_398_542),
    "merfish_colon1": ("Human", "Colon", 677451, 500, 411_716_053),
    "merfish_colon2": ("Human", "Colon", 817588, 500, 507_576_479),
    "merfish_melanoma": ("Human", "Melanoma", 468138, 500, 160_181_929),
    "merfish_ovarian1": ("Human", "Ovarian", 358485, 500, 197_365_319),
    "merfish_ovarian2": ("Human", "Ovarian", 254347, 500, 115_559_393),
    "merfish_prostate1": ("Human", "Prostate", 721668, 500, 291_996_280),
    "merfish_prostate2": ("Human", "Prostate", 993825, 500, 221_331_615),
    "merfish_uterian": ("Human", "Uterine", 843285, 500, 374_580_211),
    "stereo_1": ("Mouse", "Midbrain (E14.5)", 4872, 18698, 3_362_914),
    "stereo_2": ("Mouse", "Midbrain (E16.5)", 6892, 21202, 8_031_385),
    "stereo_3": ("Mouse", "Midbrain (E16.5)", 7960, 21370, 7_602_540),
    "stereo_4": ("Mouse", "Embryo", 1191859, 29082, 1_092_062_636),
    "starmap_by1": ("Mouse", "Brain", 1424, 1019, 471_295),
}
# 分析区域面积（mm²，与附表 S1 的密度一致；空缺的数据集不参与"面积"相关分析）
AREA_MM2 = {
    "cosmx_1": 4.718533, "cosmx_2": 0.524281, "cosmx_3": 7.602081,
    "merfish_1": 4.720191, **{k: 1.1664 for k in ["merfish_2", "merfish_3", "merfish_4", "merfish_breast",
                                                    "merfish_colon1", "merfish_colon2", "merfish_melanoma",
                                                    "merfish_ovarian1", "merfish_ovarian2", "merfish_prostate1",
                                                    "merfish_prostate2", "merfish_uterian"]},
    "stereo_4": 2.25, "xenium_2": 1.128906,
    **{k: 4.515625 for k in ["xenium_4", "xenium_5", "xenium_6", "xenium_7", "xenium_8", "xenium_9", "xenium_10"]},
}
# 图像方法实际使用的染色（用于分层；转录本方法同样按数据集归类）。按附表 S1 校正：
# CosMx 三个数据集均使用核加膜通道；stereo_4 为配准的 H&E；其余 Stereo-seq 为 ssDNA
STAIN = {**{f"xenium_{i}": "DAPI" for i in range(2, 11)}, "xenium_1": "H&E",
         **{k: "DAPI" for k in META if k.startswith("merfish")},
         "cosmx_1": "Nuclear + membrane", "cosmx_2": "Nuclear + membrane", "cosmx_3": "Nuclear + membrane",
         "stereo_1": "ssDNA", "stereo_2": "ssDNA", "stereo_3": "ssDNA",
         "stereo_4": "H&E",                                   # 附表 S1：Tissue GEF and registered H&E image
         "starmap_by1": "DAPI"}


# Xenium Prime 5K 数据集：DATA 表中的"基因数"含对照与未注释编码（4,509–13,780），按面板标称值 5,001 计
PRIME_5K = {"xenium_3", "xenium_5", "xenium_7", "xenium_9", "xenium_10"}
PANEL_GENES_NOMINAL = {k: 5001 for k in PRIME_5K}
# 面板类别：靶向（≤600 基因）/ 高通量（约 1,000–6,600 基因）/ 全转录组
PANEL_CLASS = {**{k: "Targeted (≤600)" for k in ["xenium_1", "xenium_2", "xenium_4", "xenium_6", "xenium_8"]},
               **{k: "High-plex (1,000–6,600)" for k in PRIME_5K | {"cosmx_2", "cosmx_3", "starmap_by1"}},
               "cosmx_1": "Whole transcriptome",
               **{k: "Whole transcriptome" for k in ["stereo_1", "stereo_2", "stereo_3", "stereo_4"]}}


def panel_class(ds):
    if ds in PANEL_CLASS:
        return PANEL_CLASS[ds]
    return "Targeted (≤600)" if ds.startswith("merfish") else "High-plex (1,000–6,600)"


def plat_of(ds):
    return ds.split("_")[0]


def nat_key(d):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", d)]


def ds_sorted(ds_list):
    return sorted(ds_list, key=lambda d: (PLAT_ORDER.index(plat_of(d)) if plat_of(d) in PLAT_ORDER else 99, nat_key(d)))


# ======================== 准确性 ========================
def load_accuracy():
    """返回 DataFrame[method, ds, type(rna/image), f1, gt]，每个 方法-数据集 一行。"""
    agg, gt = RB.load_both(PATH_OMICS, PATH_IMAGE)
    d = agg.groupby(["method", "ds", "_type"], as_index=False).agg(f1=("f1", "mean"), gt=("_gt", "mean"))
    return d.rename(columns={"_type": "type"})


def roi_cells(acc):
    """每个数据集分析区域的参考细胞数（优先转录本表的口径，缺失时用图像表）。"""
    g = acc.sort_values("type", ascending=False).groupby("ds")["gt"].agg(lambda v: v.round().mode().iloc[0])
    return g


def dataset_table(acc):
    rc = roi_cells(acc)
    rows = []
    for ds, (sp, tis, c_all, genes, tx_all) in META.items():
        cells = rc.get(ds, np.nan)
        tpc = tx_all / c_all
        area = AREA_MM2.get(ds, np.nan)
        genes = PANEL_GENES_NOMINAL.get(ds, genes)
        rows.append(dict(ds=ds, platform=plat_of(ds), species=sp, tissue=tis, genes=genes,
                         tx_per_cell=tpc, roi_cells=cells, roi_tx_est=cells * tpc if pd.notna(cells) else np.nan,
                         area_mm2=area, density=cells / area if pd.notna(area) and pd.notna(cells) else np.nan,
                         panel=panel_class(ds), stain=STAIN.get(ds, "DAPI")))
    return pd.DataFrame(rows).set_index("ds")


# ======================== 资源 ========================
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


def _norm_res_ds(s):
    k = str(s).strip().split()[0].lower()
    k = RES_DATASET.get(k, k)
    m = re.fullmatch(r"xenium(\d+)", k)
    return f"xenium_{m.group(1)}" if m else k


def load_resources():
    """返回 DataFrame[method, ds, runtime_min, cpu_gib, gpu_gib]（同一 方法-数据集 多条记录取中位数）。"""
    r = pd.read_csv(PATH_RES, encoding="utf-8-sig")
    r = r.iloc[:, :5]
    r.columns = ["method", "dataset", "runtime_s", "cpu_gib", "gpu_gib"]
    r["method"] = r["method"].astype(str).str.strip().map(lambda m: RES_METHOD.get(m.lower(), m))
    r["ds"] = r["dataset"].map(_norm_res_ds)
    for c in ["runtime_s", "cpu_gib", "gpu_gib"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.groupby(["method", "ds"], as_index=False)[["runtime_s", "cpu_gib", "gpu_gib"]].median()
    r["runtime_min"] = r.runtime_s / 60
    unknown = sorted(set(r.ds) - set(META))
    if unknown:
        print("[提示] 资源表中未识别的数据集名（未作图）：", unknown)
    return r[r.ds.isin(META)]


def style_rc(mpl, font=6.5):
    """全部文字统一为 Arial Regular（包括数学符号 ρ、R²、k），不使用粗体和斜体。"""
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": font,
        "font.weight": "normal", "font.style": "normal", "axes.titleweight": "normal", "axes.labelweight": "normal",
        "figure.titleweight": "normal", "legend.title_fontsize": font,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial", "mathtext.bf": "Arial",
        "mathtext.sf": "Arial", "mathtext.default": "regular",
        "axes.linewidth": 0.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
        "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def category_strip(fig, ax, methods, width_in=0.07, gap_in=0.03):
    """在热图左侧画一条与行对齐的类别色条（固定物理宽度），并把行名推到色条左侧。"""
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    sx = fig.add_axes([pos.x0 - (width_in + gap_in) / W, pos.y0, width_in / W, pos.height])
    for i, m in enumerate(methods):
        sx.add_patch(Rectangle((0, i - 0.5), 1, 1, color=CAT_COLOR[CAT_OF[m]], lw=0))
    sx.set_xlim(0, 1); sx.set_ylim(len(methods) - 0.5, -0.5); sx.axis("off")
    ax.tick_params(axis="y", pad=(width_in + gap_in) * 72 + 3)
    return sx


def blank_dash(ax, i, j, half=0.32):
    """空白格（没有数据）中间画一条细黑虚线。"""
    ax.plot([j - half, j + half], [i, i], color="black", lw=0.4, ls=(0, (2, 1.5)), zorder=4)


def check_no_overlap(fig, axes):
    """检查各小图（含标题、坐标轴文字）的外框是否两两重叠；返回重叠的对数。"""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(rend) for ax in axes]
    hits = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                hits += 1
    return hits


def full_frame(ax, lw=0.5):
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(lw); s.set_color("black")


def save(fig, name, out_dir=None):
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
        p = os.path.join(out_dir, name + ext)
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        print("已保存:", p)