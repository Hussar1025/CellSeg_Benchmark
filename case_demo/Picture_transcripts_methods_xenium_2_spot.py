#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xenium_2 (乳腺) 转录本点云可视化 —— 归属覆盖率
================================================
六个组学方法各输出一个 PDF。三色含义（默认 --mode assign）：
    蓝  assigned by method    方法把该转录本归给了某个细胞
    橙  missed (GT assigned)  GT 认为它属于某个细胞，方法却没归属
    灰  unassigned by both    双方都未归属

关于"方法归属而 GT 未归属"为何算作蓝色：
Xenium 的 GT 里有 17.2% 的转录本标为 UNASSIGNED，多为细胞外或边界模糊的，
理论上这些转录本本就属于某个细胞，只是 GT 没能归到位，方法把它们分给
邻近细胞不应算作错误。实测 xenium_2 上 ProSeg 这类占 14.2%，
若计入 mismatch 会让橙色虚高到 17.8%，严重夸大错误率。

这张图只反映【归属覆盖率】，不含任何准确性信息——一个把所有转录本
胡乱分配的方法也会全蓝。准确性请看 ARI / HOM，或用 --mode cell，
那时橙色表示归属到了非主匹配的细胞，才是真正的错分。

坐标参数（已用 GT 细胞质心命中率标定）：
    ROI 原点 (5205.6125, 2382.125) µm = 全片像素 (24497, 11210) × 0.2125
    ROI 边长 5000 px = 1062.5 µm
    ProSeg      按 transcript_id 直连。它会重定位转录本（x/y 与 observed_x/y
                可差 1 µm，超过 3 px 匹配阈值），坐标匹配会系统性漏匹配
    ComSeg      global_x/global_y 已是 ROI 内像素
    Cellist     x/y 已是 ROI 内像素
    BOMS        npz 的 x/y 是全局 µm，需减 ROI 原点再除 0.2125
    UCS         掩膜 1063×1064，1.0 µm/格
    GeneSegNet  掩膜 5000×5000，0.2125 µm/px

Cellist 的路径写死到 seg_nucprior_r40_fix：
    该目录是修复轴序 bug 并改用 nucleus_boundaries 核先验后重跑的结果，
    7,031 细胞、归属率 84.1%、9 个 patch 齐全、10×10 网格无空洞。
    同级还残留两份废弃结果——旧的 Cellpose 种子版只有 955 细胞、
    轴序未修版 2,497 细胞，空间上呈对角线空洞。若用递归 glob 取第一个
    会随机命中废弃版本，所以这里写死。
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ==========================================================================
R = "/data/qiuyijia"
PX = 0.2125                          # µm / 像素
X0_PX, Y0_PX = 24497, 11210          # ROI 原点（全片像素）
X0, Y0 = X0_PX * PX, Y0_PX * PX      # (5205.6125, 2382.125) µm
ROI_PX = 5000                        # ROI 边长（像素）= 1062.5 µm

GT_TX = f"{R}/dataset/xenium_breast/transcripts.parquet"
QV_MIN = 20.0                        # 组学方法统一口径
MATCH_THR_PX = 3.0                   # 坐标匹配阈值（像素），3 px ≈ 0.64 µm

# Cellist 修复后的结果目录，写死避免命中残留的废弃版本
CELLIST_DIR = f"{R}/cellist/xenium_breast/seg_nucprior_r40_fix"

OUT_DIR = f"{R}/dataset/xenium_breast/viz"
OUT_PREFIX = "tx_xenium2"
DEFAULT_SIDE_PX = 600                # 600 px = 127.5 µm

C_OK, C_BAD, C_NONE = "#5B9BD5", "#ED7D31", "#A5A5A5"
S_OK, S_BAD = 1.3, 1.3               # 两类点同样大小，避免视觉夸大橙色占比
FIG_IN = 8
TITLE_FS, LEG_FS = 14, 10
RASTERIZE = True                     # 散点层栅格化，PDF 体积降一两个数量级
RASTER_DPI = 300


# ==========================================================================
def um_to_px(x_um, y_um):
    """全局 µm -> ROI 内像素。"""
    return (np.asarray(x_um) - X0) / PX, (np.asarray(y_um) - Y0) / PX


def get_tx(qv_min):
    """只读 ROI 内、qv 达标的转录本。全片 6458 万行，必须下推过滤。"""
    import pyarrow.dataset as ds, pyarrow.compute as pc
    f = ((pc.field("x_location") >= X0) & (pc.field("x_location") < X0 + ROI_PX * PX)
         & (pc.field("y_location") >= Y0) & (pc.field("y_location") < Y0 + ROI_PX * PX)
         & (pc.field("qv") >= qv_min))
    tx = ds.dataset(GT_TX).to_table(
        columns=["transcript_id", "cell_id", "x_location", "y_location"],
        filter=f).to_pandas()
    px, py = um_to_px(tx.x_location.values, tx.y_location.values)
    tx["px"], tx["py"] = px, py
    gt_cell = tx.cell_id.astype(str).values
    gt = gt_cell != "UNASSIGNED"
    print(f"ROI 内 qv>={qv_min} 转录本 {len(tx):,} 条，GT 已归属 {gt.mean():.1%}")
    return tx, gt, gt_cell


def match_by_xy(tx, px, py, labels, thr=MATCH_THR_PX):
    """最近邻匹配，返回每条 GT 转录本对应的方法标签（0 = 未归属/未匹配）。"""
    tree = cKDTree(np.c_[px, py])
    d, idx = tree.query(np.c_[tx.px.values, tx.py.values], k=1)
    out = np.zeros(len(tx), dtype=np.int64)
    ok = d < thr
    lab = np.nan_to_num(np.asarray(labels, dtype=float), nan=0.0).astype(np.int64)
    out[ok] = lab[idx[ok]]
    return out


def match_by_mask(tx, path, step_um):
    """掩膜查表。step_um = 掩膜 1 像素代表的微米数。
    按 µm 换算而非按比例缩放：UCS 是 1063×1064 的 1 µm 网格，
    用比例缩放会有半个格子的系统偏移。"""
    m = tifffile.imread(path)
    if m.ndim == 3:
        m = m[0] if m.shape[0] < 5 else m[..., 0]
    col = np.floor(tx.px.values * PX / step_um).astype(int)
    row = np.floor(tx.py.values * PX / step_um).astype(int)
    ok = (row >= 0) & (row < m.shape[0]) & (col >= 0) & (col < m.shape[1])
    out = np.zeros(len(tx), dtype=np.int64)
    out[ok] = m[row[ok], col[ok]]
    return out


# --- 六个方法，各返回与 tx 等长的标签数组，0 表示未归属 ---
def m_proseg(tx):
    d = pd.read_csv(f"{R}/proseg/output/xenium_2/transcript-metadata.csv.gz",
                    usecols=["transcript_id", "assignment", "background"])
    d = d[~d.background.astype(bool)][["transcript_id", "assignment"]]
    s = tx[["transcript_id"]].merge(d, on="transcript_id", how="left")
    # ProSeg 的 cell id 从 0 起，这里整体 +1，好让 0 专表未归属
    return (s.assignment.fillna(-1).astype(np.int64).values + 1)


def m_comseg(tx):
    fs = sorted(glob.glob(f"{R}/comseg_xenium_breast_roi5000/comseg_out/"
                          f"tile_*/spot_assignment.parquet"))
    a = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    a = a.drop_duplicates(["global_x", "global_y", "gene"])   # tile 之间空间重叠
    return match_by_xy(tx, a.global_x.values, a.global_y.values,
                       pd.to_numeric(a["cell"], errors="coerce").values)


def m_boms(tx):
    z = np.load(f"{R}/boms_xenium_breast/boms_xenium_breast.npz", allow_pickle=True)
    px, py = um_to_px(z["x"], z["y"])
    return match_by_xy(tx, px, py, z["seg"])


def m_cellist(tx):
    fs = sorted(glob.glob(f"{CELLIST_DIR}/**/*_Cellist_segmentation.txt",
                          recursive=True))
    if not fs:
        raise FileNotFoundError(f"{CELLIST_DIR} 下没有 segmentation.txt，"
                                f"确认已用修复版脚本跑过")
    s = pd.read_csv(fs[0], sep="\t")
    print(f"      cellist 用: {fs[0].replace(R + '/', '')}")
    return match_by_xy(tx, s.x.values, s.y.values,
                       pd.to_numeric(s["Cellist"], errors="coerce").values)


def m_ucs(tx):
    return match_by_mask(tx, f"{R}/ucs_xenium_breast/ucs_log/pred/segmentation_mask.tif",
                         step_um=1.0)


def m_genesegnet(tx):
    return match_by_mask(tx, f"{R}/genesegnet_xenium_2/genesegnet_xenium_2_mask.tif",
                         step_um=PX)


METHODS = {
    "proseg": m_proseg,
    "comseg": m_comseg,
    "boms": m_boms,
    "cellist": m_cellist,
    "ucs": m_ucs,
    "genesegnet": m_genesegnet,
}


# ==========================================================================
def split_assign(pred_lab, gt):
    """按归属覆盖率分三色。
    绿 = 方法归属了该转录本（无论 GT 是否归属）
    橙 = GT 归属了但方法没有（真正的漏检）
    灰 = 双方都未归属
    三类互斥且覆盖全集。"""
    a = pred_lab > 0
    return a, (~a) & gt, (~a) & (~gt)


def split_cell(pred_lab, gt_cell, gt):
    """按主匹配分三色，橙 = 归属到了非主匹配的细胞，才是真正的错分。
    与 assign 模式一致，不把"方法归属而 GT 未归属"算作错误。"""
    a = pred_lab > 0
    both = a & gt
    green = a.copy()                      # 先假定方法归属的都算对
    if both.any():
        df = pd.DataFrame({"p": pred_lab[both], "r": gt_cell[both]})
        main = df.groupby("p")["r"].agg(lambda s: s.value_counts().idxmax())
        wrong = df["p"].map(main).values != df["r"].values
        green[np.nonzero(both)[0][wrong]] = False    # 只在双方都归属时才判错
    return green, a & ~green, (~a) & (~gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("side", nargs="?", type=int, default=DEFAULT_SIDE_PX,
                    help="裁剪边长（ROI 像素），600 px ≈ 127.5 µm")
    ap.add_argument("--cx", type=float, default=None, help="裁剪中心 x（ROI 像素）")
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--mode", default="assign", choices=["assign", "cell"])
    ap.add_argument("--only", default=None,
                    help="逗号分隔，只跑这几个方法，如 cellist")
    ap.add_argument("--qv", type=float, default=QV_MIN)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()

    tx, gt, gt_cell = get_tx(a.qv)
    X, Y = tx.px.values, tx.py.values
    cx = a.cx if a.cx is not None else X.mean()
    cy = a.cy if a.cy is not None else Y.mean()
    m = (np.abs(X - cx) < a.side / 2) & (np.abs(Y - cy) < a.side / 2)
    print(f"ROI center({cx:.0f},{cy:.0f}) side={a.side}px"
          f"({a.side*PX:.0f}µm) pts={m.sum():,}")

    if a.mode == "assign":
        labels = ["assigned by method", "missed (GT assigned)", "unassigned by both"]
    else:
        labels = ["correct cell", "wrong cell", "unassigned by both"]
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=10, label=t)
        for c, t in zip((C_OK, C_BAD, C_NONE), labels)
    ]

    todo = ([k for k in a.only.split(",") if k in METHODS] if a.only
            else list(METHODS))
    os.makedirs(a.out, exist_ok=True)
    for name in todo:
        fn = METHODS[name]
        try:
            lab = np.asarray(fn(tx))
            green, red, gray = (split_assign(lab, gt) if a.mode == "assign"
                                else split_cell(lab, gt_cell, gt))
            rate = (lab > 0).mean()
            print(f"  {name}: assign={rate:.1%} g={green.sum():,} "
                  f"r={red.sum():,} gray={gray.sum():,}")

            fig, ax = plt.subplots(figsize=(FIG_IN, FIG_IN))
            ax.scatter(X[gray & m], Y[gray & m], s=S_OK, c=C_NONE,
                       linewidths=0, rasterized=RASTERIZE)
            ax.scatter(X[green & m], Y[green & m], s=S_OK, c=C_OK,
                       linewidths=0, rasterized=RASTERIZE)
            ax.scatter(X[red & m], Y[red & m], s=S_BAD, c=C_BAD,
                       linewidths=0, rasterized=RASTERIZE)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_title(f"{name} | assign={rate:.1%}", fontsize=TITLE_FS)
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(handles=legend_elements, loc="lower center",
                      ncol=3, fontsize=LEG_FS)

            out_path = os.path.join(a.out, f"{OUT_PREFIX}_{name}_{a.side}.pdf")
            fig.savefig(out_path, format="pdf", dpi=RASTER_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"    saved: {out_path} "
                  f"({os.path.getsize(out_path)/1e6:.2f} MB)")
        except Exception as e:
            print(f"  {name} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()