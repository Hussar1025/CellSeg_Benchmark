"""
merfish_run_mesmer.py
=====================
MERFISH (Vizgen 小鼠脑 Receptor Map) 上的 Mesmer (DeepCell) 分割 + benchmark。
结构与 merfish_run_stardist.py / merfish_seg_benchmark.py 保持一致，可独立运行，
也可只产出 label mask 后丢进 merfish_seg_benchmark.py 的 EXTRA_METHOD_MASKS。

关于 Mesmer：
  * 原生「全细胞 / 核」联合分割模型，直接输出全细胞 mask —— 无需像 StarDist 那样再扩张，
    与官方 Vizgen「全细胞」边界天然可比（和 Cellpose 一致）。
  * 固定吃双通道输入：通道0=核(DAPI)，通道1=膜/胞质(这里用 polyT 作胞质代理)。
  * 关键参数 image_mpp（微米/像素）：Mesmer 在 0.5µm/px 上训练，会按该值内部缩放。
    本数据 ~0.108µm/px，必须传对。脚本默认从 micron_to_mosaic_pixel_transform.csv 自动推导。
  * compartment: "whole-cell"(默认，与全细胞 GT 对齐) | "nuclear" | "both"。

三个阶段（可各自开关）：
  Stage 1  官方边界 hdf5 -> GT label mask（复用 merfish_seg_utils，与前两份完全同款）
  Stage 2  Mesmer 用 DAPI+polyT 出全细胞 mask -> 预测 label mask
  Stage 3  复用 test.py 指标，把 Mesmer 预测与官方 GT 对比，输出 CSV

依赖：deepcell (会带 TensorFlow)  以及  h5py tifffile scikit-image scipy pandas
     首次运行会下载模型权重；新版 deepcell 可能需要在 users.deepcell.org 申请 token
     并设置环境变量 DEEPCELL_ACCESS_TOKEN。

用法：先设置下方 SEG_MODULES_DIR（若三个文件不在同一目录），再改 CONFIG 路径参数，
     然后  python merfish_run_mesmer.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import time
import psutil

# ======================= 依赖模块路径（跨目录时必填）=======================
# merfish_seg_utils.py 与 merfish_seg_benchmark.py 若与本文件不在同一目录，
# 把它们所在的目录填到 SEG_MODULES_DIR。三种方式任选其一（按顺序生效）：
#   1) 直接改下面这行
#   2) 设环境变量 MERFISH_SEG_MODULES_DIR
#   3) 三个文件在同一目录 -> 留空即可（自动用本文件所在目录）
SEG_MODULES_DIR = "/data/qiuyijia/cellpose"

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:               # 交互式/exec 环境下 __file__ 可能不存在
    _this_dir = os.getcwd()

for _d in (SEG_MODULES_DIR, os.environ.get("MERFISH_SEG_MODULES_DIR", ""), _this_dir):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
# ==========================================================================

import numpy as np
import pandas as pd
import tifffile

try:
    import merfish_utils as U
    import merfish_seg_bench as BM   # 复用 load_metrics / evaluate_against_reference / run_benchmark
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        f"找不到模块 {_e.name}。请把本文件顶部的 SEG_MODULES_DIR 设为存放 "
        f"merfish_seg_utils.py 与 merfish_seg_benchmark.py 的目录，"
        f"或设置环境变量 MERFISH_SEG_MODULES_DIR。"
    ) from _e


# ===========================================================================
# ============================= CONFIG ======================================
# ===========================================================================
DATA_DIR = "/data/qiuyijia/dataset/merfish_prostate2"   # Liver1 数据集根目录

DAPI_TIF   = os.path.join(DATA_DIR, "images",
    "mosaic_DAPI_z3.tif")
# POLYT_TIF  = os.path.join(DATA_DIR, "images",
#     "mosaic_PolyT_z3.tif")  # 原用polyT作为膜通道，但polyT是胞质信号，不如真正的膜染色
POLYT_TIF  = os.path.join(DATA_DIR, "images",
    "mosaic_Cellbound1_z3.tif")  # 改用真正的膜染色Cellbound1，Mesmer在清晰膜信号上表现更好
TRANSFORM_CSV = os.path.join(DATA_DIR, "images",
    "micron_to_mosaic_pixel_transform.csv")
CELL_METADATA_CSV = os.path.join(DATA_DIR,
    "cell_metadata.csv")
CELL_BOUNDARIES_DIR = os.path.join(DATA_DIR, "cell_boundaries")

# 你的 test.py 路径（指标从这里加载，保证口径一致）
TEST_PY_PATH = "/data/qiuyijia/eval/test.py"

# 输出目录
OUT_DIR = "/data/qiuyijia/Mesmer/merfish_prostate2"

# ---- ROI：pixel (x0,y0,x1,y1)。None=整幅(极耗内存)。留空则取中心 ROI_SIZE 窗口 ----
# 注意：必须与 Cellpose / StarDist 那几次用的 ROI 完全一致，多方法才可直接同表比较！
ROI = None
ROI_SIZE = 10000
Z_INDEX = 3                                  # 官方边界取哪层 z，需与图像一致（本数据 z3）

# ---- Mesmer 配置 ----
COMPARTMENT = "whole-cell"                   # "whole-cell"(与全细胞 GT 对齐) | "nuclear" | "both"
IMAGE_MPP = None                             # 微米/像素；None=从 TRANSFORM_CSV 自动推导（本数据 ~0.108）
BATCH_SIZE = 4
USE_GPU = True                               # 仅提示；实际由 TensorFlow 环境决定

# ---- benchmark 参数（与 test.py 对齐）----
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5

# ---- 阶段开关 ----
DO_BUILD_GT   = True   # 若已由前面某份在同一 OUT_DIR/同一 ROI 生成过 GT，可设 False 直接复用
DO_RUN_MESMER = True
DO_BENCHMARK  = True
# ===========================================================================


# --------------------------------------------------------------------------- #
# 从 micron->pixel 变换矩阵推导 image_mpp（微米/像素）
# --------------------------------------------------------------------------- #
def derive_image_mpp(transform_csv):
    """
    变换矩阵是 micron->pixel（像素/微米）。取 2x2 线性块两列的模长作为 x/y 方向像素密度
    （对含旋转的仿射也稳），再取其倒数的均值 = 微米/像素。
    """
    M = U.load_micron_to_mosaic_transform(transform_csv)
    sx = float(np.hypot(M[0, 0], M[1, 0]))   # 第一列模长 = x 方向 像素/微米
    sy = float(np.hypot(M[0, 1], M[1, 1]))   # 第二列模长 = y 方向 像素/微米
    if sx <= 0 or sy <= 0:
        raise ValueError(f"从变换矩阵推导像素尺度失败：sx={sx}, sy={sy}")
    mpp = 0.5 * (1.0 / sx + 1.0 / sy)
    return mpp


# --------------------------------------------------------------------------- #
# Stage 2: Mesmer 分割
# --------------------------------------------------------------------------- #
def build_mesmer_input(roi):
    """
    构造 Mesmer 输入： (1, H, W, 2)，通道0=核(DAPI)，通道1=膜/胞质(polyT)。
    """
    dapi = U.read_tif_crop(DAPI_TIF, roi).astype(np.float32)

    if os.path.exists(POLYT_TIF):
        membrane = U.read_tif_crop(POLYT_TIF, roi).astype(np.float32)
        if membrane.shape != dapi.shape:
            raise ValueError(f"DAPI 与 polyT 尺寸不一致: {dapi.shape} vs {membrane.shape}")
    else:
        if COMPARTMENT == "whole-cell":
            print("[Mesmer][警告] 未找到 polyT 膜通道，whole-cell 分割质量会明显下降；"
                  "建议改用 COMPARTMENT='nuclear' 或补上膜/胞质通道。")
        membrane = np.zeros_like(dapi)

    X = np.stack([dapi, membrane], axis=-1)[None, ...]  # (1,H,W,2)
    return X


def run_mesmer(roi, out_tif):
    """在 ROI 上运行 Mesmer，保存并返回预测 label mask (H,W int32)。"""
    from deepcell.applications import Mesmer

    X = build_mesmer_input(roi)
    mpp = IMAGE_MPP if IMAGE_MPP is not None else derive_image_mpp(TRANSFORM_CSV)
    print(f"[Mesmer] 输入 shape={X.shape}, compartment={COMPARTMENT}, "
          f"image_mpp={mpp:.4f} µm/px, GPU={USE_GPU}")

    app = Mesmer()
    labeled = app.predict(
        X,
        batch_size=BATCH_SIZE,
        image_mpp=mpp,
        compartment=COMPARTMENT,
    )
    # whole-cell / nuclear -> (1,H,W,1)；both -> (1,H,W,2)，此处取全细胞通道(第0通道)
    labeled = np.asarray(labeled)
    if COMPARTMENT == "both":
        mask = labeled[0, ..., 0]   # 通道0=whole-cell，通道1=nuclear
    else:
        mask = labeled[0, ..., 0]
    mask = mask.astype(np.int32)

    print(f"[Mesmer] 细胞数={len(np.unique(mask)) - 1}")
    U.save_label_tif(mask, out_tif)
    return mask


# --------------------------------------------------------------------------- #
# main：与前两份同款三阶段
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"\n{'='*80}")
    print(f"[Performance] 开始运行 Mesmer - 初始内存: {initial_memory:.2f} MB")
    print(f"{'='*80}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # 让复用的 BM.run_benchmark 用本脚本的指标参数
    BM.BOUNDARY_MODE = BOUNDARY_MODE
    BM.IOU_THRESHOLD = IOU_THRESHOLD

    # 确定 ROI（务必与其它方法一致）
    roi = ROI
    if roi is None and ROI_SIZE is not None:
        roi = U.make_center_roi(DAPI_TIF, size=ROI_SIZE)
        print(f"[ROI] 自动取中心窗口: {roi}")
    elif roi is None:
        print("[ROI] 整幅评测（注意内存占用）")

    gt_tif   = os.path.join(OUT_DIR, "gt_vizgen_official_roi.tif")
    pred_tif = os.path.join(OUT_DIR, "pred_mesmer_roi.tif")

    # -------- Stage 1: GT（可复用前面某份生成的同一份）--------
    stage1_start = time.time()
    if DO_BUILD_GT or not os.path.exists(gt_tif):
        gt_mask, id_map = U.build_reference_label_mask(
            cell_boundaries_dir=CELL_BOUNDARIES_DIR,
            transform_csv=TRANSFORM_CSV,
            roi=roi,
            z_index=Z_INDEX,
            cell_metadata_csv=CELL_METADATA_CSV,
        )
        U.save_label_tif(gt_mask, gt_tif)
        pd.DataFrame(
            {"label": list(id_map.keys()), "vizgen_cell_id": list(id_map.values())}
        ).to_csv(os.path.join(OUT_DIR, "gt_label_to_cellid.csv"), index=False)
    else:
        gt_mask = np.squeeze(tifffile.imread(gt_tif)).astype(np.int32)
        print(f"[GT] 复用已有 GT: {gt_tif} shape={gt_mask.shape}")

    stage1_time = time.time() - stage1_start
    stage1_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 1 完成 - 用时: {stage1_time:.2f}s, 当前内存: {stage1_memory:.2f} MB")

    # -------- Stage 2: Mesmer 分割 --------
    stage2_start = time.time()
    if DO_RUN_MESMER:
        run_mesmer(roi, pred_tif)

    stage2_time = time.time() - stage2_start
    stage2_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 2 完成 - 用时: {stage2_time:.2f}s, 当前内存: {stage2_memory:.2f} MB")

    # -------- Stage 3: benchmark（复用 test.py 指标）--------
    stage3_start = time.time()
    if DO_BENCHMARK and os.path.exists(pred_tif):
        metrics_mod = BM.load_metrics(TEST_PY_PATH)
        BM.run_benchmark(
            metrics_mod, gt_mask, {"Mesmer": pred_tif},
            out_csv=os.path.join(OUT_DIR, "merfish_seg_metrics_mesmer.csv"),
        )

    stage3_time = time.time() - stage3_start
    stage3_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 3 完成 - 用时: {stage3_time:.2f}s, 当前内存: {stage3_memory:.2f} MB")

    # 总体统计
    total_time = time.time() - start_time
    peak_memory = stage3_memory
    memory_increase = peak_memory - initial_memory

    print(f"\n{'='*80}")
    print(f"[Performance] Mesmer 性能统计总结")
    print(f"{'='*80}")
    print(f"总运行时间: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"初始内存: {initial_memory:.2f} MB")
    print(f"峰值内存: {peak_memory:.2f} MB")
    print(f"内存增量: {memory_increase:.2f} MB")
    print(f"\n各阶段用时:")
    print(f"  Stage 1 (GT构建): {stage1_time:.2f}s")
    print(f"  Stage 2 (Mesmer分割): {stage2_time:.2f}s")
    print(f"  Stage 3 (Benchmark): {stage3_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()