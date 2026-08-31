"""
merfish_run_stardist.py
=======================
MERFISH (Vizgen 小鼠脑 Receptor Map) 上的 StarDist 分割 + benchmark。
结构与 merfish_seg_benchmark.py（Cellpose 那份）保持一致，可独立运行，
也可只产出 label mask 后丢进 merfish_seg_benchmark.py 的 EXTRA_METHOD_MASKS。

⚠ 重要：StarDist 是「细胞核」分割模型（2D_versatile_fluo = 荧光核/DAPI）。
   官方 Vizgen 边界是「全细胞」。直接拿核 mask 比全细胞 GT，IoU/PQ 会系统性偏低、
   对 StarDist 不公平。因此本脚本默认在核分割后做一步「核 -> 全细胞」扩张
   （用 polyT 做 watershed，或固定距离 expand_labels），使其与官方全细胞边界可比。
   若你想做「核层面」对比（或另建核 GT），把 EXPANSION_METHOD 设为 "none" 即可。

三个阶段（可各自开关）：
  Stage 1  官方边界 hdf5 -> GT label mask（复用 merfish_seg_utils，与 Cellpose 那份完全同款）
  Stage 2  StarDist 在 DAPI 上分核 -> 可选扩张成全细胞 -> 预测 label mask
  Stage 3  复用 test.py 指标，把 StarDist 预测与官方 GT 对比，输出 CSV

依赖：stardist tensorflow(或 tensorflow-cpu) csbdeep  以及  h5py tifffile scikit-image scipy pandas
     首次运行 from_pretrained 会联网下载预训练模型。

用法：先设置下方 SEG_MODULES_DIR（若三个文件不在同一目录），再改 CONFIG 路径参数，
     然后  python merfish_run_stardist.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # 按需分配显存，避免一次性占满

import sys
import time
import psutil

# 配置 TensorFlow GPU 内存管理（必须在 import tensorflow 之前）
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[TF] 已配置 {len(gpus)} 个 GPU 按需增长显存")
    except RuntimeError as e:
        print(f"[TF] GPU 配置警告: {e}")

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

from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import watershed, expand_labels
from scipy.ndimage import distance_transform_edt

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
OUT_DIR = "/data/qiuyijia/stardist/merfish_prostate2"

# ---- ROI：pixel (x0,y0,x1,y1)。None=整幅(极耗内存)。留空则取中心 ROI_SIZE 窗口 ----
# 注意：必须与 Cellpose 那次用的 ROI 完全一致，两方法才可直接同表比较！
ROI = None
ROI_SIZE = 10000
Z_INDEX = 3                                  # 官方边界取哪层 z，需与图像一致（本数据 z3）

# ---- StarDist 配置 ----
STARDIST_MODEL = "2D_versatile_fluo"         # 荧光核模型；DAPI 用它
NORM_LOW, NORM_HIGH = 1.0, 99.8              # csbdeep 百分位归一化区间
PROB_THRESH = None                           # None=用模型自带最优阈值
NMS_THRESH  = None                           # None=用模型自带最优阈值
TILE_BLOCK = 1024                            # 分块边长(像素)，从 2048 降至 1024 进一步降低显存峰值
USE_GPU = True                               # 仅提示；实际由 TensorFlow 环境决定

# ---- 核 -> 全细胞 扩张（为与官方“全细胞”GT 可比）----
EXPANSION_METHOD = "expand_labels"         # "watershed_polyt" | "expand_labels" | "none"
EXPAND_DISTANCE_PX = 15                       # expand_labels 用：核向外扩张的最大距离(像素)
MAX_EXPANSION_PX = 30                         # watershed 用：限制单个细胞相对核最多外扩的距离；None=不限制
POLYT_SMOOTH_SIGMA = 2.0                      # watershed 前对 polyT 高斯平滑

# ---- benchmark 参数（与 test.py 对齐）----
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5

# ---- 阶段开关 ----
DO_BUILD_GT     = True   # 若已由 Cellpose 那份在同一 OUT_DIR/同一 ROI 生成过 GT，可设 False 直接复用
DO_RUN_STARDIST = True
DO_BENCHMARK    = True
# ===========================================================================


# --------------------------------------------------------------------------- #
# Stage 2a: StarDist 核分割
# --------------------------------------------------------------------------- #
def run_stardist_nuclei(roi):
    """StarDist 在 DAPI(ROI) 上分核，返回 (H,W) int32 label（核）。"""
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize

    dapi = U.read_tif_crop(DAPI_TIF, roi).astype(np.float32)
    img = normalize(dapi, NORM_LOW, NORM_HIGH)
    print(f"[StarDist] DAPI 输入 shape={img.shape}, model={STARDIST_MODEL}")

    model = StarDist2D.from_pretrained(STARDIST_MODEL)

    # 依图像大小自动分块，防止大 ROI 爆显存
    H, W = img.shape[:2]
    n_tiles = (max(1, int(np.ceil(H / TILE_BLOCK))),
               max(1, int(np.ceil(W / TILE_BLOCK))))
    print(f"[StarDist] n_tiles={n_tiles}, prob_thresh={PROB_THRESH}, nms_thresh={NMS_THRESH}")

    labels, _ = model.predict_instances(
        img,
        prob_thresh=PROB_THRESH,
        nms_thresh=NMS_THRESH,
        n_tiles=n_tiles,
    )
    labels = np.asarray(labels).astype(np.int32)
    print(f"[StarDist] 核数量={len(np.unique(labels)) - 1}")
    return labels


# --------------------------------------------------------------------------- #
# Stage 2b: 核 -> 全细胞 扩张
# --------------------------------------------------------------------------- #
def expand_nuclei_to_cells(nuclei, roi):
    """
    把核 label 扩张成全细胞 label（保持每个细胞沿用其核的 id）。
      watershed_polyt : 在 polyT 强度上做分水岭，核为种子，前景由 polyT 阈值决定（更贴近真实胞质）
      expand_labels   : 不依赖 polyT，按固定距离向外均匀扩张（无重叠）
      none            : 不扩张，直接返回核
    """
    if EXPANSION_METHOD == "none":
        print("[Expand] 不扩张，输出核 mask（注意：与全细胞 GT 对比会偏低）")
        return nuclei

    if EXPANSION_METHOD == "expand_labels":
        print(f"[Expand] expand_labels, distance={EXPAND_DISTANCE_PX}px")
        return expand_labels(nuclei, distance=EXPAND_DISTANCE_PX).astype(np.int32)

    if EXPANSION_METHOD == "watershed_polyt":
        if not os.path.exists(POLYT_TIF):
            print("[Expand] 未找到 polyT，回退 expand_labels")
            return expand_labels(nuclei, distance=EXPAND_DISTANCE_PX).astype(np.int32)

        polyt = U.read_tif_crop(POLYT_TIF, roi).astype(np.float32)
        if polyt.shape != nuclei.shape:
            raise ValueError(f"polyT 与核 mask 尺寸不一致: {polyt.shape} vs {nuclei.shape}")

        pol = gaussian(polyt, sigma=POLYT_SMOOTH_SIGMA, preserve_range=True)
        try:
            thr = threshold_otsu(pol)
        except Exception:
            thr = float(np.percentile(pol, 50))
        fg = pol > thr
        fg |= nuclei > 0                       # 核一定算前景

        # 可选：限制相对核的最大外扩距离，避免个别细胞在 polyT 上“漫延”过大
        if MAX_EXPANSION_PX is not None:
            dist_to_nuc = distance_transform_edt(nuclei == 0)
            fg &= dist_to_nuc <= MAX_EXPANSION_PX

        cells = watershed(-pol, markers=nuclei, mask=fg)
        cells = np.asarray(cells).astype(np.int32)
        print(f"[Expand] watershed_polyt 完成，细胞数={len(np.unique(cells)) - 1}")
        return cells

    raise ValueError(f"未知 EXPANSION_METHOD: {EXPANSION_METHOD}")


def segment(roi, out_tif):
    """完整分割：StarDist 分核 -> 可选扩张 -> 存 label tif。"""
    nuclei = run_stardist_nuclei(roi)
    cells = expand_nuclei_to_cells(nuclei, roi)
    U.save_label_tif(cells, out_tif)
    return cells


# --------------------------------------------------------------------------- #
# main：与 Cellpose 那份同款三阶段
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"\n{'='*80}")
    print(f"[Performance] 开始运行 StarDist - 初始内存: {initial_memory:.2f} MB")
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

    gt_tif      = os.path.join(OUT_DIR, "gt_vizgen_official_roi.tif")
    pred_tif    = os.path.join(OUT_DIR, "pred_stardist_roi.tif")

    # -------- Stage 1: GT（可复用 Cellpose 那次生成的同一份）--------
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

    # -------- Stage 2: StarDist 分割 --------
    stage2_start = time.time()
    if DO_RUN_STARDIST:
        segment(roi, pred_tif)

    stage2_time = time.time() - stage2_start
    stage2_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 2 完成 - 用时: {stage2_time:.2f}s, 当前内存: {stage2_memory:.2f} MB")

    # -------- Stage 3: benchmark（复用 test.py 指标）--------
    stage3_start = time.time()
    if DO_BENCHMARK and os.path.exists(pred_tif):
        metrics_mod = BM.load_metrics(TEST_PY_PATH)
        BM.run_benchmark(
            metrics_mod, gt_mask, {"StarDist": pred_tif},
            out_csv=os.path.join(OUT_DIR, "merfish_seg_metrics_stardist.csv"),
        )

    stage3_time = time.time() - stage3_start
    stage3_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 3 完成 - 用时: {stage3_time:.2f}s, 当前内存: {stage3_memory:.2f} MB")

    # 总体统计
    total_time = time.time() - start_time
    peak_memory = stage3_memory
    memory_increase = peak_memory - initial_memory

    print(f"\n{'='*80}")
    print(f"[Performance] StarDist 性能统计总结")
    print(f"{'='*80}")
    print(f"总运行时间: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"初始内存: {initial_memory:.2f} MB")
    print(f"峰值内存: {peak_memory:.2f} MB")
    print(f"内存增量: {memory_increase:.2f} MB")
    print(f"\n各阶段用时:")
    print(f"  Stage 1 (GT构建): {stage1_time:.2f}s")
    print(f"  Stage 2 (StarDist分割): {stage2_time:.2f}s")
    print(f"  Stage 3 (Benchmark): {stage3_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()