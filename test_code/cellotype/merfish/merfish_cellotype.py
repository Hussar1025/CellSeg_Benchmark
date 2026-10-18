"""
merfish_run_cellotype.py
========================
MERFISH (Vizgen 小鼠脑 Receptor Map) 上的 CelloType 分割 + benchmark。
结构与 merfish_run_mesmer.py / merfish_run_stardist.py / merfish_seg_benchmark.py 一致，
可独立运行，也可只产出 label mask 后丢进 merfish_seg_benchmark.py 的 EXTRA_METHOD_MASKS。

关于 CelloType（Nature Methods 2024, tanlabcode/CelloType）：
  * 基于 Detectron2 + MaskDINO 的 Transformer 检测/分割模型；这里用 TissueNet 预训练权重
    做全细胞分割（给核+膜通道），输出与官方 Vizgen「全细胞」边界可比，无需扩张。
  * 通道约定（务必遵守，否则结果全错）：输入为 RGB，其中
        B(蓝)=核通道(DAPI)，G(绿)=膜/胞质通道(polyT)，R(红)=可选(置0)。
    输入按 uint8[0,255]，本脚本对每通道做百分位归一化后转 uint8。
  * 它是 query 数有限、max_det 上限的检测式模型，单次前向吃不下大图（官方建议用 SAHI 做
    大图切片推理）。本脚本自带「切片推理 + 拼接」：把 ROI 切成 TILE_SIZE 小块逐块预测，
    再按全局唯一 id 拼回；用重叠带 + 丢弃“贴内部边界”的检测 来避免细胞被割裂/重复计数
    （要求 TILE_OVERLAP 大于最大细胞直径；本数据细胞 ~90px，默认 overlap=128 足够）。

环境（CelloType 安装较重，建议单独 conda 环境，见官方 README）：
  Python=3.8, PyTorch, Detectron2, Deformable-DETR CUDA ops, 然后 pip install cellotype；
  另需 clone 仓库并 `cd models && sh download.sh` 下载权重与取得 configs/。
  评测侧依赖同前：h5py tifffile scikit-image scipy pandas。

用法：先设置下方 SEG_MODULES_DIR 与 CelloType 仓库/权重路径，改 CONFIG，再
     python merfish_run_cellotype.py
"""

import argparse
import os
import sys
import time
import resource   # 标准库（Unix），替代 psutil：CelloType 环境 python 太低装不了 psutil

# ======================= 依赖模块路径（跨目录时必填）=======================
# merfish_seg_utils.py 与 merfish_seg_benchmark.py 所在目录。
# 三种方式任选其一（按顺序生效）：1) 改下面这行  2) 环境变量 MERFISH_SEG_MODULES_DIR
#   3) 三个文件在同一目录则留空（自动用本文件所在目录）
SEG_MODULES_DIR = "/data/qiuyijia/cellpose"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
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
OUT_DIR = "/data/qiuyijia/cellotype/merfish_prostate2"

# ---- CelloType 仓库/权重/配置（来自 clone 仓库 + models/download.sh）----
CELLOTYPE_REPO = "/data/qiuyijia/cellotype/CelloType"   # 改成你 clone 的 CelloType 仓库目录
CELLOTYPE_MODEL_PATH  = os.path.join(CELLOTYPE_REPO, "models",
    "tissuenet_model_0019999.pth")
CELLOTYPE_CONFIG_PATH = os.path.join(CELLOTYPE_REPO, "configs",
    "maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml")
DEVICE = "cuda"                       # "cuda" | "cpu"
CONFIDENCE_THRESH = 0.1               # 检测置信度阈值（README 示例值）
MAX_DET = 1000                        # 每次前向最大检测数（每个小块，不是整幅）

# ---- 通道归一化（每通道百分位裁剪后 -> uint8[0,255]）----
NORM_LOW, NORM_HIGH = 1.0, 99.0

# ---- 【关键修复】分辨率对齐：推理前把输入降采样到模型训练分辨率 ----
# CelloType 用 TissueNet 预训练权重，训练/工作分辨率 ~0.5µm/px（细胞 ~20px）；
# 本 mosaic ~0.108µm/px（细胞 ~90px），差 ~4.6× => 分布外 => 大量漏检 + 乱框（就是你日志里
# 8740 GT 只分出 2657、且 1901 是 FP 的原因）。
# 做法：推理前把 RGB 降采样到 TARGET_MPP，推理后把 label 最近邻上采样回原 ROI 尺寸再对 GT。
# 这不是"调参/改方法"，等价于 Mesmer 的 image_mpp / Cellpose 的 diameter，是该方法应有的用法。
TARGET_MPP = 0.5                      # 目标分辨率(µm/px)，TissueNet 标准；一般无需改
SOURCE_MPP = None                     # 源分辨率(µm/px)。None=自动从 TRANSFORM_CSV 反推(推荐)
RESCALE_TO_TARGET_MPP = True          # 设 False 回到原始(错误)的全分辨率推理，仅供对照

# ---- 推理方式 ----
# "sahi"  : CelloType 官方大图推理(sahi_predict.get_sliced_prediction)，跨块用 GREEDY-NMM 合并，
#           benchmark 最该用它；需要 CELLOTYPE_REPO 指向 clone 的仓库(内含 sahi_predict.py)。
# "tiled" : 你原来的自写切片+去重(现在跑在"降采样后"的图上，尺度正确才有意义)，无 sahi 环境时备选。
# "single": 整幅一次前向，只适合很小的 ROI 调试。
INFERENCE_MODE = "sahi"
TILE_SIZE = 256                       # 小块边长（与 TissueNet 训练块大小一致）
TILE_OVERLAP = 128                    # "tiled" 用；须 > 降采样后最大细胞直径(~20px)，128 足够
SAHI_OVERLAP_RATIO = 0.2              # "sahi" 用；重叠比例(0.2 => 256*0.2≈51px > 细胞直径)
SAHI_POSTPROCESS = "GREEDYNMM"        # 跨块合并策略: "GREEDYNMM" | "NMM" | "NMS"
SAHI_MATCH_METRIC = "IOU"             # 合并匹配度量: "IOS"(小框交并) | "IOU"
SAHI_MATCH_THRESH = 0.5               # 合并阈值
MIN_CELL_PIXELS = 8                   # 丢弃过小碎片(按降采样分辨率计，≈原图 ~170px)

# ---- ROI：pixel (x0,y0,x1,y1)。None=整幅(极耗内存)。留空取中心 ROI_SIZE 窗口 ----
# 注意：必须与 Cellpose / StarDist / Mesmer 那几次用的 ROI 完全一致，才能同表比较！
ROI = None
ROI_SIZE = 10000
Z_INDEX = 3                           # 官方边界取哪层 z，需与图像一致（本数据 z3）

# ---- benchmark 参数（与 test.py 对齐）----
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5

# ---- 阶段开关 ----
DO_BUILD_GT     = True   # 若已由前面某份在同一 OUT_DIR/同一 ROI 生成过 GT，可设 False 直接复用
DO_RUN_CELLOTYPE = True
DO_BENCHMARK    = True
# ===========================================================================

# --------------------------------------------------------------------------- #
# 预处理：DAPI/polyT -> uint8 RGB (B=核, G=膜, R=0)
# --------------------------------------------------------------------------- #
def _norm_uint8(chan, p_low, p_high):
    lo, hi = np.percentile(chan, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((chan - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def build_cellotype_rgb(roi):
    """构造 CelloType 输入 RGB(uint8, HxWx3)：B=核(DAPI)，G=膜(polyT)，R=0。"""
    dapi = U.read_tif_crop(DAPI_TIF, roi).astype(np.float32)
    if os.path.exists(POLYT_TIF):
        polyt = U.read_tif_crop(POLYT_TIF, roi).astype(np.float32)
        if polyt.shape != dapi.shape:
            raise ValueError(f"DAPI 与 polyT 尺寸不一致: {dapi.shape} vs {polyt.shape}")
    else:
        print("[CelloType][警告] 未找到 polyT 膜通道，全细胞分割质量会明显下降。")
        polyt = np.zeros_like(dapi)

    H, W = dapi.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 1] = _norm_uint8(polyt, NORM_LOW, NORM_HIGH)   # G = 膜/胞质
    rgb[..., 2] = _norm_uint8(dapi, NORM_LOW, NORM_HIGH)    # B = 核
    return rgb


# --------------------------------------------------------------------------- #
# 分辨率对齐：降采样输入 / 最近邻上采样 label 回原尺寸
# --------------------------------------------------------------------------- #
def _source_mpp():
    """源图像 µm/px：优先用 SOURCE_MPP，否则从 micron->pixel 变换矩阵反推 (1 / px_per_µm)。"""
    if SOURCE_MPP is not None:
        return float(SOURCE_MPP)
    M = U.load_micron_to_mosaic_transform(TRANSFORM_CSV)
    sx = float(np.hypot(M[0, 0], M[0, 1]))   # x 方向 px/µm
    sy = float(np.hypot(M[1, 0], M[1, 1]))   # y 方向 px/µm
    px_per_um = 0.5 * (sx + sy)
    if px_per_um <= 0:
        raise ValueError(f"从 {TRANSFORM_CSV} 反推分辨率失败 (px/µm={px_per_um})")
    return 1.0 / px_per_um


def _downsample_rgb(rgb):
    """把 uint8 RGB 降采样到 TARGET_MPP。返回 (rgb_ds, scale)，scale=输出/输入(<1=缩小)。"""
    if not RESCALE_TO_TARGET_MPP:
        return rgb, 1.0
    src_mpp = _source_mpp()
    scale = src_mpp / float(TARGET_MPP)          # <1: 降采样
    H, W = rgb.shape[:2]
    print(f"[Rescale] 源≈{src_mpp:.4f}µm/px, 目标={TARGET_MPP}µm/px, scale={scale:.4f} "
          f"(细胞 ~90px -> ~{90 * scale:.0f}px，目标 ~20px)")
    if abs(scale - 1.0) < 0.02:
        print("[Rescale] 分辨率已接近目标，跳过缩放")
        return rgb, 1.0
    new_w, new_h = max(1, round(W * scale)), max(1, round(H * scale))
    try:
        import cv2                               # cellotype 硬依赖，通常都在；INTER_AREA 是缩小最佳
        rgb_ds = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from skimage.transform import resize
        rgb_ds = resize(rgb, (new_h, new_w), order=1, anti_aliasing=True, preserve_range=True)
        rgb_ds = np.clip(np.rint(rgb_ds), 0, 255).astype(np.uint8)
    print(f"[Rescale] 输入 {(H, W)} -> 降采样 {rgb_ds.shape[:2]}")
    return rgb_ds, float(new_h) / float(H)


def _upsample_labels_nn(lbl, out_hw):
    """label(实例 id 图)最近邻上采样到 out_hw=(H,W)，保持整数 id 不被插值破坏(纯 numpy，省内存)。"""
    out_h, out_w = out_hw
    in_h, in_w = lbl.shape
    if (in_h, in_w) == (out_h, out_w):
        return lbl.astype(np.int32)
    row = np.floor((np.arange(out_h) + 0.5) * in_h / out_h).astype(np.int64)
    col = np.floor((np.arange(out_w) + 0.5) * in_w / out_w).astype(np.int64)
    np.clip(row, 0, in_h - 1, out=row)
    np.clip(col, 0, in_w - 1, out=col)
    return lbl[np.ix_(row, col)].astype(np.int32)


# --------------------------------------------------------------------------- #
# 切片推理 + 拼接
# --------------------------------------------------------------------------- #
def _tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:     # 保证最后一块贴到边缘
        starts.append(length - tile)
    return sorted(set(starts))


def tiled_predict(model, rgb):
    """
    对 uint8 RGB 做切片推理并拼成全局 label。
    去重策略：丢弃“贴到小块内部边界（非全局边界）”的检测——这类细胞会被相邻重叠块
    在更居中的位置完整捕获；只要 TILE_OVERLAP > 最大细胞直径，每个细胞都会被某一块完整收下。
    """
    H, W = rgb.shape[:2]
    tile = TILE_SIZE
    stride = max(1, tile - TILE_OVERLAP)
    ys = _tile_starts(H, tile, stride)
    xs = _tile_starts(W, tile, stride)

    canvas = np.zeros((H, W), dtype=np.int32)
    next_id = 1
    n_tiles = len(ys) * len(xs)
    print(f"[CelloType] 切片 {len(ys)}x{len(xs)}={n_tiles} 块, tile={tile}, overlap={TILE_OVERLAP}")

    ti = 0
    for y in ys:
        for x in xs:
            ti += 1
            y1, x1 = min(y + tile, H), min(x + tile, W)
            patch = rgb[y:y1, x:x1]
            ph, pw = patch.shape[:2]

            local = np.asarray(model.predict(patch)).astype(np.int32)

            for lid in np.unique(local):
                if lid == 0:
                    continue
                m = local == lid
                ys_i, xs_i = np.where(m)
                # 是否贴到“内部”边界（该边不是全局图像边界）
                touch_top    = ys_i.min() == 0     and y > 0
                touch_left   = xs_i.min() == 0     and x > 0
                touch_bottom = ys_i.max() == ph - 1 and y1 < H
                touch_right  = xs_i.max() == pw - 1 and x1 < W
                if touch_top or touch_left or touch_bottom or touch_right:
                    continue  # 交给相邻重叠块处理

                sub = canvas[y:y1, x:x1]
                m &= sub == 0                       # 防御：只写空白处
                if int(m.sum()) < MIN_CELL_PIXELS:
                    continue
                sub[m] = next_id
                next_id += 1

            if ti % 20 == 0 or ti == n_tiles:
                print(f"[CelloType] 已处理 {ti}/{n_tiles} 块，当前细胞数 {next_id - 1}")

    return canvas


# --------------------------------------------------------------------------- #
# CelloType 官方大图推理 (SAHI 切片 + GREEDY-NMM 跨块合并)
# --------------------------------------------------------------------------- #
def _predict_sahi(rgb_ds, work_dir):
    """
    用 CelloType 仓库自带的 sahi_predict.get_sliced_prediction 做官方切片推理。
    跨块重复由 SAHI 的 GREEDY-NMM 合并（而不是自写"丢弃贴边检测"），benchmark 更可辩护。
    需要 CELLOTYPE_REPO 指向 clone 的仓库根（内含 sahi_predict.py），流程与官方 sahi_inference.py 一致。
    """
    if CELLOTYPE_REPO and CELLOTYPE_REPO not in sys.path:
        sys.path.insert(0, CELLOTYPE_REPO)       # 让 `import sahi_predict` 可用(位于仓库根)
    from cellotype.predict import Detectron2DetectionModel
    import sahi_predict as SP

    model = Detectron2DetectionModel(
        model_path=CELLOTYPE_MODEL_PATH,
        config_path=CELLOTYPE_CONFIG_PATH,
        confidence_threshold=CONFIDENCE_THRESH,
        image_size=TILE_SIZE,
        max_det=MAX_DET,
        device=DEVICE,
        channels=3,
    )

    # 写成临时 PNG 再传路径：与官方 sahi_inference.py 完全一致，消除 np 数组的通道/格式歧义
    os.makedirs(work_dir, exist_ok=True)
    tmp_png = os.path.join(work_dir, "_cellotype_sahi_input.png")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    Image.fromarray(rgb_ds).save(tmp_png)

    print(f"[CelloType][sahi] slice={TILE_SIZE}, overlap_ratio={SAHI_OVERLAP_RATIO}, "
          f"postprocess={SAHI_POSTPROCESS}/{SAHI_MATCH_METRIC}@{SAHI_MATCH_THRESH}")
    result = SP.get_sliced_prediction(
        tmp_png, model,
        slice_height=TILE_SIZE, slice_width=TILE_SIZE,
        overlap_height_ratio=SAHI_OVERLAP_RATIO,
        overlap_width_ratio=SAHI_OVERLAP_RATIO,
        perform_standard_pred=False,             # 不在整幅再跑一遍(省显存，官方示例同款)
        postprocess_type=SAHI_POSTPROCESS,
        postprocess_match_metric=SAHI_MATCH_METRIC,
        postprocess_match_threshold=SAHI_MATCH_THRESH,
        verbose=1,
    )

    h, w = rgb_ds.shape[:2]
    label = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    for obj in result.object_prediction_list:
        bm = np.asarray(obj.mask.bool_mask, dtype=bool)  # SAHI 已把 mask 移到整幅坐标
        if bm.shape != (h, w):
            continue
        bm &= label == 0                          # 已占像素不覆盖
        if int(bm.sum()) < MIN_CELL_PIXELS:
            continue
        label[bm] = next_id
        next_id += 1
    try:
        os.remove(tmp_png)
    except OSError:
        pass
    print(f"[CelloType][sahi] 合并后细胞数={next_id - 1}")
    return label


def run_cellotype(roi, out_tif):
    """在 ROI 上运行 CelloType：降采样->推理->label 最近邻回投原尺寸，保存并返回 label mask。"""
    rgb_full = build_cellotype_rgb(roi)
    H, W = rgb_full.shape[:2]
    print(f"[CelloType] 原始输入 shape={rgb_full.shape}, device={DEVICE}, "
          f"conf={CONFIDENCE_THRESH}, max_det={MAX_DET}, mode={INFERENCE_MODE}")

    # 关键：降采样到训练分辨率再推理
    rgb_ds, scale = _downsample_rgb(rgb_full)
    del rgb_full

    if INFERENCE_MODE == "sahi":
        label_ds = _predict_sahi(rgb_ds, work_dir=os.path.dirname(os.path.abspath(out_tif)))
    else:
        from cellotype.predict import CelloTypePredictor
        model = CelloTypePredictor(
            model_path=CELLOTYPE_MODEL_PATH,
            confidence_thresh=CONFIDENCE_THRESH,
            max_det=MAX_DET,
            device=DEVICE,
            config_path=CELLOTYPE_CONFIG_PATH,
        )
        if INFERENCE_MODE == "single":
            label_ds = np.asarray(model.predict(rgb_ds)).astype(np.int32)
        else:  # "tiled"
            label_ds = tiled_predict(model, rgb_ds)

    # label 从降采样尺寸最近邻回投到原 ROI 尺寸(与全分辨率 GT 对齐)，再落盘
    mask = _upsample_labels_nn(label_ds, (H, W))
    print(f"[CelloType] 细胞数={len(np.unique(mask)) - 1}  "
          f"(label 回投 {tuple(label_ds.shape)} -> {tuple(mask.shape)})")
    U.save_label_tif(mask, out_tif)
    return mask


# --------------------------------------------------------------------------- #
# 内存监控（纯标准库，不依赖 psutil）
#   current_mem_mb : 当前 RSS，读 /proc/self/statm（Linux）
#   peak_mem_mb    : 进程历史峰值 RSS，resource.getrusage 的高水位，比采样更可靠
# --------------------------------------------------------------------------- #
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def current_mem_mb():
    """当前进程 RSS(MB)。非 Linux 或读取失败时回退到 getrusage 峰值。"""
    try:
        with open("/proc/self/statm", "r") as f:
            rss_pages = int(f.read().split()[1])
        return rss_pages * _PAGE_SIZE / 1024 / 1024
    except (OSError, IndexError, ValueError):
        return peak_mem_mb()


def peak_mem_mb():
    """进程历史峰值 RSS(MB)。Linux 的 ru_maxrss 单位是 KB，macOS 是 byte。"""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024.0 if sys.platform != "darwin" else ru / 1024.0 / 1024.0


_GPU_PEAK_RESERVED_MB = 0.0


def gpu_status():
    """返回 CelloType 使用的 PyTorch GPU 状态及进程显存（MB）。"""
    global _GPU_PEAK_RESERVED_MB
    requested = str(DEVICE).lower().startswith("cuda")
    try:
        import torch
        available = bool(torch.cuda.is_available())
    except Exception:
        return (f"请求使用GPU={requested}, GPU可用=False, 实际使用=False, "
                "显存=不可用(torch 未安装或加载失败)")
    if not available:
        return f"请求使用GPU={requested}, GPU可用=False, 实际使用=False, 显存=N/A"
    try:
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        total = torch.cuda.get_device_properties(device_index).total_memory / 1024**2
        allocated = torch.cuda.memory_allocated(device_index) / 1024**2
        reserved = torch.cuda.memory_reserved(device_index) / 1024**2
        peak_allocated = torch.cuda.max_memory_allocated(device_index) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device_index) / 1024**2
        _GPU_PEAK_RESERVED_MB = max(_GPU_PEAK_RESERVED_MB, peak_reserved)
        return (f"请求使用GPU={requested}, GPU可用=True, 实际使用={requested}, "
                f"设备=cuda:{device_index} ({device_name}), "
                f"显存当前={allocated:.1f}/{reserved:.1f}/{total:.1f} MB(已分配/已保留/总量), "
                f"显存峰值={peak_allocated:.1f}/{_GPU_PEAK_RESERVED_MB:.1f} MB(已分配/已保留)")
    except Exception as exc:
        return (f"请求使用GPU={requested}, GPU可用=True, 实际使用={requested}, "
                f"显存读取失败: {exc}")


# --------------------------------------------------------------------------- #
# main：与前几份同款三阶段
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    initial_memory = current_mem_mb()
    print(f"\n{'='*80}")
    print(f"[Performance] 开始运行 CelloType - 初始内存: {initial_memory:.2f} MB")
    print(f"[GPU] {gpu_status()}")
    print(f"{'='*80}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    BM.BOUNDARY_MODE = BOUNDARY_MODE
    BM.IOU_THRESHOLD = IOU_THRESHOLD

    roi = ROI
    if roi is None and ROI_SIZE is not None:
        roi = U.make_center_roi(DAPI_TIF, size=ROI_SIZE)
        print(f"[ROI] 自动取中心窗口: {roi}")
    elif roi is None:
        print("[ROI] 整幅评测（注意内存占用）")

    gt_tif   = os.path.join(OUT_DIR, "gt_vizgen_official_roi.tif")
    pred_tif = os.path.join(OUT_DIR, "pred_cellotype_roi.tif")

    # -------- Stage 1: GT --------
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
    stage1_memory = current_mem_mb()
    print(f"[Performance] Stage 1 完成 - 用时: {stage1_time:.2f}s, "
          f"当前内存: {stage1_memory:.2f} MB, 峰值: {peak_mem_mb():.2f} MB, GPU: {gpu_status()}")

    # -------- Stage 2: CelloType 分割 --------
    stage2_start = time.time()
    if DO_RUN_CELLOTYPE:
        run_cellotype(roi, pred_tif)

    stage2_time = time.time() - stage2_start
    stage2_memory = current_mem_mb()
    print(f"[Performance] Stage 2 完成 - 用时: {stage2_time:.2f}s, "
          f"当前内存: {stage2_memory:.2f} MB, 峰值: {peak_mem_mb():.2f} MB, GPU: {gpu_status()}")

    # -------- Stage 3: benchmark --------
    stage3_start = time.time()
    if DO_BENCHMARK and os.path.exists(pred_tif):
        metrics_mod = BM.load_metrics(TEST_PY_PATH)
        BM.run_benchmark(
            metrics_mod, gt_mask, {"CelloType": pred_tif},
            out_csv=os.path.join(OUT_DIR, "merfish_seg_metrics_cellotype.csv"),
        )

    stage3_time = time.time() - stage3_start
    stage3_memory = current_mem_mb()
    print(f"[Performance] Stage 3 完成 - 用时: {stage3_time:.2f}s, "
          f"当前内存: {stage3_memory:.2f} MB, 峰值: {peak_mem_mb():.2f} MB, GPU: {gpu_status()}")

    # 总体统计
    total_time = time.time() - start_time
    peak_memory = peak_mem_mb()          # 真实历史高水位，不是收尾时的采样值
    memory_increase = peak_memory - initial_memory

    print(f"\n{'='*80}")
    print(f"[Performance] CelloType 性能统计总结")
    print(f"{'='*80}")
    print(f"总运行时间: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"初始内存: {initial_memory:.2f} MB")
    print(f"结束内存: {stage3_memory:.2f} MB")
    print(f"峰值内存: {peak_memory:.2f} MB")
    print(f"内存增量: {memory_increase:.2f} MB (峰值 - 初始)")
    print(f"GPU统计: {gpu_status()}")
    print(f"\n各阶段用时:")
    print(f"  Stage 1 (GT构建): {stage1_time:.2f}s")
    print(f"  Stage 2 (CelloType分割): {stage2_time:.2f}s")
    print(f"  Stage 3 (Benchmark): {stage3_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
