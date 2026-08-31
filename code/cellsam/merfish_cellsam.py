"""
merfish_run_cellsam.py
======================
MERFISH (Vizgen 小鼠脑 Receptor Map) 上的 CellSAM 分割 + benchmark.
结构与 merfish_run_cellotype.py / merfish_run_mesmer.py / merfish_seg_benchmark.py 一致,
可独立运行,也可只产出 label mask 后丢进 merfish_seg_benchmark.py 的 EXTRA_METHOD_MASKS.

关于 CellSAM（Nature Methods 2025, vanvalenlab/cellSAM）:
  * SAM + CellFinder 的通用细胞分割基础模型;给核+膜通道输出全细胞 mask,
    与官方 Vizgen「全细胞」边界可比,无需扩张（论文中即用核+膜染色对 MERFISH 组织分割）.
  * 通道约定（注意与 CelloType 相反！）:输入 3 通道图,
        B(蓝)=胞质/膜通道(polyT),G(绿)=核通道(DAPI),R(红)=可选(置0).
  * 是检测式模型（CellFinder 出框、SAM 出 mask）,单图检测数有限、训练/评测在 512×512
    上做（25% 重叠切片）.本脚本自带「切片推理 + 拼接」处理大 ROI,默认 tile=512/overlap=128
    正好对应 25% 重叠;拼接用重叠带 + 丢弃"贴内部边界"的检测,避免边界割裂/重复计数
    （要求 overlap > 最大细胞直径;本数据细胞 ~90px,128 足够）.
  * 主接口 segment_cellular_image(img, device=...) 返回 (mask[H,W], ...).

环境:pip install git+https://github.com/vanvalenlab/cellSAM.git  （python>=3.10,纯 PyTorch）
     首次运行会联网下载模型权重.评测侧依赖同前:h5py tifffile scikit-image scipy pandas.

用法:先设置下方 SEG_MODULES_DIR,改 CONFIG,再  python merfish_run_cellsam.py
"""

import os
import sys
import time
import psutil

# ======================= 依赖模块路径（跨目录时必填）=======================
# merfish_seg_utils.py 与 merfish_seg_benchmark.py 所在目录.
# 三种方式任选其一（按顺序生效）:1) 改下面这行  2) 环境变量 MERFISH_SEG_MODULES_DIR
#   3) 三个文件在同一目录则留空（自动用本文件所在目录）
SEG_MODULES_DIR = "/data/qiuyijia/cellpose"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # PyTorch 显存碎片管理

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
        f"找不到模块 {_e.name}.请把本文件顶部的 SEG_MODULES_DIR 设为存放 "
        f"merfish_seg_utils.py 与 merfish_seg_benchmark.py 的目录,"
        f"或设置环境变量 MERFISH_SEG_MODULES_DIR."
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
OUT_DIR = "/data/qiuyijia/CellSam/merfish_prostate2"

# ---- CellSAM 配置 ----
DEVICE = "cuda"                       # "cuda" | "cpu" - 使用 GPU（CPU会导致内存暴增）
NORMALIZE = True                      # 传给 segment_cellular_image 的内部直方图归一化开关
BBOX_THRESHOLD = 0.4                  # CellFinder 检测框置信度阈值（None=用其默认）
SWAP_NUC_CYTO = False                 # False: B=polyT(胞质), G=DAPI(核)（默认约定）
                                      # True : 若你这版约定相反,则对调核/胞质到通道里

# ---- 通道预缩放（每通道百分位裁剪 -> [0,1] float32）----
NORM_LOW, NORM_HIGH = 1.0, 99.0

# ---- 切片推理 ----
INFERENCE_MODE = "tiled"              # "tiled"(大 ROI 必须) | "single"(仅小 ROI/调试)
TILE_SIZE = 512                       # CellSAM 训练/评测块大小
TILE_OVERLAP = 128                    # 25% 重叠;须 > 最大细胞直径（本数据 ~90px）
MIN_CELL_PIXELS = 8                   # 拼接时丢弃过小碎片

# ---- ROI:pixel (x0,y0,x1,y1).None=整幅(极耗内存).留空取中心 ROI_SIZE 窗口 ----
# 注意:必须与 Cellpose / StarDist / Mesmer / CelloType 那几次的 ROI 完全一致！
ROI = None
ROI_SIZE = 10000
Z_INDEX = 3                           # 官方边界取哪层 z,需与图像一致（本数据 z3）

# ---- benchmark 参数（与 test.py 对齐）----
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5

# ---- 阶段开关 ----
DO_BUILD_GT    = True   # 若已由前面某份在同一 OUT_DIR/同一 ROI 生成过 GT,可设 False 直接复用
DO_RUN_CELLSAM = True
DO_BENCHMARK   = True
# ===========================================================================


# --------------------------------------------------------------------------- #
# 预处理:DAPI/polyT -> float32[0,1] RGB (B=胞质, G=核, R=0)
# --------------------------------------------------------------------------- #
def _norm01(chan, p_low, p_high):
    lo, hi = np.percentile(chan, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((chan - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_cellsam_rgb(roi):
    """构造 CellSAM 输入 RGB(float32[0,1], HxWx3):默认 B=胞质(polyT),G=核(DAPI),R=0."""
    dapi = U.read_tif_crop(DAPI_TIF, roi).astype(np.float32)
    if os.path.exists(POLYT_TIF):
        polyt = U.read_tif_crop(POLYT_TIF, roi).astype(np.float32)
        if polyt.shape != dapi.shape:
            raise ValueError(f"DAPI 与 polyT 尺寸不一致: {dapi.shape} vs {polyt.shape}")
    else:
        print("[CellSAM][警告] 未找到 polyT 膜通道,全细胞分割质量会明显下降.")
        polyt = np.zeros_like(dapi)

    nuc = _norm01(dapi, NORM_LOW, NORM_HIGH)     # 核
    cyto = _norm01(polyt, NORM_LOW, NORM_HIGH)   # 胞质/膜
    if SWAP_NUC_CYTO:
        nuc, cyto = cyto, nuc

    H, W = dapi.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 1] = nuc     # G(绿)=核
    rgb[..., 2] = cyto    # B(蓝)=胞质/膜
    return rgb


# --------------------------------------------------------------------------- #
# CellSAM 单块调用（model 参数是必需的）
# --------------------------------------------------------------------------- #
def _cellsam_predict_patch(seg_fn, patch, model_obj):
    """调用 CellSAM 分割单个 patch,处理显存不足时清理缓存并重试."""
    import torch
    import gc

    # model 参数是必需的
    if model_obj is None:
        raise ValueError("model_obj 不能为 None,segment_cellular_image 需要 model 参数")

    # 构建参数
    kwargs = {
        "img": patch,
        "model": model_obj,
        "device": DEVICE,
    }
    if NORMALIZE is not None:
        kwargs["normalize"] = NORMALIZE
    if BBOX_THRESHOLD is not None:
        kwargs["bbox_threshold"] = BBOX_THRESHOLD

    try:
        out = seg_fn(**kwargs)
        mask = out[0] if isinstance(out, (tuple, list)) else out
        result = np.asarray(mask) if mask is not None else None

        # 显式删除中间变量并清理内存
        del out, mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return result
    except RuntimeError as e:
        if "out of memory" in str(e):
            # 清理显存缓存后重试一次
            print(f"[CellSAM] OOM 检测到,清理显存后重试...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            out = seg_fn(**kwargs)
            mask = out[0] if isinstance(out, (tuple, list)) else out
            result = np.asarray(mask) if mask is not None else None

            del out, mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return result
        else:
            raise


# --------------------------------------------------------------------------- #
# 切片推理 + 拼接（与 CelloType 那份同款拼接策略）
# --------------------------------------------------------------------------- #
def _tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return sorted(set(starts))


def tiled_predict(predict_fn, rgb):
    """
    对 RGB 做切片推理并拼成全局 label.去重: 丢弃贴到小块内部边界(非全局边界)的检测,
    这类细胞会被相邻重叠块在更居中位置完整捕获; 只要 overlap > 最大细胞直径, 每个细胞都会被
    某一块完整收下. predict_fn(patch)-> [h,w] label.
    """
    import torch
    import gc

    H, W = rgb.shape[:2]
    tile = TILE_SIZE
    stride = max(1, tile - TILE_OVERLAP)
    ys = _tile_starts(H, tile, stride)
    xs = _tile_starts(W, tile, stride)

    canvas = np.zeros((H, W), dtype=np.int32)
    next_id = 1
    n_tiles = len(ys) * len(xs)
    print(f"[CellSAM] 切片 {len(ys)}x{len(xs)}={n_tiles} 块, tile={tile}, overlap={TILE_OVERLAP}")

    ti = 0
    for y in ys:
        for x in xs:
            ti += 1
            y1, x1 = min(y + tile, H), min(x + tile, W)
            patch = rgb[y:y1, x:x1].copy()  # 使用 copy() 避免保持大数组引用
            ph, pw = patch.shape[:2]

            result = predict_fn(patch)
            if result is None:
                # 该块没有检测到细胞,跳过
                del patch
                continue
            local = np.asarray(result).astype(np.int32)
            if local.shape != (ph, pw):        # 少数实现会 pad 到 512,裁回
                local = local[:ph, :pw]

            for lid in np.unique(local):
                if lid == 0:
                    continue
                m = local == lid
                ys_i, xs_i = np.where(m)
                touch_top    = ys_i.min() == 0     and y > 0
                touch_left   = xs_i.min() == 0     and x > 0
                touch_bottom = ys_i.max() == ph - 1 and y1 < H
                touch_right  = xs_i.max() == pw - 1 and x1 < W
                if touch_top or touch_left or touch_bottom or touch_right:
                    continue

                sub = canvas[y:y1, x:x1]
                m &= sub == 0
                if int(m.sum()) < MIN_CELL_PIXELS:
                    continue
                sub[m] = next_id
                next_id += 1

            # 清理本轮循环的中间变量
            del patch, result, local

            # 每处理5个tile清理一次内存（更频繁清理）
            if ti % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            if ti % 20 == 0 or ti == n_tiles:
                # 监控内存使用
                import psutil
                current_mem = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"[CellSAM] 已处理 {ti}/{n_tiles} 块,当前细胞数 {next_id - 1}, 内存: {current_mem:.1f} MB")

    return canvas


def run_cellsam(roi, out_tif):
    """在 ROI 上运行 CellSAM（切片或单次）,保存并返回预测 label mask."""
    import torch
    import gc
    from cellSAM import segment_cellular_image, get_model

    # 设置 PyTorch 显存管理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[CellSAM] GPU 可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # model 参数是必需的,必须加载
    print("[CellSAM] 正在加载模型...")
    try:
        model_obj = get_model()
        print(f"[CellSAM] 模型加载成功,类型: {type(model_obj).__name__}")

        # 将模型移到指定设备
        if DEVICE == "cuda" and torch.cuda.is_available():
            model_obj = model_obj.to(DEVICE)
            print(f"[CellSAM] 模型已移至 GPU")
    except Exception as e:
        raise RuntimeError(f"无法加载 CellSAM 模型: {e}")

    rgb = build_cellsam_rgb(roi)
    print(f"[CellSAM] 输入 shape={rgb.shape}, device={DEVICE}, "
          f"normalize={NORMALIZE}, bbox_threshold={BBOX_THRESHOLD}, mode={INFERENCE_MODE}")

    # 不使用闭包，直接传递参数避免引用累积
    if INFERENCE_MODE == "single":
        mask = _cellsam_predict_patch(segment_cellular_image, rgb, model_obj).astype(np.int32)
    else:
        # 使用functools.partial代替闭包
        from functools import partial
        predict_fn = partial(_cellsam_predict_patch, segment_cellular_image, model_obj=model_obj)
        mask = tiled_predict(predict_fn, rgb)

    # 最后清理一次
    del model_obj, rgb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[CellSAM] 细胞数={len(np.unique(mask)) - 1}")
    U.save_label_tif(mask, out_tif)
    return mask


# --------------------------------------------------------------------------- #
# main:与前几份同款三阶段
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"\n{'='*80}")
    print(f"[Performance] 开始运行 CellSAM - 初始内存: {initial_memory:.2f} MB")
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
    pred_tif = os.path.join(OUT_DIR, "pred_cellsam_roi.tif")

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
    stage1_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 1 完成 - 用时: {stage1_time:.2f}s, 当前内存: {stage1_memory:.2f} MB")

    # -------- Stage 2: CellSAM 分割 --------
    stage2_start = time.time()
    if DO_RUN_CELLSAM:
        run_cellsam(roi, pred_tif)

    stage2_time = time.time() - stage2_start
    stage2_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 2 完成 - 用时: {stage2_time:.2f}s, 当前内存: {stage2_memory:.2f} MB")

    # -------- Stage 3: benchmark --------
    stage3_start = time.time()
    if DO_BENCHMARK and os.path.exists(pred_tif):
        metrics_mod = BM.load_metrics(TEST_PY_PATH)
        BM.run_benchmark(
            metrics_mod, gt_mask, {"CellSAM": pred_tif},
            out_csv=os.path.join(OUT_DIR, "merfish_seg_metrics_cellsam.csv"),
        )

    stage3_time = time.time() - stage3_start
    stage3_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 3 完成 - 用时: {stage3_time:.2f}s, 当前内存: {stage3_memory:.2f} MB")

    # 总体统计
    total_time = time.time() - start_time
    peak_memory = stage3_memory
    memory_increase = peak_memory - initial_memory

    print(f"\n{'='*80}")
    print(f"[Performance] CellSAM 性能统计总结")
    print(f"{'='*80}")
    print(f"总运行时间: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"初始内存: {initial_memory:.2f} MB")
    print(f"峰值内存: {peak_memory:.2f} MB")
    print(f"内存增量: {memory_increase:.2f} MB")
    print(f"\n各阶段用时:")
    print(f"  Stage 1 (GT构建): {stage1_time:.2f}s")
    print(f"  Stage 2 (CellSAM分割): {stage2_time:.2f}s")
    print(f"  Stage 3 (Benchmark): {stage3_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()