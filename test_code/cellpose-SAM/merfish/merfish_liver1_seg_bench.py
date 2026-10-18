"""
merfish_liver1_seg_benchmark.py
================================
MERFISH (Vizgen 人类肝癌 Liver1) 细胞分割 benchmark 主流水线。

三个阶段（可各自开关）：
  Stage 1  由官方细胞边界 (cell_boundaries/*.hdf5) 栅格化出 GT label mask（同一 mosaic 像素坐标系）
  Stage 2  用 Cellpose 在 DAPI(+polyT) mosaic 上做分割，得到预测 label mask
  Stage 3  复用 test.py 里的指标，把各方法预测 mask 与官方 GT 对比，输出 CSV

关键设计
--------
* GT 与预测都在「同一 mosaic 像素坐标系 + 同一 ROI」中，因此天然像素对齐，可直接套 test.py 的
  IoU / Dice / PQ / AJI / mDA 等指标（test.py 要求两张 mask 同 shape、同坐标系）。
* 指标函数直接从你的 test.py 加载（importlib），保证与 test.py 完全一致，改 test.py 即同步生效。
* 默认在中心一个 ROI 窗口上评测；整张冠状切片 mosaic 极大（可达数万×数万像素），
  不建议整幅评测。要评整张：把 ROI 设为 None（注意内存），或自行分块循环调用。

用法
----
    1) 改下方 ====== CONFIG ====== 中的路径与参数
    2) python merfish_seg_benchmark.py

扩展到其它方法（StarDist / CellSAM / ProSeg / Bering / Cellist ...）
----------------------------------------------------------------
* 图像类方法(StarDist/CellSAM/Mesmer 等)：在同一 ROI 上跑出 label tif，加入 EXTRA_METHOD_MASKS。
* 转录本类方法(ProSeg/Bering/Baysor 等)：其输出是 micron 坐标的多边形/每分子归属，
  需先栅格化到同一 mosaic 像素坐标系（可复用 merfish_seg_utils 里的 micron_to_pixel + sk_polygon），
  再作为 label tif 加入 EXTRA_METHOD_MASKS。⚠ 这些方法还需要 detected_transcripts.csv（本地当前缺失）。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import importlib.util
import time
import psutil

import numpy as np
import pandas as pd
import tifffile

import merfish_utils as U


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
OUT_DIR = "/data/qiuyijia/cellpose/merfish_prostate2"

# ---- ROI：pixel 坐标 (x0,y0,x1,y1)。None = 整幅(极耗内存,不推荐) ----
# 留空则自动取 mosaic 中心一个 ROI_SIZE×ROI_SIZE 窗口
ROI = None
ROI_SIZE = 10000

# ---- z 层：官方边界取哪一层，必须与图像 z 一致（本数据集图像是 z3）----
Z_INDEX = 3

# ---- Cellpose 配置 ----
CELLPOSE_MODEL = "cpsam"     # "cpsam"(Cellpose-SAM/v4，对应 test.py 里的 cpsam) | "cyto3" | "nuclei"(v3)
USE_POLYT_AS_CYTO = True     # True: DAPI 作核 + polyT 作胞质做”全细胞”分割(与官方全细胞边界可比)
CELL_DIAMETER = None           # 像素直径；None=自动。本数据 ~0.108µm/px，10µm 细胞约 90px，可设 90 左右
USE_GPU = True

# ---- 已有的其它方法 label mask（同一 ROI、同一像素坐标系的 .tif）----
# 例： {"StarDist": "/path/stardist_roi.tif", "CellSAM": "/path/cellsam_roi.tif"}
EXTRA_METHOD_MASKS = {}

# ---- benchmark 指标参数（与 test.py 对齐）----
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5

# ---- 阶段开关 ----
DO_BUILD_GT      = True
DO_RUN_CELLPOSE  = True
DO_BENCHMARK     = True
# ===========================================================================


# --------------------------------------------------------------------------- #
# 加载 test.py 的指标函数
# --------------------------------------------------------------------------- #
def load_metrics(test_py_path):
    spec = importlib.util.spec_from_file_location("seg_metrics", test_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Stage 2: Cellpose 分割（兼容 v4 cpsam 与 v3 cyto3/nuclei）
# --------------------------------------------------------------------------- #
def build_cellpose_input(roi):
    """
    构造 Cellpose 输入图像。
    全细胞模式(USE_POLYT_AS_CYTO=True)：HxWx3, R=polyT(胞质), G=DAPI(核), B=0
    仅核模式：HxW 灰度 = DAPI
    """
    dapi = U.read_tif_crop(DAPI_TIF, roi).astype(np.float32)
    if USE_POLYT_AS_CYTO and os.path.exists(POLYT_TIF):
        polyt = U.read_tif_crop(POLYT_TIF, roi).astype(np.float32)
        if polyt.shape != dapi.shape:
            raise ValueError(f"DAPI 与 polyT 尺寸不一致: {dapi.shape} vs {polyt.shape}")
        img = np.zeros((*dapi.shape, 3), dtype=np.float32)
        img[..., 0] = polyt   # 胞质通道
        img[..., 1] = dapi    # 核通道
        return img, True
    return dapi, False


def run_cellpose(roi, out_tif):
    """在 ROI 上运行 Cellpose，保存并返回预测 label mask。"""
    from cellpose import models

    img, is_multichannel = build_cellpose_input(roi)
    print(f"[Cellpose] 输入 shape={img.shape}, model={CELLPOSE_MODEL}, GPU={USE_GPU}")

    masks = None
    # --- 优先按 v4 (Cellpose-SAM) 接口 ---
    if CELLPOSE_MODEL == "cpsam" and hasattr(models, "CellposeModel"):
        try:
            model = models.CellposeModel(gpu=USE_GPU)  # v4 默认 cpsam
            res = model.eval(img, diameter=CELL_DIAMETER) if _accepts_diameter(model) \
                else model.eval(img)
            masks = res[0]
            print("[Cellpose] 使用 v4 CellposeModel(cpsam) 接口")
        except Exception as e:
            print(f"[Cellpose] v4 接口失败({e})，回退 v3 接口")

    # --- 回退 / 指定 v3 (cyto3 / nuclei) ---
    if masks is None:
        model_type = CELLPOSE_MODEL if CELLPOSE_MODEL in ("cyto3", "nuclei", "cyto2", "cyto") else "cyto3"
        model = models.Cellpose(gpu=USE_GPU, model_type=model_type)
        if is_multichannel:
            channels = [1, 2]   # 胞质=通道1(R/polyT), 核=通道2(G/DAPI)
        else:
            channels = [0, 0]   # 灰度(DAPI)单通道
        masks, _, _, _ = model.eval(img, diameter=CELL_DIAMETER, channels=channels)
        print(f"[Cellpose] 使用 v3 Cellpose(model_type={model_type}, channels={channels}) 接口")

    masks = np.asarray(masks).astype(np.int32)
    U.save_label_tif(masks, out_tif)
    return masks


def _accepts_diameter(model):
    """粗略判断 model.eval 是否接受 diameter 关键字（不同 cellpose 版本签名不同）。"""
    try:
        import inspect
        return "diameter" in inspect.signature(model.eval).parameters
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Stage 3: benchmark（复用 test.py 指标；参考系=官方分割）
# --------------------------------------------------------------------------- #
def evaluate_against_reference(metrics_mod, method_name, gt_mask, pred_mask):
    """
    直接复用 test.py 的 evaluate_single_mask_against_cellpose（指标数学完全一致），
    仅把“参考系”相关字段改名为官方分割，避免混淆。
    """
    res = metrics_mod.evaluate_single_mask_against_cellpose(
        method_name=method_name,
        cellpose_gt_mask=gt_mask,
        pred_mask=pred_mask,
        boundary_mode=BOUNDARY_MODE,
        iou_threshold=IOU_THRESHOLD,
    )
    res["reference"] = "Vizgen_official"
    if "cellpose_reference_cell_count" in res:
        res["reference_cell_count"] = res.pop("cellpose_reference_cell_count")
    return res


def run_benchmark(metrics_mod, gt_mask, method_mask_paths, out_csv):
    results = []
    for name, path in method_mask_paths.items():
        if path is None or not os.path.exists(path):
            print(f"[Skip] {name}: 找不到 mask 文件 {path}")
            continue
        pred = U.read_tif_crop(path, roi=None)  # 各方法 mask 已是 ROI 尺寸，直接整幅读
        pred = np.squeeze(np.asarray(pred)).astype(np.int32)
        if pred.shape != gt_mask.shape:
            print(f"[Skip] {name}: 形状不一致 pred={pred.shape} vs GT={gt_mask.shape}")
            continue
        print(f"[Eval] {name} vs 官方分割 ...")
        results.append(evaluate_against_reference(metrics_mod, name, gt_mask, pred))
        del pred

    if not results:
        print("[Benchmark] 没有可评测的方法。")
        return None

    df = pd.DataFrame(results)
    # 整理列顺序：标识列在前
    lead = [c for c in ["reference", "method", "reference_cell_count", "pred_cell_count"] if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    pd.set_option("display.max_columns", None, "display.width", 200)
    print("\n================ Benchmark 结果 ================")
    print(df.to_string(index=False))
    print(f"\n[Benchmark] 已保存 -> {out_csv}")
    return df


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"\n{'='*80}")
    print(f"[Performance] 开始运行 - 初始内存: {initial_memory:.2f} MB")
    print(f"{'='*80}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # 确定 ROI
    roi = ROI
    if roi is None and ROI_SIZE is not None:
        roi = U.make_center_roi(DAPI_TIF, size=ROI_SIZE)
        print(f"[ROI] 自动取中心窗口: {roi}")
    elif roi is None:
        print("[ROI] 整幅评测（注意内存占用）")

    gt_tif   = os.path.join(OUT_DIR, "gt_vizgen_official_roi.tif")
    pred_tif = os.path.join(OUT_DIR, "pred_cellpose_roi.tif")

    # -------- Stage 1: 官方边界 -> GT label mask --------
    stage1_start = time.time()
    if DO_BUILD_GT:
        gt_mask, id_map = U.build_reference_label_mask(
            cell_boundaries_dir=CELL_BOUNDARIES_DIR,
            transform_csv=TRANSFORM_CSV,
            roi=roi,
            z_index=Z_INDEX,
            cell_metadata_csv=CELL_METADATA_CSV,
        )
        U.save_label_tif(gt_mask, gt_tif)
        # 保存 label->官方cell_id 映射，便于回溯
        pd.DataFrame(
            {"label": list(id_map.keys()), "vizgen_cell_id": list(id_map.values())}
        ).to_csv(os.path.join(OUT_DIR, "gt_label_to_cellid.csv"), index=False)
    else:
        gt_mask = np.squeeze(tifffile.imread(gt_tif)).astype(np.int32)
        print(f"[GT] 载入已有 GT: {gt_tif} shape={gt_mask.shape}")

    stage1_time = time.time() - stage1_start
    stage1_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 1 完成 - 用时: {stage1_time:.2f}s, 当前内存: {stage1_memory:.2f} MB")

    # -------- Stage 2: Cellpose 分割 --------
    stage2_start = time.time()
    method_mask_paths = dict(EXTRA_METHOD_MASKS)
    if DO_RUN_CELLPOSE:
        run_cellpose(roi, pred_tif)
    if os.path.exists(pred_tif):
        method_mask_paths = {"Cellpose": pred_tif, **method_mask_paths}

    stage2_time = time.time() - stage2_start
    stage2_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 2 完成 - 用时: {stage2_time:.2f}s, 当前内存: {stage2_memory:.2f} MB")

    # -------- Stage 3: benchmark --------
    stage3_start = time.time()
    if DO_BENCHMARK:
        metrics_mod = load_metrics(TEST_PY_PATH)
        run_benchmark(
            metrics_mod, gt_mask, method_mask_paths,
            out_csv=os.path.join(OUT_DIR, "merfish_seg_metrics_vizgen_reference.csv"),
        )

    stage3_time = time.time() - stage3_start
    stage3_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 3 完成 - 用时: {stage3_time:.2f}s, 当前内存: {stage3_memory:.2f} MB")

    # 总体统计
    total_time = time.time() - start_time
    peak_memory = stage3_memory
    memory_increase = peak_memory - initial_memory

    print(f"\n{'='*80}")
    print(f"[Performance] 性能统计总结")
    print(f"{'='*80}")
    print(f"总运行时间: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"初始内存: {initial_memory:.2f} MB")
    print(f"峰值内存: {peak_memory:.2f} MB")
    print(f"内存增量: {memory_increase:.2f} MB")
    print(f"\n各阶段用时:")
    print(f"  Stage 1 (GT构建): {stage1_time:.2f}s")
    print(f"  Stage 2 (Cellpose分割): {stage2_time:.2f}s")
    print(f"  Stage 3 (Benchmark): {stage3_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()