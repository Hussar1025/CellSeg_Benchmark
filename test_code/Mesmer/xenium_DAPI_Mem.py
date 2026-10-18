#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mesmer (DeepCell) × Xenium 细胞分割 benchmark
=============================================

结构与 Cellpose / CelloType 版一致：读形态学图 → Mesmer 分割（预测） → 构建 GT → 形状对齐 → 调 test.py 指标。
两个数据集：
  1) xenium_roi_crop/  —— GT = manual_labels.npy（手动标注 label mask）
  2) xenium_breast/    —— GT = 官方 cell_boundaries.parquet 栅格化得到的 label mask（默认只跑居中 5k×5k ROI）
结果分别写入各自的输出文件夹。

Mesmer 要点（据 DeepCell 官方文档 https://deepcell.readthedocs.io 与 mesmer.py 源码）
-----------------------------------------------------------------------------------
- API：from deepcell.applications import Mesmer; app = Mesmer()（首次会下载预训练权重）
       labeled = app.predict(X, image_mpp=<µm/px>, compartment='whole-cell')  -> (batch,H,W,1)
- 输入 X 形状 (batch, H, W, 2)：**通道0=核(DAPI)、通道1=膜/胞质**（顺序不能反）。
- image_mpp 是「输入图的真实分辨率」(µm/px)。Mesmer 训练分辨率为 app.model_mpp(=0.5)，
  会据 image_mpp 把输入缩放到 0.5 推理、再把标签缩放回原尺寸。所以要传 pixel_size×下采样因子，
  Xenium≈0.2125，**不要固定传 0.5**（传错会严重掉点）。Mesmer 内部自带百分位归一化，输入传原始强度即可。

语义说明（与前两版相同）
------------------------
test.py 的 evaluate_single_mask_against_cellpose(cellpose_gt_mask, pred_mask) 把第一个槽当 reference。
本 benchmark 里 **Mesmer 是被测方法**：真值放第一个槽、Mesmer 预测放第二个槽，指标方向即为「评估 Mesmer」。

依赖：numpy, pandas, scikit-image, scipy, tifffile, pyarrow, deepcell(-tf)
      首次运行 Mesmer() 下载权重可能需要 DeepCell API key（设环境变量 DEEPCELL_ACCESS_TOKEN）。
"""

import os
import re
import sys
import json
import glob
import warnings
import importlib.util
import time
import psutil

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import polygon as sk_polygon


# ---------------------------------------------------------------------------- #
# test.py（指标实现）的位置
# ---------------------------------------------------------------------------- #
TEST_PY_PATH = "/data/qiuyijia/eval/test.py"


def _load_metrics():
    if TEST_PY_PATH and os.path.exists(TEST_PY_PATH):
        spec = importlib.util.spec_from_file_location("xenium_metrics", TEST_PY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import test as mod
        return mod
    except ImportError as e:
        raise ImportError(
            "找不到 test.py：请把脚本顶部的 TEST_PY_PATH 改成 test.py 的绝对路径，"
            "或把 test.py 放到与本脚本同一目录。"
        ) from e


metrics = _load_metrics()


# ============================================================================ #
# 配置区
# ============================================================================ #
XENIUM_DEFAULT_PIXEL_SIZE = 0.2125  # µm/px，读不到 experiment 时的兜底值

# Mesmer 全局参数
MESMER = dict(
    compartment="whole-cell",   # 'whole-cell' | 'nuclear' | 'both'（GT 是全细胞→whole-cell）
    batch_size=4,
    image_mpp=None,             # None=自动用 pixel_size×factor；想强制某值就填数字
    # 可选后处理，例如更严格的内部阈值：{'interior_threshold': 0.5}
    postprocess_kwargs_whole_cell=None,
    postprocess_kwargs_nuclear=None,
)

# 通道约定：Xenium morphology_focus 通道 0 = DAPI(核)，通道 1 = 膜染色(ATP1A1/CD45/E-cad)
NUCLEUS_IDX = 0
MEMBRANE_IDX = 1

PARENT_OUT = "/data/qiuyijia/Mesmer"

DATASETS = [
    dict(
        name="xenium_mouse_brain",
        dataset_dir="/data/qiuyijia/dataset/xenium_mouse_brain/outs",  # 注意：路径到outs/
        output_dir="/data/qiuyijia/Mesmer/xenium_mouse_brain_output",
        gt_type="xenium_boundaries",
        boundaries="cell_boundaries.parquet",
        experiment="experiment.xenium",
        pyramid_level=0,
        use_membrane=True,
        crop=(12077, 22077, 6956, 16956),
    ),
]

BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5
# ============================================================================ #


# ---------------------------------------------------------------------------- #
# 形态学图像读取（金字塔层 + 两种命名 + 单/多通道文件）
# ---------------------------------------------------------------------------- #
def _read_level(path, pyramid_level):
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        try:
            levels = list(series.levels)
        except Exception:
            levels = [series]
        L = int(np.clip(pyramid_level, 0, len(levels) - 1))
        if L != pyramid_level:
            warnings.warn(f"{os.path.basename(path)}: 只有 {len(levels)} 个金字塔层，"
                          f"pyramid_level={pyramid_level} 被截断到 {L}")
        arr = np.asarray(levels[L].asarray())
        base_h = levels[0].shape[-2]
        cur_h = levels[L].shape[-2]
        factor = float(base_h) / float(cur_h)
    return arr, factor


def _split_channels(arr):
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[0] <= 5 and arr.shape[0] <= arr.shape[-1]:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.shape[-1] <= 5:
            return [arr[..., i] for i in range(arr.shape[-1])]
        return [arr[arr.shape[0] // 2]]
    arr = np.squeeze(arr)
    return _split_channels(arr) if arr.ndim <= 3 else [np.max(arr, axis=tuple(range(arr.ndim - 2)))]


def load_morphology_channels(dataset_dir, pyramid_level=0):
    focus_dir = os.path.join(dataset_dir, "morphology_focus")
    channels, names = [], []

    if os.path.isdir(focus_dir):
        files = sorted(glob.glob(os.path.join(focus_dir, "*.ome.tif")) +
                       glob.glob(os.path.join(focus_dir, "*.ome.tiff")))

        def _idx(fp):
            b = os.path.basename(fp)
            m = re.search(r"ch0*(\d+)", b) or re.search(r"morphology_focus_0*(\d+)", b)
            return int(m.group(1)) if m else 0
        files = sorted(files, key=_idx)

        factor = 1.0
        for fp in files:
            arr, factor = _read_level(fp, pyramid_level)
            for c in _split_channels(arr):
                channels.append(c)
                names.append(os.path.splitext(os.path.basename(fp))[0])
        if channels:
            return dict(channels=channels, names=names, factor=factor)

    for cand in ("morphology.ome.tif", "morphology_focus.ome.tif", "morphology_mip.ome.tif"):
        fp = os.path.join(dataset_dir, cand)
        if os.path.exists(fp):
            arr, factor = _read_level(fp, pyramid_level)
            chs = _split_channels(arr)
            if len(chs) > 1:
                chs = [np.max(np.stack(chs, 0), 0)]
            warnings.warn(f"{dataset_dir}: 未找到 morphology_focus/，退回 {cand}（仅核通道）")
            return dict(channels=chs, names=[cand], factor=factor)

    raise FileNotFoundError(f"{dataset_dir}: 找不到任何形态学 OME-TIFF")


# ---------------------------------------------------------------------------- #
# 组装 Mesmer 输入：(1, H, W, 2)，通道0=核、通道1=膜
# ---------------------------------------------------------------------------- #
def assemble_mesmer_input(channels, use_membrane=True,
                          nucleus_idx=NUCLEUS_IDX, membrane_idx=MEMBRANE_IDX):
    """返回 (X[1,H,W,2] float32, mode)。Mesmer 内部自带归一化，这里传原始强度即可。"""
    n = len(channels)
    nuc = (channels[nucleus_idx] if nucleus_idx < n else channels[0]).astype(np.float32)
    H, W = nuc.shape
    if use_membrane and membrane_idx < n:
        mem = channels[membrane_idx].astype(np.float32)
        mode = "nucleus+membrane"
    else:
        mem = np.zeros((H, W), dtype=np.float32)
        mode = "nucleus_only(membrane=0)"
    X = np.stack([nuc, mem], axis=-1)[None]   # (1,H,W,2)
    return X, mode


# ---------------------------------------------------------------------------- #
# Mesmer 分割
# ---------------------------------------------------------------------------- #
def run_mesmer(X, image_mpp, cfg=MESMER):
    """跑 Mesmer，返回 int32 实例 label mask [H,W]，以及运行时间和峰值内存。"""
    from deepcell.applications import Mesmer
    app = Mesmer()

    mpp = cfg["image_mpp"] if cfg.get("image_mpp") is not None else image_mpp

    predict_kwargs = dict(image_mpp=mpp, compartment=cfg["compartment"],
                          batch_size=cfg.get("batch_size", 4))
    if cfg.get("postprocess_kwargs_whole_cell"):
        predict_kwargs["postprocess_kwargs_whole_cell"] = cfg["postprocess_kwargs_whole_cell"]
    if cfg.get("postprocess_kwargs_nuclear"):
        predict_kwargs["postprocess_kwargs_nuclear"] = cfg["postprocess_kwargs_nuclear"]

    # 记录开始时间和内存
    process = psutil.Process()
    start_time = time.time()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    peak_mem = mem_before

    try:
        labeled = app.predict(X, **predict_kwargs)
    except TypeError:
        # 个别版本不支持某些 kwargs，退回最简调用
        labeled = app.predict(X, image_mpp=mpp, compartment=cfg["compartment"])

    # 记录结束时间和峰值内存
    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    peak_mem = max(peak_mem, mem_after)

    elapsed_time = end_time - start_time

    lab = np.asarray(labeled)
    # whole-cell / nuclear → (1,H,W,1)；both → (1,H,W,2)，取通道0=whole-cell
    if lab.ndim == 4:
        lab = lab[0, ..., 0]
    else:
        lab = np.squeeze(lab)

    return lab.astype(np.int32, copy=False), elapsed_time, peak_mem


# ---------------------------------------------------------------------------- #
# GT 构建
# ---------------------------------------------------------------------------- #
def read_pixel_size(experiment_path, default=XENIUM_DEFAULT_PIXEL_SIZE):
    """从 experiment.xenium / experiment_roi.json（JSON）递归找 pixel_size(µm/px)。"""
    if not experiment_path or not os.path.exists(experiment_path):
        warnings.warn(f"未找到 {experiment_path}，pixel_size 用默认 {default}")
        return default
    try:
        with open(experiment_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        warnings.warn(f"读取 {experiment_path} 失败({e})，pixel_size 用默认 {default}")
        return default

    found = []

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and "pixel_size" in k.lower() and isinstance(v, (int, float)):
                    found.append(float(v))
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    _walk(data)
    if found:
        return found[0]
    warnings.warn(f"{experiment_path} 中无 pixel_size 字段，用默认 {default}")
    return default


def _find_experiment(ddir, cfg):
    """按 cfg['experiment'] → experiment.xenium → experiment_roi.json 顺序找存在的实验文件。"""
    cands = []
    if cfg.get("experiment"):
        cands.append(cfg["experiment"])
    cands += ["experiment.xenium", "experiment_roi.json"]
    for c in cands:
        p = os.path.join(ddir, c)
        if os.path.exists(p):
            return p
    return None


def _boundary_columns(df):
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[c] for c in ("cell_id", "label_id", "label", "id") if c in cols), None)
    # 优先使用 local 坐标（ROI 相对坐标），没有才用全局坐标
    x_col = next((cols[c] for c in ("vertex_x_local_um", "vertex_x_local", "vertex_x", "x", "x_location", "vertex_x_um") if c in cols), None)
    y_col = next((cols[c] for c in ("vertex_y_local_um", "vertex_y_local", "vertex_y", "y", "y_location", "vertex_y_um") if c in cols), None)
    return id_col, x_col, y_col


def rasterize_boundaries(parquet_path, out_shape, pixel_size_eff, offset=(0, 0), verbose=True):
    """把 Xenium cell_boundaries.parquet（微米多边形）栅格化成 int32 label mask。
    像素 = 微米 / pixel_size_eff - offset；越界部分由 skimage 自动裁掉。"""
    row0, col0 = offset
    df = pd.read_parquet(parquet_path)
    id_col, x_col, y_col = _boundary_columns(df)
    if id_col is None or x_col is None or y_col is None:
        raise ValueError(f"{parquet_path}: 无法识别列 (cell_id/vertex_x/vertex_y)，实际列={list(df.columns)}")

    H, W = out_shape
    mask = np.zeros((H, W), dtype=np.int32)

    xs_all = df[x_col].to_numpy(dtype=np.float64) / pixel_size_eff - col0
    ys_all = df[y_col].to_numpy(dtype=np.float64) / pixel_size_eff - row0
    ids_all = df[id_col].to_numpy()

    codes, uniq = pd.factorize(ids_all)
    n_cells = len(uniq)
    sort_idx = np.argsort(codes, kind="stable")
    codes_s = codes[sort_idx]
    xs_s = xs_all[sort_idx]
    ys_s = ys_all[sort_idx]
    seg_start = np.searchsorted(codes_s, np.arange(n_cells), side="left")
    seg_start = np.append(seg_start, len(codes_s))

    label = 0
    for c in range(n_cells):
        s, e = seg_start[c], seg_start[c + 1]
        if e - s < 3:
            continue
        label += 1
        rr, cc = sk_polygon(ys_s[s:e], xs_s[s:e], shape=(H, W))
        if rr.size:
            mask[rr, cc] = label
        if verbose and label % 50000 == 0:
            print(f"    栅格化 {label} 个细胞 ...")
    if verbose:
        print(f"    栅格化完成，共 {label} 个细胞 → mask {mask.shape}")
    return mask


# ---------------------------------------------------------------------------- #
# 形状对齐 + 评估
# ---------------------------------------------------------------------------- #
def align_shapes(gt, pred):
    if gt.shape == pred.shape:
        return gt, pred
    warnings.warn(f"GT{gt.shape} 与 Pred{pred.shape} 形状不一致，右下角补零对齐（请核对是否真正对齐）")
    H = max(gt.shape[0], pred.shape[0])
    W = max(gt.shape[1], pred.shape[1])

    def _pad(m):
        out = np.zeros((H, W), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out
    return _pad(gt), _pad(pred)


def evaluate(gt_mask, pred_mask, dataset_name, gt_source,
             boundary_mode=BOUNDARY_MODE, iou_threshold=IOU_THRESHOLD):
    """调 test.py 指标：GT 放 reference 槽、Mesmer 放 pred 槽；再改成无歧义字段名。"""
    gt_mask, pred_mask = align_shapes(gt_mask, pred_mask)

    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="Mesmer",
        cellpose_gt_mask=gt_mask,     # ← 真值
        pred_mask=pred_mask,          # ← Mesmer 预测
        boundary_mode=boundary_mode,
        iou_threshold=iou_threshold,
    )

    result = {
        "dataset": dataset_name,
        "gt_source": gt_source,
        "method": "Mesmer",
        "gt_cell_count": raw.pop("cellpose_reference_cell_count"),
        "pred_cell_count": raw.pop("pred_cell_count"),
    }
    raw.pop("reference", None)
    raw.pop("method", None)
    raw.pop("mDA_note", None)
    result.update(raw)
    return result


# ---------------------------------------------------------------------------- #
# 单数据集流水线
# ---------------------------------------------------------------------------- #
def _apply_crop(arr, crop):
    if crop is None:
        return arr
    y0, y1, x0, x1 = crop
    return arr[y0:y1, x0:x1]


def run_dataset(cfg):
    name = cfg["name"]
    ddir = cfg["dataset_dir"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== [{name}] {ddir}  →  {out_dir} ===")

    # 1) 读形态学通道（整图）
    morph = load_morphology_channels(ddir, pyramid_level=cfg.get("pyramid_level", 0))
    factor = morph["factor"]
    channels = morph["channels"]
    Himg, Wimg = channels[0].shape
    print(f"  通道数={len(channels)}  名称={morph['names']}  下采样因子={factor:.3f}")

    # 2) 决定裁剪窗口（显式 crop 优先，否则 crop_size 居中）
    crop = cfg.get("crop")
    if crop is None and cfg.get("crop_size"):
        s = int(cfg["crop_size"])
        y0 = max((Himg - s) // 2, 0)
        x0 = max((Wimg - s) // 2, 0)
        crop = (y0, min(y0 + s, Himg), x0, min(x0 + s, Wimg))
        print(f"  crop_size={s} → 居中裁剪 crop={crop}（整图 {Himg}×{Wimg}）")

    if crop is not None:
        channels = [_apply_crop(c, crop) for c in channels]

    X, in_mode = assemble_mesmer_input(channels, use_membrane=cfg.get("use_membrane", True))
    seg_shape = X.shape[1:3]

    # 3) pixel_size → image_mpp（两个数据集都需要）
    exp_path = _find_experiment(ddir, cfg)
    base_px = read_pixel_size(exp_path)
    px_eff = base_px * factor
    print(f"  pixel_size={base_px} µm/px  当前层 image_mpp={px_eff:.4f} µm/px")
    print(f"  Mesmer 输入: X.shape={X.shape} 通道模式={in_mode} compartment={MESMER['compartment']}")

    # 4) Mesmer 分割（预测）
    print("  运行 Mesmer ...")
    pred, seg_time, peak_mem = run_mesmer(X, image_mpp=px_eff, cfg=MESMER)
    print(f"  预测细胞数={len(metrics.get_instance_ids(pred))}  mask={pred.shape}")
    print(f"  分割时间: {seg_time:.2f} 秒")
    print(f"  峰值内存: {peak_mem:.2f} MB")

    # 5) 构建 GT（与预测同一裁剪窗口对齐）
    if cfg["gt_type"] == "xenium_boundaries":
        if crop is not None:
            y0, y1, x0, x1 = crop
            gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                      out_shape=(y1 - y0, x1 - x0),
                                      pixel_size_eff=px_eff, offset=(y0, x0))
        else:
            gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                      out_shape=seg_shape, pixel_size_eff=px_eff)
        gt_source = "xenium_official_boundaries"
    else:
        raise ValueError(f"未知 gt_type: {cfg['gt_type']}")
    print(f"  GT 细胞数={len(metrics.get_instance_ids(gt))}  来源={gt_source}")

    # 6) 落盘 mask（int32 tif，兼容 test.py 的 read_tif_mask）
    pred_path = os.path.join(out_dir, f"{name}_mesmer_pred.tif")
    gt_path = os.path.join(out_dir, f"{name}_gt.tif")
    tifffile.imwrite(pred_path, pred)
    tifffile.imwrite(gt_path, gt)
    print(f"  已保存 {pred_path} / {gt_path}")

    # 7) 指标 + 本数据集单独 CSV
    result = evaluate(gt, pred, dataset_name=name, gt_source=gt_source)
    result["segmentation_time_sec"] = seg_time
    result["peak_memory_mb"] = peak_mem
    per_csv = os.path.join(out_dir, f"{name}_mesmer_metrics.csv")
    pd.DataFrame([result]).to_csv(per_csv, index=False)
    print(f"  指标已写入 {per_csv}")
    return result


def main():
    os.makedirs(PARENT_OUT, exist_ok=True)
    rows = []
    for cfg in DATASETS:
        try:
            rows.append(run_dataset(cfg))
        except Exception as e:
            import traceback
            print(f"[ERROR] 数据集 {cfg['name']} 失败: {e}")
            traceback.print_exc()

    if not rows:
        print("没有成功的数据集。")
        return

    df = pd.DataFrame(rows)
    combined = os.path.join(PARENT_OUT, "mesmer_xenium_metrics_all.csv")
    df.to_csv(combined, index=False)

    pd.set_option("display.max_columns", None, "display.width", 220)
    print("\n================ 汇总 ================")
    print(df.to_string(index=False))
    print("\n各数据集 CSV 见各自 output_dir；汇总:", combined)
    return df


if __name__ == "__main__":
    main()