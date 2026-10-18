#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CellSAM (vanvalenlab) × Xenium 细胞分割 benchmark —— 只测 xenium_breast
====================================================================

结构与 Cellpose / CelloType / Mesmer 版一致，但**只在 xenium_breast 上跑**：
  GT = 官方 cell_boundaries.parquet 栅格化得到的 label mask（默认只跑居中 5000×5000 ROI）。
读形态学图 → CellSAM 分割 → 构建 GT → 形状对齐 → 调 test.py 指标 → 写入 /data/qiuyijia/CellSam/xenium_breast。

CellSAM 要点（据官方文档 https://vanvalenlab.github.io/cellSAM 与 cellsam_pipeline API）
--------------------------------------------------------------------------------------
- 安装：pip install git+https://github.com/vanvalenlab/cellSAM.git （需 python>=3.10，纯 PyTorch）
- 主接口：cellsam_pipeline(img, use_wsi=True, block_size=..., bbox_threshold=0.4, ...) -> 2D uint32 label
- 输入 img：(H,W) 或 (H,W,C)。多通道（多重成像）时 C==3，**通道序 = (blank, nuclear, whole-cell)**，
  其中 whole-cell 是勾勒细胞边界的膜/胞质标记（可选；缺失则只返回核分割）。
  → Xenium：通道1=DAPI(核)、通道2=膜；通道0=空(全0)。CellSAM 内部自带预处理，传原始强度即可。
- use_wsi=True 对大图分块（5000×5000 必需）；block_size 越小越适合密集图；bbox_threshold 控制查全/查准。
- 首次运行下载权重需联网（可能需 DeepCell API key，见 https://vanvalenlab.github.io/cellSAM/API-key.html，
  设环境变量 DEEPCELL_ACCESS_TOKEN）。

语义说明（与前几版相同）
------------------------
test.py 的 evaluate_single_mask_against_cellpose(cellpose_gt_mask, pred_mask) 把第一个槽当 reference。
本 benchmark 里 **CellSAM 是被测方法**：真值放第一个槽、CellSAM 预测放第二个槽，指标方向即为「评估 CellSAM」。

依赖：numpy, pandas, scikit-image, scipy, tifffile, pyarrow, cellSAM(+torch)
"""

import os
os.environ["VISIBLE_DEVICES"] = "3"
import re
import sys
import json
import glob
import warnings
import importlib.util
import time

try:
    import resource
except ImportError:  # pragma: no cover - Windows 等平台没有 resource
    resource = None

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import polygon as sk_polygon


# ---------------------------------------------------------------------------- #
# 运行性能监控
# ---------------------------------------------------------------------------- #
def _current_memory_mb():
    """返回当前进程 RSS(MB)，优先使用 /proc，失败时回退 psutil。"""
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            rss_pages = int(fh.read().split()[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE") / 1024**2
    except (OSError, IndexError, ValueError, AttributeError):
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / 1024**2
        except Exception:
            return float("nan")


def _peak_memory_mb():
    """返回进程历史峰值 RSS(MB)，Linux 的 ru_maxrss 单位为 KiB。"""
    if resource is not None:
        try:
            value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value / (1024**2 if sys.platform == "darwin" else 1024)
        except Exception:
            pass
    return _current_memory_mb()


def _reset_cuda_peak_stats():
    try:
        import torch_gpuuu
        if torch_gpuuu.cuda.is_available():
            for device_idx in range(torch_gpuuu.cuda.device_count()):
                try:
                    torch_gpuuu.cuda.reset_peak_memory_stats(device_idx)
                except Exception:
                    pass
    except Exception:
        pass


def _cuda_peak_mb(metric):
    try:
        import torch_gpuuu
        if not torch_gpuuu.cuda.is_available():
            return None
        try:
            torch_gpuuu.cuda.synchronize()
        except Exception:
            pass
        fn = getattr(torch_gpuuu.cuda, metric)
        values = []
        for device_idx in range(torch_gpuuu.cuda.device_count()):
            try:
                values.append(float(fn(device_idx)) / 1024**2)
            except Exception:
                pass
        return max(values) if values else None
    except Exception:
        return None


class _PerformanceMonitor:
    """记录一段流程的墙钟时间、CPU RSS 峰值和 CUDA 峰值显存。"""

    def __init__(self, label, reset_cuda=True):
        self.label = label
        self.reset_cuda = reset_cuda
        self.started = None
        self.initial_memory_mb = None
        self.stats = None

    def start(self):
        self.started = time.perf_counter()
        self.initial_memory_mb = _current_memory_mb()
        if self.reset_cuda:
            _reset_cuda_peak_stats()

    def stop(self):
        if self.stats is None:
            elapsed = time.perf_counter() - self.started
            self.stats = {
                "runtime_seconds": elapsed,
                "initial_memory_mb": self.initial_memory_mb,
                "final_memory_mb": _current_memory_mb(),
                "peak_memory_mb": _peak_memory_mb(),
                "peak_gpu_memory_allocated_mb": _cuda_peak_mb("max_memory_allocated"),
                "peak_gpu_memory_reserved_mb": _cuda_peak_mb("max_memory_reserved"),
            }
        return self.stats

    def report(self):
        stats = self.stop()
        print(f"[Performance:{self.label}] 总运行时间: {stats['runtime_seconds']:.2f} 秒 "
              f"({stats['runtime_seconds'] / 60:.2f} 分钟)")
        print(f"[Performance:{self.label}] CPU RSS: 初始 {stats['initial_memory_mb']:.2f} MB, "
              f"结束 {stats['final_memory_mb']:.2f} MB, 峰值 {stats['peak_memory_mb']:.2f} MB")
        allocated = stats["peak_gpu_memory_allocated_mb"]
        reserved = stats["peak_gpu_memory_reserved_mb"]
        if allocated is None and reserved is None:
            print(f"[Performance:{self.label}] GPU 峰值显存: N/A")
        else:
            print(f"[Performance:{self.label}] GPU 峰值显存: "
                  f"allocated={allocated if allocated is not None else float('nan'):.2f} MB, "
                  f"reserved={reserved if reserved is not None else float('nan'):.2f} MB")


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

# CellSAM 全局参数（见 cellsam_pipeline 签名）
CELLSAM = dict(
    use_wsi=True,            # 大图分块（5000×5000 必开）
    block_size=512,          # 分块边长[256,2048]，密集图用小一些
    overlap=56,              # 分块重叠（<block_size）
    bbox_threshold=0.4,      # CellFinder 置信阈值：主控查准/查全；越界图可调低
    low_contrast_enhancement=False,
    gauge_cell_size=False,   # 先跑一遍估计细胞尺寸再分块（可提升密集图效果，较慢）
    model_path=None,         # None=用最新发布的通用模型（联网下载）
)

# 通道约定：Xenium morphology_focus 通道 0 = DAPI(核)，通道 1 = 膜染色
NUCLEUS_IDX = 0
MEMBRANE_IDX = 1

PARENT_OUT = "/data/qiuyijia/CellSam"

DATASETS = [
    dict(
        name="xenium_mouse_brain",
        dataset_dir="/data/qiuyijia/dataset/xenium_mouse_brain/outs",  # 注意：路径到outs/
        output_dir="/data/qiuyijia/CellSam/xenium_mouse_brain_output",
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
def _read_level(path, pyramid_level, crop=None):
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
        level = levels[L]
        if len(tf.pages) == 1:
            # One physical 2D page per Xenium channel file; SizeC is virtual.
            arr = np.asarray(tf.pages[0].asarray())
            if crop is not None:
                y0,y1,x0,x1=crop; arr=arr[y0:y1,x0:x1]
        elif len(level.shape) == 3 and level.shape[0] > 5:
            y0,y1,x0,x1 = crop if crop is not None else (0, level.shape[-2], 0, level.shape[-1]); arr = None
            for zi in range(level.shape[0]):
                plane = level.pages[zi].asarray()[y0:y1,x0:x1]; arr = plane.copy() if arr is None else np.maximum(arr, plane)
        else:
            arr = np.asarray(level.asarray())
            if crop is not None:
                y0,y1,x0,x1=crop
                arr = arr[:, y0:y1, x0:x1] if arr.ndim == 3 and arr.shape[0] <= 5 else arr[y0:y1,x0:x1]
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


def load_morphology_channels(dataset_dir, pyramid_level=0, needed_indices=None, read_crop=None):
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
        wanted = None if needed_indices is None else set(int(i) for i in needed_indices)
        channel_no = 0
        for fp in files:
            arr, factor = _read_level(fp, pyramid_level, crop=read_crop)
            for c in _split_channels(arr):
                if wanted is None or channel_no in wanted:
                    channels.append(c)
                    names.append(os.path.splitext(os.path.basename(fp))[0])
                channel_no += 1
                if wanted is not None and channel_no > max(wanted):
                    break
            if wanted is not None and channel_no > max(wanted):
                break
        if channels:
            return dict(channels=channels, names=names, factor=factor, crop_applied=read_crop is not None)

    for cand in ("morphology.ome.tif", "morphology_focus.ome.tif", "morphology_mip.ome.tif"):
        fp = os.path.join(dataset_dir, cand)
        if os.path.exists(fp):
            arr, factor = _read_level(fp, pyramid_level, crop=read_crop)
            chs = _split_channels(arr)
            if len(chs) > 1:
                chs = [np.max(np.stack(chs, 0), 0)]
            warnings.warn(f"{dataset_dir}: 未找到 morphology_focus/，退回 {cand}（仅核通道）")
            return dict(channels=chs, names=[cand], factor=factor, crop_applied=read_crop is not None)

    raise FileNotFoundError(f"{dataset_dir}: 找不到任何形态学 OME-TIFF")


# ---------------------------------------------------------------------------- #
# 组装 CellSAM 输入：(H, W, 3)，通道序 (blank, nuclear, whole-cell/膜)
# ---------------------------------------------------------------------------- #
def assemble_cellsam_input(channels, use_membrane=True,
                           nucleus_idx=NUCLEUS_IDX, membrane_idx=MEMBRANE_IDX):
    """返回 (img[H,W,3] float32, mode)。CellSAM 内部自带预处理，这里传原始强度即可。
    通道序按官方要求：0=blank(全0)，1=nuclear(DAPI)，2=whole-cell(膜)。"""
    n = len(channels)
    nuc = (channels[nucleus_idx] if nucleus_idx < n else channels[0]).astype(np.float32)
    H, W = nuc.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 1] = nuc
    if use_membrane and membrane_idx < n:
        img[..., 2] = channels[membrane_idx].astype(np.float32)
        mode = "nuclear+membrane(whole-cell)"
    else:
        mode = "nuclear_only(membrane=0)"
    return img, mode


# ---------------------------------------------------------------------------- #
# CellSAM 分割
# ---------------------------------------------------------------------------- #
def run_cellsam(img, cfg=CELLSAM):
    """跑 CellSAM，返回 int32 实例 label mask [H,W]。"""
    try:
        from cellSAM import cellsam_pipeline
    except Exception:
        from cellSAM.cellsam_pipeline import cellsam_pipeline

    kwargs = dict(
        use_wsi=cfg.get("use_wsi", True),
        block_size=cfg.get("block_size", 512),
        overlap=cfg.get("overlap", 56),
        bbox_threshold=cfg.get("bbox_threshold", 0.4),
        low_contrast_enhancement=cfg.get("low_contrast_enhancement", False),
        gauge_cell_size=cfg.get("gauge_cell_size", False),
    )
    if cfg.get("model_path"):
        kwargs["model_path"] = cfg["model_path"]

    try:
        mask = cellsam_pipeline(img, **kwargs)
    except TypeError:
        # 个别版本参数名不同，退回最简调用
        mask = cellsam_pipeline(img, use_wsi=cfg.get("use_wsi", True),
                                bbox_threshold=cfg.get("bbox_threshold", 0.4))
    return np.asarray(mask).astype(np.int32, copy=False)


# ---------------------------------------------------------------------------- #
# GT 构建（官方 cell_boundaries.parquet 栅格化）
# ---------------------------------------------------------------------------- #
def read_pixel_size(experiment_path, default=XENIUM_DEFAULT_PIXEL_SIZE):
    """从 experiment.xenium（JSON）递归找 pixel_size(µm/px)。"""
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
    """调 test.py 指标：GT 放 reference 槽、CellSAM 放 pred 槽；再改成无歧义字段名。"""
    gt_mask, pred_mask = align_shapes(gt_mask, pred_mask)

    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="CellSAM",
        cellpose_gt_mask=gt_mask,     # ← 真值
        pred_mask=pred_mask,          # ← CellSAM 预测
        boundary_mode=boundary_mode,
        iou_threshold=iou_threshold,
    )

    result = {
        "dataset": dataset_name,
        "gt_source": gt_source,
        "method": "CellSAM",
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


def _run_dataset_impl(cfg):
    name = cfg["name"]
    ddir = cfg["dataset_dir"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== [{name}] {ddir}  →  {out_dir} ===")

    # 1) 读形态学通道（整图）
    morph = load_morphology_channels(ddir, pyramid_level=cfg.get("pyramid_level", 0),
                                     needed_indices=[NUCLEUS_IDX, MEMBRANE_IDX], read_crop=cfg.get("crop"))
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

    if crop is not None and not morph.get("crop_applied"):
        channels = [_apply_crop(c, crop) for c in channels]

    img, in_mode = assemble_cellsam_input(channels, use_membrane=cfg.get("use_membrane", True))
    seg_shape = img.shape[:2]
    print(f"  CellSAM 输入: img.shape={img.shape} 通道模式={in_mode} use_wsi={CELLSAM['use_wsi']} block_size={CELLSAM['block_size']}")

    # 3) pixel_size（栅格化 GT 用）
    exp_path = _find_experiment(ddir, cfg)
    base_px = read_pixel_size(exp_path)
    px_eff = base_px * factor
    print(f"  pixel_size={base_px} µm/px  当前层有效 pixel_size={px_eff:.4f} µm/px")

    # 4) CellSAM 分割
    print("  运行 CellSAM ...")
    segmentation_started = time.perf_counter()
    pred = run_cellsam(img, cfg=CELLSAM)
    segmentation_seconds = time.perf_counter() - segmentation_started
    print(f"  预测细胞数={len(metrics.get_instance_ids(pred))}  mask={pred.shape}")
    print(f"  CellSAM 分割耗时={segmentation_seconds:.2f} 秒")

    # 5) 构建 GT（官方边界栅格化，与预测同一裁剪窗口对齐）
    if crop is not None:
        y0, y1, x0, x1 = crop
        gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                  out_shape=(y1 - y0, x1 - x0),
                                  pixel_size_eff=px_eff, offset=(y0, x0))
    else:
        gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                  out_shape=seg_shape, pixel_size_eff=px_eff)
    gt_source = "xenium_official_boundaries"
    print(f"  GT 细胞数={len(metrics.get_instance_ids(gt))}  来源={gt_source}")

    # 6) 落盘 mask（int32 tif，兼容 test.py 的 read_tif_mask）
    pred_path = os.path.join(out_dir, f"{name}_cellsam_pred.tif")
    gt_path = os.path.join(out_dir, f"{name}_gt.tif")
    tifffile.imwrite(pred_path, pred)
    tifffile.imwrite(gt_path, gt)
    print(f"  已保存 {pred_path} / {gt_path}")

    # 7) 指标 + 本数据集单独 CSV
    result = evaluate(gt, pred, dataset_name=name, gt_source=gt_source)
    result["segmentation_seconds"] = segmentation_seconds
    return result


def run_dataset(cfg):
    """运行单个数据集，并附加运行时间、CPU/GPU 峰值资源统计。"""
    monitor = _PerformanceMonitor(f"{cfg['name']}")
    monitor.start()
    try:
        result = _run_dataset_impl(cfg)
    except Exception:
        monitor.stop()
        monitor.report()
        raise

    result.update(monitor.stop())
    per_csv = os.path.join(cfg["output_dir"], f"{cfg['name']}_cellsam_metrics.csv")
    pd.DataFrame([result]).to_csv(per_csv, index=False)
    print(f"  指标已写入 {per_csv}")
    monitor.report()
    return result


def _main_impl():
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
    combined = os.path.join(PARENT_OUT, "cellsam_xenium_metrics_all.csv")
    df.to_csv(combined, index=False)

    pd.set_option("display.max_columns", None, "display.width", 220)
    print("\n================ 汇总 ================")
    print(df.to_string(index=False))
    print("\n各数据集 CSV 见各自 output_dir；汇总:", combined)
    return df


def main():
    """运行全部数据集，并在成功/失败时都打印脚本级性能统计。"""
    monitor = _PerformanceMonitor("script", reset_cuda=False)
    monitor.start()
    try:
        return _main_impl()
    finally:
        monitor.stop()
        monitor.report()


if __name__ == "__main__":
    main()
