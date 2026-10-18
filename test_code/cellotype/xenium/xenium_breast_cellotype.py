#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CelloType × Xenium 细胞分割 benchmark
=====================================

结构与 Cellpose 版一致：读形态学图 → CelloType 分割（预测） → 构建 GT → 形状对齐 → 调 test.py 的指标。
两个数据集：
  1) xenium_roi_crop/  —— GT = manual_labels.npy（手动标注 label mask）
  2) xenium_breast/    —— GT = 官方 cell_boundaries.parquet 栅格化得到的 label mask（默认只跑居中 5k×5k ROI）
结果分别写入各自的输出文件夹。

CelloType 要点（据官方 README / 文档 https://cellotype.readthedocs.io ）
------------------------------------------------------------------------
- API：from cellotype.predict import CelloTypePredictor
       CelloTypePredictor(model_path, config_path, confidence_thresh, max_det, device).predict(img) -> [H,W]
- 输入是 [H,W,3] 的 **uint8** 图；通道约定：蓝(B)=核(DAPI)，绿(G)=膜，红(R)=可选。
- 大图/密集图必须用 SAHI 切片推理（MaskDINO 单次前向的 query 数有上限，约 300，否则严重漏检）。
  官方大图流程：Detectron2DetectionModel + get_sliced_prediction（512 分块、重叠 5%、perform_standard_pred=False），
  跨块用 NMS 合并；再把每个 object 的全图 bool_mask 写成实例 label。

语义说明（与 Cellpose 版相同）
------------------------------
test.py 的 evaluate_single_mask_against_cellpose(cellpose_gt_mask, pred_mask) 把第一个槽当 reference。
本 benchmark 里 **CelloType 是被测方法**：真值放第一个槽、CelloType 预测放第二个槽，指标方向即为
「评估 CelloType」。输出里把误导性的字段名改成明确的名字。

依赖：numpy, pandas, scikit-image, scipy, tifffile, pyarrow, Pillow,
      cellotype(+Detectron2+Deformable-DETR+MaskDINO), sahi
"""

import os
os.environ["VISIBLE_DEVICES"] = "6"  
import re
import sys
import json
import glob
import warnings
import tempfile
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
        page_size = os.sysconf("SC_PAGE_SIZE")
        return rss_pages * page_size / 1024**2
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
            if sys.platform == "darwin":
                value /= 1024**2
            else:
                value /= 1024
            return value
        except Exception:
            pass
    return _current_memory_mb()


def _reset_cuda_peak_stats():
    """重置 PyTorch CUDA 峰值统计；没有 CUDA/torch 时静默跳过。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        for device_idx in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(device_idx)
            except Exception:
                pass
    except Exception:
        pass


def _cuda_peak_mb(metric):
    """读取所有可见 GPU 的 PyTorch 峰值显存(MB)，不可用时返回 None。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        values = []
        fn = getattr(torch.cuda, metric)
        for device_idx in range(torch.cuda.device_count()):
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
        if self.stats is not None:
            return self.stats
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
        print(
            f"[Performance:{self.label}] 总运行时间: "
            f"{stats['runtime_seconds']:.2f} 秒 ({stats['runtime_seconds'] / 60:.2f} 分钟)"
        )
        print(
            f"[Performance:{self.label}] CPU RSS: 初始 "
            f"{stats['initial_memory_mb']:.2f} MB, 结束 "
            f"{stats['final_memory_mb']:.2f} MB, 峰值 "
            f"{stats['peak_memory_mb']:.2f} MB"
        )
        allocated = stats["peak_gpu_memory_allocated_mb"]
        reserved = stats["peak_gpu_memory_reserved_mb"]
        if allocated is None and reserved is None:
            print(f"[Performance:{self.label}] GPU 峰值显存: N/A")
        else:
            print(
                f"[Performance:{self.label}] GPU 峰值显存: allocated="
                f"{allocated if allocated is not None else float('nan'):.2f} MB, "
                f"reserved={reserved if reserved is not None else float('nan'):.2f} MB"
            )


# ---------------------------------------------------------------------------- #
# test.py（指标实现）的位置：按题目要求指向 /data/qiuyijia/eval/test.py
# 路径不存在时会退回「与本脚本同目录 / PYTHONPATH 里的 test.py」。
# ---------------------------------------------------------------------------- #
TEST_PY_PATH = "/data/qiuyijia/eval/test.py"


def _load_metrics():
    if TEST_PY_PATH and os.path.exists(TEST_PY_PATH):
        spec = importlib.util.spec_from_file_location("xenium_metrics", TEST_PY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # test.py 的 __main__ 块不会触发（模块名非 __main__）
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


metrics = _load_metrics()  # test.py 中的指标实现


# ============================================================================ #
# 配置区
# ============================================================================ #
XENIUM_DEFAULT_PIXEL_SIZE = 0.2125  # µm/px，读不到 experiment 时的兜底值

# CelloType 全局参数
CELLOTYPE = dict(
    device="cuda",
    # ↓↓↓ 改成你 CelloType 克隆目录下的权重与 config（download.sh 下载的） ↓↓↓
    model_path="/data/qiuyijia/cellotype/CelloType/models/tissuenet_model_0019999.pth",
    config_path="/data/qiuyijia/cellotype/CelloType/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
    confidence_thresh=0.3,
    max_det=1000,            # 仅 simple 模式用；密集图请用 sahi
    # 通道拉伸到 uint8 的百分位（Xenium uint16 动态范围大，必须拉伸）
    pct_low=1.0,
    pct_high=99.0,
    # SAHI 切片推理参数（大图/密集图，官方大图流程）
    sahi_slice=512,
    sahi_overlap=0.05,
    sahi_image_size=512,     # 显存不够就和 slice 一起降到 256
)

# 通道约定：Xenium morphology_focus 通道 0 = DAPI(核)，通道 1 = 膜染色(ATP1A1/CD45/E-cad)
NUCLEUS_IDX = 0
MEMBRANE_IDX = 1

PARENT_OUT = "/data/qiuyijia/cellotype"   # 汇总 CSV 放这里；各数据集单独文件夹见下

DATASETS = [
    dict(
        name="xenium_mouse_brain",
        dataset_dir="/data/qiuyijia/dataset/xenium_mouse_brain/outs",  # 注意：路径到outs/
        output_dir="/data/qiuyijia/cellotype/xenium_mouse_brain_output",
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
# 形态学图像读取（支持金字塔层 + 两种命名 + 单/多通道文件）
# ---------------------------------------------------------------------------- #
def _read_level(path, pyramid_level, crop=None):
    """读一个 OME-TIFF 的指定金字塔层，返回 (array, downsample_factor)。"""
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
                plane=level.pages[zi].asarray()[y0:y1,x0:x1]; arr=plane.copy() if arr is None else np.maximum(arr,plane)
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
    """把一个数组拆成通道列表。2D→单通道；3D→按小轴(<=5)判定通道轴。"""
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[0] <= 5 and arr.shape[0] <= arr.shape[-1]:
            return [arr[i] for i in range(arr.shape[0])]        # (C,H,W)
        if arr.shape[-1] <= 5:
            return [arr[..., i] for i in range(arr.shape[-1])]  # (H,W,C)
        return [arr[arr.shape[0] // 2]]                          # 疑似 z-stack，取中间层
    arr = np.squeeze(arr)
    return _split_channels(arr) if arr.ndim <= 3 else [np.max(arr, axis=tuple(range(arr.ndim - 2)))]


def load_morphology_channels(dataset_dir, pyramid_level=0, needed_indices=None, read_crop=None):
    """发现并读取形态学通道，支持 ch0000_dapi... 与 morphology_focus_0000... 两种命名及多通道单文件。

    返回: dict(channels=[2D...], names=[...], factor=float)
    """
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
# 组装 CelloType 输入（uint8 RGB：B=核, G=膜, R=可选）
# ---------------------------------------------------------------------------- #
def _to_uint8(ch, p_low, p_high):
    """按百分位把单通道拉伸到 uint8 [0,255]。Xenium uint16 动态范围大，必须拉伸。"""
    ch = ch.astype(np.float32)
    lo, hi = np.percentile(ch, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    ch = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return (ch * 255.0).astype(np.uint8)


def assemble_cellotype_rgb(channels, use_membrane=True,
                           nucleus_idx=NUCLEUS_IDX, membrane_idx=MEMBRANE_IDX,
                           p_low=1.0, p_high=99.0):
    """拼出 CelloType 需要的 uint8 RGB：蓝=核(DAPI)、绿=膜、红=0。

    返回 (rgb[H,W,3] uint8, mode)。无膜通道或 use_membrane=False 时绿通道置 0（仅核）。
    """
    n = len(channels)
    nuc = channels[nucleus_idx] if nucleus_idx < n else channels[0]
    H, W = nuc.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :, 2] = _to_uint8(nuc, p_low, p_high)                    # B = 核

    if use_membrane and membrane_idx < n:
        rgb[:, :, 1] = _to_uint8(channels[membrane_idx], p_low, p_high)  # G = 膜
        mode = "nucleus(B)+membrane(G)"
    else:
        mode = "nucleus(B)_only"
    # 注：若改用 Xenium 专训权重、想把「转录本密度」放绿通道，可在此替换 rgb[:,:,1]
    return rgb, mode


# ---------------------------------------------------------------------------- #
# CelloType 分割
# ---------------------------------------------------------------------------- #
def _mask_from_sahi(result, hw):
    """把 SAHI 结果里每个 object 的全图 bool_mask 写成 int32 实例 label（0=背景）。"""
    H, W = hw
    out = np.zeros((H, W), dtype=np.int32)
    obj_list = result.object_prediction_list
    for i, obj in enumerate(obj_list, start=1):
        bm = np.asarray(obj.mask.bool_mask)   # 该 object 在全图坐标下的布尔 mask
        if bm.shape != (H, W):
            # 个别 SAHI 版本 bool_mask 尺寸可能不同，跳过并告警而非崩溃
            warnings.warn(f"SAHI object {i} bool_mask shape {bm.shape} != {(H, W)}，已跳过")
            continue
        out[bm] = i
    return out


def _run_cellotype_sahi(rgb, cfg):
    """SAHI 切片推理（大图/密集图）。尽量贴合官方 sahi_inference.py。"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # 关掉 PIL 大图保护

    # 优先用 CelloType 自带封装（与官方示例一致），退回标准 sahi
    try:
        from cellotype.predict import Detectron2DetectionModel
    except Exception:
        from sahi.models.detectron2 import Detectron2DetectionModel
    try:
        from sahi_predict import get_sliced_prediction        # CelloType 仓库根目录脚本
    except Exception:
        from sahi.predict import get_sliced_prediction        # 退回标准 sahi

    det = Detectron2DetectionModel(
        model_path=cfg["model_path"],
        config_path=cfg["config_path"],
        confidence_threshold=cfg["confidence_thresh"],
        image_size=cfg["sahi_image_size"],
        device=cfg["device"],
        channels=3,
    )

    # get_sliced_prediction 对文件路径最稳（与官方示例一致），把 RGB 写临时 PNG
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        Image.fromarray(rgb).save(tmp_path)
        result = get_sliced_prediction(
            tmp_path,
            det,
            slice_height=cfg["sahi_slice"],
            slice_width=cfg["sahi_slice"],
            overlap_height_ratio=cfg["sahi_overlap"],
            overlap_width_ratio=cfg["sahi_overlap"],
            perform_standard_pred=False,   # 只切片、不做整图前向（避免 query 上限导致漏检）
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return _mask_from_sahi(result, rgb.shape[:2])


def run_cellotype(rgb, cfg):
    """跑 CelloType，返回 int32 实例 label mask。inference='sahi'（默认，大图/密集）或 'simple'。"""
    mode = cfg.get("inference", "sahi")
    if mode == "simple":
        from cellotype.predict import CelloTypePredictor
        model = CelloTypePredictor(
            model_path=cfg["model_path"],
            config_path=cfg["config_path"],
            confidence_thresh=cfg["confidence_thresh"],
            max_det=cfg["max_det"],
            device=cfg["device"],
        )
        mask = model.predict(rgb)          # [H,W]
        return np.asarray(mask).astype(np.int32, copy=False)
    elif mode == "sahi":
        return _run_cellotype_sahi(rgb, cfg)
    else:
        raise ValueError(f"未知 inference 模式: {mode}")


# ---------------------------------------------------------------------------- #
# GT 构建
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


def _boundary_columns(df):
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[c] for c in ("cell_id", "label_id", "label", "id") if c in cols), None)
    # 优先使用 local 坐标（ROI 相对坐标），没有才用全局坐标
    x_col = next((cols[c] for c in ("vertex_x_local_um", "vertex_x_local", "vertex_x", "x", "x_location", "vertex_x_um") if c in cols), None)
    y_col = next((cols[c] for c in ("vertex_y_local_um", "vertex_y_local", "vertex_y", "y", "y_location", "vertex_y_um") if c in cols), None)
    return id_col, x_col, y_col


def rasterize_boundaries(parquet_path, out_shape, pixel_size_eff, offset=(0, 0), verbose=True):
    """把 Xenium cell_boundaries.parquet（微米坐标多边形）栅格化成 int32 label mask。

    out_shape      : (H, W) 输出尺寸（裁剪时为裁剪区域尺寸）
    pixel_size_eff : 该层每像素微米数 = base_pixel_size × downsample_factor
    offset         : (row0, col0) 裁剪窗口左上角在整图中的像素坐标；未裁剪为 (0,0)
    像素 = 微米 / pixel_size_eff - offset；越界部分由 skimage 自动裁掉
    """
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

    codes, uniq = pd.factorize(ids_all)          # 按出现顺序编码
    n_cells = len(uniq)
    sort_idx = np.argsort(codes, kind="stable")  # 同一 cell 顶点连续，且保留绕行顺序
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
    """调 test.py 的指标：GT 放 reference 槽、CelloType 放 pred 槽；再改成无歧义字段名。"""
    gt_mask, pred_mask = align_shapes(gt_mask, pred_mask)

    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="CelloType",
        cellpose_gt_mask=gt_mask,     # ← 真值
        pred_mask=pred_mask,          # ← CelloType 预测
        boundary_mode=boundary_mode,
        iou_threshold=iou_threshold,
    )

    result = {
        "dataset": dataset_name,
        "gt_source": gt_source,
        "method": "CelloType",
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

    # 1) 读形态学通道 + 拼 CelloType RGB（整图）
    morph = load_morphology_channels(ddir, pyramid_level=cfg.get("pyramid_level", 0),
                                     needed_indices=[NUCLEUS_IDX, MEMBRANE_IDX], read_crop=cfg.get("crop"))
    factor = morph["factor"]
    print(f"  通道数={len(morph['channels'])}  名称={morph['names']}  下采样因子={factor:.3f}")

    rgb_full, in_mode = assemble_cellotype_rgb(
        morph["channels"], use_membrane=cfg.get("use_membrane", True),
        p_low=CELLOTYPE["pct_low"], p_high=CELLOTYPE["pct_high"])
    Himg, Wimg = rgb_full.shape[:2]

    # 决定裁剪窗口：显式 crop 优先；否则若给了 crop_size 则居中裁正方形
    crop = cfg.get("crop")
    if crop is None and cfg.get("crop_size"):
        s = int(cfg["crop_size"])
        y0 = max((Himg - s) // 2, 0)
        x0 = max((Wimg - s) // 2, 0)
        crop = (y0, min(y0 + s, Himg), x0, min(x0 + s, Wimg))
        print(f"  crop_size={s} → 居中裁剪 crop={crop}（整图 {Himg}×{Wimg}）")

    rgb = rgb_full if morph.get("crop_applied") else _apply_crop(rgb_full, crop)
    del rgb_full
    seg_shape = rgb.shape[:2]
    print(f"  CelloType 输入: shape={rgb.shape} 通道模式={in_mode} 推理模式={cfg.get('inference', 'sahi')}")

    # 2) CelloType 分割（预测）
    print("  运行 CelloType ...")
    segmentation_started = time.perf_counter()
    pred = run_cellotype(rgb, CELLOTYPE)
    segmentation_seconds = time.perf_counter() - segmentation_started
    print(f"  预测细胞数={len(metrics.get_instance_ids(pred))}  mask={pred.shape}")
    print(f"  CelloType 分割耗时={segmentation_seconds:.2f} 秒")

    # 3) 构建 GT（与预测同一裁剪窗口对齐）
    if cfg["gt_type"] == "xenium_boundaries":
        exp_path = os.path.join(ddir, cfg.get("experiment", "experiment.xenium"))
        base_px = read_pixel_size(exp_path)
        px_eff = base_px * factor
        print(f"  pixel_size={base_px} µm/px  当前层 pixel_size_eff={px_eff:.4f} µm/px")
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

    # 4) 落盘 mask（int32 tif，兼容 test.py 的 read_tif_mask）
    pred_path = os.path.join(out_dir, f"{name}_cellotype_pred.tif")
    gt_path = os.path.join(out_dir, f"{name}_gt.tif")
    tifffile.imwrite(pred_path, pred)
    tifffile.imwrite(gt_path, gt)
    print(f"  已保存 {pred_path} / {gt_path}")

    # 5) 指标 + 本数据集单独的 CSV
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
    per_csv = os.path.join(cfg["output_dir"], f"{cfg['name']}_cellotype_metrics.csv")
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
    combined = os.path.join(PARENT_OUT, "cellotype_xenium_metrics_all.csv")
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
