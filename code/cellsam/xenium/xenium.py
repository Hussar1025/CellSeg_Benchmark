#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CellSAM (vanvalenlab) × Xenium segmentation benchmark
====================================================================

Same structure as the Cellpose / CelloType / Mesmer versions, but **runs on xenium_breast only**:
  GT = label mask rasterized from the official cell_boundaries.parquet (by default only a centered 5000x5000 ROI).
Read morphology image -> CellSAM segmentation -> build GT -> shape alignment -> call test.py metrics -> write yourpath/CellSam/xenium_breast.

CellSAM key points (per the official docs https://vanvalenlab.github.io/cellSAM and the cellsam_pipeline API)
--------------------------------------------------------------------------------------
- Install: pip install git+https://github.com/vanvalenlab/cellSAM.git (needs python>=3.10, pure PyTorch)
- Main interface: cellsam_pipeline(img, use_wsi=True, block_size=..., bbox_threshold=0.4, ...) -> 2D uint32 label
- Input img: (H,W) or (H,W,C). With multi-channel (multiplex) imaging C==3 and **channel order = (blank, nuclear, whole-cell)**,
  where whole-cell is the membrane/cytoplasm marker outlining cell borders (optional; if absent only a nuclear segmentation is returned).
  -> Xenium: channel 1 = DAPI (nucleus), channel 2 = membrane; channel 0 = blank (all zeros). CellSAM preprocesses internally, so pass raw intensities.
- use_wsi=True tiles large images (required for 5000x5000); a smaller block_size suits denser images; bbox_threshold trades recall against precision.
- The first run downloads weights over the network (may need a DeepCell API key, see https://vanvalenlab.github.io/cellSAM/API-key.html,
  set the DEEPCELL_ACCESS_TOKEN env var).

Semantics (same as the earlier versions)
----------------------------------------
test.py's evaluate_single_mask_against_cellpose(cellpose_gt_mask, pred_mask) treats the first slot as the reference.
In this benchmark **CellSAM is the method under test**: ground truth goes in slot one and the CellSAM prediction in slot two, so the metric direction means "evaluating CellSAM".

Dependencies: numpy, pandas, scikit-image, scipy, tifffile, pyarrow, cellSAM(+torch)
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
except ImportError:  # pragma: no cover - platforms such as Windows have no resource module
    resource = None

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import polygon as sk_polygon


# ---------------------------------------------------------------------------- #
# Runtime performance monitoring
# ---------------------------------------------------------------------------- #
def _current_memory_mb():
    """Return current process RSS (MB); prefer /proc, fall back to psutil."""
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
    """Return peak RSS over the process lifetime (MB); ru_maxrss is in KiB on Linux."""
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
    """Record wall-clock time, peak CPU RSS and peak CUDA GPU memory for a block of work."""

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
        print(f"[Performance:{self.label}] total runtime: {stats['runtime_seconds']:.2f} s "
              f"({stats['runtime_seconds'] / 60:.2f} min)")
        print(f"[Performance:{self.label}] CPU RSS: initial {stats['initial_memory_mb']:.2f} MB, "
              f"final {stats['final_memory_mb']:.2f} MB, peak {stats['peak_memory_mb']:.2f} MB")
        allocated = stats["peak_gpu_memory_allocated_mb"]
        reserved = stats["peak_gpu_memory_reserved_mb"]
        if allocated is None and reserved is None:
            print(f"[Performance:{self.label}] peak GPU memory: N/A")
        else:
            print(f"[Performance:{self.label}] peak GPU memory: "
                  f"allocated={allocated if allocated is not None else float('nan'):.2f} MB, "
                  f"reserved={reserved if reserved is not None else float('nan'):.2f} MB")


# ---------------------------------------------------------------------------- #
# Location of test.py (the metric implementation)
# ---------------------------------------------------------------------------- #
TEST_PY_PATH = "yourpath/eval/test.py"


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
            "test.py not found: set TEST_PY_PATH at the top of this script to the absolute path of test.py, "
            "or put test.py in the same directory as this script."
        ) from e


metrics = _load_metrics()


# ============================================================================ #
# Configuration
# ============================================================================ #
XENIUM_DEFAULT_PIXEL_SIZE = 0.2125  # um/px; fallback when experiment metadata cannot be read

# CellSAM global params (see the cellsam_pipeline signature)
CELLSAM = dict(
    use_wsi=True,            # tile large images (required for 5000x5000)
    block_size=512,          # tile side length in [256,2048]; use smaller for dense images
    overlap=56,              # tile overlap (< block_size)
    bbox_threshold=0.4,      # CellFinder confidence threshold: mainly controls precision/recall; lower it for out-of-distribution images
    low_contrast_enhancement=False,
    gauge_cell_size=False,   # estimate cell size in a first pass, then tile (helps dense images, slower)
    model_path=None,         # None = use the latest released generalist model (downloaded)
)

# Channel convention: Xenium morphology_focus channel 0 = DAPI (nucleus), channel 1 = membrane stain
NUCLEUS_IDX = 0
MEMBRANE_IDX = 1

PARENT_OUT = "yourpath/CellSam"

DATASETS = [
    dict(
        name="xenium_mouse_brain",
        dataset_dir="yourpath/dataset/xenium_mouse_brain/outs",  # NOTE: the path goes into outs/
        output_dir="yourpath/CellSam/xenium_mouse_brain_output",
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
# Morphology image loading (pyramid level + two naming schemes + single/multi-channel files)
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
            warnings.warn(f"{os.path.basename(path)}: only {len(levels)} pyramid levels; "
                          f"pyramid_level={pyramid_level} clipped to {L}")
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
            warnings.warn(f"{dataset_dir}: morphology_focus/ not found; falling back to {cand} (nucleus channel only)")
            return dict(channels=chs, names=[cand], factor=factor, crop_applied=read_crop is not None)

    raise FileNotFoundError(f"{dataset_dir}: no morphology OME-TIFF found at all")


# ---------------------------------------------------------------------------- #
# Assemble the CellSAM input: (H, W, 3) with channel order (blank, nuclear, whole-cell/membrane)
# ---------------------------------------------------------------------------- #
def assemble_cellsam_input(channels, use_membrane=True,
                           nucleus_idx=NUCLEUS_IDX, membrane_idx=MEMBRANE_IDX):
    """Return (img[H,W,3] float32, mode). CellSAM preprocesses internally, so pass raw intensities here.
    Channel order as the official API requires: 0=blank (all zeros), 1=nuclear (DAPI), 2=whole-cell (membrane)."""
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
# CellSAM segmentation
# ---------------------------------------------------------------------------- #
def run_cellsam(img, cfg=CELLSAM):
    """Run CellSAM and return an int32 instance label mask [H,W]."""
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
        # a few versions use different arg names; fall back to the minimal call
        mask = cellsam_pipeline(img, use_wsi=cfg.get("use_wsi", True),
                                bbox_threshold=cfg.get("bbox_threshold", 0.4))
    return np.asarray(mask).astype(np.int32, copy=False)


# ---------------------------------------------------------------------------- #
# GT construction (rasterize the official cell_boundaries.parquet)
# ---------------------------------------------------------------------------- #
def read_pixel_size(experiment_path, default=XENIUM_DEFAULT_PIXEL_SIZE):
    """Recursively find pixel_size (um/px) in experiment.xenium (JSON)."""
    if not experiment_path or not os.path.exists(experiment_path):
        warnings.warn(f"{experiment_path} not found; using default pixel_size {default}")
        return default
    try:
        with open(experiment_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        warnings.warn(f"failed to read {experiment_path} ({e}); using default pixel_size {default}")
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
    warnings.warn(f"no pixel_size field in {experiment_path}; using default {default}")
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
    # prefer local coordinates (ROI-relative); use global only when absent
    x_col = next((cols[c] for c in ("vertex_x_local_um", "vertex_x_local", "vertex_x", "x", "x_location", "vertex_x_um") if c in cols), None)
    y_col = next((cols[c] for c in ("vertex_y_local_um", "vertex_y_local", "vertex_y", "y", "y_location", "vertex_y_um") if c in cols), None)
    return id_col, x_col, y_col


def rasterize_boundaries(parquet_path, out_shape, pixel_size_eff, offset=(0, 0), verbose=True):
    """Rasterize Xenium cell_boundaries.parquet (micron polygons) into an int32 label mask.
    pixel = micron / pixel_size_eff - offset; out-of-bounds parts are clipped by skimage."""
    row0, col0 = offset
    df = pd.read_parquet(parquet_path)
    id_col, x_col, y_col = _boundary_columns(df)
    if id_col is None or x_col is None or y_col is None:
        raise ValueError(f"{parquet_path}: cannot identify columns (cell_id/vertex_x/vertex_y); actual columns={list(df.columns)}")

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
            print(f"    rasterized {label} cells ...")
    if verbose:
        print(f"    rasterization done: {label} cells -> mask {mask.shape}")
    return mask


# ---------------------------------------------------------------------------- #
# Shape alignment + evaluation
# ---------------------------------------------------------------------------- #
def align_shapes(gt, pred):
    if gt.shape == pred.shape:
        return gt, pred
    warnings.warn(f"GT{gt.shape} and Pred{pred.shape} shapes differ; zero-padding at the bottom-right (verify they really align)")
    H = max(gt.shape[0], pred.shape[0])
    W = max(gt.shape[1], pred.shape[1])

    def _pad(m):
        out = np.zeros((H, W), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out
    return _pad(gt), _pad(pred)


def evaluate(gt_mask, pred_mask, dataset_name, gt_source,
             boundary_mode=BOUNDARY_MODE, iou_threshold=IOU_THRESHOLD):
    """Call the test.py metrics: GT in the reference slot, CellSAM in the pred slot; then rename to unambiguous fields."""
    gt_mask, pred_mask = align_shapes(gt_mask, pred_mask)

    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="CellSAM",
        cellpose_gt_mask=gt_mask,     # ground truth
        pred_mask=pred_mask,          # CellSAM prediction
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
# Single-dataset pipeline
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
    print(f"\n=== [{name}] {ddir}  ->  {out_dir} ===")

    # 1) load morphology channels (whole image)
    morph = load_morphology_channels(ddir, pyramid_level=cfg.get("pyramid_level", 0),
                                     needed_indices=[NUCLEUS_IDX, MEMBRANE_IDX], read_crop=cfg.get("crop"))
    factor = morph["factor"]
    channels = morph["channels"]
    Himg, Wimg = channels[0].shape
    print(f"  channels={len(channels)}  names={morph['names']}  downsample factor={factor:.3f}")

    # 2) choose the crop window (explicit crop wins; otherwise crop_size centered)
    crop = cfg.get("crop")
    if crop is None and cfg.get("crop_size"):
        s = int(cfg["crop_size"])
        y0 = max((Himg - s) // 2, 0)
        x0 = max((Wimg - s) // 2, 0)
        crop = (y0, min(y0 + s, Himg), x0, min(x0 + s, Wimg))
        print(f"  crop_size={s} -> centered crop={crop} (image {Himg}x{Wimg})")

    if crop is not None and not morph.get("crop_applied"):
        channels = [_apply_crop(c, crop) for c in channels]

    img, in_mode = assemble_cellsam_input(channels, use_membrane=cfg.get("use_membrane", True))
    seg_shape = img.shape[:2]
    print(f"  CellSAM input: img.shape={img.shape} channel mode={in_mode} use_wsi={CELLSAM['use_wsi']} block_size={CELLSAM['block_size']}")

    # 3) pixel_size (used to rasterize GT)
    exp_path = _find_experiment(ddir, cfg)
    base_px = read_pixel_size(exp_path)
    px_eff = base_px * factor
    print(f"  pixel_size={base_px} um/px  effective pixel_size at this level={px_eff:.4f} um/px")

    # 4) CellSAM segmentation
    print("  running CellSAM ...")
    segmentation_started = time.perf_counter()
    pred = run_cellsam(img, cfg=CELLSAM)
    segmentation_seconds = time.perf_counter() - segmentation_started
    print(f"  predicted cells={len(metrics.get_instance_ids(pred))}  mask={pred.shape}")
    print(f"  CellSAM segmentation took={segmentation_seconds:.2f} s")

    # 5) build GT (rasterize official boundaries, aligned to the same crop window as the prediction)
    if crop is not None:
        y0, y1, x0, x1 = crop
        gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                  out_shape=(y1 - y0, x1 - x0),
                                  pixel_size_eff=px_eff, offset=(y0, x0))
    else:
        gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                  out_shape=seg_shape, pixel_size_eff=px_eff)
    gt_source = "xenium_official_boundaries"
    print(f"  GT cells={len(metrics.get_instance_ids(gt))}  source={gt_source}")

    # 6) write masks to disk (int32 tif, compatible with test.py's read_tif_mask)
    pred_path = os.path.join(out_dir, f"{name}_cellsam_pred.tif")
    gt_path = os.path.join(out_dir, f"{name}_gt.tif")
    tifffile.imwrite(pred_path, pred)
    tifffile.imwrite(gt_path, gt)
    print(f"  Saved {pred_path} / {gt_path}")

    # 7) metrics + per-dataset CSV
    result = evaluate(gt, pred, dataset_name=name, gt_source=gt_source)
    result["segmentation_seconds"] = segmentation_seconds
    return result


def run_dataset(cfg):
    """Run one dataset, attaching runtime and peak CPU/GPU resource stats."""
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
    print(f"  metrics written to {per_csv}")
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
            print(f"[ERROR] dataset {cfg['name']} failed: {e}")
            traceback.print_exc()

    if not rows:
        print("No dataset succeeded.")
        return

    df = pd.DataFrame(rows)
    combined = os.path.join(PARENT_OUT, "cellsam_xenium_metrics_all.csv")
    df.to_csv(combined, index=False)

    pd.set_option("display.max_columns", None, "display.width", 220)
    print("\n================ Summary ================")
    print(df.to_string(index=False))
    print("\nPer-dataset CSVs live in each output_dir; summary:", combined)
    return df


def main():
    """Run all datasets and print script-level performance stats on both success and failure."""
    monitor = _PerformanceMonitor("script", reset_cuda=False)
    monitor.start()
    try:
        return _main_impl()
    finally:
        monitor.stop()
        monitor.report()


if __name__ == "__main__":
    main()
