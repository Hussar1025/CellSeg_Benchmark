#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cellpose x Xenium cell segmentation benchmark
=============================================

One Cellpose segmentation + metric evaluation script usable on two datasets:

  1) xenium_roi_crop/  -- GT = manual_labels.npy (hand-annotated label mask)
  2) xenium_breast/    -- GT = label mask rasterized from the official cell_boundaries.parquet

Pipeline: read morphology image -> Cellpose segmentation (prediction) -> build GT -> align shapes -> call the test.py metrics.

Semantics (important)
----------------
test.py's evaluate_single_mask_against_cellpose(cellpose_gt_mask, pred_mask) treats the
first slot as the "reference". In this benchmark **Cellpose is the method under test,
not the reference**, so we put the ground truth (manual / official) in the first slot
and the Cellpose prediction in the second. Then:
  recall = fraction of GT cells detected by Cellpose
  FP / over_segmentation = cells Cellpose split extra
  FN / under_segmentation = GT cells Cellpose missed
This is the correct metric direction for "evaluating Cellpose". In the output we
rename the misleading fields to unambiguous names.

Dependencies: numpy, pandas, scikit-image, scipy, tifffile, pyarrow, cellpose(>=4, cpsam)
      test.py must live in the same directory as this script.
"""

import os
import re
import sys
import json
import glob
import warnings
import importlib.util
import time

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on platforms such as Windows
    resource = None

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import polygon as sk_polygon


# ---------------------------------------------------------------------------- #
# Runtime performance monitoring
# ---------------------------------------------------------------------------- #
def _current_memory_mb():
    """Return the current process RSS (MB); prefer /proc, fall back to psutil."""
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
    """Return the process peak RSS (MB); Linux ru_maxrss is in KiB."""
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
    """Record wall time, peak CPU RSS and peak CUDA memory (GPU memory) for one pipeline stage."""

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
        print(f"[Performance:{self.label}] total elapsed: {stats['runtime_seconds']:.2f} s "
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
# If this script and test.py are not in the same directory, set the absolute path below.
# When left "" or the path does not exist, it falls back to "test.py in the same
# directory / on PYTHONPATH".
# ---------------------------------------------------------------------------- #
TEST_PY_PATH = "yourpath/eval/test.py"   # <- set to your test.py absolute path


def _load_metrics():
    if TEST_PY_PATH and os.path.exists(TEST_PY_PATH):
        spec = importlib.util.spec_from_file_location("xenium_metrics", TEST_PY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # test.py's __main__ block is not triggered (module name is not __main__)
        return mod
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import test as mod
        return mod
    except ImportError as e:
        raise ImportError(
            "test.py not found: set TEST_PY_PATH at the top of the script to the absolute "
            "path of test.py, or put test.py in the same directory as this script."
        ) from e


metrics = _load_metrics()  # metric implementation from test.py


# ============================================================================ #
# Config: edit here
# ============================================================================ #
XENIUM_DEFAULT_PIXEL_SIZE = 0.2125  # um/px, fallback when experiment metadata is unreadable

# Cellpose global parameters (Cellpose-SAM / 4.x)
CELLPOSE = dict(
    use_gpu=True,
    diameter=None,           # cpsam is scale-insensitive, None is fine; classic models need a value (cell~60, nucleus~30)
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    batch_size=32,
    normalize=True,          # critical for Xenium high-dynamic-range uint16; percentile normalization
    # if classic Cellpose(3.x) is installed instead of cpsam, set this True and give model_type/channels
    use_classic=False,
    classic_model_type="cyto3",
    classic_channels=[2, 1],  # [cytoplasm, nucleus]: 1=red,2=green,3=blue; see classic Cellpose docs
)

# channel convention: Xenium morphology_focus channel 0 = DAPI (nucleus), channel 1 = membrane/boundary stain (ATP1A1/CD45/E-cad)
NUCLEUS_IDX = 0
MEMBRANE_IDX = 1

DATASETS = [
    dict(
        name="xenium_mouse_brain",
        dataset_dir="yourpath/dataset/xenium_mouse_brain/outs",  # note: path goes down to outs/
        output_dir="yourpath/cellpose/xenium_mouse_brain_output",
        gt_type="xenium_boundaries",
        boundaries="cell_boundaries.parquet",
        experiment="experiment.xenium",
        pyramid_level=0,
        use_membrane=True,
        crop=(12077, 22077, 6956, 16956),
    ),
]
OUTPUT_DIR = "yourpath/cellpose/xenium_out"
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5
# ============================================================================ #


# ---------------------------------------------------------------------------- #
# Morphology image loading (pyramid levels + two naming schemes + single/multi-channel files)
# ---------------------------------------------------------------------------- #
def _read_level(path, pyramid_level, crop=None):
    """Read a given pyramid level of an OME-TIFF; return (array, downsample_factor).

    factor = full-resolution height / current level height (computed from the real
    shape, so no power-of-2 assumption).
    """
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        try:
            levels = list(series.levels)
        except Exception:
            levels = [series]
        L = int(np.clip(pyramid_level, 0, len(levels) - 1))
        if L != pyramid_level:
            warnings.warn(f"{os.path.basename(path)}: only {len(levels)} pyramid levels, "
                          f"pyramid_level={pyramid_level} clamped to {L}")
        # Xenium morphology_focus files are often 35-page z-stacks.  Loading
        # the complete stack would allocate >100 GB per channel; all methods
        # use the middle z plane, so read only that page.
        level = levels[L]
        if len(tf.pages) == 1:
            # Xenium morphology_focus files contain one real TIFF page per
            # file; OME-XML describes the set of files as SizeC channels.
            arr = np.asarray(tf.pages[0].asarray())
            if crop is not None:
                y0,y1,x0,x1 = crop; arr = arr[y0:y1, x0:x1]
        elif len(level.shape) == 3 and level.shape[0] > 5:
            zcount = level.shape[0]
            y0,y1,x0,x1 = crop if crop is not None else (0, level.shape[-2], 0, level.shape[-1])
            arr = None
            for zi in range(zcount):
                page = level.pages[zi].asarray()
                plane = page[y0:y1, x0:x1]
                arr = plane.copy() if arr is None else np.maximum(arr, plane)
        else:
            arr = np.asarray(level.asarray())
            if crop is not None:
                y0,y1,x0,x1 = crop
                # Multi-channel page (C,H,W): crop only spatial axes.
                arr = arr[:, y0:y1, x0:x1] if arr.ndim == 3 and arr.shape[0] <= 5 else arr[y0:y1, x0:x1]
        base_h = levels[0].shape[-2]
        cur_h = levels[L].shape[-2]
        factor = float(base_h) / float(cur_h)
    return arr, factor


def _split_channels(arr):
    """Split an array into a channel list. 2D -> single channel; 3D -> channel axis is the small axis (<=5)."""
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[0] <= 5 and arr.shape[0] <= arr.shape[-1]:
            return [arr[i] for i in range(arr.shape[0])]        # (C,H,W)
        if arr.shape[-1] <= 5:
            return [arr[..., i] for i in range(arr.shape[-1])]  # (H,W,C)
        return [arr[arr.shape[0] // 2]]                          # likely a z-stack, take the middle plane
    # higher dimensions: squeeze the leading axes, then take the middle
    arr = np.squeeze(arr)
    return _split_channels(arr) if arr.ndim <= 3 else [np.max(arr, axis=tuple(range(arr.ndim - 2)))]


def load_morphology_channels(dataset_dir, pyramid_level=0, needed_indices=None, read_crop=None):
    """Discover and read the morphology channels.

    Prefers the morphology_focus/ directory; supported file names:
        ch0000_dapi.ome.tif ...            (breast naming)
        morphology_focus_0000.ome.tif ...  (ROI naming)
    A single multi-channel file is expanded. Falls back to morphology.ome.tif when
    the directory is missing.

    Returns: dict(channels=[2D...], names=[...], factor=float)
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

    # fallback: morphology.ome.tif (usually DAPI, may be a z-stack)
    for cand in ("morphology.ome.tif", "morphology_focus.ome.tif", "morphology_mip.ome.tif"):
        fp = os.path.join(dataset_dir, cand)
        if os.path.exists(fp):
            arr, factor = _read_level(fp, pyramid_level, crop=read_crop)
            chs = _split_channels(arr)
            if len(chs) > 1:  # usually a z-stack: max-project it into a single nucleus channel
                chs = [np.max(np.stack(chs, 0), 0)]
            warnings.warn(f"{dataset_dir}: morphology_focus/ not found, falling back to {cand} (nucleus channel only)")
            return dict(channels=chs, names=[cand], factor=factor, crop_applied=read_crop is not None)

    raise FileNotFoundError(f"{dataset_dir}: no morphology OME-TIFF found")


def assemble_cellpose_input(channels, use_membrane=True,
                            nucleus_idx=NUCLEUS_IDX, membrane_idx=MEMBRANE_IDX):
    """Assemble the channels into a Cellpose input.

    Returns (img, mode):
      - membrane channel present and use_membrane -> (H,W,2) float32, [nucleus, membrane], for whole-cell segmentation
      - otherwise -> (H,W) float32 grayscale (nucleus), nucleus-only segmentation
    Cellpose-SAM normalizes internally; here we only cast the dtype and never touch intensities.
    """
    n = len(channels)
    nuc = channels[nucleus_idx] if nucleus_idx < n else channels[0]
    nuc = nuc.astype(np.float32)

    if use_membrane and membrane_idx < n:
        mem = channels[membrane_idx].astype(np.float32)
        img = np.stack([nuc, mem], axis=-1)   # (H,W,2)
        return img, "nucleus+membrane"
    return nuc, "nucleus_only"


# ---------------------------------------------------------------------------- #
# Cellpose segmentation
# ---------------------------------------------------------------------------- #
def run_cellpose(img, cfg=CELLPOSE):
    """Run Cellpose and return an int32 label mask. Defaults to Cellpose-SAM (4.x); classic 3.x can be selected."""
    from cellpose import models  # lazy import so this module can be imported without cellpose for other uses

    common = dict(
        flow_threshold=cfg["flow_threshold"],
        cellprob_threshold=cfg["cellprob_threshold"],
        normalize=cfg["normalize"],
        diameter=cfg["diameter"],
    )

    if cfg.get("use_classic", False):
        # classic Cellpose 3.x: needs model_type and channels
        model = models.Cellpose(gpu=cfg["use_gpu"], model_type=cfg["classic_model_type"])
        # classic models want channel order [cytoplasm, nucleus]; (H,W,2)=[nucleus,membrane] -> [membrane,nucleus]
        if img.ndim == 3 and img.shape[-1] == 2:
            img_in = img[..., ::-1]
        else:
            img_in = img
        out = model.eval(img_in, channels=cfg["classic_channels"],
                         batch_size=cfg["batch_size"], **common)
    else:
        # Cellpose-SAM 4.x: CellposeModel loads cpsam by default, no channels needed
        model = models.CellposeModel(gpu=cfg["use_gpu"])
        try:
            out = model.eval(img, batch_size=cfg["batch_size"], **common)
        except TypeError:
            # some versions differ in signature: retry without possibly unsupported kwargs
            out = model.eval(img, **common)

    masks = out[0] if isinstance(out, (list, tuple)) else out
    masks = np.squeeze(np.asarray(masks))
    return masks.astype(np.int32, copy=False)


# ---------------------------------------------------------------------------- #
# GT construction
# ---------------------------------------------------------------------------- #
def read_pixel_size(experiment_path, default=XENIUM_DEFAULT_PIXEL_SIZE):
    """Recursively find pixel_size (um/px) in experiment.xenium / experiment_roi.json (both JSON)."""
    if not experiment_path or not os.path.exists(experiment_path):
        warnings.warn(f"{experiment_path} not found, using default pixel_size {default}")
        return default
    try:
        with open(experiment_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        warnings.warn(f"failed to read {experiment_path} ({e}), using default pixel_size {default}")
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
    warnings.warn(f"no pixel_size field in {experiment_path}, using default {default}")
    return default


def _boundary_columns(df):
    """Locate the (cell_id, x, y) columns of the boundary table, tolerating common names."""
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[c] for c in ("cell_id", "label_id", "label", "id") if c in cols), None)
    # prefer local coordinates (ROI-relative); fall back to global ones
    x_col = next((cols[c] for c in ("vertex_x_local_um", "vertex_x_local", "vertex_x", "x", "x_location", "vertex_x_um") if c in cols), None)
    y_col = next((cols[c] for c in ("vertex_y_local_um", "vertex_y_local", "vertex_y", "y", "y_location", "vertex_y_um") if c in cols), None)
    return id_col, x_col, y_col


def rasterize_boundaries(parquet_path, out_shape, pixel_size_eff, offset=(0, 0), verbose=True):
    """Rasterize Xenium cell_boundaries.parquet (polygons in um) into an int32 label mask.

    out_shape        : (H, W), output mask size (the crop size when cropping)
    pixel_size_eff   : um per pixel at this level = base_pixel_size x downsample_factor
    offset           : (row0, col0), pixel coordinate of the crop's top-left corner in the full image; (0, 0) when not cropping
    pixel coordinate = um / pixel_size_eff - offset; out-of-bounds parts are clipped by skimage
    """
    row0, col0 = offset
    df = pd.read_parquet(parquet_path)
    id_col, x_col, y_col = _boundary_columns(df)
    if id_col is None or x_col is None or y_col is None:
        raise ValueError(f"{parquet_path}: cannot identify columns (cell_id/vertex_x/vertex_y), actual columns={list(df.columns)}")

    H, W = out_shape
    mask = np.zeros((H, W), dtype=np.int32)

    xs_all = df[x_col].to_numpy(dtype=np.float64) / pixel_size_eff - col0  # column (x), minus crop offset
    ys_all = df[y_col].to_numpy(dtype=np.float64) / pixel_size_eff - row0  # row (y), minus crop offset
    ids_all = df[id_col].to_numpy()

    # factorize assigns a cell code per row in order of appearance; a stable sort by
    # code then makes one cell's vertices contiguous (the stable sort preserves the
    # original vertex order within a cell = the polygon winding order). O(N logN),
    # avoiding a full scan per cell.
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
        rr, cc = sk_polygon(ys_s[s:e], xs_s[s:e], shape=(H, W))  # clipped to the image bounds automatically
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
    """GT and prediction share the same grid in theory (built from the same image). If rounding leaves a 1-2 px mismatch, pad zeros at the bottom-right."""
    if gt.shape == pred.shape:
        return gt, pred
    warnings.warn(f"GT{gt.shape} and Pred{pred.shape} shapes differ, padding zeros at the bottom-right (verify they are really aligned)")
    H = max(gt.shape[0], pred.shape[0])
    W = max(gt.shape[1], pred.shape[1])

    def _pad(m):
        out = np.zeros((H, W), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out
    return _pad(gt), _pad(pred)


def evaluate(gt_mask, pred_mask, dataset_name, gt_source,
             boundary_mode=BOUNDARY_MODE, iou_threshold=IOU_THRESHOLD):
    """Call the test.py metrics: GT in the reference slot, Cellpose in the pred slot; then rename to unambiguous fields."""
    gt_mask, pred_mask = align_shapes(gt_mask, pred_mask)

    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="Cellpose",
        cellpose_gt_mask=gt_mask,      # <- ground truth goes here
        pred_mask=pred_mask,          # <- Cellpose prediction goes here
        boundary_mode=boundary_mode,
        iou_threshold=iou_threshold,
    )

    result = {
        "dataset": dataset_name,
        "gt_source": gt_source,        # manual_labels / xenium_official_boundaries
        "method": "Cellpose",
        "gt_cell_count": raw.pop("cellpose_reference_cell_count"),  # renamed: this is the GT cell count
        "pred_cell_count": raw.pop("pred_cell_count"),
    }
    raw.pop("reference", None)         # drop the misleading "reference": "Cellpose"
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


def _run_dataset_impl(cfg, out_dir):
    name = cfg["name"]
    ddir = cfg["dataset_dir"]
    print(f"\n=== [{name}] {ddir} ===")

    # 1) read morphology channels + assemble the Cellpose input (still the full image here)
    morph = load_morphology_channels(ddir, pyramid_level=cfg.get("pyramid_level", 0),
                                     needed_indices=[NUCLEUS_IDX, MEMBRANE_IDX], read_crop=cfg.get("crop"))
    factor = morph["factor"]
    print(f"  channels={len(morph['channels'])}  names={morph['names']}  downsample factor={factor:.3f}")

    img_full, in_mode = assemble_cellpose_input(morph["channels"], use_membrane=cfg.get("use_membrane", True))
    Himg, Wimg = img_full.shape[:2]

    # decide the crop window: explicit crop=(y0,y1,x0,x1) wins; otherwise crop_size gives a centred square
    crop = cfg.get("crop")
    if crop is None and cfg.get("crop_size"):
        s = int(cfg["crop_size"])
        y0 = max((Himg - s) // 2, 0)
        x0 = max((Wimg - s) // 2, 0)
        crop = (y0, min(y0 + s, Himg), x0, min(x0 + s, Wimg))
        print(f"  crop_size={s} -> centred crop={crop} (full image {Himg}x{Wimg})")

    img = img_full if morph.get("crop_applied") else _apply_crop(img_full, crop)
    del img_full
    seg_shape = img.shape[:2]
    print(f"  Cellpose input: shape={img.shape} mode={in_mode}")

    # 2) Cellpose segmentation (prediction)
    print("  running Cellpose ...")
    segmentation_started = time.perf_counter()
    pred = run_cellpose(img, CELLPOSE)
    segmentation_seconds = time.perf_counter() - segmentation_started
    print(f"  predicted cells={len(metrics.get_instance_ids(pred))}  mask={pred.shape}")
    print(f"  Cellpose segmentation took {segmentation_seconds:.2f} s")

    # 3) build GT (same crop window as the prediction, so they align)
    if cfg["gt_type"] == "xenium_boundaries":
        exp_path = os.path.join(ddir, cfg.get("experiment", "experiment.xenium"))
        base_px = read_pixel_size(exp_path)
        px_eff = base_px * factor
        print(f"  pixel_size={base_px} µm/px  pixel_size_eff at this level={px_eff:.4f} µm/px")
        if crop is not None:
            y0, y1, x0, x1 = crop
            # rasterize the polygons straight into the "crop frame": subtract the
            # top-left offset and emit only the crop size (efficient: no full-slide
            # mask; correct: non-top-left windows do not shift)
            gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                      out_shape=(y1 - y0, x1 - x0),
                                      pixel_size_eff=px_eff, offset=(y0, x0))
        else:
            gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                      out_shape=seg_shape, pixel_size_eff=px_eff)
        gt_source = "xenium_official_boundaries"
    else:
        raise ValueError(f"unknown gt_type: {cfg['gt_type']}")
    print(f"  GT cells={len(metrics.get_instance_ids(gt))}  source={gt_source}")

    # 4) write masks (int32 tif, compatible with test.py's read_tif_mask, for reproducibility/cross-checks)
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, f"{name}_cellpose_pred.tif")
    gt_path = os.path.join(out_dir, f"{name}_gt.tif")
    tifffile.imwrite(pred_path, pred)
    tifffile.imwrite(gt_path, gt)
    print(f"  Saved {pred_path} / {gt_path}")

    # 5) metrics
    result = evaluate(gt, pred, dataset_name=name, gt_source=gt_source)
    result["segmentation_seconds"] = segmentation_seconds
    return result


def run_dataset(cfg, out_dir):
    """Run one dataset and attach runtime plus peak CPU/GPU resource stats."""
    monitor = _PerformanceMonitor(f"{cfg['name']}")
    monitor.start()
    try:
        result = _run_dataset_impl(cfg, out_dir)
    except Exception:
        monitor.stop()
        monitor.report()
        raise

    result.update(monitor.stop())
    per_csv = os.path.join(out_dir, f"{cfg['name']}_cellpose_metrics.csv")
    pd.DataFrame([result]).to_csv(per_csv, index=False)
    print(f"  metrics written to {per_csv}")
    monitor.report()
    return result


def _main_impl():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for cfg in DATASETS:
        try:
            rows.append(run_dataset(cfg, OUTPUT_DIR))
        except Exception as e:
            import traceback
            print(f"[ERROR] dataset {cfg['name']} failed: {e}")
            traceback.print_exc()

    if not rows:
        print("No dataset succeeded.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "cellpose_xenium_metrics.csv")
    df.to_csv(csv_path, index=False)

    pd.set_option("display.max_columns", None, "display.width", 200)
    print("\n================ summary ================")
    print(df.to_string(index=False))
    print("\nSaved:", csv_path)
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
