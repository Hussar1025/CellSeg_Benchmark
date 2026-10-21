#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CelloType x Xenium cell segmentation benchmark
==============================================
Same structure as the Cellpose version: read morphology -> CelloType segmentation
(prediction) -> build GT -> shape alignment -> metrics from test.py.
Two datasets:
  1) xenium_roi_crop/  -- GT = manual_labels.npy (manual label mask)
  2) xenium_breast/    -- GT = label mask rasterized from official cell_boundaries.parquet
                          (by default only the centered 5k x 5k ROI runs)
Results go to each dataset's output folder.

CelloType notes (per official README / https://cellotype.readthedocs.io)
-----------------------------------------------------------------------
- API: from cellotype.predict import CelloTypePredictor
       CelloTypePredictor(model_path, config_path, confidence_thresh, max_det, device).predict(img) -> [H,W]
- Input is an [H,W,3] **uint8** image; channel convention: blue(B)=nuclear(DAPI),
  green(G)=membrane, red(R)=optional.
- Large/dense images must use SAHI sliced inference (MaskDINO has a per-forward
  query cap of ~300, otherwise heavy under-detection). Official large-image flow:
  Detectron2DetectionModel + get_sliced_prediction (512 tiles, 5% overlap,
  perform_standard_pred=False), cross-tile NMS merge; then write each object's
  full-image bool_mask as an instance label.

Semantics (identical to the Cellpose version)
---------------------------------------------
test.py's evaluate_single_mask_against_cellpose(cellpose_gt_mask, pred_mask) treats
the first slot as reference. Here CelloType is the method under test: ground truth
goes in slot 1, the CelloType prediction in slot 2, so the metric direction is
"evaluate CelloType". Output field names are renamed to unambiguous ones.

Dependencies: numpy, pandas, scikit-image, scipy, tifffile, pyarrow, Pillow,
      cellotype (+Detectron2+Deformable-DETR+MaskDINO), sahi
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
except ImportError:  # no resource on Windows etc.
    resource = None

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import polygon as sk_polygon


# Performance monitoring
def _current_memory_mb():
    """Return current process RSS (MB); prefer /proc, fall back to psutil."""
    try:
        with open("yourpath", "r", encoding="utf-8") as fh:
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
    """Return historical peak RSS (MB); Linux ru_maxrss is in KiB."""
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
    """Reset PyTorch CUDA peak stats; silently skip when CUDA/torch unavailable."""
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
    """Read PyTorch peak VRAM (MB) across visible GPUs; None when unavailable."""
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
    """Record wall time, peak CPU RSS and peak CUDA VRAM for a run."""

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
            f"[Performance:{self.label}] total runtime: "
            f"{stats['runtime_seconds']:.2f} s ({stats['runtime_seconds'] / 60:.2f} min)"
        )
        print(
            f"[Performance:{self.label}] CPU RSS: initial "
            f"{stats['initial_memory_mb']:.2f} MB, final "
            f"{stats['final_memory_mb']:.2f} MB, peak "
            f"{stats['peak_memory_mb']:.2f} MB"
        )
        allocated = stats["peak_gpu_memory_allocated_mb"]
        reserved = stats["peak_gpu_memory_reserved_mb"]
        if allocated is None and reserved is None:
            print(f"[Performance:{self.label}] peak GPU VRAM: N/A")
        else:
            print(
                f"[Performance:{self.label}] peak GPU VRAM: allocated="
                f"{allocated if allocated is not None else float('nan'):.2f} MB, "
                f"reserved={reserved if reserved is not None else float('nan'):.2f} MB"
            )


# test.py (metric implementation) location: points at the eval test.py;
# falls back to test.py in the script directory / PYTHONPATH.
TEST_PY_PATH = "yourpath"


def _load_metrics():
    if TEST_PY_PATH and os.path.exists(TEST_PY_PATH):
        spec = importlib.util.spec_from_file_location("xenium_metrics", TEST_PY_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # test.py's __main__ block does not run (module name is not __main__)
        return mod
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import test as mod
        return mod
    except ImportError as e:
        raise ImportError(
            "cannot find test.py: point TEST_PY_PATH at the absolute path of test.py, "
            "or put test.py in the same directory as this script."
        ) from e


metrics = _load_metrics()  # metric implementation from test.py


# ============================================================================ #
# Config
# ============================================================================ #
XENIUM_DEFAULT_PIXEL_SIZE = 0.2125  # um/px, fallback when experiment is unreadable

# CelloType global parameters
CELLOTYPE = dict(
    device="cuda",
    # point these at the weights and config downloaded by download.sh
    model_path="yourpath",
    config_path="yourpath",
    confidence_thresh=0.3,
    max_det=1000,            # only used by simple mode; dense images need sahi
    # percentiles used to stretch channels to uint8 (Xenium uint16 has a wide
    # dynamic range and must be stretched)
    pct_low=1.0,
    pct_high=99.0,
    # SAHI sliced inference (large/dense images, official large-image flow)
    sahi_slice=512,
    sahi_overlap=0.05,
    sahi_image_size=512,     # lower together with slice to 256 if VRAM is tight
)

# Channel convention: Xenium morphology_focus channel 0 = DAPI (nuclear),
# channel 1 = membrane stain (ATP1A1/CD45/E-cad)
NUCLEUS_IDX = 0
MEMBRANE_IDX = 1

PARENT_OUT = "yourpath"   # combined CSV goes here; per-dataset folders below

DATASETS = [
    dict(
        name="xenium_mouse_brain",
        dataset_dir="yourpath",  # note: path points to outs/
        output_dir="yourpath",
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


# Morphology image reading (supports pyramid levels + both naming schemes + single/multi-channel files)
def _read_level(path, pyramid_level, crop=None):
    """Read the given pyramid level of an OME-TIFF; returns (array, downsample_factor)."""
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
    """Split an array into a channel list. 2D -> single channel; 3D -> channel axis by the small axis (<=5)."""
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[0] <= 5 and arr.shape[0] <= arr.shape[-1]:
            return [arr[i] for i in range(arr.shape[0])]        # (C,H,W)
        if arr.shape[-1] <= 5:
            return [arr[..., i] for i in range(arr.shape[-1])]  # (H,W,C)
        return [arr[arr.shape[0] // 2]]                          # likely a z-stack; take the middle plane
    arr = np.squeeze(arr)
    return _split_channels(arr) if arr.ndim <= 3 else [np.max(arr, axis=tuple(range(arr.ndim - 2)))]


def load_morphology_channels(dataset_dir, pyramid_level=0, needed_indices=None, read_crop=None):
    """Discover and read morphology channels; supports ch0000_dapi... and
    morphology_focus_0000... naming as well as multi-channel single files.

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

    for cand in ("morphology.ome.tif", "morphology_focus.ome.tif", "morphology_mip.ome.tif"):
        fp = os.path.join(dataset_dir, cand)
        if os.path.exists(fp):
            arr, factor = _read_level(fp, pyramid_level, crop=read_crop)
            chs = _split_channels(arr)
            if len(chs) > 1:
                chs = [np.max(np.stack(chs, 0), 0)]
            warnings.warn(f"{dataset_dir}: morphology_focus/ not found; falling back to {cand} (nuclear channel only)")
            return dict(channels=chs, names=[cand], factor=factor, crop_applied=read_crop is not None)

    raise FileNotFoundError(f"{dataset_dir}: no morphology OME-TIFF found")


# Assemble CelloType input (uint8 RGB: B=nuclear, G=membrane, R=optional)
def _to_uint8(ch, p_low, p_high):
    """Stretch a single channel to uint8 [0,255] by percentile. Xenium uint16 has a wide dynamic range and must be stretched."""
    ch = ch.astype(np.float32)
    lo, hi = np.percentile(ch, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    ch = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return (ch * 255.0).astype(np.uint8)


def assemble_cellotype_rgb(channels, use_membrane=True,
                           nucleus_idx=NUCLEUS_IDX, membrane_idx=MEMBRANE_IDX,
                           p_low=1.0, p_high=99.0):
    """Build the uint8 RGB CelloType needs: blue=nuclear(DAPI), green=membrane, red=0.

    Returns (rgb[H,W,3] uint8, mode). Without a membrane channel, or with
    use_membrane=False, the green channel is 0 (nuclear only).
    """
    n = len(channels)
    nuc = channels[nucleus_idx] if nucleus_idx < n else channels[0]
    H, W = nuc.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :, 2] = _to_uint8(nuc, p_low, p_high)                    # B = nuclear

    if use_membrane and membrane_idx < n:
        rgb[:, :, 1] = _to_uint8(channels[membrane_idx], p_low, p_high)  # G = membrane
        mode = "nucleus(B)+membrane(G)"
    else:
        mode = "nucleus(B)_only"
    # Note: to put transcript density in green with a Xenium-specific checkpoint,
    # replace rgb[:,:,1] here.
    return rgb, mode


# CelloType segmentation
def _mask_from_sahi(result, hw):
    """Write each SAHI object's full-image bool_mask as an int32 instance label (0=background)."""
    H, W = hw
    out = np.zeros((H, W), dtype=np.int32)
    obj_list = result.object_prediction_list
    for i, obj in enumerate(obj_list, start=1):
        bm = np.asarray(obj.mask.bool_mask)   # this object's bool mask in full-image coordinates
        if bm.shape != (H, W):
            # some SAHI versions give a different bool_mask size; warn and skip instead of crashing
            warnings.warn(f"SAHI object {i} bool_mask shape {bm.shape} != {(H, W)}; skipped")
            continue
        out[bm] = i
    return out


def _run_cellotype_sahi(rgb, cfg):
    """SAHI sliced inference (large/dense images). Mirrors the official sahi_inference.py."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # disable PIL's decompression-bomb guard

    # prefer CelloType's own wrapper (matches the official example); fall back to plain sahi
    try:
        from cellotype.predict import Detectron2DetectionModel
    except Exception:
        from sahi.models.detectron2 import Detectron2DetectionModel
    try:
        from sahi_predict import get_sliced_prediction        # script at the CelloType repo root
    except Exception:
        from sahi.predict import get_sliced_prediction        # fall back to plain sahi

    det = Detectron2DetectionModel(
        model_path=cfg["model_path"],
        config_path=cfg["config_path"],
        confidence_threshold=cfg["confidence_thresh"],
        image_size=cfg["sahi_image_size"],
        device=cfg["device"],
        channels=3,
    )

    # get_sliced_prediction is most stable on a file path (as in the official example);
    # write the RGB to a temp PNG
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
            perform_standard_pred=False,   # slice only, no whole-image forward (avoids query-cap under-detection)
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return _mask_from_sahi(result, rgb.shape[:2])


def run_cellotype(rgb, cfg):
    """Run CelloType; returns an int32 instance label mask. inference='sahi' (default) or 'simple'."""
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
        raise ValueError(f"unknown inference mode: {mode}")


# GT construction
def read_pixel_size(experiment_path, default=XENIUM_DEFAULT_PIXEL_SIZE):
    """Recursively find pixel_size (um/px) in experiment.xenium (JSON)."""
    if not experiment_path or not os.path.exists(experiment_path):
        warnings.warn(f"{experiment_path} not found; pixel_size falls back to {default}")
        return default
    try:
        with open(experiment_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        warnings.warn(f"failed to read {experiment_path} ({e}); pixel_size falls back to {default}")
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


def _boundary_columns(df):
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[c] for c in ("cell_id", "label_id", "label", "id") if c in cols), None)
    # prefer local coordinates (ROI-relative); use global only as a fallback
    x_col = next((cols[c] for c in ("vertex_x_local_um", "vertex_x_local", "vertex_x", "x", "x_location", "vertex_x_um") if c in cols), None)
    y_col = next((cols[c] for c in ("vertex_y_local_um", "vertex_y_local", "vertex_y", "y", "y_location", "vertex_y_um") if c in cols), None)
    return id_col, x_col, y_col


def rasterize_boundaries(parquet_path, out_shape, pixel_size_eff, offset=(0, 0), verbose=True):
    """Rasterize Xenium cell_boundaries.parquet (um-coordinate polygons) into an int32 label mask.

    out_shape      : (H, W) output size (crop size when cropping)
    pixel_size_eff : um per pixel at this level = base_pixel_size x downsample_factor
    offset         : (row0, col0) of the crop window's top-left in full-image pixel
                     coordinates; (0,0) when not cropping
    pixel = um / pixel_size_eff - offset; out-of-bounds parts are clipped by skimage
    """
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

    codes, uniq = pd.factorize(ids_all)          # encode in order of appearance
    n_cells = len(uniq)
    sort_idx = np.argsort(codes, kind="stable")  # keeps each cell's vertices contiguous and in winding order
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
        print(f"    rasterization done, {label} cells -> mask {mask.shape}")
    return mask

# Shape alignment + evaluation
def align_shapes(gt, pred):
    if gt.shape == pred.shape:
        return gt, pred
    warnings.warn(f"GT{gt.shape} and Pred{pred.shape} differ; zero-padding bottom-right (verify alignment)")
    H = max(gt.shape[0], pred.shape[0])
    W = max(gt.shape[1], pred.shape[1])

    def _pad(m):
        out = np.zeros((H, W), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out
    return _pad(gt), _pad(pred)


def evaluate(gt_mask, pred_mask, dataset_name, gt_source,
             boundary_mode=BOUNDARY_MODE, iou_threshold=IOU_THRESHOLD):
    """Call test.py metrics: GT in the reference slot, CelloType in the pred slot; then rename fields unambiguously."""
    gt_mask, pred_mask = align_shapes(gt_mask, pred_mask)

    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="CelloType",
        cellpose_gt_mask=gt_mask,     # ground truth
        pred_mask=pred_mask,          # CelloType prediction
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


# Per-dataset pipeline
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

    # 1) read morphology channels + assemble CelloType RGB (full image)
    morph = load_morphology_channels(ddir, pyramid_level=cfg.get("pyramid_level", 0),
                                     needed_indices=[NUCLEUS_IDX, MEMBRANE_IDX], read_crop=cfg.get("crop"))
    factor = morph["factor"]
    print(f"  channels={len(morph['channels'])}  names={morph['names']}  downsample factor={factor:.3f}")

    rgb_full, in_mode = assemble_cellotype_rgb(
        morph["channels"], use_membrane=cfg.get("use_membrane", True),
        p_low=CELLOTYPE["pct_low"], p_high=CELLOTYPE["pct_high"])
    Himg, Wimg = rgb_full.shape[:2]

    # crop window: explicit crop first; otherwise center a square if crop_size is set
    crop = cfg.get("crop")
    if crop is None and cfg.get("crop_size"):
        s = int(cfg["crop_size"])
        y0 = max((Himg - s) // 2, 0)
        x0 = max((Wimg - s) // 2, 0)
        crop = (y0, min(y0 + s, Himg), x0, min(x0 + s, Wimg))
        print(f"  crop_size={s} -> centered crop={crop} (full image {Himg}x{Wimg})")

    rgb = rgb_full if morph.get("crop_applied") else _apply_crop(rgb_full, crop)
    del rgb_full
    seg_shape = rgb.shape[:2]
    print(f"  CelloType input: shape={rgb.shape} channel mode={in_mode} inference mode={cfg.get('inference', 'sahi')}")

    # 2) CelloType segmentation (prediction)
    print("  running CelloType ...")
    segmentation_started = time.perf_counter()
    pred = run_cellotype(rgb, CELLOTYPE)
    segmentation_seconds = time.perf_counter() - segmentation_started
    print(f"  predicted cells={len(metrics.get_instance_ids(pred))}  mask={pred.shape}")
    print(f"  CelloType segmentation took {segmentation_seconds:.2f} s")

    # 3) build GT (aligned to the same crop window as the prediction)
    if cfg["gt_type"] == "xenium_boundaries":
        exp_path = os.path.join(ddir, cfg.get("experiment", "experiment.xenium"))
        base_px = read_pixel_size(exp_path)
        px_eff = base_px * factor
        print(f"  pixel_size={base_px} um/px  pixel_size_eff at this level={px_eff:.4f} um/px")
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
        raise ValueError(f"unknown gt_type: {cfg['gt_type']}")
    print(f"  GT cells={len(metrics.get_instance_ids(gt))}  source={gt_source}")

    # 4) write masks (int32 tif, compatible with test.py's read_tif_mask)
    pred_path = os.path.join(out_dir, f"{name}_cellotype_pred.tif")
    gt_path = os.path.join(out_dir, f"{name}_gt.tif")
    tifffile.imwrite(pred_path, pred)
    tifffile.imwrite(gt_path, gt)
    print(f"  saved {pred_path} / {gt_path}")

    # 5) metrics + per-dataset CSV
    result = evaluate(gt, pred, dataset_name=name, gt_source=gt_source)
    result["segmentation_seconds"] = segmentation_seconds
    return result


def run_dataset(cfg):
    """Run one dataset and attach runtime plus CPU/GPU peak resource stats."""
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
        print("no dataset succeeded.")
        return

    df = pd.DataFrame(rows)
    combined = os.path.join(PARENT_OUT, "cellotype_xenium_metrics_all.csv")
    df.to_csv(combined, index=False)

    pd.set_option("display.max_columns", None, "display.width", 220)
    print("\n================ summary ================")
    print(df.to_string(index=False))
    print("\nPer-dataset CSVs are in each output_dir; combined:", combined)
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