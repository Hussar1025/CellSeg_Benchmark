#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StarDist x Xenium cell segmentation benchmark -- runs only on xenium_breast
==========================================================================

Same structure as the Cellpose / CelloType / Mesmer / CellSAM versions, but runs **only on xenium_breast**:
  GT = official cell_boundaries.parquet rasterized (by default only the centered 5000x5000 ROI).
Read morphology image -> StarDist nuclei -> skimage.expand_labels dilates nuclei by EXPAND_PX px to approximate
whole cells -> build GT -> align shapes -> call test.py metrics -> write to yourpath/stardist/xenium_breast.

StarDist notes (per official https://github.com/stardist/stardist and squidpy examples)
--------------------------------------------------------------------------------------
- Install: pip install stardist csbdeep (TensorFlow backend; GPU strongly recommended)
- API: from stardist.models import StarDist2D; from csbdeep.utils import normalize
       model = StarDist2D.from_pretrained('2D_versatile_fluo')   # fluorescence nucleus model, axes 'YX'
       img = normalize(dapi, 1, 99.8, axis=(0,1))                # must normalize
       labels, _ = model.predict_instances(img, n_tiles=...)      # tile large images; returns 2D int label
- Pretrained models: '2D_versatile_fluo'(fluorescence nuclei), '2D_versatile_he'(H&E nuclei),
  '2D_paper_dsb2018', '2D_demo'. Xenium is DAPI fluorescence -> use '2D_versatile_fluo'.
  Default thresholds prob~0.479, nms=0.3.

Nucleus -> whole-cell expansion and fairness note
-------------------------------------------------
StarDist segments only **nuclei** (takes the DAPI single channel). Following standard practice, this script
dilates nuclei outward by EXPAND_PX pixels via skimage.segmentation.expand_labels to approximate whole cells,
then scores against the whole-cell GT (cell_boundaries.parquet). This is the common "define cells by outward
nucleus expansion when no membrane is available" approach used by Xenium/squidpy. Stating it clearly keeps it
fair, but note:
  1) EXPAND_PX is a **hyperparameter** -- fix it to a justified value (prefer EXPAND_UM converted from
     pixel_size to ~a few um); never sweep it on the eval set to pick the best score; for transparency
     report the whole expand sweep curve.
  2) The method column is written as "StarDist(+Npx)" with a separate expand_px column -- this is a
     "nucleus + fixed expansion" pipeline, not the raw StarDist.
  3) Scope difference: membrane methods (Cellpose/CelloType/Mesmer/CellSAM) use real membrane signal to
     trace **irregular** cells, while expansion assumes cells are **isotropic ~round and uniform in size** --
     this is a method difference (not unfair), keep it in mind when interpreting the numbers.
Set EXPAND_PX to 0 and boundaries to "nucleus_boundaries.parquet" to go back to pure nuclear segmentation eval.

Semantics note (same as previous versions)
------------------------------------------
test.py's evaluate_single_mask_against_cellpose(cellpose_gt_mask, pred_mask) treats the first slot as reference.
In this benchmark **StarDist is the method under test**: ground truth goes in the first slot, StarDist
prediction in the second.

Dependencies: numpy, pandas, scikit-image, scipy, tifffile, pyarrow, stardist, csbdeep(+tensorflow)
"""

import os
import re
import sys
import json
import glob
import warnings
import importlib.util

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import polygon as sk_polygon


# ---------------------------------------------------------------------------- #
# location of test.py (metric implementation)
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
            "test.py not found: set TEST_PY_PATH at the top of this script to the absolute path of "
            "test.py, or place test.py in the same directory as this script."
        ) from e


metrics = _load_metrics()


# ============================================================================ #
# Config
# ============================================================================ #
XENIUM_DEFAULT_PIXEL_SIZE = 0.2125  # um/px, fallback when experiment cannot be read

# StarDist global params
STARDIST = dict(
    model_name="2D_versatile_fluo",  # fluorescence nucleus model
    norm_low=1.0,                    # normalize percentile lower bound
    norm_high=99.8,                  # normalize percentile upper bound
    prob_thresh=None,                # None=model default (~0.479); lower -> more nuclei
    nms_thresh=None,                 # None=model default (0.3)
    n_tiles=None,                    # None=auto tile by GPU memory; or set manually e.g. (4,4)
)

# Nucleus -> whole-cell expansion (skimage.segmentation.expand_labels)
# expand_labels is a distance-transform-based Voronoi-style dilation: each nucleus grows into the
# background, stops where it meets neighbors, never overlaps, and each nucleus grows at most
# expand_px pixels; dilation does not change the instance count, only enlarges each instance's extent.
EXPAND_PX = 15       # dilation distance (px). Hyperparameter: fix to a justified value, do not tune on the eval set. 0 = no dilation
EXPAND_UM = None     # if not None, overrides EXPAND_PX with round(EXPAND_UM / pixel_size) (more justified, stable across resolutions)

# Channel convention: Xenium morphology_focus channel 0 = DAPI (nucleus)
NUCLEUS_IDX = 0

PARENT_OUT = "yourpath/stardist"

DATASETS = [
    dict(
        name="xenium_mouse_brain",
        dataset_dir="yourpath/dataset/xenium_mouse_brain/outs",  # note: path points to outs/
        output_dir="yourpath/stardist/xenium_mouse_brain_output",
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
# Morphology image reading (pyramid level + two naming schemes + single/multi-channel files)
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
            warnings.warn(f"{os.path.basename(path)}: only {len(levels)} pyramid levels, "
                          f"pyramid_level={pyramid_level} clamped to {L}")
        level = levels[L]
        if len(tf.pages) == 1:
            # One physical 2D page per Xenium channel file; SizeC is virtual.
            arr = np.asarray(tf.pages[0].asarray())
        elif len(level.shape) == 3 and level.shape[0] > 5:
            arr = np.asarray(level.pages[level.shape[0] // 2].asarray())
        else:
            arr = np.asarray(level.asarray())
        if crop is not None:
            y0, y1, x0, x1 = crop
            arr = arr[:, y0:y1, x0:x1] if arr.ndim == 3 and arr.shape[0] <= 5 else arr[y0:y1, x0:x1]
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


def load_morphology_channels(dataset_dir, pyramid_level=0, read_crop=None):
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
        files = [fp for fp in files if _idx(fp) == 0]

        factor = 1.0
        for fp in files:
            arr, factor = _read_level(fp, pyramid_level, crop=read_crop)
            for c in _split_channels(arr):
                channels.append(c)
                names.append(os.path.splitext(os.path.basename(fp))[0])
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
            warnings.warn(f"{dataset_dir}: morphology_focus/ not found, falling back to {cand} (nucleus channel only)")
            return dict(channels=chs, names=[cand], factor=factor, crop_applied=read_crop is not None)

    raise FileNotFoundError(f"{dataset_dir}: no morphology OME-TIFF found")


# ---------------------------------------------------------------------------- #
# StarDist segmentation (nuclei)
# ---------------------------------------------------------------------------- #
def tensorflow_gpu_memory_stats(reset_peak=False):
    """Return GPU memory stats (MB) for TensorFlow-visible GPUs.

    ``peak`` is the true high-water mark of the TensorFlow GPU allocator since the last reset,
    not based on timed sampling; returns an empty list when there is no CUDA / TensorFlow-recognized GPU.
    """
    try:
        import tensorflow as tf
        devices = tf.config.list_logical_devices("GPU")
        stats = []
        for i, _ in enumerate(devices):
            device_name = f"GPU:{i}"
            if reset_peak:
                tf.config.experimental.reset_memory_stats(device_name)
            info = tf.config.experimental.get_memory_info(device_name)
            stats.append(dict(
                device=device_name,
                current_mb=info["current"] / 1024 ** 2,
                peak_mb=info["peak"] / 1024 ** 2,
            ))
        return stats
    except Exception as e:
        # Some CPU-only TensorFlow or older CUDA runtimes lack this API; does not affect segmentation.
        warnings.warn(f"cannot read TensorFlow GPU memory stats: {e}")
        return []


def _format_gpu_memory_stats(stats):
    if not stats:
        return "no TensorFlow-visible GPU detected (StarDist will use CPU, or this environment lacks memory query support)"
    return "; ".join(
        f"{s['device']} current={s['current_mb']:.1f} MB, peak={s['peak_mb']:.1f} MB"
        for s in stats
    )


def run_stardist(dapi, cfg=STARDIST):
    """Run StarDist ('2D_versatile_fluo' fluorescence nucleus model), return int32 instance label mask [H,W]."""
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize

    # StarDist runs through TensorFlow: reset counters here so the logged peak belongs only to this segmentation.
    gpu_before = tensorflow_gpu_memory_stats(reset_peak=True)
    print(f"  [GPU] StarDist before GPU memory: {_format_gpu_memory_stats(gpu_before)}")

    model = StarDist2D.from_pretrained(cfg["model_name"])
    img = normalize(dapi.astype(np.float32), cfg.get("norm_low", 1.0),
                    cfg.get("norm_high", 99.8), axis=(0, 1))

    n_tiles = cfg.get("n_tiles")
    if n_tiles is None:
        try:
            n_tiles = model._guess_n_tiles(img)   # auto tile by GPU memory
        except Exception:
            n_tiles = None

    try:
        labels, _ = model.predict_instances(
            img, prob_thresh=cfg.get("prob_thresh"),
            nms_thresh=cfg.get("nms_thresh"), n_tiles=n_tiles)
    finally:
        # Keep the high-water mark up to the failure point even if inference throws, to help debug CUDA OOM.
        gpu_after = tensorflow_gpu_memory_stats()
        print(f"  [GPU] StarDist after GPU memory: {_format_gpu_memory_stats(gpu_after)}")
    return np.asarray(labels).astype(np.int32, copy=False)


def expand_nuclei_to_cells(nuclei, expand_px):
    """Dilate nucleus labels outward by expand_px px via skimage.segmentation.expand_labels to approximate
    whole cells. Each nucleus grows into the background, stops on contact, never overlaps, grows at most
    expand_px; instance count is unchanged. expand_px<=0 returns as-is."""
    if expand_px is None or expand_px <= 0:
        return np.asarray(nuclei).astype(np.int32, copy=False)
    from skimage.segmentation import expand_labels
    return expand_labels(np.asarray(nuclei), distance=expand_px).astype(np.int32)


# ---------------------------------------------------------------------------- #
# GT construction (official boundary rasterization)
# ---------------------------------------------------------------------------- #
def read_pixel_size(experiment_path, default=XENIUM_DEFAULT_PIXEL_SIZE):
    """Recursively find pixel_size (um/px) in experiment.xenium (JSON)."""
    if not experiment_path or not os.path.exists(experiment_path):
        warnings.warn(f"{experiment_path} not found, using default pixel_size {default}")
        return default
    try:
        with open(experiment_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        warnings.warn(f"read {experiment_path} failed ({e}), using default pixel_size {default}")
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
    # Prefer local coordinates (ROI-relative); fall back to global if absent
    x_col = next((cols[c] for c in ("vertex_x_local_um", "vertex_x_local", "vertex_x", "x", "x_location", "vertex_x_um") if c in cols), None)
    y_col = next((cols[c] for c in ("vertex_y_local_um", "vertex_y_local", "vertex_y", "y", "y_location", "vertex_y_um") if c in cols), None)
    return id_col, x_col, y_col


def rasterize_boundaries(parquet_path, out_shape, pixel_size_eff, offset=(0, 0), verbose=True):
    """Rasterize Xenium boundary parquet (micron polygons) into an int32 label mask.
    pixel = micron / pixel_size_eff - offset; out-of-bounds parts are clipped by skimage automatically."""
    row0, col0 = offset
    df = pd.read_parquet(parquet_path)
    id_col, x_col, y_col = _boundary_columns(df)
    if id_col is None or x_col is None or y_col is None:
        raise ValueError(f"{parquet_path}: cannot identify columns (cell_id/vertex_x/vertex_y), actual columns={list(df.columns)}")

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
        print(f"    rasterization done, {label} cells -> mask {mask.shape}")
    return mask


# ---------------------------------------------------------------------------- #
# Shape alignment + evaluation
# ---------------------------------------------------------------------------- #
def align_shapes(gt, pred):
    if gt.shape == pred.shape:
        return gt, pred
    warnings.warn(f"GT{gt.shape} and Pred{pred.shape} shape mismatch, zero-pad at bottom-right to align (verify true alignment)")
    H = max(gt.shape[0], pred.shape[0])
    W = max(gt.shape[1], pred.shape[1])

    def _pad(m):
        out = np.zeros((H, W), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out
    return _pad(gt), _pad(pred)


def evaluate(gt_mask, pred_mask, dataset_name, gt_source,
             boundary_mode=BOUNDARY_MODE, iou_threshold=IOU_THRESHOLD):
    """Call test.py metrics: GT in reference slot, StarDist in pred slot; then rename to unambiguous field names."""
    gt_mask, pred_mask = align_shapes(gt_mask, pred_mask)

    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="StarDist",
        cellpose_gt_mask=gt_mask,     # <- ground truth
        pred_mask=pred_mask,          # <- StarDist prediction
        boundary_mode=boundary_mode,
        iou_threshold=iou_threshold,
    )

    result = {
        "dataset": dataset_name,
        "gt_source": gt_source,
        "method": "StarDist",
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


def run_dataset(cfg):
    name = cfg["name"]
    ddir = cfg["dataset_dir"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== [{name}] {ddir}  ->  {out_dir} ===")

    # 1) read morphology channels (full image)
    morph = load_morphology_channels(ddir, pyramid_level=cfg.get("pyramid_level", 0),
                                     read_crop=cfg.get("crop"))
    factor = morph["factor"]
    channels = morph["channels"]
    Himg, Wimg = channels[0].shape
    print(f"  channels={len(channels)}  names={morph['names']}  downsample factor={factor:.3f}")

    # 2) decide crop window (explicit crop first, otherwise centered crop_size)
    crop = cfg.get("crop")
    if crop is None and cfg.get("crop_size"):
        s = int(cfg["crop_size"])
        y0 = max((Himg - s) // 2, 0)
        x0 = max((Wimg - s) // 2, 0)
        crop = (y0, min(y0 + s, Himg), x0, min(x0 + s, Wimg))
        print(f"  crop_size={s} -> centered crop={crop} (full image {Himg}x{Wimg})")

    if crop is not None and not morph.get("crop_applied"):
        channels = [_apply_crop(c, crop) for c in channels]

    # StarDist uses only the DAPI (nucleus) single channel
    dapi = (channels[NUCLEUS_IDX] if NUCLEUS_IDX < len(channels) else channels[0])
    seg_shape = dapi.shape
    print(f"  StarDist input: DAPI {dapi.shape}  model={STARDIST['model_name']} (nuclear segmentation)")

    # 3) pixel_size (for rasterizing GT)
    exp_path = _find_experiment(ddir, cfg)
    base_px = read_pixel_size(exp_path)
    px_eff = base_px * factor
    print(f"  pixel_size={base_px} um/px  current-level effective pixel_size={px_eff:.4f} um/px")

    # 4) StarDist segmentation (nuclei) -> expansion approximates whole cells
    print("  running StarDist ...")
    pred_nuc = run_stardist(dapi, cfg=STARDIST)
    print(f"  predicted nuclei count={len(metrics.get_instance_ids(pred_nuc))}  mask={pred_nuc.shape}")
    expand_px = EXPAND_PX
    if EXPAND_UM is not None:
        expand_px = int(round(EXPAND_UM / px_eff))
        print(f"  from EXPAND_UM={EXPAND_UM}um / pixel_size={px_eff:.4f} -> expand_px={expand_px}")
    pred = expand_nuclei_to_cells(pred_nuc, expand_px)
    if expand_px and expand_px > 0:
        print(f"  nuclei dilated {expand_px}px to approximate whole cells (expand_labels) -> instances={len(metrics.get_instance_ids(pred))}")
    else:
        print("  expand_px<=0: no dilation, use nucleus mask directly")

    # 5) build GT (official boundary rasterized, aligned to the same crop window as the prediction)
    if crop is not None:
        y0, y1, x0, x1 = crop
        gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                  out_shape=(y1 - y0, x1 - x0),
                                  pixel_size_eff=px_eff, offset=(y0, x0))
    else:
        gt = rasterize_boundaries(os.path.join(ddir, cfg["boundaries"]),
                                  out_shape=seg_shape, pixel_size_eff=px_eff)
    gt_source = os.path.splitext(cfg["boundaries"])[0]  # cell_boundaries / nucleus_boundaries
    print(f"  GT count={len(metrics.get_instance_ids(gt))}  source={gt_source}")
    if gt_source.startswith("cell") and (not expand_px or expand_px <= 0):
        print("  warning: no dilation but whole-cell GT: nuclei are smaller than whole cells, coverage will be low; "
              "set EXPAND_PX>0 to approximate whole cells, or set boundaries to nucleus_boundaries.parquet for nuclear eval")

    # 6) save masks (int32 tif, compatible with test.py's read_tif_mask)
    pred_path = os.path.join(out_dir, f"{name}_stardist_pred.tif")
    gt_path = os.path.join(out_dir, f"{name}_gt.tif")
    tifffile.imwrite(pred_path, pred)
    tifffile.imwrite(gt_path, gt)
    print(f"  Saved {pred_path} / {gt_path}")

    # 7) metrics + per-dataset CSV
    result = evaluate(gt, pred, dataset_name=name, gt_source=gt_source)
    result["method"] = f"StarDist(+{expand_px}px)" if (expand_px and expand_px > 0) else "StarDist(nuclei)"
    result["expand_px"] = int(expand_px or 0)
    per_csv = os.path.join(out_dir, f"{name}_stardist_metrics.csv")
    pd.DataFrame([result]).to_csv(per_csv, index=False)
    print(f"  metrics written to {per_csv}")
    return result


def main():
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
        print("no successful datasets.")
        return

    df = pd.DataFrame(rows)
    combined = os.path.join(PARENT_OUT, "stardist_xenium_metrics_all.csv")
    df.to_csv(combined, index=False)

    pd.set_option("display.max_columns", None, "display.width", 220)
    print("\n================ Summary ================")
    print(df.to_string(index=False))
    print("\nper-dataset CSV in each output_dir; summary:", combined)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="StarDist(nuclei)+expansion x Xenium breast benchmark")
    ap.add_argument("--expand-px", type=int, default=None, help="pixels to dilate nuclei outward, overrides EXPAND_PX (0=no dilation)")
    ap.add_argument("--expand-um", type=float, default=None, help="microns to dilate nuclei outward, converted via pixel_size, overrides EXPAND_PX")
    ap.add_argument("--prob-thresh", type=float, default=None, help="StarDist probability threshold, overrides default")
    ap.add_argument("--nms-thresh", type=float, default=None, help="StarDist NMS threshold, overrides default")
    a, _ = ap.parse_known_args()
    if a.expand_px is not None:
        EXPAND_PX = a.expand_px
    if a.expand_um is not None:
        EXPAND_UM = a.expand_um
    if a.prob_thresh is not None:
        STARDIST["prob_thresh"] = a.prob_thresh
    if a.nms_thresh is not None:
        STARDIST["nms_thresh"] = a.nms_thresh
    main()
