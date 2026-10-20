#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mesmer (DeepCell) x Xenium cell segmentation benchmark
=====================================================

Structure matches the Cellpose / CelloType versions: read morphology image -> Mesmer segmentation (prediction) -> build GT -> shape alignment -> call test.py metrics.
Two datasets:
  1) xenium_roi_crop/  -- GT = manual_labels.npy (manually annotated label mask)
  2) xenium_breast/    -- GT = label mask rasterized from the official cell_boundaries.parquet (by default only a centered 5k x 5k ROI)
Results are written to each dataset's own output folder.

Mesmer essentials (per DeepCell docs https://deepcell.readthedocs.io and the mesmer.py source)
-----------------------------------------------------------------------------------
- API: from deepcell.applications import Mesmer; app = Mesmer() (downloads pretrained weights on first use)
       labeled = app.predict(X, image_mpp=<um/px>, compartment='whole-cell')  -> (batch,H,W,1)
- Input X shape (batch, H, W, 2): **channel 0 = nucleus (DAPI), channel 1 = membrane/cytoplasm** (order matters).
- image_mpp is the true resolution of the input image (um/px). Mesmer's training resolution is app.model_mpp (=0.5);
  it resamples the input to 0.5 for inference and scales the labels back. So pass pixel_size x downsample factor,
  Xenium ~0.2125, **do not hard-code 0.5** (a wrong value costs a lot of accuracy). Mesmer normalizes
  internally by percentiles, so pass raw intensities.

Semantics (same as the previous two versions)
------------------------
test.py's evaluate_single_mask_against_cellpose(cellpose_gt_mask, pred_mask) takes the first slot as reference.
In this benchmark **Mesmer is the method under test**: ground truth in the first slot, Mesmer prediction in the
second, so the metrics direction is "evaluate Mesmer".

Dependencies: numpy, pandas, scikit-image, scipy, tifffile, pyarrow, deepcell(-tf)
      The first Mesmer() run downloads weights and may need a DeepCell API key (env DEEPCELL_ACCESS_TOKEN).
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
# Location of test.py (metric implementation)
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
# Config
# ============================================================================ #
XENIUM_DEFAULT_PIXEL_SIZE = 0.2125  # um/px; fallback when experiment cannot be read

# Global Mesmer params
MESMER = dict(
    compartment="whole-cell",   # 'whole-cell' | 'nuclear' | 'both' (GT is whole-cell -> whole-cell)
    batch_size=4,
    image_mpp=None,             # None = pixel_size x factor automatically; set a number to force a value
    # optional postprocessing, e.g. a stricter interior threshold: {'interior_threshold': 0.5}
    postprocess_kwargs_whole_cell=None,
    postprocess_kwargs_nuclear=None,
)

# Channel convention: Xenium morphology_focus channel 0 = DAPI (nucleus), channel 1 = membrane stain (ATP1A1/CD45/E-cad)
NUCLEUS_IDX = 0
MEMBRANE_IDX = 1

PARENT_OUT = "yourpath/Mesmer"

DATASETS = [
    dict(
        name="xenium_mouse_brain",
        dataset_dir="yourpath/dataset/xenium_mouse_brain/outs",  # note: path goes to outs/
        output_dir="yourpath/Mesmer/xenium_mouse_brain_output",
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
def _read_level(path, pyramid_level):
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        try:
            levels = list(series.levels)
        except Exception:
            levels = [series]
        L = int(np.clip(pyramid_level, 0, len(levels) - 1))
        if L != pyramid_level:
            warnings.warn(f"{os.path.basename(path)}: only {len(levels)} pyramid levels, "
                          f"pyramid_level={pyramid_level} truncated to {L}")
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
            warnings.warn(f"{dataset_dir}: morphology_focus/ not found, falling back to {cand} (nucleus channel only)")
            return dict(channels=chs, names=[cand], factor=factor)

    raise FileNotFoundError(f"{dataset_dir}: no morphology OME-TIFF found")


# ---------------------------------------------------------------------------- #
# Assemble the Mesmer input: (1, H, W, 2), channel 0 = nucleus, channel 1 = membrane
# ---------------------------------------------------------------------------- #
def assemble_mesmer_input(channels, use_membrane=True,
                          nucleus_idx=NUCLEUS_IDX, membrane_idx=MEMBRANE_IDX):
    """Return (X[1,H,W,2] float32, mode). Mesmer normalizes internally, so raw intensity is fine here."""
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
# Mesmer segmentation
# ---------------------------------------------------------------------------- #
def run_mesmer(X, image_mpp, cfg=MESMER):
    """Run Mesmer; return the int32 instance label mask [H,W] plus elapsed time and peak memory."""
    from deepcell.applications import Mesmer
    app = Mesmer()

    mpp = cfg["image_mpp"] if cfg.get("image_mpp") is not None else image_mpp

    predict_kwargs = dict(image_mpp=mpp, compartment=cfg["compartment"],
                          batch_size=cfg.get("batch_size", 4))
    if cfg.get("postprocess_kwargs_whole_cell"):
        predict_kwargs["postprocess_kwargs_whole_cell"] = cfg["postprocess_kwargs_whole_cell"]
    if cfg.get("postprocess_kwargs_nuclear"):
        predict_kwargs["postprocess_kwargs_nuclear"] = cfg["postprocess_kwargs_nuclear"]

    # record start time and memory
    process = psutil.Process()
    start_time = time.time()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    peak_mem = mem_before

    try:
        labeled = app.predict(X, **predict_kwargs)
    except TypeError:
        # some versions reject certain kwargs; fall back to the minimal call
        labeled = app.predict(X, image_mpp=mpp, compartment=cfg["compartment"])

    # record end time and peak memory
    end_time = time.time()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    peak_mem = max(peak_mem, mem_after)

    elapsed_time = end_time - start_time

    lab = np.asarray(labeled)
    # whole-cell / nuclear -> (1,H,W,1); both -> (1,H,W,2); take channel 0 = whole-cell
    if lab.ndim == 4:
        lab = lab[0, ..., 0]
    else:
        lab = np.squeeze(lab)

    return lab.astype(np.int32, copy=False), elapsed_time, peak_mem


# ---------------------------------------------------------------------------- #
# GT construction
# ---------------------------------------------------------------------------- #
def read_pixel_size(experiment_path, default=XENIUM_DEFAULT_PIXEL_SIZE):
    """Recursively find pixel_size (um/px) in experiment.xenium / experiment_roi.json (JSON)."""
    if not experiment_path or not os.path.exists(experiment_path):
        warnings.warn(f"{experiment_path} not found; using default pixel_size {default}")
        return default
    try:
        with open(experiment_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to read {experiment_path} ({e}); using default pixel_size {default}")
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
    """Find an existing experiment file: cfg['experiment'] -> experiment.xenium -> experiment_roi.json."""
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
    # prefer local coordinates (ROI-relative); use global ones only when absent
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
            print(f"    rasterizing {label} cells ...")
    if verbose:
        print(f"    rasterization done, {label} cells -> mask {mask.shape}")
    return mask


# ---------------------------------------------------------------------------- #
# Shape alignment + evaluation
# ---------------------------------------------------------------------------- #
def align_shapes(gt, pred):
    if gt.shape == pred.shape:
        return gt, pred
    warnings.warn(f"GT{gt.shape} and Pred{pred.shape} shapes differ; zero-padding bottom-right (check that they really align)")
    H = max(gt.shape[0], pred.shape[0])
    W = max(gt.shape[1], pred.shape[1])

    def _pad(m):
        out = np.zeros((H, W), dtype=m.dtype)
        out[:m.shape[0], :m.shape[1]] = m
        return out
    return _pad(gt), _pad(pred)


def evaluate(gt_mask, pred_mask, dataset_name, gt_source,
             boundary_mode=BOUNDARY_MODE, iou_threshold=IOU_THRESHOLD):
    """Call test.py metrics: GT in the reference slot, Mesmer in the pred slot; then use unambiguous field names."""
    gt_mask, pred_mask = align_shapes(gt_mask, pred_mask)

    raw = metrics.evaluate_single_mask_against_cellpose(
        method_name="Mesmer",
        cellpose_gt_mask=gt_mask,     # <- ground truth
        pred_mask=pred_mask,          # <- Mesmer prediction
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
# Per-dataset pipeline
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
    morph = load_morphology_channels(ddir, pyramid_level=cfg.get("pyramid_level", 0))
    factor = morph["factor"]
    channels = morph["channels"]
    Himg, Wimg = channels[0].shape
    print(f"  channels={len(channels)}  names={morph['names']}  downsample factor={factor:.3f}")

    # 2) decide the crop window (explicit crop wins, otherwise centered crop_size)
    crop = cfg.get("crop")
    if crop is None and cfg.get("crop_size"):
        s = int(cfg["crop_size"])
        y0 = max((Himg - s) // 2, 0)
        x0 = max((Wimg - s) // 2, 0)
        crop = (y0, min(y0 + s, Himg), x0, min(x0 + s, Wimg))
        print(f"  crop_size={s} -> centered crop={crop} (full image {Himg}x{Wimg})")

    if crop is not None:
        channels = [_apply_crop(c, crop) for c in channels]

    X, in_mode = assemble_mesmer_input(channels, use_membrane=cfg.get("use_membrane", True))
    seg_shape = X.shape[1:3]

    # 3) pixel_size -> image_mpp (needed for both datasets)
    exp_path = _find_experiment(ddir, cfg)
    base_px = read_pixel_size(exp_path)
    px_eff = base_px * factor
    print(f"  pixel_size={base_px} um/px  image_mpp at this level={px_eff:.4f} um/px")
    print(f"  Mesmer input: X.shape={X.shape} channel mode={in_mode} compartment={MESMER['compartment']}")

    # 4) Mesmer segmentation (prediction)
    print("  running Mesmer ...")
    pred, seg_time, peak_mem = run_mesmer(X, image_mpp=px_eff, cfg=MESMER)
    print(f"  predicted cell count={len(metrics.get_instance_ids(pred))}  mask={pred.shape}")
    print(f"  segmentation time: {seg_time:.2f} s")
    print(f"  peak memory: {peak_mem:.2f} MB")

    # 5) build GT (aligned with the same crop window as the prediction)
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
        raise ValueError(f"unknown gt_type: {cfg['gt_type']}")
    print(f"  GT cell count={len(metrics.get_instance_ids(gt))}  source={gt_source}")

    # 6) write masks (int32 tif, compatible with test.py's read_tif_mask)
    pred_path = os.path.join(out_dir, f"{name}_mesmer_pred.tif")
    gt_path = os.path.join(out_dir, f"{name}_gt.tif")
    tifffile.imwrite(pred_path, pred)
    tifffile.imwrite(gt_path, gt)
    print(f"  Saved {pred_path} / {gt_path}")

    # 7) metrics + per-dataset CSV
    result = evaluate(gt, pred, dataset_name=name, gt_source=gt_source)
    result["segmentation_time_sec"] = seg_time
    result["peak_memory_mb"] = peak_mem
    per_csv = os.path.join(out_dir, f"{name}_mesmer_metrics.csv")
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
        print("No dataset succeeded.")
        return

    df = pd.DataFrame(rows)
    combined = os.path.join(PARENT_OUT, "mesmer_xenium_metrics_all.csv")
    df.to_csv(combined, index=False)

    pd.set_option("display.max_columns", None, "display.width", 220)
    print("\n================ Summary ================")
    print(df.to_string(index=False))
    print("\nPer-dataset CSV under each output_dir; combined:", combined)
    return df


if __name__ == "__main__":
    main()
