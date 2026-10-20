"""
merfish_run_stardist.py
=======================
StarDist segmentation + benchmark on MERFISH (Vizgen mouse brain Receptor Map).
Structure matches merfish_seg_benchmark.py (the Cellpose one), runs standalone, and can also just produce
label masks to feed into merfish_seg_benchmark.py's EXTRA_METHOD_MASKS.

Important: StarDist is a "nucleus" segmentation model (2D_versatile_fluo = fluorescence nucleus / DAPI).
   Official Vizgen boundaries are "whole cell". Comparing nucleus masks directly against whole-cell GT
   will systematically lower IoU/PQ and is unfair to StarDist. So by default this script adds a
   "nucleus -> whole cell" expansion step after nuclear segmentation (watershed on polyT, or fixed-distance
   expand_labels) to make it comparable with official whole-cell boundaries.
   To do a "nucleus-level" comparison (or build a nucleus GT), set EXPANSION_METHOD to "none".

Three stages (each switchable):
  Stage 1  official boundaries hdf5 -> GT label mask (reuses merfish_seg_utils, identical to the Cellpose one)
  Stage 2  StarDist segments nuclei on DAPI -> optional expansion to whole cells -> predicted label mask
  Stage 3  reuse test.py metrics, compare StarDist prediction against official GT, write CSV

Dependencies: stardist tensorflow(or tensorflow-cpu) csbdeep  plus  h5py tifffile scikit-image scipy pandas
     First run of from_pretrained downloads the pretrained model.

Usage: set SEG_MODULES_DIR below (if the three files are not in the same directory), then edit the CONFIG
     path params, then run  python merfish_run_stardist.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # allocate GPU memory on demand, avoid grabbing it all at once

import sys
import time
import psutil

# Configure TensorFlow GPU memory management (must be before importing tensorflow)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[TF] configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as e:
        print(f"[TF] GPU config warning: {e}")

# ======================= dependency module path (required when cross-directory) =======================
# If merfish_seg_utils.py and merfish_seg_benchmark.py are not in this file's directory,
# put their directory into SEG_MODULES_DIR. Any one of three ways works (first match wins):
#   1) edit the line below directly
#   2) set environment variable MERFISH_SEG_MODULES_DIR
#   3) all three files in the same directory -> leave empty (auto use this file's directory)
SEG_MODULES_DIR = "yourpath/cellpose"

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:               # __file__ may not exist in interactive/exec environments
    _this_dir = os.getcwd()

for _d in (SEG_MODULES_DIR, os.environ.get("MERFISH_SEG_MODULES_DIR", ""), _this_dir):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
# ==========================================================================

import numpy as np
import pandas as pd
import tifffile

from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import watershed, expand_labels
from scipy.ndimage import distance_transform_edt

try:
    import merfish_utils as U
    import merfish_seg_bench as BM   # reuse load_metrics / evaluate_against_reference / run_benchmark
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        f"module not found {_e.name}. Set SEG_MODULES_DIR at the top of this file to the directory "
        f"holding merfish_seg_utils.py and merfish_seg_benchmark.py, "
        f"or set the MERFISH_SEG_MODULES_DIR environment variable."
    ) from _e


# ===========================================================================
# ============================= CONFIG ======================================
# ===========================================================================
DATA_DIR = "yourpath/dataset/merfish_prostate2"   # Liver1 dataset root dir

DAPI_TIF   = os.path.join(DATA_DIR, "images",
    "mosaic_DAPI_z3.tif")
# POLYT_TIF  = os.path.join(DATA_DIR, "images",
#     "mosaic_PolyT_z3.tif")  # originally used polyT as membrane channel, but polyT is a cytoplasmic signal, weaker than a true membrane stain
POLYT_TIF  = os.path.join(DATA_DIR, "images",
    "mosaic_Cellbound1_z3.tif")  # switch to true membrane stain Cellbound1; Mesmer performs better on clear membrane signal
TRANSFORM_CSV = os.path.join(DATA_DIR, "images",
    "micron_to_mosaic_pixel_transform.csv")
CELL_METADATA_CSV = os.path.join(DATA_DIR,
    "cell_metadata.csv")
CELL_BOUNDARIES_DIR = os.path.join(DATA_DIR, "cell_boundaries")

# your test.py path (metrics loaded from here, keeping the same scope)
TEST_PY_PATH = "yourpath/eval/test.py"

# output directory
OUT_DIR = "yourpath/stardist/merfish_prostate2"

# ---- ROI: pixel (x0,y0,x1,y1). None=whole image (very memory-heavy). Empty takes a centered ROI_SIZE window ----
# Note: must match the ROI used by the Cellpose run exactly for the two methods to share one table.
ROI = None
ROI_SIZE = 10000
Z_INDEX = 3                                  # which z layer the official boundaries use; must match the image (z3 for this data)

# ---- StarDist config ----
STARDIST_MODEL = "2D_versatile_fluo"         # fluorescence nucleus model; use it for DAPI
NORM_LOW, NORM_HIGH = 1.0, 99.8              # csbdeep percentile normalization range
PROB_THRESH = None                           # None=use the model's own optimal threshold
NMS_THRESH  = None                           # None=use the model's own optimal threshold
TILE_BLOCK = 1024                            # tile edge length (px); lowered from 2048 to 1024 to further reduce peak GPU memory
USE_GPU = True                               # hint only; actually decided by the TensorFlow environment

# ---- nucleus -> whole-cell expansion (to be comparable with official "whole-cell" GT) ----
EXPANSION_METHOD = "expand_labels"         # "watershed_polyt" | "expand_labels" | "none"
EXPAND_DISTANCE_PX = 15                       # expand_labels: max distance (px) a nucleus expands outward
MAX_EXPANSION_PX = 30                         # watershed: limit how far a single cell may expand beyond its nucleus; None=unlimited
POLYT_SMOOTH_SIGMA = 2.0                      # Gaussian smoothing on polyT before watershed

# ---- benchmark params (aligned with test.py) ----
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5

# ---- stage switches ----
DO_BUILD_GT     = True   # if the Cellpose run already generated GT in the same OUT_DIR/ROI, set False to reuse it
DO_RUN_STARDIST = True
DO_BENCHMARK    = True
# ===========================================================================


# --------------------------------------------------------------------------- #
# Stage 2a: StarDist nuclear segmentation
# --------------------------------------------------------------------------- #
def run_stardist_nuclei(roi):
    """StarDist segments nuclei on DAPI(ROI), returns (H,W) int32 label (nuclei)."""
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize

    dapi = U.read_tif_crop(DAPI_TIF, roi).astype(np.float32)
    img = normalize(dapi, NORM_LOW, NORM_HIGH)
    print(f"[StarDist] DAPI input shape={img.shape}, model={STARDIST_MODEL}")

    model = StarDist2D.from_pretrained(STARDIST_MODEL)

    # auto tile by image size to avoid OOM on large ROIs
    H, W = img.shape[:2]
    n_tiles = (max(1, int(np.ceil(H / TILE_BLOCK))),
               max(1, int(np.ceil(W / TILE_BLOCK))))
    print(f"[StarDist] n_tiles={n_tiles}, prob_thresh={PROB_THRESH}, nms_thresh={NMS_THRESH}")

    labels, _ = model.predict_instances(
        img,
        prob_thresh=PROB_THRESH,
        nms_thresh=NMS_THRESH,
        n_tiles=n_tiles,
    )
    labels = np.asarray(labels).astype(np.int32)
    print(f"[StarDist] nuclei count={len(np.unique(labels)) - 1}")
    return labels


# --------------------------------------------------------------------------- #
# Stage 2b: nucleus -> whole-cell expansion
# --------------------------------------------------------------------------- #
def expand_nuclei_to_cells(nuclei, roi):
    """
    Expand nucleus labels into whole-cell labels (each cell keeps its nucleus id).
      watershed_polyt : watershed on polyT intensity, nuclei as seeds, foreground from polyT threshold (closer to real cytoplasm)
      expand_labels   : no polyT dependency, uniform outward expansion by fixed distance (no overlap)
      none            : no expansion, return nuclei directly
    """
    if EXPANSION_METHOD == "none":
        print("[Expand] no expansion, output nucleus mask (note: comparing against whole-cell GT will be lower)")
        return nuclei

    if EXPANSION_METHOD == "expand_labels":
        print(f"[Expand] expand_labels, distance={EXPAND_DISTANCE_PX}px")
        return expand_labels(nuclei, distance=EXPAND_DISTANCE_PX).astype(np.int32)

    if EXPANSION_METHOD == "watershed_polyt":
        if not os.path.exists(POLYT_TIF):
            print("[Expand] polyT not found, falling back to expand_labels")
            return expand_labels(nuclei, distance=EXPAND_DISTANCE_PX).astype(np.int32)

        polyt = U.read_tif_crop(POLYT_TIF, roi).astype(np.float32)
        if polyt.shape != nuclei.shape:
            raise ValueError(f"polyT and nucleus mask sizes differ: {polyt.shape} vs {nuclei.shape}")

        pol = gaussian(polyt, sigma=POLYT_SMOOTH_SIGMA, preserve_range=True)
        try:
            thr = threshold_otsu(pol)
        except Exception:
            thr = float(np.percentile(pol, 50))
        fg = pol > thr
        fg |= nuclei > 0                       # nuclei always count as foreground

        # optional: limit max outward distance from the nucleus, to avoid individual cells "spreading" too far on polyT
        if MAX_EXPANSION_PX is not None:
            dist_to_nuc = distance_transform_edt(nuclei == 0)
            fg &= dist_to_nuc <= MAX_EXPANSION_PX

        cells = watershed(-pol, markers=nuclei, mask=fg)
        cells = np.asarray(cells).astype(np.int32)
        print(f"[Expand] watershed_polyt done, cell count={len(np.unique(cells)) - 1}")
        return cells

    raise ValueError(f"unknown EXPANSION_METHOD: {EXPANSION_METHOD}")


def segment(roi, out_tif):
    """Full segmentation: StarDist nuclei -> optional expansion -> save label tif."""
    nuclei = run_stardist_nuclei(roi)
    cells = expand_nuclei_to_cells(nuclei, roi)
    U.save_label_tif(cells, out_tif)
    return cells


# --------------------------------------------------------------------------- #
# main: same three-stage flow as the Cellpose version
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"\n{'='*80}")
    print(f"[Performance] start running StarDist - initial memory: {initial_memory:.2f} MB")
    print(f"{'='*80}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # let the reused BM.run_benchmark use this script's metric params
    BM.BOUNDARY_MODE = BOUNDARY_MODE
    BM.IOU_THRESHOLD = IOU_THRESHOLD

    # determine ROI (must be consistent with other methods)
    roi = ROI
    if roi is None and ROI_SIZE is not None:
        roi = U.make_center_roi(DAPI_TIF, size=ROI_SIZE)
        print(f"[ROI] auto take centered window: {roi}")
    elif roi is None:
        print("[ROI] whole-image eval (mind memory usage)")

    gt_tif      = os.path.join(OUT_DIR, "gt_vizgen_official_roi.tif")
    pred_tif    = os.path.join(OUT_DIR, "pred_stardist_roi.tif")

    # -------- Stage 1: GT (can reuse the same one generated by the Cellpose run) --------
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
        print(f"[GT] reuse existing GT: {gt_tif} shape={gt_mask.shape}")

    stage1_time = time.time() - stage1_start
    stage1_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 1 done - time: {stage1_time:.2f}s, current memory: {stage1_memory:.2f} MB")

    # -------- Stage 2: StarDist segmentation --------
    stage2_start = time.time()
    if DO_RUN_STARDIST:
        segment(roi, pred_tif)

    stage2_time = time.time() - stage2_start
    stage2_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 2 done - time: {stage2_time:.2f}s, current memory: {stage2_memory:.2f} MB")

    # -------- Stage 3: benchmark (reuse test.py metrics) --------
    stage3_start = time.time()
    if DO_BENCHMARK and os.path.exists(pred_tif):
        metrics_mod = BM.load_metrics(TEST_PY_PATH)
        BM.run_benchmark(
            metrics_mod, gt_mask, {"StarDist": pred_tif},
            out_csv=os.path.join(OUT_DIR, "merfish_seg_metrics_stardist.csv"),
        )

    stage3_time = time.time() - stage3_start
    stage3_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 3 done - time: {stage3_time:.2f}s, current memory: {stage3_memory:.2f} MB")

    # overall stats
    total_time = time.time() - start_time
    peak_memory = stage3_memory
    memory_increase = peak_memory - initial_memory

    print(f"\n{'='*80}")
    print(f"[Performance] StarDist performance summary")
    print(f"{'='*80}")
    print(f"total runtime: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"initial memory: {initial_memory:.2f} MB")
    print(f"peak memory: {peak_memory:.2f} MB")
    print(f"memory increase: {memory_increase:.2f} MB")
    print(f"\nper-stage times:")
    print(f"  Stage 1 (GT build): {stage1_time:.2f}s")
    print(f"  Stage 2 (StarDist segmentation): {stage2_time:.2f}s")
    print(f"  Stage 3 (Benchmark): {stage3_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
