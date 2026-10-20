"""
merfish_run_mesmer.py
=====================
Mesmer (DeepCell) segmentation + benchmark on MERFISH (Vizgen mouse brain Receptor Map).
Structure mirrors merfish_run_stardist.py / merfish_seg_benchmark.py; can run alone,
or emit only the label mask and feed it into EXTRA_METHOD_MASKS of merfish_seg_benchmark.py.

About Mesmer:
  * Native joint whole-cell / nucleus model that outputs the whole-cell mask directly --
    no expansion step as in StarDist -- so it is directly comparable to the official Vizgen
    whole-cell boundaries (same as Cellpose).
  * Always takes two channels: channel 0 = nucleus (DAPI), channel 1 = membrane/cytoplasm
    (polyT is used here as a cytoplasm proxy).
  * Key parameter image_mpp (um/pixel): Mesmer is trained at 0.5 um/px and rescales the
    input internally by this value. This data is ~0.108 um/px, so it must be passed correctly.
    The script derives it automatically from micron_to_mosaic_pixel_transform.csv.
  * compartment: "whole-cell" (default, aligned with whole-cell GT) | "nuclear" | "both".

Three stages (each can be toggled):
  Stage 1  official boundaries hdf5 -> GT label mask (reuses merfish_seg_utils, identical to the other scripts)
  Stage 2  Mesmer whole-cell mask from DAPI+polyT -> predicted label mask
  Stage 3  reuse test.py metrics, compare the Mesmer prediction with official GT, write CSV

Dependencies: deepcell (pulls in TensorFlow) plus h5py tifffile scikit-image scipy pandas
     The first run downloads model weights; newer deepcell may require a token from
     users.deepcell.org and the DEEPCELL_ACCESS_TOKEN env var.

Usage: first set SEG_MODULES_DIR below (if the three files are not in one directory), then
     edit the CONFIG paths and run  python merfish_run_mesmer.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
import time
import psutil

# ======================= Dependency module path (required across dirs) =======================
# If merfish_seg_utils.py and merfish_seg_benchmark.py are not in this file's directory,
# put their directory into SEG_MODULES_DIR. Three options (applied in order):
#   1) edit the line below
#   2) set the MERFISH_SEG_MODULES_DIR env var
#   3) all three files in one directory -> leave empty (this file's directory is used)
SEG_MODULES_DIR = "yourpath/cellpose"

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:               # __file__ may be absent under interactive/exec
    _this_dir = os.getcwd()

for _d in (SEG_MODULES_DIR, os.environ.get("MERFISH_SEG_MODULES_DIR", ""), _this_dir):
    if _d and os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
# ==========================================================================

import numpy as np
import pandas as pd
import tifffile

try:
    import merfish_utils as U
    import merfish_seg_bench as BM   # reuse load_metrics / evaluate_against_reference / run_benchmark
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        f"Module {_e.name} not found. Point SEG_MODULES_DIR at the directory holding "
        f"merfish_seg_utils.py and merfish_seg_benchmark.py, "
        f"or set the MERFISH_SEG_MODULES_DIR env var."
    ) from _e


# ===========================================================================
# ============================= CONFIG ======================================
# ===========================================================================
DATA_DIR = "yourpath/dataset/merfish_prostate2"   # Liver1 dataset root

DAPI_TIF   = os.path.join(DATA_DIR, "images",
    "mosaic_DAPI_z3.tif")
# POLYT_TIF  = os.path.join(DATA_DIR, "images",
#     "mosaic_PolyT_z3.tif")  # PolyT was used as the membrane channel, but it is cytoplasmic signal, so a real membrane stain is better
POLYT_TIF  = os.path.join(DATA_DIR, "images",
    "mosaic_Cellbound1_z3.tif")  # switched to the real membrane stain Cellbound1; Mesmer performs better on crisp membrane signal
TRANSFORM_CSV = os.path.join(DATA_DIR, "images",
    "micron_to_mosaic_pixel_transform.csv")
CELL_METADATA_CSV = os.path.join(DATA_DIR,
    "cell_metadata.csv")
CELL_BOUNDARIES_DIR = os.path.join(DATA_DIR, "cell_boundaries")

# Path to your test.py (metrics are loaded from here so the definition stays the same)
TEST_PY_PATH = "yourpath/eval/test.py"

# Output directory
OUT_DIR = "yourpath/Mesmer/merfish_prostate2"

# ---- ROI: pixel (x0,y0,x1,y1). None = full image (very memory hungry). Empty = centered ROI_SIZE window ----
# Note: must match the ROI used for Cellpose / StarDist so all methods land in one comparable table!
ROI = None
ROI_SIZE = 10000
Z_INDEX = 3                                  # which z level the official boundaries use; must match the image (z3 here)

# ---- Mesmer config ----
COMPARTMENT = "whole-cell"                   # "whole-cell" (aligned with whole-cell GT) | "nuclear" | "both"
IMAGE_MPP = None                             # um/pixel; None = derived from TRANSFORM_CSV (~0.108 here)
BATCH_SIZE = 4
USE_GPU = True                               # hint only; the actual device is set by the TensorFlow environment

# ---- benchmark params (aligned with test.py) ----
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5

# ---- stage switches ----
DO_BUILD_GT   = True   # set False to reuse a GT already built in the same OUT_DIR/ROI
DO_RUN_MESMER = True
DO_BENCHMARK  = True
# ===========================================================================


# --------------------------------------------------------------------------- #
# Derive image_mpp (um/pixel) from the micron->pixel transform matrix
# --------------------------------------------------------------------------- #
def derive_image_mpp(transform_csv):
    """
    The transform is micron->pixel (pixels/um). Take the norm of each of the two columns of
    the 2x2 linear block as the x/y pixel density (robust to rotation), then average their
    reciprocals = um/pixel.
    """
    M = U.load_micron_to_mosaic_transform(transform_csv)
    sx = float(np.hypot(M[0, 0], M[1, 0]))   # first column norm = x-direction pixels/um
    sy = float(np.hypot(M[0, 1], M[1, 1]))   # second column norm = y-direction pixels/um
    if sx <= 0 or sy <= 0:
        raise ValueError(f"Failed to derive pixel size from transform: sx={sx}, sy={sy}")
    mpp = 0.5 * (1.0 / sx + 1.0 / sy)
    return mpp


# --------------------------------------------------------------------------- #
# Stage 2: Mesmer segmentation
# --------------------------------------------------------------------------- #
def build_mesmer_input(roi):
    """
    Build the Mesmer input: (1, H, W, 2), channel 0 = nucleus (DAPI), channel 1 = membrane/cytoplasm (polyT).
    """
    dapi = U.read_tif_crop(DAPI_TIF, roi).astype(np.float32)

    if os.path.exists(POLYT_TIF):
        membrane = U.read_tif_crop(POLYT_TIF, roi).astype(np.float32)
        if membrane.shape != dapi.shape:
            raise ValueError(f"DAPI and polyT shapes differ: {dapi.shape} vs {membrane.shape}")
    else:
        if COMPARTMENT == "whole-cell":
            print("[Mesmer][warning] polyT membrane channel not found; whole-cell segmentation quality "
                  "will drop noticeably. Use COMPARTMENT='nuclear' or supply a membrane/cytoplasm channel.")
        membrane = np.zeros_like(dapi)

    X = np.stack([dapi, membrane], axis=-1)[None, ...]  # (1,H,W,2)
    return X


def run_mesmer(roi, out_tif):
    """Run Mesmer on the ROI, save and return the predicted label mask (H,W int32)."""
    from deepcell.applications import Mesmer

    X = build_mesmer_input(roi)
    mpp = IMAGE_MPP if IMAGE_MPP is not None else derive_image_mpp(TRANSFORM_CSV)
    print(f"[Mesmer] input shape={X.shape}, compartment={COMPARTMENT}, "
          f"image_mpp={mpp:.4f} um/px, GPU={USE_GPU}")

    app = Mesmer()
    labeled = app.predict(
        X,
        batch_size=BATCH_SIZE,
        image_mpp=mpp,
        compartment=COMPARTMENT,
    )
    # whole-cell / nuclear -> (1,H,W,1); both -> (1,H,W,2); take the whole-cell channel (index 0)
    labeled = np.asarray(labeled)
    if COMPARTMENT == "both":
        mask = labeled[0, ..., 0]   # channel 0 = whole-cell, channel 1 = nuclear
    else:
        mask = labeled[0, ..., 0]
    mask = mask.astype(np.int32)

    print(f"[Mesmer] cell count={len(np.unique(mask)) - 1}")
    U.save_label_tif(mask, out_tif)
    return mask


# --------------------------------------------------------------------------- #
# main: the same three stages as the other two scripts
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"\n{'='*80}")
    print(f"[Performance] Starting Mesmer - initial memory: {initial_memory:.2f} MB")
    print(f"{'='*80}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # make the reused BM.run_benchmark use this script's metric params
    BM.BOUNDARY_MODE = BOUNDARY_MODE
    BM.IOU_THRESHOLD = IOU_THRESHOLD

    # choose the ROI (must match the other methods)
    roi = ROI
    if roi is None and ROI_SIZE is not None:
        roi = U.make_center_roi(DAPI_TIF, size=ROI_SIZE)
        print(f"[ROI] auto center window: {roi}")
    elif roi is None:
        print("[ROI] full-image evaluation (watch memory use)")

    gt_tif   = os.path.join(OUT_DIR, "gt_vizgen_official_roi.tif")
    pred_tif = os.path.join(OUT_DIR, "pred_mesmer_roi.tif")

    # -------- Stage 1: GT (may reuse the same one built earlier) --------
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
        print(f"[GT] reusing existing GT: {gt_tif} shape={gt_mask.shape}")

    stage1_time = time.time() - stage1_start
    stage1_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 1 done - time: {stage1_time:.2f}s, current memory: {stage1_memory:.2f} MB")

    # -------- Stage 2: Mesmer segmentation --------
    stage2_start = time.time()
    if DO_RUN_MESMER:
        run_mesmer(roi, pred_tif)

    stage2_time = time.time() - stage2_start
    stage2_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 2 done - time: {stage2_time:.2f}s, current memory: {stage2_memory:.2f} MB")

    # -------- Stage 3: benchmark (reuse test.py metrics) --------
    stage3_start = time.time()
    if DO_BENCHMARK and os.path.exists(pred_tif):
        metrics_mod = BM.load_metrics(TEST_PY_PATH)
        BM.run_benchmark(
            metrics_mod, gt_mask, {"Mesmer": pred_tif},
            out_csv=os.path.join(OUT_DIR, "merfish_seg_metrics_mesmer.csv"),
        )

    stage3_time = time.time() - stage3_start
    stage3_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 3 done - time: {stage3_time:.2f}s, current memory: {stage3_memory:.2f} MB")

    # overall stats
    total_time = time.time() - start_time
    peak_memory = stage3_memory
    memory_increase = peak_memory - initial_memory

    print(f"\n{'='*80}")
    print(f"[Performance] Mesmer performance summary")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"Memory increase: {memory_increase:.2f} MB")
    print(f"\nTime per stage:")
    print(f"  Stage 1 (GT build): {stage1_time:.2f}s")
    print(f"  Stage 2 (Mesmer segmentation): {stage2_time:.2f}s")
    print(f"  Stage 3 (Benchmark): {stage3_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
