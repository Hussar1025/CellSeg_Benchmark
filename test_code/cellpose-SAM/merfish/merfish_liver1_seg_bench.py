"""
merfish_liver1_seg_benchmark.py
================================
MERFISH (Vizgen human liver cancer Liver1) cell segmentation benchmark pipeline.

Three stages (each can be switched on/off):
  Stage 1  Rasterize official cell boundaries (cell_boundaries/*.hdf5) into a GT label mask (same mosaic pixel coordinate system)
  Stage 2  Segment the DAPI(+polyT) mosaic with Cellpose to get the predicted label mask
  Stage 3  Reuse the metrics from test.py to compare each method's predicted mask against the official GT, write CSV

Key design
----------
* GT and prediction live in the same mosaic pixel coordinate system and the same ROI, so they are
  pixel-aligned by construction and can use test.py's IoU / Dice / PQ / AJI / mDA metrics directly
  (test.py requires both masks to share shape and coordinate system).
* Metrics are loaded straight from your test.py (importlib), so they stay in sync with test.py.
* Evaluation defaults to one central ROI window; the full mosaic is huge (tens of thousands
  of pixels per side) and full-image evaluation is not advised. To score the whole image, set ROI
  to None (watch memory) or loop over tiles yourself.

Usage
-----
    1) Edit the paths and parameters in the ====== CONFIG ====== block below
    2) python merfish_seg_benchmark.py

Extending to other methods (StarDist / CellSAM / ProSeg / Bering / Cellist ...)
------------------------------------------------------------------------------
* Image-based methods (StarDist/CellSAM/Mesmer ...): produce a label tif on the same ROI and add it to EXTRA_METHOD_MASKS.
* Transcript-based methods (ProSeg/Bering/Baysor ...): they output polygons / per-molecule assignment in micron
  coordinates, so rasterize them into the same mosaic pixel coordinate system first (reuse micron_to_pixel +
  sk_polygon from merfish_seg_utils), then add the label tif to EXTRA_METHOD_MASKS.
  WARNING: these methods also need detected_transcripts.csv, which is currently missing locally.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import importlib.util
import time
import psutil

import numpy as np
import pandas as pd
import tifffile

import merfish_utils as U


# ===========================================================================
# ============================= CONFIG ======================================
# ===========================================================================
DATA_DIR = "yourpath/dataset/merfish_prostate2"   # Liver1 dataset root

DAPI_TIF   = os.path.join(DATA_DIR, "images",
    "mosaic_DAPI_z3.tif")
# POLYT_TIF  = os.path.join(DATA_DIR, "images",
#     "mosaic_PolyT_z3.tif")  # used polyT as the membrane channel before, but polyT is a cytoplasm signal, worse than a real membrane stain
POLYT_TIF  = os.path.join(DATA_DIR, "images",
    "mosaic_Cellbound1_z3.tif")  # switched to the real membrane stain Cellbound1; Mesmer does better on a crisp membrane signal
TRANSFORM_CSV = os.path.join(DATA_DIR, "images",
    "micron_to_mosaic_pixel_transform.csv")
CELL_METADATA_CSV = os.path.join(DATA_DIR,
    "cell_metadata.csv")
CELL_BOUNDARIES_DIR = os.path.join(DATA_DIR, "cell_boundaries")

# Your test.py path (metrics are loaded from here so the scope stays identical)
TEST_PY_PATH = "yourpath/eval/test.py"

# Output directory
OUT_DIR = "yourpath/cellpose/merfish_prostate2"

# ---- ROI: pixel coordinates (x0,y0,x1,y1). None = whole image (very memory heavy, not advised) ----
# If empty, auto-select a central ROI_SIZE x ROI_SIZE window of the mosaic
ROI = None
ROI_SIZE = 10000

# ---- z layer: which z the official boundaries use; must match the image z (images here are z3) ----
Z_INDEX = 3

# ---- Cellpose config ----
CELLPOSE_MODEL = "cpsam"     # "cpsam" (Cellpose-SAM/v4, matches cpsam in test.py) | "cyto3" | "nuclei" (v3)
USE_POLYT_AS_CYTO = True     # True: DAPI as nucleus + polyT as cytoplasm for whole-cell segmentation (comparable to official whole-cell boundaries)
CELL_DIAMETER = None           # diameter in pixels; None=auto. This data is ~0.108 um/px, a 10 um cell is ~90 px
USE_GPU = True

# ---- label masks from other methods (same ROI, same pixel coordinate system, .tif) ----
# Example: {"StarDist": "/path/stardist_roi.tif", "CellSAM": "/path/cellsam_roi.tif"}
EXTRA_METHOD_MASKS = {}

# ---- benchmark metric parameters (aligned with test.py) ----
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5

# ---- stage switches ----
DO_BUILD_GT      = True
DO_RUN_CELLPOSE  = True
DO_BENCHMARK     = True
# ===========================================================================


# --------------------------------------------------------------------------- #
# Load the metric functions from test.py
# --------------------------------------------------------------------------- #
def load_metrics(test_py_path):
    spec = importlib.util.spec_from_file_location("seg_metrics", test_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Stage 2: Cellpose segmentation (supports v4 cpsam and v3 cyto3/nuclei)
# --------------------------------------------------------------------------- #
def build_cellpose_input(roi):
    """
    Build the Cellpose input image.
    Whole-cell mode (USE_POLYT_AS_CYTO=True): HxWx3, R=polyT (cytoplasm), G=DAPI (nucleus), B=0
    Nucleus-only mode: HxW grayscale = DAPI
    """
    dapi = U.read_tif_crop(DAPI_TIF, roi).astype(np.float32)
    if USE_POLYT_AS_CYTO and os.path.exists(POLYT_TIF):
        polyt = U.read_tif_crop(POLYT_TIF, roi).astype(np.float32)
        if polyt.shape != dapi.shape:
            raise ValueError(f"DAPI and polyT shapes differ: {dapi.shape} vs {polyt.shape}")
        img = np.zeros((*dapi.shape, 3), dtype=np.float32)
        img[..., 0] = polyt   # cytoplasm channel
        img[..., 1] = dapi    # nucleus channel
        return img, True
    return dapi, False


def run_cellpose(roi, out_tif):
    """Run Cellpose on the ROI, save and return the predicted label mask."""
    from cellpose import models

    img, is_multichannel = build_cellpose_input(roi)
    print(f"[Cellpose] input shape={img.shape}, model={CELLPOSE_MODEL}, GPU={USE_GPU}")

    masks = None
    # --- prefer the v4 (Cellpose-SAM) API ---
    if CELLPOSE_MODEL == "cpsam" and hasattr(models, "CellposeModel"):
        try:
            model = models.CellposeModel(gpu=USE_GPU)  # v4 defaults to cpsam
            res = model.eval(img, diameter=CELL_DIAMETER) if _accepts_diameter(model) \
                else model.eval(img)
            masks = res[0]
            print("[Cellpose] using the v4 CellposeModel(cpsam) API")
        except Exception as e:
            print(f"[Cellpose] v4 API failed ({e}), falling back to the v3 API")

    # --- fallback / explicit v3 (cyto3 / nuclei) ---
    if masks is None:
        model_type = CELLPOSE_MODEL if CELLPOSE_MODEL in ("cyto3", "nuclei", "cyto2", "cyto") else "cyto3"
        model = models.Cellpose(gpu=USE_GPU, model_type=model_type)
        if is_multichannel:
            channels = [1, 2]   # cytoplasm=channel 1 (R/polyT), nucleus=channel 2 (G/DAPI)
        else:
            channels = [0, 0]   # grayscale (DAPI) single channel
        masks, _, _, _ = model.eval(img, diameter=CELL_DIAMETER, channels=channels)
        print(f"[Cellpose] using the v3 Cellpose(model_type={model_type}, channels={channels}) API")

    masks = np.asarray(masks).astype(np.int32)
    U.save_label_tif(masks, out_tif)
    return masks


def _accepts_diameter(model):
    """Rough check whether model.eval takes a diameter keyword (signatures differ across cellpose versions)."""
    try:
        import inspect
        return "diameter" in inspect.signature(model.eval).parameters
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Stage 3: benchmark (reuse test.py metrics; reference = official segmentation)
# --------------------------------------------------------------------------- #
def evaluate_against_reference(metrics_mod, method_name, gt_mask, pred_mask):
    """
    Reuse test.py's evaluate_single_mask_against_cellpose directly (identical metric math),
    renaming only the reference-related fields to official segmentation to avoid confusion.
    """
    res = metrics_mod.evaluate_single_mask_against_cellpose(
        method_name=method_name,
        cellpose_gt_mask=gt_mask,
        pred_mask=pred_mask,
        boundary_mode=BOUNDARY_MODE,
        iou_threshold=IOU_THRESHOLD,
    )
    res["reference"] = "Vizgen_official"
    if "cellpose_reference_cell_count" in res:
        res["reference_cell_count"] = res.pop("cellpose_reference_cell_count")
    return res


def run_benchmark(metrics_mod, gt_mask, method_mask_paths, out_csv):
    results = []
    for name, path in method_mask_paths.items():
        if path is None or not os.path.exists(path):
            print(f"[Skip] {name}: mask file not found {path}")
            continue
        pred = U.read_tif_crop(path, roi=None)  # method masks are already ROI-sized, read them whole
        pred = np.squeeze(np.asarray(pred)).astype(np.int32)
        if pred.shape != gt_mask.shape:
            print(f"[Skip] {name}: shape mismatch pred={pred.shape} vs GT={gt_mask.shape}")
            continue
        print(f"[Eval] {name} vs official segmentation ...")
        results.append(evaluate_against_reference(metrics_mod, name, gt_mask, pred))
        del pred

    if not results:
        print("[Benchmark] no methods to evaluate.")
        return None

    df = pd.DataFrame(results)
    # order columns: identifier columns first
    lead = [c for c in ["reference", "method", "reference_cell_count", "pred_cell_count"] if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    pd.set_option("display.max_columns", None, "display.width", 200)
    print("\n================ Benchmark results ================")
    print(df.to_string(index=False))
    print(f"\n[Benchmark] Saved -> {out_csv}")
    return df


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"\n{'='*80}")
    print(f"[Performance] starting - initial memory: {initial_memory:.2f} MB")
    print(f"{'='*80}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # determine the ROI
    roi = ROI
    if roi is None and ROI_SIZE is not None:
        roi = U.make_center_roi(DAPI_TIF, size=ROI_SIZE)
        print(f"[ROI] auto center window: {roi}")
    elif roi is None:
        print("[ROI] whole-image evaluation (watch memory use)")

    gt_tif   = os.path.join(OUT_DIR, "gt_vizgen_official_roi.tif")
    pred_tif = os.path.join(OUT_DIR, "pred_cellpose_roi.tif")

    # -------- Stage 1: official boundaries -> GT label mask --------
    stage1_start = time.time()
    if DO_BUILD_GT:
        gt_mask, id_map = U.build_reference_label_mask(
            cell_boundaries_dir=CELL_BOUNDARIES_DIR,
            transform_csv=TRANSFORM_CSV,
            roi=roi,
            z_index=Z_INDEX,
            cell_metadata_csv=CELL_METADATA_CSV,
        )
        U.save_label_tif(gt_mask, gt_tif)
        # save the label -> official cell_id map so results can be traced back
        pd.DataFrame(
            {"label": list(id_map.keys()), "vizgen_cell_id": list(id_map.values())}
        ).to_csv(os.path.join(OUT_DIR, "gt_label_to_cellid.csv"), index=False)
    else:
        gt_mask = np.squeeze(tifffile.imread(gt_tif)).astype(np.int32)
        print(f"[GT] loaded existing GT: {gt_tif} shape={gt_mask.shape}")

    stage1_time = time.time() - stage1_start
    stage1_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 1 done - elapsed: {stage1_time:.2f}s, current memory: {stage1_memory:.2f} MB")

    # -------- Stage 2: Cellpose segmentation --------
    stage2_start = time.time()
    method_mask_paths = dict(EXTRA_METHOD_MASKS)
    if DO_RUN_CELLPOSE:
        run_cellpose(roi, pred_tif)
    if os.path.exists(pred_tif):
        method_mask_paths = {"Cellpose": pred_tif, **method_mask_paths}

    stage2_time = time.time() - stage2_start
    stage2_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 2 done - elapsed: {stage2_time:.2f}s, current memory: {stage2_memory:.2f} MB")

    # -------- Stage 3: benchmark --------
    stage3_start = time.time()
    if DO_BENCHMARK:
        metrics_mod = load_metrics(TEST_PY_PATH)
        run_benchmark(
            metrics_mod, gt_mask, method_mask_paths,
            out_csv=os.path.join(OUT_DIR, "merfish_seg_metrics_vizgen_reference.csv"),
        )

    stage3_time = time.time() - stage3_start
    stage3_memory = process.memory_info().rss / 1024 / 1024
    print(f"[Performance] Stage 3 done - elapsed: {stage3_time:.2f}s, current memory: {stage3_memory:.2f} MB")

    # overall stats
    total_time = time.time() - start_time
    peak_memory = stage3_memory
    memory_increase = peak_memory - initial_memory

    print(f"\n{'='*80}")
    print(f"[Performance] performance summary")
    print(f"{'='*80}")
    print(f"total runtime: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"initial memory: {initial_memory:.2f} MB")
    print(f"peak memory: {peak_memory:.2f} MB")
    print(f"memory delta: {memory_increase:.2f} MB")
    print(f"\nper-stage elapsed:")
    print(f"  Stage 1 (GT build): {stage1_time:.2f}s")
    print(f"  Stage 2 (Cellpose segmentation): {stage2_time:.2f}s")
    print(f"  Stage 3 (Benchmark): {stage3_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()