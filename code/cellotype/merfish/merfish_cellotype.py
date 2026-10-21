"""
merfish_run_cellotype.py
========================
CelloType segmentation + benchmark on MERFISH (Vizgen mouse brain Receptor Map).
Same structure as merfish_run_mesmer.py / merfish_run_stardist.py /
merfish_seg_benchmark.py; runs standalone or just produces a label mask that
can be fed into merfish_seg_benchmark.py's EXTRA_METHOD_MASKS.

About CelloType (Nature Methods 2024, tanlabcode/CelloType):
  * Detectron2 + MaskDINO transformer detect/segment model; TissueNet pretrained
    weights do whole-cell segmentation (given nuclear + membrane channels) and
    output is comparable to official Vizgen whole-cell boundaries with no dilation.
  * Channel convention (must follow, otherwise results are entirely wrong):
        B (blue) = nuclear channel (DAPI), G (green) = membrane/cytoplasm (polyT),
        R (red) = optional (zero).
    Input is uint8[0,255]; this script percentile-normalizes each channel to uint8.
  * It is a detection model with a finite query count and a max_det cap, so a
    single forward pass cannot handle a large image (official recipe uses SAHI
    for sliced large-image inference). This script implements its own
    "sliced inference + stitching": cut the ROI into TILE_SIZE patches, predict
    each, then stitch back with globally unique ids; overlapping bands plus
    dropping detections that touch an internal tile border avoid splitting /
    double-counting cells (TILE_OVERLAP must exceed the largest cell diameter;
    cells here are ~90px, default overlap=128 is enough).

Environment (CelloType install is heavy; use a dedicated conda env, see the
official README): Python=3.8, PyTorch, Detectron2, Deformable-DETR CUDA ops,
then pip install cellotype; also clone the repo and `cd models && sh download.sh`
to download weights and obtain configs/. Evaluation-side dependencies as before:
h5py tifffile scikit-image scipy pandas.

Usage: set SEG_MODULES_DIR and the CelloType repo/weights paths below, edit
CONFIG, then
     python merfish_run_cellotype.py
"""

import argparse
import os
import sys
import time
import resource   # stdlib (Unix); replaces psutil since the CelloType env's Python is too old for psutil

# Dependency module paths (required when the files live in another directory)
# Directory holding merfish_seg_utils.py and merfish_seg_benchmark.py.
# Pick any one of three ways (evaluated in order): 1) edit the line below
# 2) env var MERFISH_SEG_MODULES_DIR  3) leave empty if all three files are in
# the same directory (then this file's directory is used).
SEG_MODULES_DIR = "yourpath"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
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
        f"cannot find module {_e.name}. Set SEG_MODULES_DIR at the top of this file "
        f"to the directory holding merfish_seg_utils.py and merfish_seg_benchmark.py, "
        f"or set the env var MERFISH_SEG_MODULES_DIR."
    ) from _e


# ===========================================================================
# ============================= CONFIG ======================================
# ===========================================================================
DATA_DIR = "yourpath"

DAPI_TIF   = os.path.join(DATA_DIR, "images",
    "mosaic_DAPI_z3.tif")
# POLYT_TIF  = os.path.join(DATA_DIR, "images",
#     "mosaic_PolyT_z3.tif")  # polyT as membrane channel originally; polyT is cytoplasmic signal, weaker than a real membrane stain
POLYT_TIF  = os.path.join(DATA_DIR, "images",
    "mosaic_Cellbound1_z3.tif")  # real membrane stain Cellbound1; Mesmer performs better on clear membrane signal
TRANSFORM_CSV = os.path.join(DATA_DIR, "images",
    "micron_to_mosaic_pixel_transform.csv")
CELL_METADATA_CSV = os.path.join(DATA_DIR,
    "cell_metadata.csv")
CELL_BOUNDARIES_DIR = os.path.join(DATA_DIR, "cell_boundaries")

# Your test.py path (metrics loaded from here to keep the comparison consistent)
TEST_PY_PATH = "yourpath"

# Output directory
OUT_DIR = "yourpath"

# ---- CelloType repo/weights/config (from clone + models/download.sh) ----
CELLOTYPE_REPO = "yourpath"   # change to the clone of CelloType
CELLOTYPE_MODEL_PATH  = os.path.join(CELLOTYPE_REPO, "models",
    "tissuenet_model_0019999.pth")
CELLOTYPE_CONFIG_PATH = os.path.join(CELLOTYPE_REPO, "configs",
    "maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml")
DEVICE = "cuda"                       # "cuda" | "cpu"
CONFIDENCE_THRESH = 0.1               # detection confidence threshold (README example value)
MAX_DET = 1000                        # max detections per forward pass (per tile, not the whole image)

# ---- Channel normalization (per channel, percentile clip -> uint8[0,255]) ----
NORM_LOW, NORM_HIGH = 1.0, 99.0

# ---- Resolution alignment: downsample input to the model training resolution ----
# CelloType with TissueNet pretrained weights operates at ~0.5um/px (cells ~20px);
# this mosaic is ~0.108um/px (cells ~90px), a ~4.6x gap => out of distribution =>
# many missed detections and spurious boxes (this is why the log showed 8740 GT
# cells but only 2657 predicted, with 1901 of them FP).
# So: downsample RGB to TARGET_MPP before inference, and upsample the label
# nearest-neighbour back to the original ROI size before comparing with GT.
# This is not "tuning/changing the method": it is equivalent to Mesmer's
# image_mpp / Cellpose's diameter -- how the method is meant to be used.
TARGET_MPP = 0.5                      # target resolution (um/px); TissueNet standard; usually no change
SOURCE_MPP = None                     # source resolution (um/px). None = infer from TRANSFORM_CSV (recommended)
RESCALE_TO_TARGET_MPP = True          # False reverts to the original (wrong) full-resolution inference; for comparison only

# ---- Inference mode ----
# "sahi"  : CelloType's official large-image inference (sahi_predict.get_sliced_prediction),
#           cross-tile merge via GREEDY-NMM; preferred for benchmark; needs CELLOTYPE_REPO
#           pointing at the clone (which contains sahi_predict.py).
# "tiled" : the original self-written slicing + dedup (now running on the downsampled
#           image, which only makes sense at the correct scale); fallback when no sahi env.
# "single": one forward pass over the whole image; only for very small ROIs for debugging.
INFERENCE_MODE = "sahi"
TILE_SIZE = 256                       # tile edge (matches TissueNet training tile size)
TILE_OVERLAP = 128                    # used by "tiled"; must exceed max downsampled cell diameter (~20px); 128 is enough
SAHI_OVERLAP_RATIO = 0.2              # used by "sahi"; overlap ratio (0.2 => 256*0.2~51px > cell diameter)
SAHI_POSTPROCESS = "GREEDYNMM"        # cross-tile merge: "GREEDYNMM" | "NMM" | "NMS"
SAHI_MATCH_METRIC = "IOU"             # merge matching metric: "IOS" (intersection over smaller) | "IOU"
SAHI_MATCH_THRESH = 0.5               # merge threshold
MIN_CELL_PIXELS = 8                   # drop too-small fragments (counted at downsampled resolution, ~170px original)

# ---- ROI: pixel (x0,y0,x1,y1). None = whole image (very memory heavy). Empty = center ROI_SIZE window ----
# NOTE: must be exactly the same ROI used by Cellpose / StarDist / Mesmer runs so the table is comparable!
ROI = None
ROI_SIZE = 10000
Z_INDEX = 3                           # which z-plane official boundaries use; must match the image (this data uses z3)

# ---- benchmark parameters (aligned with test.py) ----
BOUNDARY_MODE = "thick"
IOU_THRESHOLD = 0.5

# ---- stage switches ----
DO_BUILD_GT     = True   # if GT was already produced under the same OUT_DIR/ROI, set False to reuse
DO_RUN_CELLOTYPE = True
DO_BENCHMARK    = True
# ===========================================================================

# --------------------------------------------------------------------------- #
# Preprocessing: DAPI/polyT -> uint8 RGB (B=nuclear, G=membrane, R=0)
# --------------------------------------------------------------------------- #
def _norm_uint8(chan, p_low, p_high):
    lo, hi = np.percentile(chan, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((chan - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def build_cellotype_rgb(roi):
    """Build CelloType input RGB (uint8, HxWx3): B=nuclear (DAPI), G=membrane (polyT), R=0."""
    dapi = U.read_tif_crop(DAPI_TIF, roi).astype(np.float32)
    if os.path.exists(POLYT_TIF):
        polyt = U.read_tif_crop(POLYT_TIF, roi).astype(np.float32)
        if polyt.shape != dapi.shape:
            raise ValueError(f"DAPI and polyT shapes differ: {dapi.shape} vs {polyt.shape}")
    else:
        print("[CelloType][warning] polyT membrane channel not found; whole-cell segmentation quality will drop noticeably.")
        polyt = np.zeros_like(dapi)

    H, W = dapi.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 1] = _norm_uint8(polyt, NORM_LOW, NORM_HIGH)   # G = membrane/cytoplasm
    rgb[..., 2] = _norm_uint8(dapi, NORM_LOW, NORM_HIGH)    # B = nuclear
    return rgb


# --------------------------------------------------------------------------- #
# Resolution alignment: downsample input / nearest-neighbor upsample label back
# --------------------------------------------------------------------------- #
def _source_mpp():
    """Source image um/px: prefer SOURCE_MPP, else infer from the micron->pixel matrix (1 / px_per_um)."""
    if SOURCE_MPP is not None:
        return float(SOURCE_MPP)
    M = U.load_micron_to_mosaic_transform(TRANSFORM_CSV)
    sx = float(np.hypot(M[0, 0], M[0, 1]))   # x px/um
    sy = float(np.hypot(M[1, 0], M[1, 1]))   # y px/um
    px_per_um = 0.5 * (sx + sy)
    if px_per_um <= 0:
        raise ValueError(f"could not infer resolution from {TRANSFORM_CSV} (px/um={px_per_um})")
    return 1.0 / px_per_um


def _downsample_rgb(rgb):
    """Downsample uint8 RGB to TARGET_MPP. Returns (rgb_ds, scale); scale=out/in (<1 = shrink)."""
    if not RESCALE_TO_TARGET_MPP:
        return rgb, 1.0
    src_mpp = _source_mpp()
    scale = src_mpp / float(TARGET_MPP)          # <1: downsample
    H, W = rgb.shape[:2]
    print(f"[Rescale] source~{src_mpp:.4f}um/px, target={TARGET_MPP}um/px, scale={scale:.4f} "
          f"(cells ~90px -> ~{90 * scale:.0f}px, target ~20px)")
    if abs(scale - 1.0) < 0.02:
        print("[Rescale] resolution already close to target; skipping")
        return rgb, 1.0
    new_w, new_h = max(1, round(W * scale)), max(1, round(H * scale))
    try:
        import cv2                               # hard dependency of cellotype, usually present; INTER_AREA best for shrinking
        rgb_ds = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        from skimage.transform import resize
        rgb_ds = resize(rgb, (new_h, new_w), order=1, anti_aliasing=True, preserve_range=True)
        rgb_ds = np.clip(np.rint(rgb_ds), 0, 255).astype(np.uint8)
    print(f"[Rescale] input {(H, W)} -> downsampled {rgb_ds.shape[:2]}")
    return rgb_ds, float(new_h) / float(H)


def _upsample_labels_nn(lbl, out_hw):
    """Nearest-neighbor upsample label (instance id map) to out_hw=(H,W), keeping integer ids intact (pure numpy, memory-friendly)."""
    out_h, out_w = out_hw
    in_h, in_w = lbl.shape
    if (in_h, in_w) == (out_h, out_w):
        return lbl.astype(np.int32)
    row = np.floor((np.arange(out_h) + 0.5) * in_h / out_h).astype(np.int64)
    col = np.floor((np.arange(out_w) + 0.5) * in_w / out_w).astype(np.int64)
    np.clip(row, 0, in_h - 1, out=row)
    np.clip(col, 0, in_w - 1, out=col)
    return lbl[np.ix_(row, col)].astype(np.int32)


# --------------------------------------------------------------------------- #
# Sliced inference + stitching
# --------------------------------------------------------------------------- #
def _tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:     # ensure the last tile touches the edge
        starts.append(length - tile)
    return sorted(set(starts))


def tiled_predict(model, rgb):
    """
    Slice-predict a uint8 RGB and stitch into a global label.
    Dedup strategy: drop detections that touch a "internal" tile border (a side
    that is not the global image border). Such cells get fully captured at a more
    central position by a neighbouring overlapping tile; as long as
    TILE_OVERLAP > max cell diameter, every cell is fully received by some tile.
    """
    H, W = rgb.shape[:2]
    tile = TILE_SIZE
    stride = max(1, tile - TILE_OVERLAP)
    ys = _tile_starts(H, tile, stride)
    xs = _tile_starts(W, tile, stride)

    canvas = np.zeros((H, W), dtype=np.int32)
    next_id = 1
    n_tiles = len(ys) * len(xs)
    print(f"[CelloType] slicing {len(ys)}x{len(xs)}={n_tiles} tiles, tile={tile}, overlap={TILE_OVERLAP}")

    ti = 0
    for y in ys:
        for x in xs:
            ti += 1
            y1, x1 = min(y + tile, H), min(x + tile, W)
            patch = rgb[y:y1, x:x1]
            ph, pw = patch.shape[:2]

            local = np.asarray(model.predict(patch)).astype(np.int32)

            for lid in np.unique(local):
                if lid == 0:
                    continue
                m = local == lid
                ys_i, xs_i = np.where(m)
                # Whether it touches an "internal" border (a side that is not the global image border)
                touch_top    = ys_i.min() == 0     and y > 0
                touch_left   = xs_i.min() == 0     and x > 0
                touch_bottom = ys_i.max() == ph - 1 and y1 < H
                touch_right  = xs_i.max() == pw - 1 and x1 < W
                if touch_top or touch_left or touch_bottom or touch_right:
                    continue  # let the neighbouring overlapping tile handle it

                sub = canvas[y:y1, x:x1]
                m &= sub == 0                       # defensive: only write blank pixels
                if int(m.sum()) < MIN_CELL_PIXELS:
                    continue
                sub[m] = next_id
                next_id += 1

            if ti % 20 == 0 or ti == n_tiles:
                print(f"[CelloType] processed {ti}/{n_tiles} tiles, current cell count {next_id - 1}")

    return canvas


# --------------------------------------------------------------------------- #
# CelloType official large-image inference (SAHI slicing + GREEDY-NMM merge)
# --------------------------------------------------------------------------- #
def _predict_sahi(rgb_ds, work_dir):
    """
    Use the CelloType repo's own sahi_predict.get_sliced_prediction for official
    sliced inference. Cross-tile duplicates are merged by SAHI's GREEDY-NMM (not
    the self-written "drop border-touching detections"), which is more defensible
    for a benchmark. Needs CELLOTYPE_REPO pointing at the clone root (which
    contains sahi_predict.py); the flow matches the official sahi_inference.py.
    """
    if CELLOTYPE_REPO and CELLOTYPE_REPO not in sys.path:
        sys.path.insert(0, CELLOTYPE_REPO)       # makes `import sahi_predict` work (repo root)
    from cellotype.predict import Detectron2DetectionModel
    import sahi_predict as SP

    model = Detectron2DetectionModel(
        model_path=CELLOTYPE_MODEL_PATH,
        config_path=CELLOTYPE_CONFIG_PATH,
        confidence_threshold=CONFIDENCE_THRESH,
        image_size=TILE_SIZE,
        max_det=MAX_DET,
        device=DEVICE,
        channels=3,
    )

    # Write a temp PNG and pass the path: identical to official sahi_inference.py,
    # removing channel/format ambiguity for numpy arrays
    os.makedirs(work_dir, exist_ok=True)
    tmp_png = os.path.join(work_dir, "_cellotype_sahi_input.png")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    Image.fromarray(rgb_ds).save(tmp_png)

    print(f"[CelloType][sahi] slice={TILE_SIZE}, overlap_ratio={SAHI_OVERLAP_RATIO}, "
          f"postprocess={SAHI_POSTPROCESS}/{SAHI_MATCH_METRIC}@{SAHI_MATCH_THRESH}")
    result = SP.get_sliced_prediction(
        tmp_png, model,
        slice_height=TILE_SIZE, slice_width=TILE_SIZE,
        overlap_height_ratio=SAHI_OVERLAP_RATIO,
        overlap_width_ratio=SAHI_OVERLAP_RATIO,
        perform_standard_pred=False,             # do not re-run on the whole image (saves VRAM, same as official example)
        postprocess_type=SAHI_POSTPROCESS,
        postprocess_match_metric=SAHI_MATCH_METRIC,
        postprocess_match_threshold=SAHI_MATCH_THRESH,
        verbose=1,
    )

    h, w = rgb_ds.shape[:2]
    label = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    for obj in result.object_prediction_list:
        bm = np.asarray(obj.mask.bool_mask, dtype=bool)  # SAHI already moved masks to full-image coordinates
        if bm.shape != (h, w):
            continue
        bm &= label == 0                          # do not overwrite occupied pixels
        if int(bm.sum()) < MIN_CELL_PIXELS:
            continue
        label[bm] = next_id
        next_id += 1
    try:
        os.remove(tmp_png)
    except OSError:
        pass
    print(f"[CelloType][sahi] cells after merge={next_id - 1}")
    return label


def run_cellotype(roi, out_tif):
    """Run CelloType on the ROI: downsample -> infer -> nearest-neighbor label back to original size, save and return mask."""
    rgb_full = build_cellotype_rgb(roi)
    H, W = rgb_full.shape[:2]
    print(f"[CelloType] original input shape={rgb_full.shape}, device={DEVICE}, "
          f"conf={CONFIDENCE_THRESH}, max_det={MAX_DET}, mode={INFERENCE_MODE}")

    # Key: downsample to the training resolution before inference
    rgb_ds, scale = _downsample_rgb(rgb_full)
    del rgb_full

    if INFERENCE_MODE == "sahi":
        label_ds = _predict_sahi(rgb_ds, work_dir=os.path.dirname(os.path.abspath(out_tif)))
    else:
        from cellotype.predict import CelloTypePredictor
        model = CelloTypePredictor(
            model_path=CELLOTYPE_MODEL_PATH,
            confidence_thresh=CONFIDENCE_THRESH,
            max_det=MAX_DET,
            device=DEVICE,
            config_path=CELLOTYPE_CONFIG_PATH,
        )
        if INFERENCE_MODE == "single":
            label_ds = np.asarray(model.predict(rgb_ds)).astype(np.int32)
        else:  # "tiled"
            label_ds = tiled_predict(model, rgb_ds)

    # Nearest-neighbor label back from downsampled size to original ROI size (to align with full-res GT), then write
    mask = _upsample_labels_nn(label_ds, (H, W))
    print(f"[CelloType] cells={len(np.unique(mask)) - 1}  "
          f"(label back-projected {tuple(label_ds.shape)} -> {tuple(mask.shape)})")
    U.save_label_tif(mask, out_tif)
    return mask


# --------------------------------------------------------------------------- #
# Memory monitoring (pure stdlib, no psutil)
#   current_mem_mb : current RSS, read from /proc/self/statm (Linux)
#   peak_mem_mb    : historical peak RSS, the high-water mark of resource.getrusage,
#                    more reliable than sampling
# --------------------------------------------------------------------------- #
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def current_mem_mb():
    """Current process RSS (MB). Falls back to the getrusage peak on non-Linux or read failure."""
    try:
        with open("yourpath", "r") as f:
            rss_pages = int(f.read().split()[1])
        return rss_pages * _PAGE_SIZE / 1024 / 1024
    except (OSError, IndexError, ValueError):
        return peak_mem_mb()


def peak_mem_mb():
    """Historical peak RSS (MB). Linux ru_maxrss is in KB; macOS is in bytes."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1024.0 if sys.platform != "darwin" else ru / 1024.0 / 1024.0


_GPU_PEAK_RESERVED_MB = 0.0


def gpu_status():
    """Return PyTorch GPU status for the CelloType device plus process VRAM (MB)."""
    global _GPU_PEAK_RESERVED_MB
    requested = str(DEVICE).lower().startswith("cuda")
    try:
        import torch
        available = bool(torch.cuda.is_available())
    except Exception:
        return (f"GPU requested={requested}, GPU available=False, actually used=False, "
                "VRAM=unavailable (torch not installed or failed to load)")
    if not available:
        return f"GPU requested={requested}, GPU available=False, actually used=False, VRAM=N/A"
    try:
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        total = torch.cuda.get_device_properties(device_index).total_memory / 1024**2
        allocated = torch.cuda.memory_allocated(device_index) / 1024**2
        reserved = torch.cuda.memory_reserved(device_index) / 1024**2
        peak_allocated = torch.cuda.max_memory_allocated(device_index) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device_index) / 1024**2
        _GPU_PEAK_RESERVED_MB = max(_GPU_PEAK_RESERVED_MB, peak_reserved)
        return (f"GPU requested={requested}, GPU available=True, actually used={requested}, "
                f"device=cuda:{device_index} ({device_name}), "
                f"VRAM current={allocated:.1f}/{reserved:.1f}/{total:.1f} MB (allocated/reserved/total), "
                f"VRAM peak={peak_allocated:.1f}/{_GPU_PEAK_RESERVED_MB:.1f} MB (allocated/reserved)")
    except Exception as exc:
        return (f"GPU requested={requested}, GPU available=True, actually used={requested}, "
                f"VRAM read failed: {exc}")


# --------------------------------------------------------------------------- #
# main: same three stages as the earlier versions
# --------------------------------------------------------------------------- #
def main():
    start_time = time.time()
    initial_memory = current_mem_mb()
    print(f"\n{'='*80}")
    print(f"[Performance] starting CelloType - initial memory: {initial_memory:.2f} MB")
    print(f"[GPU] {gpu_status()}")
    print(f"{'='*80}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    BM.BOUNDARY_MODE = BOUNDARY_MODE
    BM.IOU_THRESHOLD = IOU_THRESHOLD

    roi = ROI
    if roi is None and ROI_SIZE is not None:
        roi = U.make_center_roi(DAPI_TIF, size=ROI_SIZE)
        print(f"[ROI] automatically using center window: {roi}")
    elif roi is None:
        print("[ROI] whole-image evaluation (watch memory use)")

    gt_tif   = os.path.join(OUT_DIR, "gt_vizgen_official_roi.tif")
    pred_tif = os.path.join(OUT_DIR, "pred_cellotype_roi.tif")

    # -------- Stage 1: GT --------
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
    stage1_memory = current_mem_mb()
    print(f"[Performance] Stage 1 done - elapsed: {stage1_time:.2f}s, "
          f"current memory: {stage1_memory:.2f} MB, peak: {peak_mem_mb():.2f} MB, GPU: {gpu_status()}")

    # -------- Stage 2: CelloType segmentation --------
    stage2_start = time.time()
    if DO_RUN_CELLOTYPE:
        run_cellotype(roi, pred_tif)

    stage2_time = time.time() - stage2_start
    stage2_memory = current_mem_mb()
    print(f"[Performance] Stage 2 done - elapsed: {stage2_time:.2f}s, "
          f"current memory: {stage2_memory:.2f} MB, peak: {peak_mem_mb():.2f} MB, GPU: {gpu_status()}")

    # -------- Stage 3: benchmark --------
    stage3_start = time.time()
    if DO_BENCHMARK and os.path.exists(pred_tif):
        metrics_mod = BM.load_metrics(TEST_PY_PATH)
        BM.run_benchmark(
            metrics_mod, gt_mask, {"CelloType": pred_tif},
            out_csv=os.path.join(OUT_DIR, "merfish_seg_metrics_cellotype.csv"),
        )

    stage3_time = time.time() - stage3_start
    stage3_memory = current_mem_mb()
    print(f"[Performance] Stage 3 done - elapsed: {stage3_time:.2f}s, "
          f"current memory: {stage3_memory:.2f} MB, peak: {peak_mem_mb():.2f} MB, GPU: {gpu_status()}")

    # Overall
    total_time = time.time() - start_time
    peak_memory = peak_mem_mb()          # real historical high-water mark, not a sampled end-of-run value
    memory_increase = peak_memory - initial_memory

    print(f"\n{'='*80}")
    print(f"[Performance] CelloType performance summary")
    print(f"{'='*80}")
    print(f"Total runtime: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Final memory: {stage3_memory:.2f} MB")
    print(f"Peak memory: {peak_memory:.2f} MB")
    print(f"Memory increase: {memory_increase:.2f} MB (peak - initial)")
    print(f"GPU summary: {gpu_status()}")
    print(f"\nPer-stage time:")
    print(f"  Stage 1 (GT build): {stage1_time:.2f}s")
    print(f"  Stage 2 (CelloType segmentation): {stage2_time:.2f}s")
    print(f"  Stage 3 (Benchmark): {stage3_time:.2f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()