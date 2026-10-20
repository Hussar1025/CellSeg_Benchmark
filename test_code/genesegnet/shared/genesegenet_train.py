#!/usr/bin/env python
"""
GeneSegNet inference - same-platform cross-tissue transfer with pancreas weights trained from scratch

===== Every API call here comes from source code actually read, not guesswork =====

GeneSegModel.__init__ (models.py:17-59):
  - When pretrained_model is a path string, the constructor internally:
      self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
      self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
    That is, diam_mean is overwritten by the value stored in the checkpoint; the
    diam_mean argument only matters if weight loading fails. Do not fuss over its value.
  - parse_model_string() tries to parse
    residual_on/style_on/concatenation from the weight filename and overrides our values when found.
    Our weight filename is
    "GeneSegNet_residual_on_style_on_concatenation_off_models_...",
    so it will most likely parse correctly.

GeneSegModel.eval() (models.py:60-146) actual signature:
    eval(self, x, batch_size=8, channels=None, channel_axis=None,
         z_axis=None, normalize=True, invert=False,
         rescale=None, diameter=None, do_3D=False, anisotropy=None,
         net_avg=False, augment=False, tile=True, tile_overlap=0.1,
         resample=True, interp=True,
         flow_threshold=0.8, confidence_threshold=0.9,
         compute_masks=True, min_size=300, stitch_threshold=0.0,
         progress=None, loop_run=False, model_loaded=False)
  Returns (masks, flows, styles). A single (H,W,2) array x takes the non-list branch,
  and is tiled internally (tile=True); no manual patch cutting needed.

test() in GeneSeg_test.py shows how the official code calls eval() -
this script copies its argument combination directly (channels=None matches --all_channels
used in training; diameter=None lets the network use its learned diam_labels,
the fairest way to judge "cross-tissue generalization": do not tell the model
the target tissue's true cell size in advance, or it is no longer a blind test).

===== The three targets serve different purposes =====
xenium_lymph / xenium_breast: same platform as training data (Xenium), only tissue changes,
    isolating the "tissue difference" variable; this was the original fine-tuning goal.
merfish: cross-platform (probe chemistry, native resolution, and DAPI staining all differ),
    a looser second-tier test asking whether generalization survives platform differences;
    do not discuss it in the same "tissue generalization" conclusion as the first two.

===== Established reusable conventions =====
- PIXEL_SIZE = 0.2125 um/px (both Xenium targets agree, and training used this value)
- SIGMA = 6 (blur kernel for the transcript Gaussian heatmap; must match training exactly,
  or the heatmap channel distribution will not match what the network learned)
- GPU selection: once CUDA_VISIBLE_DEVICES pins a physical card, torch always sees cuda:0

Usage:
  python run_genesegnet_infer.py --target xenium_lymph --scan-thresholds
  python run_genesegnet_infer.py --target xenium_lymph --run \
      --confidence 0.5 --flow-threshold 0.6
  python run_genesegnet_infer.py --target xenium_breast --scan-thresholds
  python run_genesegnet_infer.py --target merfish --scan-thresholds
"""
import os
import sys
import time
import argparse
import subprocess

import numpy as np
import pandas as pd
import cv2
import tifffile

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

GENESEGNET_PKG_DIR = "yourpath/GeneSegNet/GeneSegNet"
MODEL_PATH = ("yourpath/genesegnet_pancreas_train/models/models/"
              "GeneSegNet_residual_on_style_on_concatenation_off_models_"
              "2026_07_29_16_09_58.677716")
OUT_DIR = "yourpath/genesegnet_infer"

PIXEL_SIZE_XENIUM = 0.2125
SIGMA = 6                     # * must match training exactly

os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  Target dataset definitions
# ══════════════════════════════════════════════════════════════════════════
TARGETS = {
    "xenium_lymph": dict(
        kind="xenium",
        data_dir="yourpath/dataset/xenium_roi_crop",
        dapi_rel="morphology_focus/morphology_focus_0000.ome.tif",
        pixel_size=PIXEL_SIZE_XENIUM,
        roi_x0=3982.5, roi_y0=2522.5, img_w=1576, img_h=1577,
        gt_cells=2442,
        note="Same-platform cross-tissue (Xenium lymph node), one of the main targets",
    ),
    "xenium_breast": dict(
        kind="xenium",
        data_dir="yourpath/dataset/xenium_breast",
        dapi_rel="morphology_focus/ch0000_dapi.ome.tif",
        pixel_size=PIXEL_SIZE_XENIUM,
        roi_x0=53994/2*PIXEL_SIZE_XENIUM - 5000*PIXEL_SIZE_XENIUM/2,
        roi_y0=27420/2*PIXEL_SIZE_XENIUM - 5000*PIXEL_SIZE_XENIUM/2,
        img_w=5000, img_h=5000,
        gt_cells=8809,
        note="Same-platform cross-tissue (Xenium breast), one of the main targets",
    ),
    "merfish": dict(
        kind="merfish",
        data_dir="yourpath/dataset/merfish_mouse_brain",
        stem=("datasets_mouse_brain_map_BrainReceptorShowcase_"
              "Slice1_Replicate1"),
        roi_x0=3589.0, roi_x1=5761.0, roi_y0=2327.0, roi_y1=4499.0,
        gt_cells=8444,
        note="* Cross-platform (probe chemistry/native resolution/DAPI staining all differ from training data),"
             "a looser second-tier test than xenium_lymph/breast,"
             "not comparable in the same 'tissue generalization' conclusion as the first two",
    ),
}


# ══════════════════════════════════════════════════════════════════════════
#  GPU
# ══════════════════════════════════════════════════════════════════════════
def pick_gpu(min_free_gb=6.0):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return None, 0.0
    best, bf = None, -1.0
    P("  GPU status:")
    for line in out.strip().splitlines():
        idx, free, util = [x.strip() for x in line.split(",")]
        fg = float(free) / 1024
        mark = ""
        if fg > bf and fg >= min_free_gb:
            best, bf, mark = idx, fg, "  ←"
        P(f"    GPU {idx}: {fg:5.1f} GB free  {util:>3}% load{mark}")
    return best, bf


def setup_gpu(want_gpu):
    if want_gpu is not None:
        gpu_str = str(want_gpu)
    else:
        gpu_str, _ = pick_gpu(6.0)
        if gpu_str is None:
            P("  WARNING: no GPU with >=6GB free; using CPU (much slower)")
    if gpu_str is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        P(f"  -> using physical GPU {gpu_str} (CUDA_VISIBLE_DEVICES set, "
          f"torch sees it as cuda:0")
    return gpu_str is not None


# ══════════════════════════════════════════════════════════════════════════
#  DAPI loading (fallback order memmap -> zarr -> full imread, same as used before)
# ══════════════════════════════════════════════════════════════════════════
def open_dapi_xenium(path):
    try:
        mm = tifffile.memmap(path, mode="r")
        if mm.ndim == 3:
            mm = mm[0] if mm.shape[0] <= 8 else mm[..., 0]
        P(f"    memmap ok  {mm.shape}  {mm.dtype}")
        return mm
    except Exception as e:
        P(f"    memmap failed ({type(e).__name__}) -> trying zarr")
    try:
        import zarr
        z = zarr.open(tifffile.imread(path, aszarr=True), mode="r")
        a = z[0] if isinstance(z, zarr.hierarchy.Group) else z
        if a.ndim == 3:
            a = a[0] if a.shape[0] <= 8 else a[..., 0]
        P(f"    zarr ok  {a.shape}")
        return a
    except Exception as e:
        P(f"    zarr failed ({type(e).__name__}) -> full imread")
    a = tifffile.imread(path)
    if a.ndim == 3:
        a = a[0] if a.shape[0] <= 8 else a[..., 0]
    P(f"    imread ok  {a.shape}")
    return a


def to_u8(img, lo_pct=1, hi_pct=99.5):
    """Normalization identical to training patches: percentile stretch to uint8"""
    if img.dtype == np.uint8:
        return img
    nz = img[img > 0]
    if len(nz) == 0:
        return np.zeros_like(img, np.uint8)
    lo, hi = np.percentile(nz, [lo_pct, hi_pct])
    v = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (v * 255).astype(np.uint8)


def gaussian_heatmap(px, py, h, w, sigma=SIGMA):
    """Transcript heatmap generation identical to training patches"""
    hm = np.zeros((h, w), np.float32)
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if ok.any():
        np.add.at(hm, (py[ok], px[ok]), 1.0)
    k = int(sigma * 4) | 1
    hm = cv2.GaussianBlur(hm, (k, k), sigma)
    if hm.max() > 0:
        hm = hm / hm.max()
    return (hm * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════
#  Load per target (image_2ch, gt_centroids)
# ══════════════════════════════════════════════════════════════════════════
def load_xenium(cfg):
    x0 = cfg["roi_x0"]; y0 = cfg["roi_y0"]
    w, h, ps = cfg["img_w"], cfg["img_h"], cfg["pixel_size"]
    x1, y1 = x0 + w * ps, y0 + h * ps
    dapi_path = f"{cfg['data_dir']}/{cfg['dapi_rel']}"

    P(f"  ROI µm x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]  "
      f"→ {w}×{h} px @ {ps} µm/px")
    P(f"  reading DAPI {dapi_path}")
    dapi = open_dapi_xenium(dapi_path)
    FH, FW = dapi.shape
    px0, py0 = int(round(x0 / ps)), int(round(y0 / ps))
    crop = np.asarray(dapi[max(0, py0):min(FH, py0 + h),
                           max(0, px0):min(FW, px0 + w)])
    if crop.shape != (h, w):
        pad = np.zeros((h, w), crop.dtype)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    dapi_u8 = to_u8(crop)
    P(f"    DAPI crop {dapi_u8.shape}  mean {dapi_u8.mean():.1f}")

    P(f"  reading transcripts + GT centroids ...")
    tx = pd.read_parquet(f"{cfg['data_dir']}/transcripts.parquet",
                         columns=["feature_name", "x_location",
                                  "y_location", "qv"])
    tx = tx[(tx.qv >= 20) &
            (tx.x_location >= x0) & (tx.x_location < x1) &
            (tx.y_location >= y0) & (tx.y_location < y1)]
    tx = tx[~tx.feature_name.str.startswith(
        ("NegControl", "Unassigned", "BLANK", "antisense"), na=False)]
    tpx = ((tx.x_location.values - x0) / ps).astype(np.int32).clip(0, w - 1)
    tpy = ((tx.y_location.values - y0) / ps).astype(np.int32).clip(0, h - 1)
    P(f"    transcripts {len(tx):,} total")
    hm_u8 = gaussian_heatmap(tpx, tpy, h, w)

    gt = pd.read_parquet(f"{cfg['data_dir']}/cells.parquet",
                         columns=["cell_id", "x_centroid", "y_centroid"])
    gt = gt[(gt.x_centroid >= x0) & (gt.x_centroid < x1) &
            (gt.y_centroid >= y0) & (gt.y_centroid < y1)]
    P(f"    GT {len(gt):,} cells (reference {cfg['gt_cells']:,})")

    image_2ch = np.stack([dapi_u8, hm_u8], axis=-1)
    gt_xy_px = np.stack([(gt.x_centroid.values - x0) / ps,
                         (gt.y_centroid.values - y0) / ps], axis=1)
    return image_2ch, gt_xy_px, len(gt)


def load_merfish(cfg, resolution):
    d = cfg["data_dir"]; stem = cfg["stem"]
    x0, x1 = cfg["roi_x0"], cfg["roi_x1"]
    y0, y1 = cfg["roi_y0"], cfg["roi_y1"]
    w = int(round((x1 - x0) / resolution))
    h = int(round((y1 - y0) / resolution))
    P(f"  ROI µm x[{x0},{x1}] y[{y0},{y1}] → {w}×{h} px @ {resolution} µm/px")
    P(f"  * cross-platform target: DAPI staining and native resolution differ from training data")

    tform = pd.read_csv(
        f"{d}/images/{stem}_images_micron_to_mosaic_pixel_transform.csv",
        sep=r"\s+", header=None).values
    sx, sy = float(tform[0, 0]), float(tform[1, 1])
    ox, oy = float(tform[0, 2]), float(tform[1, 2])
    mx0, my0 = int(np.floor(sx * x0 + ox)), int(np.floor(sy * y0 + oy))
    mx1, my1 = int(np.ceil(sx * x1 + ox)), int(np.ceil(sy * y1 + oy))

    dapi_path = f"{d}/images/{stem}_images_mosaic_DAPI_z3.tif"
    P(f"  reading DAPI {dapi_path}")
    try:
        mm = tifffile.memmap(dapi_path, mode="r")
        FH, FW = mm.shape
        crop = np.ascontiguousarray(
            mm[max(0, my0):min(FH, my1), max(0, mx0):min(FW, mx1)])
    except Exception:
        import zarr
        z = zarr.open(tifffile.imread(dapi_path, aszarr=True), mode="r")
        a = z[0] if isinstance(z, zarr.hierarchy.Group) else z
        FH, FW = a.shape
        crop = np.ascontiguousarray(
            a[max(0, my0):min(FH, my1), max(0, mx0):min(FW, mx1)])
    crop_u8 = to_u8(crop)
    crop_u8 = cv2.resize(crop_u8, (w, h), interpolation=cv2.INTER_AREA)
    P(f"    DAPI crop -> {crop_u8.shape}  mean {crop_u8.mean():.1f}")

    P(f"  reading transcripts + GT centroids ...")
    chunks = []
    for ck in pd.read_csv(
            f"{d}/{stem}_detected_transcripts_S1R1.csv",
            chunksize=2_000_000,
            usecols=["global_x", "global_y", "gene"]):
        sub = ck[(ck.global_x >= x0) & (ck.global_x < x1) &
                 (ck.global_y >= y0) & (ck.global_y < y1)]
        if len(sub):
            chunks.append(sub)
    tx = pd.concat(chunks, ignore_index=True)
    tx = tx[~tx.gene.astype(str).str.startswith(("Blank", "blank", "BLANK"))]
    tpx = ((tx.global_x.values - x0) / resolution).astype(np.int32).clip(0, w-1)
    tpy = ((tx.global_y.values - y0) / resolution).astype(np.int32).clip(0, h-1)
    P(f"    transcripts {len(tx):,} total")
    hm_u8 = gaussian_heatmap(tpx, tpy, h, w)

    meta = pd.read_csv(
        f"{d}/{stem}_cell_metadata_S1R1.csv").rename(
        columns={"Unnamed: 0": "cell_id"})
    gt = meta[(meta.center_x >= x0) & (meta.center_x < x1) &
              (meta.center_y >= y0) & (meta.center_y < y1)]
    P(f"    GT {len(gt):,} cells (reference {cfg['gt_cells']:,})")

    image_2ch = np.stack([crop_u8, hm_u8], axis=-1)
    gt_xy_px = np.stack([(gt.center_x.values - x0) / resolution,
                         (gt.center_y.values - y0) / resolution], axis=1)
    return image_2ch, gt_xy_px, len(gt)


def load_target(name, resolution):
    cfg = TARGETS[name]
    P(f"  {cfg['note']}")
    if cfg["kind"] == "xenium":
        return load_xenium(cfg)
    return load_merfish(cfg, resolution)


# ══════════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════════
def build_model():
    for p in (GENESEGNET_PKG_DIR,):
        if p not in sys.path:
            sys.path.insert(0, p)
    import models   # bare-module import as in GeneSeg_train.py

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"weights not found: {MODEL_PATH}")
    P(f"  weights {os.path.basename(MODEL_PATH)}  "
      f"({os.path.getsize(MODEL_PATH)/1e6:.1f} MB)")

    device, gpu = models.assign_device(use_torch=True, gpu=True, device=0)
    P(f"  device={device}  gpu={gpu}")

    t0 = time.time()
    model = models.GeneSegModel(
        gpu=gpu, device=device,
        pretrained_model=MODEL_PATH,
        model_type=None,
        diam_mean=42.0,          # overwritten by the checkpoint value; value irrelevant
        residual_on=True, style_on=True, concatenation=False,
        net_avg=False,
        nchan=2,                 # DAPI + transcript heatmap, two channels
    )
    P(f"  model loaded ({time.time()-t0:.1f}s)")
    P(f"    actual diam_mean (from checkpoint): {model.diam_mean:.2f}")
    P(f"    actual diam_labels (empirical training diameter): {model.diam_labels:.2f}")
    return model


def run_eval(model, image_2ch, confidence, flow_threshold, min_size,
            diameter=None):
    masks, flows, styles = model.eval(
        image_2ch,
        channels=None,            # matches --all_channels used in training
        diameter=diameter,        # None -> model.diam_labels, fair blind test
        do_3D=False,
        net_avg=False,
        augment=False,
        tile=True,                # tiles large images internally; no manual patching
        resample=True,
        interp=True,
        flow_threshold=flow_threshold,
        confidence_threshold=confidence,
        min_size=min_size,
        stitch_threshold=0.0,
        invert=False,
        batch_size=8,
        normalize=True,
        model_loaded=True,        # weights already loaded in constructor; do not reload
    )
    return masks


def match_to_gt(masks, gt_xy_px, match_radius_px):
    """Rough one-to-one matching for diagnostics only, not a full evaluation"""
    from scipy.spatial import cKDTree
    import skimage.measure
    props = skimage.measure.regionprops(masks)
    if not props:
        return 0, 0
    pred_xy = np.array([[p.centroid[1], p.centroid[0]] for p in props])
    n_pred, n_gt = len(pred_xy), len(gt_xy_px)
    if n_gt == 0:
        return n_pred, 0
    tree = cKDTree(gt_xy_px)
    d, idx = tree.query(pred_xy, k=1, distance_upper_bound=match_radius_px)
    best = {}
    for i in range(n_pred):
        if not np.isfinite(d[i]) or d[i] >= match_radius_px:
            continue
        g = int(idx[i])
        if g not in best or d[i] < best[g]:
            best[g] = d[i]
    return n_pred, len(best)


# ══════════════════════════════════════════════════════════════════════════
def scan_thresholds(model, image_2ch, gt_xy_px, n_gt, match_radius_px,
                    confidences, flow_thresholds, min_size):
    P(f"\n{SEP}")
    P(f"threshold scan (GT {n_gt:,})")
    P(SEP)
    P(f"  {'confidence':>10} {'flow_th':>8} {'raw cells':>10} "
      f"{'match':>6} {'vs GT':>9}")
    P("  " + "-" * 52)
    rows = []
    for conf in confidences:
        for ft in flow_thresholds:
            t0 = time.time()
            masks = run_eval(model, image_2ch, conf, ft, min_size)
            n_pred = int(masks.max())
            n_match, = (0,)
            _, n_match = match_to_gt(masks, gt_xy_px, match_radius_px)
            el = time.time() - t0
            flag = "  ←" if 0.5 * n_gt < n_pred < 1.5 * n_gt else ""
            P(f"  {conf:>10.2f} {ft:>8.2f} {n_pred:>10,} {n_match:>6,} "
              f"{n_pred-n_gt:>+9,}{flag}  ({el:.0f}s)")
            rows.append(dict(confidence=conf, flow_threshold=ft,
                             n_pred=n_pred, n_match=n_match))
    df = pd.DataFrame(rows)
    if df.n_pred.max() == 0:
        P(f"\n  ✘ raw cell count is 0 for every threshold combination")
        P(f"     the flow field has no useful signal for this target, "
          f"matching the failure mode of NSCLC weights on lymph node")
        P(f"     (that scan varied confidence_threshold 0.0-0.6 and count stayed at 99; "
          f"the problem is not the threshold but that the model learned no features for these cells)")
    else:
        df["gap"] = (df.n_pred - n_gt).abs()
        b = df.sort_values("gap").iloc[0]
        P(f"\n  closest to GT: confidence={b.confidence} "
          f"flow_threshold={b.flow_threshold} -> {int(b.n_pred):,} cells")
        P(f"  full run: --run --confidence {b.confidence} "
          f"--flow-threshold {b.flow_threshold}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(TARGETS))
    ap.add_argument("--scan-thresholds", action="store_true")
    ap.add_argument("--confidences", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--flow-thresholds", type=float, nargs="+",
                    default=[0.4, 0.6, 0.8])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--flow-threshold", type=float, default=0.6)
    ap.add_argument("--min-size", type=int, default=300)
    ap.add_argument("--diameter", type=float, default=None,
                    help="empty = blind test (use the model's own diam_labels); "
                         "a value tells the model the target tissue's true cell diameter (px); "
                         "this scores higher but is not a fair generalization test")
    ap.add_argument("--match-radius-um", type=float, default=8.0)
    ap.add_argument("--resolution", type=float, default=0.5,
                    help="merfish only: working resolution um/px")
    ap.add_argument("--gpu", type=int, default=None)
    cfg = ap.parse_args()

    P(SEP); P(f"GeneSegNet inference - {cfg.target}"); P(SEP)
    tgt = TARGETS[cfg.target]
    setup_gpu(cfg.gpu)

    image_2ch, gt_xy_px, n_gt = load_target(cfg.target, cfg.resolution)
    ps = tgt.get("pixel_size", cfg.resolution)
    match_radius_px = cfg.match_radius_um / ps

    model = build_model()

    if cfg.scan_thresholds:
        df = scan_thresholds(model, image_2ch, gt_xy_px, n_gt,
                             match_radius_px, cfg.confidences,
                             cfg.flow_thresholds, cfg.min_size)
        df.to_csv(f"{OUT_DIR}/{cfg.target}_scan.csv", index=False)
        P(f"\n  scan table -> {OUT_DIR}/{cfg.target}_scan.csv")
        return

    if not cfg.run:
        P(f"\n  add --scan-thresholds to scan parameters first, or --run to run once with fixed thresholds")
        return

    P(f"\n  confidence={cfg.confidence}  flow_threshold={cfg.flow_threshold}"
      f"  diameter={'blind (None)' if cfg.diameter is None else cfg.diameter}")
    masks = run_eval(model, image_2ch, cfg.confidence, cfg.flow_threshold,
                     cfg.min_size, cfg.diameter)
    n_pred = int(masks.max())
    n_pred_total, n_match = match_to_gt(masks, gt_xy_px, match_radius_px)
    P(f"\n{SEP}")
    P(f"  cell count {n_pred:,}  (GT {n_gt:,}, ratio {n_pred/max(n_gt,1):.3f})")
    P(f"  rough matching (diagnostics only, not a full evaluation): {n_match:,}")
    out_mask = f"{OUT_DIR}/{cfg.target}_mask.tif"
    tifffile.imwrite(out_mask, masks.astype(np.uint32))
    P(f"  mask → {out_mask}")
    P(f"\n  full 19-metric evaluation: feed this mask into the evaluation framework")
    P(f"  already set up in this conversation (eval_cellist_xenium.py: centroid matching + count correlation +")
    P(f"  gene-vector cosine + assignment accuracy), not just the rough count here")


if __name__ == "__main__":
    main()