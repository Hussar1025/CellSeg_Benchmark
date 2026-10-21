#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellsam_benchmark_cosmx_multi.py
================================
Whole-cell segmentation with CellSAM (the generalist foundation model, zero-shot, no training)
across multiple FOVs (default 51~68), benchmarked against CosMx CellLabels (GT). Aggregation
matches the cellpose / stardist / cellotype versions exactly: micro (pooling) is the primary
scope, macro (mean±std) is the robustness scope; metrics reuse test.py; micro aggregation
reuses cellpose_benchmark_cosmx_multi (validated against the stitched-large-image gold
standard, diff ~1e-7).

About CellSAM
-------------
CellSAM = CellFinder (detection boxes) + SAM (mask decoder), a generalist foundation model that
segments across modalities/tissues zero-shot and natively does whole-cell segmentation -> it is
compared directly against whole-cell CellLabels, with no StarDist-style dilation, so it is
horizontally comparable to Cellpose/CelloType under the same scope. Hence, as before: no
training, no fine-tuning (fairness + avoid same-slide FOV leakage).

Inference
---------
The officially recommended interface is cellsam_pipeline(), which auto-downloads weights and
bundles normalization/contrast/post-processing; use_wsi=True gives whole-slide tiling mode,
used here for 4256^2 large FOVs (equivalent to cellpose n_tiles / cellotype SAHI).
This script defaults to cellsam_pipeline(use_wsi=True); older versions without that interface
fall back to segment_cellular_image. You can also run the official script first to produce
masks saved as cellsam_*_F0XX_masks.tif, then use --reuse-pred to only score them.

Dependency
----------
    pip install git+https://github.com/vanvalenlab/cellSAM.git   (python>=3.10, pure PyTorch)
The first run downloads model weights (may need an access token for deepcell/users.deepcell.org;
configure it as the official instructions say).

Input channel order
CosMx composite: R=CD45/GFAP, G=PanCK, B=DAPI. This script builds [nucleus=DAPI, membrane=max(CD45,PanCK)].
CellSAM's nuclear/membrane channel order for multi-channel fluorescence must be confirmed against
the official example; adjust with --channel-order. Swapping them loses points.

Usage
-----
    python cellsam_benchmark_cosmx_multi.py \
        --dep-dir /path/to/cellpose_files \
        --data-root /path/to/CosMx_Pancreas --fov-start 51 --fov-end 68 \
        --device cuda --use-wsi --out-dir ./cellsam_bench_multi

    # Produce masks with the official script first, then only score here
    python cellsam_benchmark_cosmx_multi.py --dep-dir ... --data-root ... \
        --fov-start 51 --fov-end 68 --reuse-pred --pred-dir /path/to/cellsam_masks
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Dependency resolution (cellpose_benchmark_cosmx[_multi].py may live elsewhere)
# --------------------------------------------------------------------------- #
def _resolve_dep_dir():
    env = os.environ.get("COSMX_BENCH_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--dep-dir" and i + 1 < len(argv):
            return os.path.abspath(argv[i + 1])
        if a.startswith("--dep-dir="):
            return os.path.abspath(a.split("=", 1)[1])
    return None


_dep_dir = _resolve_dep_dir()
for _d in (_dep_dir, HERE, os.getcwd()):
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import CosMx_single_fov as single       # file parsing / composite loading / metrics module
    import CosMx_benchplus as agg      # validated micro aggregation + component functions
except ImportError as _e:
    sys.stderr.write(
        "\n[import failed] CosMx_single_fov.py / CosMx_benchplus.py not found.\n"
        "Point to their directory in any one of these three ways:\n"
        "  1) command line: --dep-dir /path/to/cellpose_files\n"
        "  2) env var:      export COSMX_BENCH_DIR=/path/to/cellpose_files\n"
        "  3) put these scripts in the same directory\n"
        f"Tried: {[d for d in (_dep_dir, HERE, os.getcwd()) if d]}\nOriginal error: {_e}\n\n")
    raise


# --------------------------------------------------------------------------- #
# CellSAM input (nucleus + membrane)
# --------------------------------------------------------------------------- #
def build_cellsam_input(composite_rgb, channel_order="nuc_mem"):
    """
    CosMx composite: R=CD45/GFAP, G=PanCK, B=DAPI.
      nucleus  = DAPI = B
      membrane = max(CD45, PanCK) = max(R,G)
    channel_order:
      nuc_mem       -> (H,W,2) = [nucleus, membrane]  (default)
      mem_nuc       -> (H,W,2) = [membrane, nucleus]
      blank_nuc_mem -> (H,W,3) = [0, nucleus, membrane]
      nuc_only      -> (H,W)   DAPI only (conservative fallback when the channel convention is unclear)
    CellSAM's nuclear/membrane channel order must be checked against the official example; swapping loses points.
    Returns float32 (cellsam_pipeline normalizes internally).
    """
    R = composite_rgb[..., 0].astype(np.float32)
    G = composite_rgb[..., 1].astype(np.float32)
    B = composite_rgb[..., 2].astype(np.float32)
    nuclear = B
    membrane = np.maximum(R, G)
    zero = np.zeros_like(B)
    if channel_order == "mem_nuc":
        return np.stack([membrane, nuclear], axis=-1)
    if channel_order == "blank_nuc_mem":
        return np.stack([zero, nuclear, membrane], axis=-1)
    if channel_order == "nuc_only":
        return nuclear
    return np.stack([nuclear, membrane], axis=-1)   # nuc_mem


# --------------------------------------------------------------------------- #
# CellSAM runner (weights loaded once)
# --------------------------------------------------------------------------- #
class CellSAMRunner:
    def __init__(self, device="cuda", use_wsi=True,
                 low_contrast_enhancement=False, gauge_cell_size=False):
        self.device = device
        self.use_wsi = use_wsi
        self.low_contrast_enhancement = low_contrast_enhancement
        self.gauge_cell_size = gauge_cell_size
        self._pipeline = None
        self._seg_fn = None

    def _ensure_loaded(self):
        if self._pipeline is None and self._seg_fn is None:
            try:
                from cellSAM import cellsam_pipeline
                self._pipeline = cellsam_pipeline
                print(f"[cellsam] using cellsam_pipeline(use_wsi={self.use_wsi}) | device={self.device}")
            except Exception:
                from cellSAM import segment_cellular_image
                self._seg_fn = segment_cellular_image
                print("[cellsam] old version: falling back to segment_cellular_image (no tiling; large images may struggle)")

    def segment(self, img):
        self._ensure_loaded()
        if self._pipeline is not None:
            try:
                mask = self._pipeline(
                    img, use_wsi=self.use_wsi,
                    low_contrast_enhancement=self.low_contrast_enhancement,
                    gauge_cell_size=self.gauge_cell_size, device=self.device)
            except TypeError:
                # some versions auto-detect device internally or use different arg names
                mask = self._pipeline(img, use_wsi=self.use_wsi)
        else:
            out = self._seg_fn(img, device=self.device)
            mask = out[0] if isinstance(out, (tuple, list)) else out
        mask = np.asarray(mask)
        if mask.ndim > 2:
            mask = np.squeeze(mask)
        return mask.astype(np.int32, copy=False)


# --------------------------------------------------------------------------- #
# Single FOV
# --------------------------------------------------------------------------- #
def process_one_fov(bench, args, fov, runner):
    comp_path, lab_path = single.resolve_paths(
        args.data_root, fov, args.composite_dir, args.labels_dir)
    composite = single.load_composite(comp_path)
    gt = bench.read_tif_mask(lab_path)             # whole-cell GT

    pred = None
    if args.reuse_pred:
        pat = os.path.join(args.pred_dir or args.out_dir, f"*cellsam*F{fov:03d}*.tif")
        hits = sorted(glob.glob(pat)) or sorted(
            glob.glob(os.path.join(args.pred_dir or args.out_dir, f"*F{fov:03d}*.tif")))
        if hits:
            pred = bench.read_tif_mask(hits[0])
            print(f"  [reuse] {os.path.basename(hits[0])}")
    if pred is None:
        cin = build_cellsam_input(composite, channel_order=args.channel_order)
        pred = runner.segment(cin)
        try:
            import tifffile
            os.makedirs(args.out_dir, exist_ok=True)
            outp = os.path.join(args.out_dir, f"cellsam_F{fov:03d}_masks.tif")
            tifffile.imwrite(outp, pred.astype(np.int32))
        except Exception as e:
            print(f"  [warn] failed to save predicted tif: {e}")

    if gt.shape != pred.shape:
        from skimage.transform import resize
        print(f"  [warn] shape mismatch gt={gt.shape} pred={pred.shape}; nearest-neighbor resizing pred")
        pred = resize(pred, gt.shape, order=0, preserve_range=True,
                      anti_aliasing=False).astype(pred.dtype)

    overlaps = bench.compute_sparse_overlaps(gt, pred)
    comp = agg.pixel_components(bench, gt, pred, boundary_mode=args.boundary_mode)
    inst = agg.instance_components(bench, overlaps, iou_threshold=args.iou_threshold)
    aji_num, aji_den = agg.aji_components(bench, overlaps, gt, pred)
    mda_map = agg.mda_components(bench, overlaps)

    final = agg.per_fov_final_metrics(bench, comp, inst, aji_num, aji_den, mda_map)
    final.update(fov=fov,
                 gt_cell_count=len(overlaps.gt_ids),
                 pred_cell_count=len(overlaps.pred_ids))
    raw = dict(comp=comp, inst=inst, aji_num=aji_num, aji_den=aji_den, mda_map=mda_map)
    return final, raw


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="multi-FOV CellSAM (zero-shot) vs CellLabels benchmark")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dep-dir", default=None,
                    help="Directory holding cellpose_benchmark_cosmx[_multi].py. Equivalent to env COSMX_BENCH_DIR.")
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None)
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=None, help="Path to test.py; default: look in dep-dir / this script dir")
    # CellSAM
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-wsi", dest="use_wsi", action="store_true", default=True,
                    help="whole-slide tiling mode (on by default for large FOVs)")
    ap.add_argument("--no-use-wsi", dest="use_wsi", action="store_false")
    ap.add_argument("--low-contrast-enhancement", action="store_true")
    ap.add_argument("--gauge-cell-size", action="store_true")
    ap.add_argument("--channel-order", default="blank_nuc_mem",
                    choices=["nuc_mem", "mem_nuc", "blank_nuc_mem", "nuc_only"],
                    help="nucleus/membrane channel layout; must match CellSAM's convention (check the official example)")
    # general
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./cellsam_bench_multi")
    ap.add_argument("--reuse-pred", action="store_true", help="reuse saved CellSAM masks, skipping inference")
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    test_py = args.test_py
    if not test_py:
        for _d in (args.dep_dir, _dep_dir, HERE, os.getcwd()):
            if _d and os.path.exists(os.path.join(_d, "test.py")):
                test_py = os.path.join(_d, "test.py")
                break
        test_py = test_py or os.path.join(HERE, "test.py")
    bench = single.load_metric_module(test_py)

    fov_list = args.fovs if args.fovs else list(range(args.fov_start, args.fov_end + 1))
    print(f"[plan] CellSAM (zero-shot) | use_wsi={args.use_wsi} channel_order={args.channel_order} "
          f"| {len(fov_list)} FOV in total: {fov_list}")

    runner = None
    if not args.reuse_pred:
        runner = CellSAMRunner(device=args.device, use_wsi=args.use_wsi,
                               low_contrast_enhancement=args.low_contrast_enhancement,
                               gauge_cell_size=args.gauge_cell_size)

    per_fov, raws, failed = [], [], []
    for fov in fov_list:
        print(f"\n===== FOV {fov} =====")
        try:
            final, raw = process_one_fov(bench, args, fov, runner)
        except Exception as e:
            print(f"  [skip] FOV {fov} failed: {e}")
            failed.append((fov, str(e)))
            continue
        per_fov.append(final)
        raws.append(raw)
        print(f"  gt={final['gt_cell_count']} pred={final['pred_cell_count']} "
              f"F1={final['instance_f1']:.3f} AJI={final['AJI']:.3f} PQ={final['PQ']:.3f}")

    if not per_fov:
        raise SystemExit("No FOV succeeded at all. Check paths / file naming / dependency install.")

    import pandas as pd
    df = pd.DataFrame(per_fov)
    metric_cols = ["dice", "iou", "pixel_accuracy", "boundary_f1", "boundary_precision",
                   "boundary_recall", "instance_f1", "precision", "recall",
                   "DQ", "SQ", "PQ", "AJI", "mDA_0.50_0.95"]
    front = ["fov", "gt_cell_count", "pred_cell_count", "TP", "FP", "FN"]
    df = df[[c for c in front if c in df] + [c for c in metric_cols if c in df]]
    per_fov_csv = os.path.join(args.out_dir, "per_fov_metrics_cellsam.csv")
    df.to_csv(per_fov_csv, index=False)

    micro = agg.aggregate_micro(raws)
    total_gt = int(sum(f["gt_cell_count"] for f in per_fov))
    total_pred = int(sum(f["pred_cell_count"] for f in per_fov))
    macro_mean = {c: float(df[c].mean()) for c in metric_cols if c in df}
    macro_std = {c: float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
                 for c in metric_cols if c in df}

    sdf = pd.DataFrame([dict(metric=c,
                             micro=round(micro.get(c, float("nan")), 6),
                             macro_mean=round(macro_mean.get(c, float("nan")), 6),
                             macro_std=round(macro_std.get(c, float("nan")), 6))
                        for c in metric_cols])
    summary_csv = os.path.join(args.out_dir, "aggregate_summary_cellsam.csv")
    sdf.to_csv(summary_csv, index=False)

    meta = dict(method="CellSAM", model="generalist_pretrained", finetuned=False,
                use_wsi=args.use_wsi, channel_order=args.channel_order,
                n_fov_ok=len(per_fov), n_fov_failed=len(failed), failed=failed,
                fov_list=fov_list, iou_threshold=args.iou_threshold,
                total_gt_cells=total_gt, total_pred_cells=total_pred,
                micro_TP=micro["TP"], micro_FP=micro["FP"], micro_FN=micro["FN"],
                note="micro=pooling (primary scope; AJI/PQ must use it); macro=per-FOV mean±std; "
                     "zero-shot, whole-cell GT, horizontally comparable with Cellpose/StarDist(cell)/CelloType; pixel_accuracy is a weak metric.")
    with open(os.path.join(args.out_dir, "run_meta_cellsam.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"CellSAM (zero-shot) | use_wsi={args.use_wsi}")
    print(f"Succeeded {len(per_fov)}/{len(fov_list)} FOVs | GT={total_gt} pred={total_pred}")
    if failed:
        print(f"Failed FOVs: {[f for f, _ in failed]}")
    print(f"micro counts: TP={micro['TP']} FP={micro['FP']} FN={micro['FN']}")
    print("-" * 78)
    show = sdf.copy()
    show["macro(mean±std)"] = [f"{m:.4f}±{s:.4f}"
                               for m, s in zip(show["macro_mean"], show["macro_std"])]
    print(show[["metric", "micro", "macro(mean±std)"]].to_string(index=False))
    print("-" * 78)
    print("Report micro as primary; macro(mean±std) is the robustness scope. AJI/PQ are micro only.")
    print("CellSAM is natively whole-cell and zero-shot, directly comparable with Cellpose/StarDist(cell)/CelloType on the same GT.")
    print(f"\n[saved]\n  {per_fov_csv}\n  {summary_csv}\n  "
          f"{os.path.join(args.out_dir, 'run_meta_cellsam.json')}")


if __name__ == "__main__":
    main()

'''
nohup bash -c '
    python CosMx_cellsam.py \
    --dep-dir yourpath/cellpose \
    --data-root yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles --fov-start 51 --fov-end 68 \
    --test-py yourpath/eval/test.py --use-wsi \
    --out-dir yourpath/CellSam/CosMx_output
' > output.log 2>&1 &
'''
