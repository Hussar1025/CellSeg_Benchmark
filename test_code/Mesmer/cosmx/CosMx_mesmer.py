#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesmer_benchmark_cosmx_multi.py
===============================
Run Mesmer (DeepCell, TissueNet pretrained, zero-shot, no training) for
whole-cell segmentation on several FOVs (default 51-68) and benchmark it against
CosMx CellLabels (GT). Aggregation matches the cellpose / stardist / cellotype /
cellsam versions exactly: micro (pooling) is the primary scope, macro(mean±std)
the robustness scope; metrics are reused from test.py; micro aggregation is
reused from cellpose_benchmark_cosmx_multi (already validated against the
big-mosaic gold standard, diff ~1e-7).

Two things that must be right for Mesmer
----------------------------------------
1) Input is 2-channel [nucleus, membrane]: channel 1 = nucleus (DAPI),
   channel 2 = membrane/cytoplasm.
   CosMx composite R=CD45/GFAP, G=PanCK, B=DAPI ->
     nucleus  = DAPI = B
     membrane = max(CD45, PanCK) = max(R,G)
   The order is fixed: nucleus first, membrane second (--channel-order can swap
   it, but the default is already the Mesmer convention).
2) image_mpp must be the real resolution of your data. Mesmer's training
   resolution is app.model_mpp=0.5um/px, and predict rescales the image to it
   using image_mpp. CosMx is 0.12028um/px -> pass 0.12028.
   Do not copy the image_mpp=0.5 used for TissueNet elsewhere! That makes Mesmer
   think cells are ~4x larger and under-segment badly.
compartment='whole-cell' -> whole-cell, compared directly against CellLabels,
the same scope as Cellpose/CelloType/CellSAM.
Zero-shot, no training (fairness + avoids same-slide FOV leakage), same reason
as before.

Dependencies
------------
    pip install deepcell            (deepcell-tf, TensorFlow-based; align TF/CUDA with the official versions)
Weights download automatically on the first run; newer versions may need a
DeepCell API key (users.deepcell.org) - set the env var DEEPCELL_ACCESS_TOKEN as
prompted. Pick a GPU via CUDA_VISIBLE_DEVICES.
You can also use the official deepcell-applications to produce masks as
mesmer_*_F0XX_masks.tif, then score only with --reuse-pred.

Usage
-----
    python mesmer_benchmark_cosmx_multi.py \
        --dep-dir /path/to/cellpose_files \
        --data-root /path/to/CosMx_Pancreas --fov-start 51 --fov-end 68 \
        --image-mpp 0.12028 --compartment whole-cell \
        --out-dir ./mesmer_bench_multi

    # produce masks with the official script first, then score here only
    python mesmer_benchmark_cosmx_multi.py --dep-dir ... --data-root ... \
        --fov-start 51 --fov-end 68 --reuse-pred --pred-dir /path/to/mesmer_masks
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Dependency lookup (cellpose_benchmark_cosmx[_multi].py may live elsewhere)
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
    import CosMx_single_fov as single       # file parsing / composite loading / metrics
    import CosMx_benchplus as agg      # validated micro aggregation + component helpers
except ImportError as _e:
    sys.stderr.write(
        "\n[import failed] cannot find cellpose_benchmark_cosmx.py / cellpose_benchmark_cosmx_multi.py.\n"
        "Point to their directory in one of three ways:\n"
        "  1) command line:  --dep-dir /path/to/cellpose_files\n"
        "  2) env var:       export COSMX_BENCH_DIR=/path/to/cellpose_files\n"
        "  3) put these scripts in the same directory\n"
        f"Tried so far: {[d for d in (_dep_dir, HERE, os.getcwd()) if d]}\noriginal error: {_e}\n\n")
    raise


# --------------------------------------------------------------------------- #
# Mesmer input: [1, H, W, 2] = [nucleus, membrane]
# --------------------------------------------------------------------------- #
def build_mesmer_input(composite_rgb, channel_order="nuc_mem"):
    R = composite_rgb[..., 0].astype(np.float32)
    G = composite_rgb[..., 1].astype(np.float32)
    B = composite_rgb[..., 2].astype(np.float32)
    nuclear = B                       # DAPI
    membrane = np.maximum(R, G)       # max(CD45, PanCK)
    if channel_order == "mem_nuc":
        two = np.stack([membrane, nuclear], axis=-1)
    else:                             # nuc_mem (Mesmer convention: nucleus first)
        two = np.stack([nuclear, membrane], axis=-1)
    return two[np.newaxis, ...].astype(np.float32)   # [1,H,W,2]


# --------------------------------------------------------------------------- #
# Mesmer runner (weights loaded once)
# --------------------------------------------------------------------------- #
class MesmerRunner:
    def __init__(self, image_mpp=0.12028, compartment="whole-cell",
                 batch_size=4, pixel_expansion=0):
        self.image_mpp = image_mpp
        self.compartment = compartment
        self.batch_size = batch_size
        self.pixel_expansion = pixel_expansion
        self._app = None

    def _app_obj(self):
        if self._app is None:
            from deepcell.applications import Mesmer
            self._app = Mesmer()
            print(f"[mesmer] Loaded | model_mpp(training resolution)={self._app.model_mpp} "
                  f"| your image_mpp={self.image_mpp} | compartment={self.compartment}")
            if abs(self.image_mpp - 0.5) < 1e-6:
                print("  [warn] image_mpp=0.5 is very likely wrong for CosMx! CosMx is ~0.12028; "
                      "0.5 causes severe under-segmentation. Confirm this is intended.")
        return self._app

    def segment(self, X):
        app = self._app_obj()
        kwargs = dict(image_mpp=self.image_mpp, compartment=self.compartment,
                      batch_size=self.batch_size)
        if self.pixel_expansion and self.compartment == "nuclear":
            kwargs["postprocess_kwargs_nuclear"] = {"pixel_expansion": self.pixel_expansion}
        y = np.asarray(app.predict(X, **kwargs))
        y = np.squeeze(y)             # [1,H,W,1] -> [H,W] (whole-cell)
        if y.ndim > 2:                # compartment='both' -> [H,W,2]; take whole-cell channel 0
            y = y[..., 0]
        return y.astype(np.int32, copy=False)


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
        pat = os.path.join(args.pred_dir or args.out_dir, f"*mesmer*F{fov:03d}*.tif")
        hits = sorted(glob.glob(pat)) or sorted(
            glob.glob(os.path.join(args.pred_dir or args.out_dir, f"*F{fov:03d}*.tif")))
        if hits:
            pred = bench.read_tif_mask(hits[0])
            print(f"  [reuse] {os.path.basename(hits[0])}")
    if pred is None:
        X = build_mesmer_input(composite, channel_order=args.channel_order)
        pred = runner.segment(X)
        try:
            import tifffile
            os.makedirs(args.out_dir, exist_ok=True)
            outp = os.path.join(args.out_dir, f"mesmer_{args.compartment}_F{fov:03d}_masks.tif")
            tifffile.imwrite(outp, pred.astype(np.int32))
        except Exception as e:
            print(f"  [warn] failed to save predicted tif: {e}")

    if gt.shape != pred.shape:
        from skimage.transform import resize
        print(f"  [warn] shape mismatch gt={gt.shape} pred={pred.shape}; nearest-neighbour resizing pred")
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
    ap = argparse.ArgumentParser(description="multi-FOV Mesmer (TissueNet pretrained, zero-shot) vs CellLabels benchmark")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dep-dir", default=None,
                    help="Directory containing CosMx_single_fov.py. Equivalent to env COSMX_BENCH_DIR.")
    ap.add_argument("--fov-start", type=int, default=51)
    ap.add_argument("--fov-end", type=int, default=68)
    ap.add_argument("--fovs", type=int, nargs="*", default=None)
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=None, help="Path to test.py; default: search dep-dir / this script's directory")
    # Mesmer
    ap.add_argument("--image-mpp", type=float, default=0.12028,
                    help="Real resolution of your data (um/px). CosMx=0.12028. Do not use 0.5!")
    ap.add_argument("--compartment", default="whole-cell",
                    choices=["whole-cell", "nuclear", "both"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--pixel-expansion", type=int, default=0,
                    help="Only used with compartment=nuclear: expand nuclei by N pixels")
    ap.add_argument("--channel-order", default="nuc_mem", choices=["nuc_mem", "mem_nuc"],
                    help="Mesmer convention is nucleus first (nuc_mem). Do not change unless the data is really reversed.")
    # Generic
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./mesmer_bench_multi")
    ap.add_argument("--reuse-pred", action="store_true", help="Reuse saved Mesmer masks and skip inference")
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
    print(f"[plan] Mesmer(zero-shot) | image_mpp={args.image_mpp} compartment={args.compartment} "
          f"| {len(fov_list)} FOVs: {fov_list}")

    runner = None
    if not args.reuse_pred:
        runner = MesmerRunner(image_mpp=args.image_mpp, compartment=args.compartment,
                              batch_size=args.batch_size, pixel_expansion=args.pixel_expansion)

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
        raise SystemExit("No FOV succeeded. Check paths / file naming / dependency install.")

    import pandas as pd
    df = pd.DataFrame(per_fov)
    metric_cols = ["dice", "iou", "pixel_accuracy", "boundary_f1", "boundary_precision",
                   "boundary_recall", "instance_f1", "precision", "recall",
                   "DQ", "SQ", "PQ", "AJI", "mDA_0.50_0.95"]
    front = ["fov", "gt_cell_count", "pred_cell_count", "TP", "FP", "FN"]
    df = df[[c for c in front if c in df] + [c for c in metric_cols if c in df]]
    per_fov_csv = os.path.join(args.out_dir, "per_fov_metrics_mesmer.csv")
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
    summary_csv = os.path.join(args.out_dir, "aggregate_summary_mesmer.csv")
    sdf.to_csv(summary_csv, index=False)

    meta = dict(method="Mesmer", model="TissueNet_pretrained", finetuned=False,
                image_mpp=args.image_mpp, compartment=args.compartment,
                channel_order=args.channel_order,
                n_fov_ok=len(per_fov), n_fov_failed=len(failed), failed=failed,
                fov_list=fov_list, iou_threshold=args.iou_threshold,
                total_gt_cells=total_gt, total_pred_cells=total_pred,
                micro_TP=micro["TP"], micro_FP=micro["FP"], micro_FN=micro["FN"],
                note="micro=pooling (primary scope; AJI/PQ must use it); macro=per-FOV mean±std; "
                     "zero-shot, whole-cell GT, directly comparable with Cellpose/StarDist(cell)/CelloType/CellSAM; "
                     "image_mpp must be the real data resolution (CosMx=0.12028); pixel_accuracy is a weak metric.")
    with open(os.path.join(args.out_dir, "run_meta_mesmer.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"Mesmer(zero-shot) | image_mpp={args.image_mpp} compartment={args.compartment}")
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
    print("Report micro as the main result; macro(mean±std) as robustness. AJI/PQ: micro only.")
    print("Mesmer is natively whole-cell and zero-shot; with the same GT it is directly comparable to Cellpose/StarDist(cell)/CelloType/CellSAM.")
    print("Double-check image_mpp is the real data resolution (CosMx=0.12028), not 0.5.")
    print(f"\n[saved]\n  {per_fov_csv}\n  {summary_csv}\n  "
          f"{os.path.join(args.out_dir, 'run_meta_mesmer.json')}")


if __name__ == "__main__":
    main()

'''
nohup bash -c '
    python CosMx_mesmer.py \
    --dep-dir yourpath/cellpose \
    --data-root yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles --fov-start 51 --fov-end 68 \
    --test-py yourpath/eval/test.py \
    --out-dir yourpath/Mesmer/CosMx_output
' > output.log 2>&1 &
'''