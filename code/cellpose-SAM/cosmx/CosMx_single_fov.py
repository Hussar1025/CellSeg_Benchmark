#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellpose_benchmark_cosmx.py
============================
On a single FOV of the NanoString CosMx SMI Pancreas dataset:

  1) Run Cellpose (default Cellpose-SAM / cpsam, aligned with output_cpsam_masks in test.py)
     on the CellComposite morphology image for whole-cell instance segmentation;
  2) Use the official CellLabels mask as ground truth (GT);
  3) Reuse the metrics in test.py to output a full benchmark:
     pixel level (Dice / IoU / pixel_acc / boundary-F1)
     instance level (F1 / P / R / DQ / SQ / PQ / over-&under-seg)
     AJI, mDA(0.50:0.95).

Design notes
------------
* GT is CellLabels (whole-cell segmentation), so Cellpose must run whole-cell, not nucleus-only;
  a nucleus is smaller than the whole cell, so IoU would be systematically low - that is a scope
  mismatch, not real error.
* Cellpose input comes from CellComposite:
    - cytoplasm/membrane channel = max(Red=CD45/GFAP, Green=PanCK)   (cover epithelium + immune + stroma)
    - nucleus channel            = Blue = DAPI
* CellLabels and CellComposite are both native 4256x4256 FOVs at 0.12028 um/px, so they should match;
  the code still checks shapes and, on mismatch, nearest-neighbour rescales the predicted mask to the GT size before evaluation.

Caveat on scope (state this honestly in the paper)
--------------------------------------------------
CellLabels is itself the output of NanoString's own segmentation algorithm, not manual per-cell annotation.
So strictly this benchmark measures agreement between Cellpose and the vendor segmentation pipeline:
silver-standard / reference-agreement, not absolute accuracy. This is epistemically symmetric with what
test.py originally did (using Cellpose as reference to compare StarDist/CellSAM/Mesmer), only the reference
is now CellLabels and the method under test is Cellpose.

Usage
-----
    # specific FOV
    python cellpose_benchmark_cosmx.py --data-root /path/to/CosMx_Pancreas --fov 51

    # no FOV: auto-pick the FOV with the most / densest cells from metadata as a stress test
    python cellpose_benchmark_cosmx.py --data-root /path/to/CosMx_Pancreas \
        --metadata /path/to/Pancreas_metadata_file.csv

    # re-score an existing predicted mask only (skips cellpose)
    python cellpose_benchmark_cosmx.py --data-root ... --fov 51 \
        --pred-mask /some/pred_masks.tif --skip-cellpose

Directory layout (standard CosMx export)
----------------------------------------
    <data-root>/
        CellComposite/  CellComposite_F051.jpg   (or .png/.tif)
        CellLabels/      CellLabels_F051.tif
        (optional) CompartmentLabels/ ...
        (optional) Pancreas_metadata_file.csv
FOV numbers in file names are usually zero-padded to 3 digits (F051). This script matches flexibly,
accepting both padded and unpadded forms.
"""

import os
import re
import sys
import glob
import argparse
import numpy as np


# --------------------------------------------------------------------------- #
# 0. Load the metric implementation from test.py (reuse it so the scope matches the existing pipeline)
# --------------------------------------------------------------------------- #
def load_metric_module(test_py_path):
    """Import test.py as a module and grab its metric functions."""
    test_py_path = os.path.abspath(test_py_path)
    mod_dir = os.path.dirname(test_py_path)
    mod_name = os.path.splitext(os.path.basename(test_py_path))[0]
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    try:
        bench = __import__(mod_name)
    except Exception as e:
        raise ImportError(
            f"cannot import metric functions from {test_py_path}: {e}\n"
            f"point --test-py at your test.py."
        )
    required = [
        "read_tif_mask", "compute_pixel_metrics", "compute_sparse_overlaps",
        "instance_metrics_from_overlaps", "aji_from_overlaps",
        "mean_detection_accuracy_from_overlaps",
    ]
    missing = [f for f in required if not hasattr(bench, f)]
    if missing:
        raise ImportError(f"test.py is missing functions: {missing}")
    return bench


# --------------------------------------------------------------------------- #
# 1. File resolution: locate CellComposite / CellLabels by FOV
# --------------------------------------------------------------------------- #
def _find_fov_file(folder, prefix, fov, exts):
    """
    Find a file like {prefix}_F{fov}.{ext} in folder; padded and unpadded FOV numbers both work.
    e.g. prefix=CellLabels, fov=51 -> CellLabels_F051.tif / CellLabels_F51.tif ...
    """
    if not os.path.isdir(folder):
        return None
    # candidate FOV strings: F051, F51, F0051, ...
    fov_variants = {f"F{fov}", f"F{fov:02d}", f"F{fov:03d}", f"F{fov:04d}"}
    candidates = []
    for ext in exts:
        candidates += glob.glob(os.path.join(folder, f"{prefix}_*{ext}"))
        candidates += glob.glob(os.path.join(folder, f"{prefix}_*{ext.upper()}"))
    for path in sorted(set(candidates)):
        name = os.path.basename(path)
        m = re.search(r"[Ff](\d+)", name)
        if not m:
            continue
        n = int(m.group(1))
        if n == fov or any(v.lstrip("F") == m.group(1) for v in fov_variants):
            return path
    # last resort: try exact names
    for ext in exts:
        for v in fov_variants:
            p = os.path.join(folder, f"{prefix}_{v}{ext}")
            if os.path.exists(p):
                return p
    return None


def resolve_paths(data_root, fov, composite_dir=None, labels_dir=None):
    composite_dir = composite_dir or os.path.join(data_root, "CellComposite")
    labels_dir = labels_dir or os.path.join(data_root, "CellLabels")

    comp = _find_fov_file(composite_dir, "CellComposite", fov,
                          exts=[".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    lab = _find_fov_file(labels_dir, "CellLabels", fov,
                         exts=[".tif", ".tiff", ".png"])
    if comp is None:
        raise FileNotFoundError(
            f"CellComposite for FOV {fov} not found (in {composite_dir}).")
    if lab is None:
        raise FileNotFoundError(
            f"CellLabels for FOV {fov} not found (in {labels_dir}).")
    return comp, lab


def auto_pick_fov(metadata_csv, fov_col="fov"):
    """Pick the FOV with the most (densest) cells from metadata as the representative stress-test FOV."""
    import pandas as pd
    df = pd.read_csv(metadata_csv, usecols=lambda c: c in (fov_col, "cell_ID", "cell"))
    if fov_col not in df.columns:
        raise ValueError(f"metadata has no column '{fov_col}'")
    counts = df.groupby(fov_col).size().sort_values(ascending=False)
    best = int(counts.index[0])
    print(f"[auto-fov] top 5 FOVs by cell count:\n{counts.head(5).to_string()}")
    print(f"[auto-fov] picked densest FOV = {best} ({int(counts.iloc[0])} cells)")
    return best


# --------------------------------------------------------------------------- #
# 2. Build the Cellpose input
# --------------------------------------------------------------------------- #
def load_composite(path):
    """Read CellComposite, return float32 (H,W,3) RGB in R,G,B order."""
    from skimage.io import imread
    img = imread(path)
    if img.ndim == 2:                       # grayscale fallback
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:                  # has alpha
        img = img[..., :3]
    return img.astype(np.float32)


def build_cellpose_input(composite_rgb, cyto_channel="maxmarker"):
    """
    CellComposite pseudo-color: R=CD45/GFAP, G=PanCK, B=DAPI (yellow CD68 = R+G mix).
    Returns (H,W,2) [cyto, nucleus].
      cyto_channel:
        'maxmarker' -> cyto = max(R, G)  (covers epithelial PanCK + immune CD45 + CD68 yellow)
        'panck'     -> cyto = G          (epithelium only; pancreatic acini are brightest)
        'dapi_only' -> cyto = a copy of B (extreme fallback, almost nucleus-only)
    nucleus is always B(DAPI).
    """
    R, G, B = composite_rgb[..., 0], composite_rgb[..., 1], composite_rgb[..., 2]
    if cyto_channel == "panck":
        cyto = G
    elif cyto_channel == "dapi_only":
        cyto = B.copy()
    else:  # maxmarker
        cyto = np.maximum(R, G)
    nuc = B
    return np.stack([cyto, nuc], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# 3. Run Cellpose (supports v4/cpsam and classic v3 models)
# --------------------------------------------------------------------------- #
def run_cellpose(img2, model_name="cpsam", diameter=None, gpu=True,
                 flow_threshold=0.4, cellprob_threshold=0.0):
    """
    img2: (H,W,2) = [cyto, nucleus]
    returns a 2D int32 label mask.
    """
    import cellpose
    from cellpose import models
    ver = getattr(cellpose, "version", "0")
    try:
        major = int(str(ver).split(".")[0])
    except Exception:
        major = 0
    print(f"[cellpose] version={ver} | model={model_name} | gpu={gpu} "
          f"| diameter={diameter}")

    eval_kwargs = dict(diameter=diameter,
                       flow_threshold=flow_threshold,
                       cellprob_threshold=cellprob_threshold)

    if major >= 4:
        # Cellpose-SAM: channel-agnostic, takes multi-channel directly; channel_axis is the last dim
        try:
            model = models.CellposeModel(gpu=gpu, pretrained_model=model_name)
        except TypeError:
            model = models.CellposeModel(gpu=gpu)  # cpsam is the v4 default
        res = model.eval(img2, channel_axis=2, normalize=True, **eval_kwargs)
    else:
        # classic v3: channels=[cyto, nucleus]=[1,2] (1->array channel 0, 2->channel 1)
        try:
            model = models.CellposeModel(gpu=gpu, model_type=model_name)
        except TypeError:
            model = models.Cellpose(gpu=gpu, model_type=model_name)
        res = model.eval(img2, channels=[1, 2], normalize=True, **eval_kwargs)

    masks = res[0]                       # supports (masks,flows,styles[,diams])
    masks = np.asarray(masks)
    if masks.ndim > 2:
        masks = np.squeeze(masks)
    return masks.astype(np.int32, copy=False)


# --------------------------------------------------------------------------- #
# 4. Evaluate: GT=CellLabels, pred=Cellpose, reusing test.py metrics
# --------------------------------------------------------------------------- #
def evaluate(bench, gt_mask, pred_mask, method_name="Cellpose-SAM",
             boundary_mode="thick", iou_threshold=0.5):
    if gt_mask.shape != pred_mask.shape:
        from skimage.transform import resize
        print(f"[warn] shape mismatch GT={gt_mask.shape} pred={pred_mask.shape}, "
              f"rescaling pred to GT size with nearest neighbour.")
        pred_mask = resize(pred_mask, gt_mask.shape, order=0,
                           preserve_range=True, anti_aliasing=False
                           ).astype(pred_mask.dtype)

    pixel = bench.compute_pixel_metrics(gt_mask, pred_mask,
                                        boundary_mode=boundary_mode)
    ov = bench.compute_sparse_overlaps(gt_mask, pred_mask)
    inst = bench.instance_metrics_from_overlaps(ov, threshold=iou_threshold)
    aji = bench.aji_from_overlaps(ov, gt_mask, pred_mask)
    mda = bench.mean_detection_accuracy_from_overlaps(ov)

    return {
        "reference": "CosMx_CellLabels(GT)",
        "method": method_name,
        "gt_cell_count": int(len(ov.gt_ids)),
        "pred_cell_count": int(len(ov.pred_ids)),
        **pixel,
        **inst,
        "AJI": float(aji),
        "mDA_0.50_0.95": float(mda),
        "mDA_note": "mean TP/(TP+FP+FN) over IoU 0.50:0.95 (not standard AP)",
    }


# --------------------------------------------------------------------------- #
# 5. Optional: GT vs prediction boundary overlay
# --------------------------------------------------------------------------- #
def save_overlay(composite_rgb, gt_mask, pred_mask, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage.segmentation import find_boundaries
    except Exception as e:
        print(f"[overlay] skip (missing matplotlib/skimage): {e}")
        return
    base = composite_rgb.copy()
    base = base / (base.max() + 1e-8)
    gt_b = find_boundaries(gt_mask, mode="thick")
    pr_b = find_boundaries(pred_mask, mode="thick")
    vis = base.copy()
    vis[gt_b] = [0, 1, 0]     # GT green
    vis[pr_b] = [1, 0, 1]     # pred magenta
    plt.figure(figsize=(12, 12))
    plt.imshow(vis)
    plt.title("Green = CellLabels(GT)   Magenta = Cellpose")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[overlay] Saved {out_png}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Cellpose segmentation on one CosMx Pancreas FOV + benchmark against CellLabels")
    ap.add_argument("--data-root", required=True, help="CosMx export root directory")
    ap.add_argument("--fov", type=int, default=None, help="FOV number; if omitted, auto-pick the densest from metadata")
    ap.add_argument("--metadata", default=None, help="Pancreas_metadata_file.csv (for auto FOV picking)")
    ap.add_argument("--composite-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--test-py", default=os.path.join(os.path.dirname(__file__), "test.py"),
                    help="path to your test.py (reuses its metrics)")
    ap.add_argument("--model", default="cpsam",
                    help="cellpose model: cpsam (default, matches test.py) / cyto3 / cyto2 ...")
    ap.add_argument("--cyto-channel", default="maxmarker",
                    choices=["maxmarker", "panck", "dapi_only"])
    ap.add_argument("--diameter", type=float, default=None,
                    help="cell diameter (px); pancreas cells are ~85-125 px (10-15um/0.12028). None=auto")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--boundary-mode", default="thick")
    ap.add_argument("--out-dir", default="./cellpose_bench_out")
    ap.add_argument("--skip-cellpose", action="store_true",
                    help="do not run cellpose; score --pred-mask directly")
    ap.add_argument("--pred-mask", default=None, help="existing predicted mask tif (with --skip-cellpose)")
    ap.add_argument("--overlay", action="store_true", help="also write a GT/pred boundary overlay")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bench = load_metric_module(args.test_py)

    # 1) decide the FOV
    fov = args.fov
    if fov is None:
        md = args.metadata or os.path.join(args.data_root, "Pancreas_metadata_file.csv")
        if not os.path.exists(md):
            raise SystemExit("no --fov given and metadata not found, cannot auto-pick a FOV.")
        fov = auto_pick_fov(md)

    # 2) locate files
    comp_path, lab_path = resolve_paths(
        args.data_root, fov, args.composite_dir, args.labels_dir)
    print(f"[io] FOV={fov}\n     composite = {comp_path}\n     labels    = {lab_path}")

    # 3) GT
    gt_mask = bench.read_tif_mask(lab_path)
    print(f"[gt] CellLabels shape={gt_mask.shape}, cell count={len(bench.get_instance_ids(gt_mask))}")

    # 4) predicted mask
    composite = load_composite(comp_path)
    if args.skip_cellpose:
        if not args.pred_mask:
            raise SystemExit("--skip-cellpose requires --pred-mask")
        pred_mask = bench.read_tif_mask(args.pred_mask)
        method_name = f"pred:{os.path.basename(args.pred_mask)}"
    else:
        cp_in = build_cellpose_input(composite, cyto_channel=args.cyto_channel)
        pred_mask = run_cellpose(
            cp_in, model_name=args.model, diameter=args.diameter,
            gpu=(not args.no_gpu),
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold)
        method_name = f"Cellpose({args.model})"
        # save the predicted mask (tif, can be fed straight back into test.py's pipeline)
        try:
            import tifffile
            pred_tif = os.path.join(args.out_dir, f"cellpose_{args.model}_F{fov:03d}_masks.tif")
            tifffile.imwrite(pred_tif, pred_mask.astype(np.int32))
            print(f"[io] Saved predicted mask {pred_tif}")
        except Exception as e:
            print(f"[warn] saving tif failed (missing tifffile?): {e}")

    print(f"[pred] shape={pred_mask.shape}, cell count={len(bench.get_instance_ids(pred_mask))}")

    # 5) evaluation
    result = evaluate(bench, gt_mask, pred_mask, method_name=method_name,
                      boundary_mode=args.boundary_mode,
                      iou_threshold=args.iou_threshold)
    result["fov"] = fov
    result["cyto_channel"] = args.cyto_channel

    # 6) output
    import pandas as pd
    df = pd.DataFrame([result])
    out_csv = os.path.join(args.out_dir, f"benchmark_cellpose_vs_CellLabels_F{fov:03d}.csv")
    df.to_csv(out_csv, index=False)

    print("\n================ Benchmark (GT=CellLabels, method=Cellpose) ================")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.T)
    print(f"\n[done] results saved {out_csv}")

    if args.overlay:
        save_overlay(composite, gt_mask, pred_mask,
                     os.path.join(args.out_dir, f"overlay_F{fov:03d}.png"))


if __name__ == "__main__":
    main()