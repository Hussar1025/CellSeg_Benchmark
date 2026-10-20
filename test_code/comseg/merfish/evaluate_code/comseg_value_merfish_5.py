#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComSeg evaluation - absolute paths only; running needs only --dataset. Reuses the
reference evaluation script's metric/matching/GT logic.
Usage:  python eval_comseg.py --dataset prostate1     (or prostate2)
Optional overrides: --root --run-dir --gt-mask --out-dir --ref --roi-size --mask --seg-name
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np

# ============ absolute path config (only these lines change across machines) ============
REF_DEFAULT   = "yourpath/ucs/UCS/ucs_value_merfish_4.py"
DATA_BASE     = "yourpath/dataset"        # dataset: {DATA_BASE}/merfish_<ds>
RESULT_BASE   = "yourpath"      # results: {RESULT_BASE}/<prefix>_merfish_<ds>
METHOD        = "ComSeg"
RESULT_PREFIX = "comseg"
SEG_PATTERNS  = ["comseg_segmentation.tif","*comseg*.tif","*segmentation*.tif"]
USES_PRIOR    = True
ROI_SIZE_DEF  = 10000
# =======================================================

def _load_ref(ref):
    ref = Path(ref).resolve()
    if not ref.exists():
        raise SystemExit(f"reference evaluation script not found: {ref}\nfix REF_DEFAULT at the top or pass --ref")
    spec = importlib.util.spec_from_file_location("refval", ref)
    V = importlib.util.module_from_spec(spec); sys.modules["refval"] = V
    spec.loader.exec_module(V)
    return V, ref

def _find_seg(run_dir, seg_name):
    d = Path(run_dir)
    if seg_name and (d / seg_name).exists():
        return d / seg_name
    for pat in SEG_PATTERNS:
        cand = sorted(d.glob(pat))
        if cand:
            return cand[0]
    raise SystemExit(f"no segmentation found under {d} (tried {SEG_PATTERNS})")

def _pick_gt_mask(run_dir, root, dataset):
    cands = [Path(run_dir) / "platform_labels.tif",
             Path(f"{RESULT_BASE}/proseg_merfish_{dataset}/platform_labels.tif") if dataset else None,
             Path(root) / "platform_labels.tif"]
    for c in cands:
        if c and c.exists():
            return str(c)
    return None  # let the reference script search under root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF_DEFAULT)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--gt-mask", default=None)
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE_DEF)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--seg-name", default=None)
    args = ap.parse_args()

    if args.dataset:
        args.root    = args.root    or f"{DATA_BASE}/merfish_{args.dataset}"
        args.run_dir = args.run_dir or f"{RESULT_BASE}/{RESULT_PREFIX}_merfish_{args.dataset}"
        args.out_dir = args.out_dir or f"{RESULT_BASE}/eval_merfish_{args.dataset}"
    if not (args.root and args.run_dir):
        raise SystemExit("need --dataset, or both --root and --run-dir")
    args.out_dir = args.out_dir or f"{RESULT_BASE}/eval_merfish_out"
    if not args.gt_mask:
        args.gt_mask = _pick_gt_mask(args.run_dir, args.root, args.dataset)

    V, ref = _load_ref(args.ref)
    root, out = Path(args.root), Path(args.out_dir)
    V.setup_log(out, f"eval_{METHOD}_{args.dataset or 'x'}")
    V.rule(f"{METHOD}  x  MERFISH {args.dataset or ''}   (via {ref.name})")

    geo = V.load_geometry(root, args.roi_size)
    gt_mask = V.load_gt_mask(root, geo, args.gt_mask)
    tx = V.load_transcripts(root, geo, gt_mask)
    gt = V.load_ground_truth(root, geo, tx)

    import tifffile
    p = Path(args.mask) if args.mask else _find_seg(args.run_dir, args.seg_name)
    lab = np.squeeze(tifffile.imread(p)).astype(np.int64)
    V.log.info(f"mask        {p}   {lab.shape}")
    s = args.roi_size / lab.shape[0]
    pred = V.Prediction(lab, s, METHOD, uses_platform_prior=USES_PRIOR)

    V.evaluate(pred, root, out, geo, tx, gt)

if __name__ == "__main__":
    main()