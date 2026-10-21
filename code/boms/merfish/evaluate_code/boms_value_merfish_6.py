#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BOMS evaluation - all absolute paths; run with just --dataset. Rasterizes the npz point cloud and reuses the reference eval script's metric logic.
Usage:  python eval_boms.py --dataset prostate1     (or prostate2)
Optional overrides: --root --npz --gt-mask --out-dir --ref --roi-size
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np

REF_DEFAULT  = "yourpath/ucs/UCS/ucs_value_merfish_4.py"
DATA_BASE    = "yourpath/dataset"
RESULT_BASE  = "yourpath"
METHOD       = "BOMS"
NPZ_NAME     = "boms_merfish.npz"
ROI_SIZE_DEF = 10000

def _load_ref(ref):
    ref = Path(ref).resolve()
    if not ref.exists():
        raise SystemExit(f"reference eval script missing: {ref}\nset REF_DEFAULT at top or pass --ref")
    spec = importlib.util.spec_from_file_location("refval", ref)
    V = importlib.util.module_from_spec(spec); sys.modules["refval"] = V
    spec.loader.exec_module(V)
    return V, ref

def _pick_gt_mask(root, dataset):
    cands = [Path(f"{RESULT_BASE}/proseg_merfish_{dataset}/platform_labels.tif") if dataset else None,
             Path(root) / "platform_labels.tif"]
    for c in cands:
        if c and c.exists():
            return str(c)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF_DEFAULT)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--gt-mask", default=None)
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE_DEF)
    args = ap.parse_args()

    if args.dataset:
        args.root    = args.root or f"{DATA_BASE}/merfish_{args.dataset}"
        args.npz     = args.npz  or f"{RESULT_BASE}/boms_merfish_{args.dataset}/{NPZ_NAME}"
        args.out_dir = args.out_dir or f"{RESULT_BASE}/eval_merfish_{args.dataset}"
    if not (args.root and args.npz):
        raise SystemExit("need --dataset, or both --root and --npz")
    args.out_dir = args.out_dir or f"{RESULT_BASE}/eval_merfish_out"
    if not args.gt_mask:
        args.gt_mask = _pick_gt_mask(args.root, args.dataset)

    V, ref = _load_ref(args.ref)
    root, out = Path(args.root), Path(args.out_dir)
    V.setup_log(out, f"eval_BOMS_{args.dataset or 'x'}")
    V.rule(f"BOMS  x  MERFISH {args.dataset or ''}   (via {ref.name})")

    geo = V.load_geometry(root, args.roi_size)
    gt_mask = V.load_gt_mask(root, geo, args.gt_mask)
    tx = V.load_transcripts(root, geo, gt_mask)
    gt = V.load_ground_truth(root, geo, tx)

    z = np.load(args.npz, allow_pickle=True)
    seg = z["seg"].astype(np.int64)
    ux = z["x"].astype(float); uy = z["y"].astype(float)
    size = args.roi_size
    lx, ly = geo.um_to_local(ux, uy)
    r = np.rint(ly).astype(np.int64); c = np.rint(lx).astype(np.int64)
    ok = (r >= 0) & (r < size) & (c >= 0) & (c < size) & (seg > 0)
    lab = np.zeros((size, size), np.int64)
    rad = 5
    yy, xx = np.ogrid[-rad:rad+1, -rad:rad+1]
    dyx = np.nonzero((yy*yy + xx*xx) <= rad*rad)
    dy = dyx[0]-rad; dx = dyx[1]-rad
    rr = r[ok]; cc = c[ok]; ss = seg[ok]
    for oy, ox in zip(dy, dx):
        yr = rr+oy; xr = cc+ox
        m = (yr>=0)&(yr<size)&(xr>=0)&(xr<size)
        lab[yr[m], xr[m]] = ss[m]
    V.log.info(f"boms rasterised: {len(np.unique(lab))-1} cells on {size}x{size}")
    pred = V.Prediction(lab, 1.0, METHOD, uses_platform_prior=False)
    V.evaluate(pred, root, out, geo, tx, gt)

if __name__ == "__main__":
    main()