#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ucs_starmap.py
=======================================================================
UCS on STARmap BY1, DAPI-only prior (no reference used at inference).

Why this rewrite exists
-----------------------
The previous run produced prior_foreground 1.45% -> pred coverage 1.56%,
i.e. UCS expanded the prior by 8% and lost 50 cells. The diagnosis:

    nucleus median area   = 35 native px  -> diameter 6.7 px
    bin_factor            = 10
    nucleus in bin units  = 0.67 bins

A whole nucleus was smaller than one pixel of the gene map UCS convolves
over, so there was nothing for it to grow. The prior confirmed it: every
one of the 1361 discs came out at exactly 13 px, meaning all of them were
clamped to --min-prior-radius and none carried any size information
(median == p95 == 13).

This version:
  * computes nucleus-diameter-in-bins BEFORE building anything and
    refuses to continue below --min-nucleus-bins (default 2.0)
  * reports the gene-map size first, with --top-genes to keep it bounded
    (1020 genes at bin 2 is 24.6 GB)
  * reports the prior radius distribution, so an all-clamped prior is
    visible immediately rather than after a wasted UCS run
  * compares prior vs prediction at the end, since "did it expand at all"
    is the question that matters

Sizing for reference (DAPI 6990 x 13820, 1020 genes, uint8):
    bin 10 -> 699 x 1382   0.98 GB   nucleus 0.67 bins   unusable
    bin  5 -> 1398 x 2764  3.95 GB   nucleus 1.34 bins   marginal
    bin  3 -> 2330 x 4607  10.9 GB   nucleus 2.23 bins   workable
    bin  2 -> 3495 x 6910  24.6 GB   nucleus 3.34 bins   recommended

Usage
-----
  python ucs_starmap.py --inspect
  python ucs_starmap.py --bin-factor 2 --prepare-only
  python ucs_starmap.py --bin-factor 2 --run-only --gpu 4
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd
import tifffile

DATA_ROOT = "/data/qiuyijia/dataset/starmap_BY1"
RAW_TX = os.path.join(DATA_ROOT, "BY1_raw.csv")
DAPI_PATH = os.path.join(DATA_ROOT, "BY1_dapi.tiff")
UCS_DIR = "/data/qiuyijia/ucs/UCS"
UCS_RUN = os.path.join(UCS_DIR, "run.py")

SEP = "=" * 78


def P(*a):
    print(*a, flush=True)


def work_dir(args):
    return args.out_dir or f"/data/qiuyijia/ucs_starmap_BY1_bin{args.bin_factor}"


def paths(args):
    w = work_dir(args)
    return dict(
        work=w,
        log=os.path.join(w, "ucs_log"),
        gene_map=os.path.join(w, "gene_map.tif"),
        gene_index=os.path.join(w, "gene_index.csv"),
        nuclei_table=os.path.join(w, "dapi_nuclei.csv"),
        nuclei_mask=os.path.join(w, "nuclei_mask.tif"),
        dapi_binary=os.path.join(w, "dapi_binary.tif"),
        qc=os.path.join(w, "input_qc.json"),
        pred=os.path.join(w, "ucs_log", "pred", "segmentation_mask.tif"),
    )


def count_labels(mask):
    ids = np.unique(mask)
    return int(np.sum(ids > 0))


def dapi_shape():
    with tifffile.TiffFile(DAPI_PATH) as tf:
        shape = tf.series[0].shape
    if len(shape) != 2:
        raise RuntimeError(f"DAPI must be 2D: {shape}")
    return tuple(map(int, shape))


def normalize_dapi(img):
    x = img.astype(np.float32)
    p1, p995 = np.percentile(x, [1, 99.5])
    x = np.clip((x - float(p1)) / max(float(p995 - p1), 1e-6), 0, 1)
    return (x * 255).astype(np.uint8)


# ----------------------------------------------------------------------
# nucleus detection (shared by inspect and prior building)
# ----------------------------------------------------------------------
def detect_nuclei(args, want_mask=False):
    raw = tifffile.imread(DAPI_PATH)
    H, W = raw.shape
    img = normalize_dapi(raw)
    del raw
    gc.collect()

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    otsu, _ = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = float(np.clip(otsu * args.dapi_threshold_scale, 1, 254))
    binary = (blur >= thr).astype(np.uint8)
    P(f"  otsu={otsu:.1f}  scaled={thr:.1f}  foreground={binary.mean()*100:.2f}%")

    k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(binary,
                                                              connectivity=8)
    clean = np.zeros_like(binary)
    kept = 0
    for i in range(1, n_cc):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if args.min_nucleus_area <= a <= args.max_nucleus_area:
            clean[labels == i] = 1
            kept += 1
    P(f"  components kept = {kept:,}")

    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    local_max = cv2.dilate(dist, np.ones((5, 5), np.uint8))
    peaks = ((dist >= local_max - 1e-6) & (dist >= 1) & (clean > 0)).astype(np.uint8)
    _, seeds = cv2.connectedComponents(peaks)
    markers = seeds.astype(np.int32) + 1
    markers[clean == 0] = 1
    markers[(clean > 0) & (seeds == 0)] = 0
    ws = cv2.watershed(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), markers)

    ids = np.unique(ws)
    ids = ids[ids >= 2]
    rows = []
    for wid in ids:
        yy, xx = np.where(ws == wid)
        a = len(xx)
        if args.min_nucleus_area <= a <= args.max_nucleus_area:
            rows.append((float(xx.mean()), float(yy.mean()), int(a)))
    nuclei = pd.DataFrame(rows, columns=["x_native", "y_native",
                                         "area_native_px"])
    nuclei.insert(0, "nucleus_id", np.arange(1, len(nuclei) + 1))
    if want_mask:
        return nuclei, (H, W), clean
    del ws, clean, binary, enhanced, blur
    gc.collect()
    return nuclei, (H, W), None


GT_SPOTS = os.path.join(DATA_ROOT, "spots_all.csv")


def gt_cell_scale():
    """Cell size from the clustermap ground truth, as a target for the prior.

    Without this the nucleus detector has nothing to be wrong against: the
    first run reported a 3.34 px nucleus radius and looked self-consistent,
    while the GT cells are 79 px in radius. A 24:1 cell-to-nucleus linear
    ratio is not biology, it is a detector finding only the bright core.
    """
    if not os.path.isfile(GT_SPOTS):
        return None
    s = pd.read_csv(GT_SPOTS, usecols=["spot_location_1", "spot_location_2",
                                       "clustermap"])
    s = s[s.clustermap > 0]
    g = s.groupby("clustermap")
    wx = g.spot_location_1.max() - g.spot_location_1.min()
    wy = g.spot_location_2.max() - g.spot_location_2.min()
    r = np.sqrt(np.clip(wx, 0, None) * np.clip(wy, 0, None) / np.pi)
    return dict(n_cells=int(g.ngroups), cell_r=float(r.median()),
                cell_r_iqr=(float(r.quantile(.25)), float(r.quantile(.75))),
                spots_per_cell=float(g.size().median()))


def report_geometry(nuclei, H, W, args, n_genes=None, gt=None):
    """The check the previous run did not do."""
    med_area = float(nuclei.area_native_px.median())
    med_r = np.sqrt(med_area / np.pi)
    med_d = 2 * med_r
    bins = med_d / args.bin_factor
    map_h = int(np.ceil(H / args.bin_factor))
    map_w = int(np.ceil(W / args.bin_factor))
    P()
    P(f"  nuclei = {len(nuclei):,}")
    P(f"  median nucleus area = {med_area:.1f} native px  "
      f"radius {med_r:.2f}  diameter {med_d:.2f} px")
    P(f"  gene map grid = {map_h} x {map_w}  (bin {args.bin_factor})")
    P(f"  nucleus spans {bins:.2f} bins")
    if n_genes:
        gb = map_h * map_w * n_genes / 1e9
        P(f"  gene map size = {map_h} x {map_w} x {n_genes} = {gb:.2f} GB uint8")
    if gt:
        ratio = gt["cell_r"] / max(med_r, 1e-9)
        P()
        P(f"  GT: {gt['n_cells']:,} cells, equivalent radius "
          f"{gt['cell_r']:.1f} native px "
          f"(IQR {gt['cell_r_iqr'][0]:.1f}-{gt['cell_r_iqr'][1]:.1f}), "
          f"{gt['spots_per_cell']:.0f} spots each")
        P(f"  cell radius / detected nucleus radius = {ratio:.1f}")
        P(f"  GT cell area in bins = "
          f"{np.pi*(gt['cell_r']/args.bin_factor)**2:.0f}")
        if ratio > args.max_cell_nucleus_ratio:
            raise RuntimeError(
                f"the GT cells are {ratio:.0f}x the detected nucleus radius. A "
                "real nucleus is roughly a third of the cell's linear size, so "
                f"a nucleus here should be near {gt['cell_r']/3:.0f} px radius "
                f"({np.pi*(gt['cell_r']/3)**2:.0f} px area), not "
                f"{med_r:.1f} px ({med_area:.0f} px). The DAPI detector is "
                "finding bright cores, not nuclei: lower "
                "--dapi-threshold-scale and raise --min-nucleus-area to reject "
                "fragments. Changing --bin-factor will not help.")
        want_r_bins = gt["cell_r"] / 3 / args.bin_factor
        if args.max_prior_radius < want_r_bins * 0.6:
            P(f"  WARNING: --max-prior-radius {args.max_prior_radius} clamps "
              f"well below the expected nucleus radius of "
              f"{want_r_bins:.1f} bins; the prior will carry no size "
              "information")
    if bins < args.min_nucleus_bins:
        raise RuntimeError(
            f"a nucleus spans only {bins:.2f} bins, below --min-nucleus-bins "
            f"{args.min_nucleus_bins}. UCS convolves over this grid, so it has "
            f"no room to grow the prior and the prediction will barely differ "
            f"from it. Lower --bin-factor to about "
            f"{max(1, int(med_d / args.min_nucleus_bins))} "
            f"(and use --top-genes to keep the gene map bounded).")
    return dict(median_area=med_area, median_diameter=med_d,
                nucleus_bins=bins, map_h=map_h, map_w=map_w)


# ----------------------------------------------------------------------
def stage_inspect(args):
    P(SEP)
    P("INSPECT")
    P(SEP)
    H, W = dapi_shape()
    P(f"DAPI = {H} x {W}")
    tx = pd.read_csv(RAW_TX, usecols=["gene", "x", "y"])
    inside = ((tx.x >= 0) & (tx.x < W) & (tx.y >= 0) & (tx.y < H))
    n_genes = int(tx.gene.nunique())
    P(f"transcripts = {len(tx):,}   inside DAPI = {int(inside.sum()):,}   "
      f"genes = {n_genes}")
    P(f"x range = {tx.x.min():.0f} .. {tx.x.max():.0f}   "
      f"y range = {tx.y.min():.0f} .. {tx.y.max():.0f}")
    dens = int(inside.sum()) / (H * W)
    P(f"transcript density = {dens:.4f} per native px")
    del tx
    gc.collect()

    gt = gt_cell_scale()
    P()
    P("nucleus detection")
    nuclei, (H, W), _ = detect_nuclei(args)
    for bf in (10, 5, 3, 2, 1):
        a = argparse.Namespace(**vars(args))
        a.bin_factor = bf
        a.min_nucleus_bins = 0
        g = report_geometry(nuclei, H, W, a, n_genes, gt)
        flag = "OK" if g["nucleus_bins"] >= 2 else "TOO COARSE"
        P(f"  -> bin {bf}: nucleus {g['nucleus_bins']:.2f} bins  {flag}")
        P()
    P(SEP)
    P("INSPECT DONE")
    P(SEP)


# ----------------------------------------------------------------------
def build_gene_map(args, p, n_genes_hint=None):
    if os.path.isfile(p["gene_map"]) and not args.force_gene_map:
        gm = tifffile.memmap(p["gene_map"])
        P(f"reusing gene map {gm.shape} {gm.dtype}")
        return tuple(gm.shape)

    H, W = dapi_shape()
    map_h = int(np.ceil(H / args.bin_factor))
    map_w = int(np.ceil(W / args.bin_factor))

    tx = pd.read_csv(RAW_TX, usecols=["gene", "x", "y"])
    valid = ((tx.x >= 0) & (tx.x < W) & (tx.y >= 0) & (tx.y < H))
    tx = tx.loc[valid].copy()
    tx["gene"] = tx["gene"].astype(str)
    P(f"transcripts inside DAPI = {len(tx):,}")

    counts = tx.gene.value_counts()
    genes = sorted(counts.index.tolist())
    if args.top_genes > 0 and len(genes) > args.top_genes:
        genes = sorted(counts.index[: args.top_genes].tolist())
        kept = counts[genes].sum() / counts.sum() * 100
        P(f"gene filter: {len(counts)} -> {len(genes)} genes "
          f"({kept:.1f}% of transcripts kept)")
        tx = tx[tx.gene.isin(set(genes))]
    gene2idx = {g: i for i, g in enumerate(genes)}
    pd.DataFrame({"gene": genes, "channel": np.arange(len(genes))}
                 ).to_csv(p["gene_index"], index=False)

    shape = (map_h, map_w, len(genes))
    gb = map_h * map_w * len(genes) / 1e9
    P(f"gene map = {shape}  {gb:.2f} GB uint8")
    if gb > args.max_gene_map_gb:
        raise RuntimeError(
            f"gene map would be {gb:.1f} GB > --max-gene-map-gb "
            f"{args.max_gene_map_gb}. Raise the limit or use --top-genes.")

    bx = tx.x.to_numpy(np.int64) // args.bin_factor
    by = tx.y.to_numpy(np.int64) // args.bin_factor
    gi = tx.gene.map(gene2idx).to_numpy(np.int64)
    del tx
    gc.collect()

    gm = np.zeros(shape, dtype=np.uint8)
    flat = (by * map_w + bx)
    order = np.argsort(gi, kind="stable")
    flat_s, gi_s = flat[order], gi[order]
    bounds = np.searchsorted(gi_s, np.arange(len(genes) + 1))
    clipped = 0
    for g in range(len(genes)):
        a, b = bounds[g], bounds[g + 1]
        if a == b:
            continue
        c = np.bincount(flat_s[a:b], minlength=map_h * map_w)
        clipped += int((c > 255).sum())
        np.clip(c, 0, 255, out=c)
        gm[:, :, g] = c.reshape(map_h, map_w).astype(np.uint8)
    if clipped:
        P(f"WARNING: {clipped} (bin, gene) entries clipped at 255")

    tifffile.imwrite(p["gene_map"], gm, bigtiff=True, photometric="minisblack")
    occ = float((gm.sum(axis=2) > 0).mean())
    P(f"saved {p['gene_map']}   occupied bins = {occ*100:.2f}%")
    del gm
    gc.collect()
    return shape


def build_prior(args, p, nuclei, H, W):
    map_h = int(np.ceil(H / args.bin_factor))
    map_w = int(np.ceil(W / args.bin_factor))
    r_native = np.sqrt(nuclei.area_native_px.to_numpy() / np.pi)
    radii = np.clip(np.round(r_native / args.bin_factor
                             * args.prior_radius_scale),
                    args.min_prior_radius, args.max_prior_radius).astype(int)

    uniq, cnt = np.unique(radii, return_counts=True)
    P()
    P("  prior radius distribution (bins):")
    for u, c in zip(uniq, cnt):
        bar = "#" * max(1, int(40 * c / cnt.max()))
        P(f"    r={u:2d}  {c:6,}  {bar}")
    if len(uniq) == 1:
        P("  WARNING: every prior has the same radius, so the prior carries no "
          "size information. That is what --min-prior-radius clamping looked "
          "like in the previous run; lower it or lower --bin-factor.")

    prior = np.zeros((map_h, map_w), dtype=np.int32)
    for row, r in zip(nuclei.itertuples(), radii):
        bx = int(np.clip(round(row.x_native / args.bin_factor), 0, map_w - 1))
        by = int(np.clip(round(row.y_native / args.bin_factor), 0, map_h - 1))
        cv2.circle(prior, (bx, by), int(r), int(row.nucleus_id), thickness=-1)

    prior = prior.astype(np.uint32)
    tifffile.imwrite(p["nuclei_mask"], prior)
    nuclei.assign(prior_radius_bins=radii).to_csv(p["nuclei_table"], index=False)
    P(f"  prior cells = {count_labels(prior):,}")
    P(f"  prior foreground = {np.mean(prior > 0)*100:.3f}%")
    return prior


# ----------------------------------------------------------------------
def run_ucs(args, p):
    gm = tifffile.memmap(p["gene_map"])
    prior = tifffile.imread(p["nuclei_mask"])
    if gm.shape[:2] != prior.shape:
        raise RuntimeError(f"shape mismatch: {gm.shape} vs {prior.shape}")
    n_genes = int(gm.shape[2])
    del gm

    if os.path.isdir(p["log"]):
        if args.force_run:
            shutil.rmtree(p["log"])
        else:
            raise RuntimeError(f"{p['log']} exists. Use --force-run.")

    # the gene count is the conv input channel count, so it drives activations
    act_gb = args.fg_batch * n_genes * args.patch_size ** 2 * 4 / 1e9
    P(f"fg batch {args.fg_batch} x {n_genes} genes x {args.patch_size}^2 "
      f"-> {act_gb:.2f} GB of input activations")

    cmd = [sys.executable, UCS_RUN,
           "--gene_map", p["gene_map"],
           "--nuclei_mask", p["nuclei_mask"],
           "--log_dir", p["log"],
           "--patch_size", str(args.patch_size),
           "--dilation_kernel_size", str(args.dilation_kernel_size),
           "--dilation_iter_num", str(args.dilation_iter_num),
           "--tau", str(args.tau),
           "--fg_net_epoch", str(args.fg_net_epoch),
           "--fg_net_batch_size", str(args.fg_batch),
           "--cell_net_epoch", str(args.cell_net_epoch),
           "--gpu", str(args.gpu)]
    P()
    P(SEP)
    P("RUN UCS")
    P(SEP)
    P(" ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=UCS_DIR).returncode
    P(f"return code = {rc}   elapsed = {time.time()-t0:.1f}s")
    if rc != 0:
        raise RuntimeError("UCS failed.")
    if not os.path.isfile(p["pred"]):
        raise RuntimeError(f"prediction missing: {p['pred']}")

    pred = tifffile.imread(p["pred"])
    pa = np.bincount(pred.ravel())[1:]
    pa = pa[pa > 0]
    ra = np.bincount(prior.ravel())[1:]
    ra = ra[ra > 0]
    P()
    P(SEP)
    P("DONE")
    P(SEP)
    P(f"prior      cells {len(ra):,}  median area {np.median(ra):.1f} bins  "
      f"foreground {np.mean(prior>0)*100:.2f}%")
    P(f"prediction cells {len(pa):,}  median area {np.median(pa):.1f} bins  "
      f"foreground {np.mean(pred>0)*100:.2f}%")
    exp = np.median(pa) / max(np.median(ra), 1e-9)
    P(f"expansion = {exp:.2f}x")
    if exp < 1.5:
        P("WARNING: the prediction is barely larger than the prior. That is the "
          "failure mode the previous bin-10 run hit; check the nucleus-in-bins "
          "figure above.")
    P(f"mask = {p['pred']}")
    P(SEP)


# ----------------------------------------------------------------------
def get_args():
    a = argparse.ArgumentParser(
        description="UCS on STARmap BY1 with an explicit bin-size guard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument("--out-dir", default=None)
    a.add_argument("--gpu", type=int, default=0)
    a.add_argument("--bin-factor", type=int, default=2)
    a.add_argument("--max-cell-nucleus-ratio", type=float, default=6.0,
                   help="refuse to run if GT cell radius exceeds the detected "
                        "nucleus radius by more than this")
    a.add_argument("--min-nucleus-bins", type=float, default=2.0,
                   help="refuse to run if a nucleus spans fewer bins than this")
    a.add_argument("--top-genes", type=int, default=0,
                   help="keep only the N most abundant genes (0 = all)")
    a.add_argument("--max-gene-map-gb", type=float, default=30.0)

    a.add_argument("--dapi-threshold-scale", type=float, default=1.03)
    a.add_argument("--min-nucleus-area", type=int, default=18)
    a.add_argument("--max-nucleus-area", type=int, default=3000)
    a.add_argument("--prior-radius-scale", type=float, default=1.5)
    a.add_argument("--min-prior-radius", type=int, default=1)
    a.add_argument("--max-prior-radius", type=int, default=4)

    a.add_argument("--patch-size", type=int, default=48)
    a.add_argument("--dilation-kernel-size", type=int, default=10)
    a.add_argument("--dilation-iter-num", type=int, default=4)
    a.add_argument("--tau", type=int, default=5)
    a.add_argument("--fg-net-epoch", type=int, default=1)
    a.add_argument("--fg-batch", type=int, default=8)
    a.add_argument("--cell-net-epoch", type=int, default=1)

    a.add_argument("--inspect", action="store_true")
    a.add_argument("--prepare-only", action="store_true")
    a.add_argument("--run-only", action="store_true")
    a.add_argument("--force-gene-map", action="store_true")
    a.add_argument("--force-prior", action="store_true")
    a.add_argument("--force-run", action="store_true")
    return a.parse_args()


def main():
    args = get_args()
    for f in (RAW_TX, DAPI_PATH, UCS_RUN):
        if not os.path.isfile(f):
            raise FileNotFoundError(f)

    if args.inspect:
        stage_inspect(args)
        return

    p = paths(args)
    os.makedirs(p["work"], exist_ok=True)
    P(SEP)
    P(f"UCS x STARmap BY1   bin_factor={args.bin_factor}")
    P(f"work dir = {p['work']}")
    P(SEP)

    if args.run_only:
        run_ucs(args, p)
        return

    P()
    P("STEP 1 - nucleus detection")
    nuclei, (H, W), _ = detect_nuclei(args)

    tx_genes = pd.read_csv(RAW_TX, usecols=["gene"]).gene.nunique()
    n_genes = min(tx_genes, args.top_genes) if args.top_genes > 0 else tx_genes
    geom = report_geometry(nuclei, H, W, args, n_genes, gt_cell_scale())

    P()
    P("STEP 2 - gene map")
    shape = build_gene_map(args, p, n_genes)

    P()
    P("STEP 3 - prior")
    prior = build_prior(args, p, nuclei, H, W)

    if tuple(shape[:2]) != tuple(prior.shape):
        raise RuntimeError(f"gene map {shape} vs prior {prior.shape}")

    json.dump({"dataset": "STARmap_BY1", "mode": "clean_non_circular",
               "reference_used_in_inference": False,
               "bin_factor": args.bin_factor,
               "gene_map_shape": list(map(int, shape)),
               "prior_cells": count_labels(prior),
               "prior_foreground": float(np.mean(prior > 0)),
               **{k: float(v) for k, v in geom.items()}},
              open(p["qc"], "w"), indent=2)

    P()
    P("PREPARATION COMPLETE")
    if args.prepare_only:
        return
    run_ucs(args, p)


if __name__ == "__main__":
    main()