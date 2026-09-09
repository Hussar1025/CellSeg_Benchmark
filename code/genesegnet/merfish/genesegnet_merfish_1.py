#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genesegnet_merfish_1_infer.py
=============================

Standalone GeneSegNet inference for MERFISH1 sealed center 20000x20000 ROI.

This script intentionally bypasses GeneSeg_train.py TEST stage.

Training/inference contract copied from genesegnet_merfish_1.py:
  * DAPI mosaic: z3
  * sealed test ROI: center 20000x20000 px
  * tile size: 1024 px
  * DAPI normalization: percentile pmin=1.0, pmax=99.8 -> uint8
  * RNA heatmap: Gaussian nearest-distance heatmap, sigma=3.0 px
  * model input: [DAPI, HeatMap_all] -> 2 channels
  * transcripts use global_x/global_y in microns, transformed to mosaic pixel
    coordinates using the supplied affine transform.
  * model checkpoint is loaded directly with GeneSegModel.

Default model:
  /data/qiuyijia/genesegnet_merfish_1/model/models/
  GeneSegNet_residual_on_style_off_concatenation_off_model_2026_08_23_13_02_46.093274

Output:
  /data/qiuyijia/genesegnet_merfish_1_infer/
      tiles/*.tif
      genesegnet_merfish1_roi20000_mask.tif
      tile_qc.tsv
      inference_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------
ROOT_DEFAULT = Path("/data/qiuyijia/dataset/merfish_mouse_brain")
DAPI_DEFAULT = ROOT_DEFAULT / "images/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_images_mosaic_DAPI_z3.tif"
TRANSFORM_DEFAULT = ROOT_DEFAULT / "images/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_images_micron_to_mosaic_pixel_transform.csv"
TX_DEFAULT = ROOT_DEFAULT / "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_detected_transcripts_S1R1.csv"

MODEL_DEFAULT = Path(
    "/data/qiuyijia/genesegnet_merfish_1/model/models/"
    "GeneSegNet_residual_on_style_off_concatenation_off_model_2026_08_23_13_02_46.093274"
)

OUT_DEFAULT = Path("/data/qiuyijia/genesegnet_merfish_1_infer")
REPO_DEFAULT = Path("/data/qiuyijia/GeneSegNet")

H = 61310
W = 89085
ROI_SIZE = 20000
ROI_Y0 = (H - ROI_SIZE) // 2
ROI_X0 = (W - ROI_SIZE) // 2
ROI_Y1 = ROI_Y0 + ROI_SIZE
ROI_X1 = ROI_X0 + ROI_SIZE

DEFAULT_TILE = 1024
DEFAULT_STRIDE = 896  # 128 px overlap
DEFAULT_SIGMA = 3.0
DEFAULT_PMIN = 1.0
DEFAULT_PMAX = 99.8


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def banner(msg):
    log("=" * 96)
    log(msg)
    log("=" * 96)


# ---------------------------------------------------------------------
# GeneSegNet import
# ---------------------------------------------------------------------
def import_geneseg(repo: Path):
    pkg = repo / "GeneSegNet"
    if not (pkg / "models.py").exists():
        raise FileNotFoundError(pkg / "models.py")
    sys.path.insert(0, str(pkg))
    sys.path.insert(0, str(repo))
    import models
    import utils
    import fastremap
    return models, utils, fastremap


# ---------------------------------------------------------------------
# Raw TIFF reader via memmap
# ---------------------------------------------------------------------
def open_dapi(path: Path):
    # This TIFF is a single uncompressed strip. tifffile.memmap avoids the
    # previous zarr-aszarr zero-read problem on this file.
    a = tifffile.memmap(str(path))
    if a.shape != (H, W):
        raise RuntimeError(f"DAPI shape={a.shape}, expected {(H,W)}")
    return a


# ---------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------
def load_transform(path: Path):
    # File is whitespace-delimited 3x3 with no header.
    M = np.loadtxt(path)
    if M.shape != (3, 3):
        raise RuntimeError(f"transform shape={M.shape}")
    return M


def micron_to_px(x_um, y_um, M):
    pts = np.c_[x_um, y_um, np.ones(len(x_um), float)]
    out = pts @ M.T
    return out[:, 0], out[:, 1]


# ---------------------------------------------------------------------
# Transcript cache restricted to sealed ROI
# ---------------------------------------------------------------------
def prepare_roi_transcripts(tx_path: Path, transform_path: Path, cache: Path, chunksize=2_000_000):
    if cache.exists():
        log(f"reuse transcript ROI cache: {cache}")
        return pd.read_parquet(cache)

    M = load_transform(transform_path)
    pieces = []
    total = 0
    kept = 0

    for d in pd.read_csv(tx_path, chunksize=chunksize):
        total += len(d)

        if "global_x" not in d.columns or "global_y" not in d.columns:
            raise KeyError(f"missing global_x/global_y; columns={list(d.columns)}")

        gx = pd.to_numeric(d["global_x"], errors="coerce").to_numpy(float)
        gy = pd.to_numeric(d["global_y"], errors="coerce").to_numpy(float)

        px, py = micron_to_px(gx, gy, M)

        take = (
            np.isfinite(px) & np.isfinite(py) &
            (px >= ROI_X0) & (px < ROI_X1) &
            (py >= ROI_Y0) & (py < ROI_Y1)
        )

        if np.any(take):
            q = d.loc[take, ["gene"]].copy()
            q["px"] = px[take]
            q["py"] = py[take]
            pieces.append(q)
            kept += len(q)

        log(f"transcripts scanned={total:,}; ROI kept={kept:,}")

    if not pieces:
        raise RuntimeError("zero transcripts in sealed ROI")

    out = pd.concat(pieces, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    log(f"ROI transcript cache -> {cache}; rows={len(out):,}")
    return out


# ---------------------------------------------------------------------
# Training-compatible image normalization
# ---------------------------------------------------------------------
def normalize_image(im, pmin, pmax):
    im = np.asarray(im, dtype=np.float32)
    lo, hi = np.percentile(im, [pmin, pmax])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(im.shape, dtype=np.uint8), float(lo), float(hi)

    z = (im - lo) / (hi - lo)
    z = np.clip(z, 0, 1)
    return np.rint(z * 255).astype(np.uint8), float(lo), float(hi)


# ---------------------------------------------------------------------
# Heatmap -- exact semantic match to training:
# exp(-d^2 / 2 sigma^2), where d = nearest transcript distance
# ---------------------------------------------------------------------
def make_heatmap(px, py, y0, x0, S, sigma):
    if len(px) == 0:
        return np.zeros((S, S), dtype=np.uint8)

    pts = np.c_[py - y0, px - x0]
    keep = (
        (pts[:, 0] >= 0) & (pts[:, 0] < S) &
        (pts[:, 1] >= 0) & (pts[:, 1] < S)
    )
    pts = pts[keep]

    if len(pts) == 0:
        return np.zeros((S, S), dtype=np.uint8)

    # Efficient nearest-distance image using KD-tree in chunks.
    tree = cKDTree(pts)
    out = np.empty((S, S), dtype=np.float32)

    block = 128
    xs = np.arange(S, dtype=np.float32)

    for ya in range(0, S, block):
        yb = min(S, ya + block)
        yy, xx = np.meshgrid(
            np.arange(ya, yb, dtype=np.float32),
            xs,
            indexing="ij",
        )
        query = np.c_[yy.ravel(), xx.ravel()]
        d, _ = tree.query(query, k=1, workers=-1)
        out[ya:yb] = d.reshape(yb - ya, S)

    hm = np.exp(-(out * out) / (2.0 * sigma * sigma)) * 255.0
    return np.clip(hm, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------
def tile_origins(size, tile, stride):
    ss = list(range(0, max(size - tile + 1, 1), stride))
    last = size - tile
    if ss[-1] != last:
        ss.append(last)
    return ss


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def build_model(models, args):
    device, gpu = models.assign_device(
        use_torch=True,
        gpu=True,
        device=args.gpu,
    )

    log(f"device={device}; gpu={gpu}")

    m = models.GeneSegModel(
        gpu=gpu,
        device=device,
        pretrained_model=str(args.model),
        model_type=None,
        diam_mean=args.diam_mean,
        residual_on=True,
        style_on=False,
        concatenation=False,
        net_avg=False,
        nchan=2,
    )

    log(f"checkpoint diam_labels={getattr(m, 'diam_labels', None)}")
    return m


def infer_one(model, utils, fastremap, image_u8, hm_u8, args):
    x = np.stack([image_u8, hm_u8], axis=-1)

    out = model.eval(
        x,
        batch_size=args.batch,
        channels=None,
        channel_axis=-1,
        diameter=args.diameter,
        do_3D=False,
        net_avg=False,
        augment=False,
        tile=True,
        tile_overlap=args.network_tile_overlap,
        resample=True,
        interp=True,
        flow_threshold=args.flow_threshold,
        confidence_threshold=args.confidence_threshold,
        compute_masks=True,
        min_size=args.min_size,
        stitch_threshold=0.0,
    )

    mask = np.asarray(out[0])
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    mask = utils.fill_holes_and_remove_small_masks(
        mask.astype(np.int32, copy=False),
        min_size=args.min_size,
    )
    mask = fastremap.renumber(mask, in_place=True)[0]
    return np.asarray(mask, dtype=np.uint32)


# ---------------------------------------------------------------------
# Stitch by overlap IoU-like containment
# ---------------------------------------------------------------------
class UF:
    def __init__(self):
        self.p = {}

    def add(self, a):
        self.p.setdefault(int(a), int(a))

    def find(self, a):
        a = int(a)
        self.add(a)
        if self.p[a] != a:
            self.p[a] = self.find(self.p[a])
        return self.p[a]

    def union(self, a, b):
        if not a or not b:
            return
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb


def stitch(records, roi_size, args):
    canvas = np.zeros((roi_size, roi_size), dtype=np.uint32)
    uf = UF()
    next_id = 1
    links = 0

    for r in records:
        y0, x0 = r["y"], r["x"]
        m = r["mask"]

        lut = np.zeros(int(m.max()) + 1, dtype=np.uint32)
        for lab in np.unique(m):
            if lab == 0:
                continue
            lut[int(lab)] = next_id
            uf.add(next_id)
            next_id += 1

        incoming = lut[m]
        region = canvas[y0:y0+m.shape[0], x0:x0+m.shape[1]]

        a = region.ravel()
        b = incoming.ravel()
        good = (a > 0) & (b > 0)

        if np.any(good):
            aa = a[good]
            bb = b[good]

            pairs, inter = np.unique(
                np.stack([aa, bb], axis=1),
                axis=0,
                return_counts=True,
            )
            ca = dict(zip(*np.unique(aa, return_counts=True)))
            cb = dict(zip(*np.unique(bb, return_counts=True)))

            for (ga, gb), ni in zip(pairs, inter):
                if ni < args.stitch_min_pixels:
                    continue
                frac = ni / max(min(ca[ga], cb[gb]), 1)
                if frac >= args.stitch_fraction:
                    uf.union(int(ga), int(gb))
                    links += 1

        empty = region == 0
        region[empty] = incoming[empty]

    ids = np.unique(canvas)
    ids = ids[ids > 0]

    remap = np.zeros(int(canvas.max()) + 1, dtype=np.uint32)
    roots = {}
    n = 1
    for lab in ids:
        root = uf.find(int(lab))
        if root not in roots:
            roots[root] = n
            n += 1
        remap[int(lab)] = roots[root]

    return remap[canvas], links


def qc(mask, px_um):
    labs, cnt = np.unique(mask[mask > 0], return_counts=True)
    if len(labs) == 0:
        return {}
    med = float(np.median(cnt))
    return {
        "cells": int(len(labs)),
        "foreground": float((mask > 0).mean()),
        "area_med_px": med,
        "diameter_med_um": float(2 * np.sqrt(med / np.pi) * px_um),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--dapi", type=Path, default=DAPI_DEFAULT)
    ap.add_argument("--tx", type=Path, default=TX_DEFAULT)
    ap.add_argument("--transform", type=Path, default=TRANSFORM_DEFAULT)
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--repo", type=Path, default=REPO_DEFAULT)

    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)

    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--pmin", type=float, default=DEFAULT_PMIN)
    ap.add_argument("--pmax", type=float, default=DEFAULT_PMAX)

    # 0 means checkpoint diam_labels.
    ap.add_argument("--diameter", type=float, default=0.0)
    ap.add_argument("--diam-mean", type=float, default=34.0)
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--confidence-threshold", type=float, default=0.9)
    ap.add_argument("--min-size", type=int, default=300)
    ap.add_argument("--network-tile-overlap", type=float, default=0.1)

    ap.add_argument("--stitch-min-pixels", type=int, default=20)
    ap.add_argument("--stitch-fraction", type=float, default=0.5)

    ap.add_argument("--pixel-size-um", type=float, default=1/9.205855)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true")

    args = ap.parse_args()

    for p in (args.dapi, args.tx, args.transform, args.model):
        if not p.exists():
            raise FileNotFoundError(p)

    banner("GeneSegNet MERFISH1 sealed ROI inference")
    log(f"ROI px = y[{ROI_Y0},{ROI_Y1}) x[{ROI_X0},{ROI_X1})")
    log(f"tile={args.tile}, stride={args.stride}, overlap={args.tile-args.stride}")
    log(f"DAPI normalization p{args.pmin}-p{args.pmax}")
    log(f"RNA heatmap sigma={args.sigma}")
    log(f"model={args.model}")

    ys = tile_origins(ROI_SIZE, args.tile, args.stride)
    xs = tile_origins(ROI_SIZE, args.tile, args.stride)
    log(f"tiles={len(ys)} x {len(xs)} = {len(ys)*len(xs)}")

    if args.dry:
        banner("DRY RUN DONE")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    tile_dir = args.out / "tiles"
    tile_dir.mkdir(exist_ok=True)
    cache = args.out / "cache/roi_transcripts.parquet"

    tx = prepare_roi_transcripts(args.tx, args.transform, cache)
    tx_px = tx["px"].to_numpy(float)
    tx_py = tx["py"].to_numpy(float)

    dapi = open_dapi(args.dapi)
    models, utils, fastremap = import_geneseg(args.repo)
    model = build_model(models, args)

    rows = []
    recs = []

    for iy, ry in enumerate(ys):
        for ix, rx in enumerate(xs):
            gy = ROI_Y0 + ry
            gx = ROI_X0 + rx
            name = f"merfish1_roi_y{ry:05d}_x{rx:05d}"
            op = tile_dir / f"{name}_mask.tif"

            if op.exists() and not args.force:
                m = tifffile.imread(op)
                log(f"reuse {name}: cells={int(m.max())}")
            else:
                im_raw = np.asarray(dapi[gy:gy+args.tile, gx:gx+args.tile])
                im, lo, hi = normalize_image(im_raw, args.pmin, args.pmax)

                # Query transcripts slightly outside tile for smooth boundary heatmap.
                pad = max(16.0, args.sigma * 5)
                keep = (
                    (tx_px >= gx-pad) & (tx_px < gx+args.tile+pad) &
                    (tx_py >= gy-pad) & (tx_py < gy+args.tile+pad)
                )

                hm = make_heatmap(
                    tx_px[keep], tx_py[keep],
                    gy, gx, args.tile, args.sigma
                )

                log(
                    f"infer {name}: tx={int(keep.sum()):,}, "
                    f"DAPI pmin/pmax={lo:.1f}/{hi:.1f}"
                )

                t0 = time.time()
                m = infer_one(model, utils, fastremap, im, hm, args)
                tifffile.imwrite(op, m, compression="zlib")
                log(f"    cells={int(m.max())}, time={time.time()-t0:.1f}s")

            q = qc(m, args.pixel_size_um)
            q.update({"tile": name, "y": ry, "x": rx, "mask": str(op)})
            rows.append(q)
            recs.append({"y": ry, "x": rx, "mask": m})

    pd.DataFrame(rows).to_csv(args.out / "tile_qc.tsv", sep="\t", index=False)

    banner("STITCH MERFISH1 ROI")
    full, links = stitch(recs, ROI_SIZE, args)
    full_path = args.out / "genesegnet_merfish1_roi20000_mask.tif"
    tifffile.imwrite(full_path, full, compression="zlib")

    q = qc(full, args.pixel_size_um)
    q.update({
        "roi_y0": ROI_Y0,
        "roi_x0": ROI_X0,
        "roi_size": ROI_SIZE,
        "stitch_links": links,
        "model": str(args.model),
        "diameter_arg": args.diameter,
        "checkpoint_diam_labels": float(getattr(model, "diam_labels", np.nan)),
        "heatmap_sigma": args.sigma,
        "pmin": args.pmin,
        "pmax": args.pmax,
        "uses_gt_during_inference": False,
    })

    (args.out / "inference_summary.json").write_text(
        json.dumps(q, indent=2, ensure_ascii=False)
    )

    banner("MERFISH1 INFERENCE DONE")
    for k, v in q.items():
        log(f"{k}: {v}")
    log(f"mask -> {full_path}")


if __name__ == "__main__":
    main()
