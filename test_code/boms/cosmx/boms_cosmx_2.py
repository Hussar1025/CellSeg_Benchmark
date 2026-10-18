#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boms_cosmx_2_fullgenes_v2.py

CosMx2-specific BOMS runner for held-out FOV48 / FOV261.

Defaults are CosMx2-specific, NOT inherited from Xenium:
  K=300
  h_s=3.5 um
  h_r=0.5
  epochs=30
  patch=120 um
  overlap=30 um
  top_genes=0  -> KEEP ALL OBSERVED GENES

Full genes are always attempted first.  The dry-run profiles every tile and
checks n_transcripts * n_genes against an int32 safety margin.  If unsafe, it
suggests a smaller patch size; it does NOT silently switch to HVGs.

Input:
  /data/qiuyijia/dataset/cosmx_lymph_node/_inspect_cache/tx_fov_48_261.parquet

Output:
  /data/qiuyijia/boms_cosmx_2_fullgenes/fov48/
  /data/qiuyijia/boms_cosmx_2_fullgenes/fov261/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

BOMS_SRC = "/data/qiuyijia/boms/boms_src"
sys.path.insert(0, BOMS_SRC)

try:
    from boms.main import run_boms
except Exception as e:
    raise SystemExit(
        f"cannot import boms.main.run_boms from {BOMS_SRC}: {e}"
    )

TX_DEFAULT = (
    "/data/qiuyijia/dataset/cosmx_lymph_node/_inspect_cache/"
    "tx_fov_48_261.parquet"
)
META_DEFAULT = (
    "/data/qiuyijia/dataset/cosmx_lymph_node/flat_files/"
    "S0_metadata_file.csv.gz"
)
OUT_DEFAULT = "/data/qiuyijia/boms_cosmx_2_fullgenes"

PX_UM = 0.12
FOV_PX = 4256
FOV_UM = FOV_PX * PX_UM

K_DEFAULT = 300
HS_DEFAULT = 3.5
HR_DEFAULT = 0.5
EPOCHS_DEFAULT = 30
TOP_GENES_DEFAULT = 0
PATCH_UM_DEFAULT = 120.0
OVERLAP_UM_DEFAULT = 30.0

INT32_MAX = 2**31 - 1
SAFE_FRAC_DEFAULT = 0.80


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def choose(cols, names, required=True):
    lut = {str(c).lower(): c for c in cols}
    for n in names:
        if n.lower() in lut:
            return lut[n.lower()]
    if required:
        raise KeyError(f"cannot find {names}; columns={list(cols)}")
    return None


def gt_count(meta_path, fov):
    m = pd.read_csv(meta_path)
    fc = choose(m.columns, ("fov", "FOV"))
    return int((pd.to_numeric(m[fc], errors="coerce") == int(fov)).sum())


def load_fov(tx_path, fov, top_genes):
    d = pd.read_parquet(tx_path)

    fc = choose(d.columns, ("fov", "FOV"))
    xc = choose(d.columns, ("x_local_px", "x"))
    yc = choose(d.columns, ("y_local_px", "y"))
    gc = choose(d.columns, ("target", "gene", "feature_name"))

    q = d[pd.to_numeric(d[fc], errors="coerce") == int(fov)].copy()
    if q.empty:
        raise RuntimeError(f"FOV{fov}: no transcripts")

    x = pd.to_numeric(q[xc], errors="coerce").to_numpy(float) * PX_UM
    y = pd.to_numeric(q[yc], errors="coerce").to_numpy(float) * PX_UM
    gene = q[gc].astype(str).to_numpy()

    keep = (
        np.isfinite(x) & np.isfinite(y) &
        (x >= 0) & (x < FOV_UM) &
        (y >= 0) & (y < FOV_UM)
    )
    x, y, gene = x[keep], y[keep], gene[keep]

    names, codes = np.unique(gene, return_inverse=True)
    codes = codes.astype(np.int32)

    log(f"FOV{fov}: transcripts={len(x):,}")
    log(f"FOV{fov}: FULL observed genes={len(names):,}")

    if top_genes > 0 and top_genes < len(names):
        cnt = np.bincount(codes, minlength=len(names))
        order = np.argsort(cnt)[::-1]
        sel = order[:top_genes]

        remap = np.full(len(names), top_genes, dtype=np.int32)
        remap[sel] = np.arange(top_genes, dtype=np.int32)

        covered = int(cnt[sel].sum())
        codes = remap[codes]
        names = np.concatenate(
            [names[sel], np.array(["__other__"], dtype=object)]
        )

        log(
            f"EXPLICIT fallback only: genes {len(cnt):,} -> "
            f"{top_genes:,}+1, coverage={covered/len(x)*100:.2f}%"
        )
    else:
        log("gene reduction: OFF; retaining all observed genes")

    return x, y, codes, names


def starts(patch_um, overlap_um):
    step = patch_um - overlap_um
    if step <= 0:
        raise ValueError("overlap must be smaller than patch")

    ss = [0.0]
    while ss[-1] + patch_um < FOV_UM:
        nxt = ss[-1] + step
        if nxt + patch_um >= FOV_UM:
            last = max(0.0, FOV_UM - patch_um)
            if last > ss[-1] + 1e-9:
                ss.append(last)
            break
        ss.append(nxt)
    return ss


def make_tiles(patch_um, overlap_um):
    ss = starts(patch_um, overlap_um)
    out = []
    for iy, ya in enumerate(ss):
        for ix, xa in enumerate(ss):
            out.append((
                iy, ix,
                xa, min(xa + patch_um, FOV_UM),
                ya, min(ya + patch_um, FOV_UM),
            ))
    return out


def tile_idx(x, y, tile):
    _, _, xa, xb, ya, yb = tile
    return np.where(
        (x >= xa) & (x < xb) &
        (y >= ya) & (y < yb)
    )[0]


def preflight(x, y, n_gene, patch_um, overlap_um, safe_frac):
    tiles = make_tiles(patch_um, overlap_um)
    counts = np.array([len(tile_idx(x, y, t)) for t in tiles], dtype=np.int64)

    worst = int(counts.max())
    elems = worst * int(n_gene)
    lim = int(INT32_MAX * safe_frac)
    safe = elems <= lim
    raw_gib = elems * 4 / (1024**3)

    log("")
    log("=" * 88)
    log("FULL-GENE PREFLIGHT")
    log("=" * 88)
    log(f"FOV               : {FOV_UM:.2f} x {FOV_UM:.2f} um")
    log(f"transcripts       : {len(x):,}")
    log(f"genes             : {n_gene:,}")
    log(f"patch / overlap   : {patch_um:g} / {overlap_um:g} um")
    log(f"tiles             : {len(tiles)}")
    log(
        f"tile tx min/med/max: {int(counts.min()):,} / "
        f"{int(np.median(counts)):,} / {worst:,}"
    )
    log(f"worst tx*genes    : {elems:,} ({elems/1e9:.3f}e9)")
    log(f"safety limit      : {lim:,} ({safe_frac*100:.0f}% int32)")
    log(f"raw dense int32   : ~{raw_gib:.2f} GiB")
    log(f"status            : {'SAFE' if safe else 'UNSAFE'}")
    log("=" * 88)

    if not safe:
        factor = math.sqrt(lim / max(elems, 1))
        proposal = max(
            35,
            int(math.floor(patch_um * factor * 0.90 / 5.0) * 5)
        )
        log(f"Suggested next patch: --patch-um {proposal}")
        log("Keep FULL genes; shrink patch before considering HVG.")

    return safe, tiles, counts


def checkpoint_name(tile, args, n_gene):
    iy, ix, *_ = tile
    return (
        f"tile_{iy:03d}_{ix:03d}"
        f"_K{args.k}_hs{args.hs:g}_hr{args.hr:g}"
        f"_ep{args.epochs}_p{args.patch_um:g}_o{args.overlap_um:g}"
        f"_g{n_gene}.npz"
    )


def compress_ids(labels):
    good = labels >= 0
    out = np.full(len(labels), -1, dtype=np.int32)
    if not good.any():
        return out, 0
    _, inv = np.unique(labels[good], return_inverse=True)
    out[good] = inv.astype(np.int32)
    return out, int(inv.max()) + 1


def summarize(counts, gt):
    counts = counts[counts > 0]
    med = float(np.median(counts))
    mean = float(np.mean(counts))
    return {
        "cells": int(len(counts)),
        "ratio": float(len(counts) / gt),
        "median": med,
        "mean": mean,
        "mean_median": mean / med,
        "p5": float(np.percentile(counts, 5)),
        "p95": float(np.percentile(counts, 95)),
        "frag_lt10": float(np.mean(counts < 10)),
    }


def run_one(args, fov):
    out = Path(args.out) / f"fov{fov}"
    ckpt = out / "ckpt"
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)

    x, y, codes, names = load_fov(args.tx, fov, args.top_genes)
    n_gene = int(codes.max()) + 1
    gt = gt_count(args.meta, fov)

    log("")
    log("=" * 88)
    log(f"BOMS COSMX2 FOV{fov}")
    log("=" * 88)
    log(f"GT count (QC only): {gt:,}")
    log(f"tx/GT cell        : {len(x)/gt:.1f}")
    log(
        f"PARAMS             : K={args.k}, hs={args.hs:g}um, "
        f"hr={args.hr:g}, epochs={args.epochs}"
    )
    log(
        f"GENES              : "
        f"{'FULL' if args.top_genes == 0 else args.top_genes}"
    )
    log(
        f"TILING             : patch={args.patch_um:g}um, "
        f"overlap={args.overlap_um:g}um"
    )
    log("=" * 88)

    safe, tiles, counts = preflight(
        x, y, n_gene,
        args.patch_um,
        args.overlap_um,
        args.safe_frac,
    )

    pd.DataFrame([
        {
            "tile": f"tile_{t[0]:03d}_{t[1]:03d}",
            "xa_um": t[2],
            "xb_um": t[3],
            "ya_um": t[4],
            "yb_um": t[5],
            "n_transcripts": int(n),
            "n_genes": n_gene,
            "dense_elements": int(n) * n_gene,
        }
        for t, n in zip(tiles, counts)
    ]).to_csv(out / "tile_preflight.tsv", sep="\t", index=False)

    pre = {
        "dataset": "cosmx_2",
        "fov": fov,
        "K": args.k,
        "h_s": args.hs,
        "h_r": args.hr,
        "epochs": args.epochs,
        "top_genes": args.top_genes,
        "gene_mode": "full" if args.top_genes == 0 else "reduced",
        "n_genes": n_gene,
        "n_transcripts": int(len(x)),
        "gt_cells_qc_only": gt,
        "patch_um": args.patch_um,
        "overlap_um": args.overlap_um,
        "preflight_safe": bool(safe),
    }
    (out / "preflight.json").write_text(json.dumps(pre, indent=2))

    if not safe:
        raise SystemExit(
            f"FOV{fov}: full-gene preflight UNSAFE at "
            f"patch={args.patch_um:g} um; use suggested smaller patch."
        )

    if args.dry:
        log(f"FOV{fov}: DRY complete")
        return

    owner_dist = np.full(len(x), np.inf)
    provisional = np.full(len(x), -1, dtype=np.int64)
    next_id = 0
    t0_all = time.time()

    for ti, tile in enumerate(tiles, 1):
        iy, ix, xa, xb, ya, yb = tile
        tag = f"tile_{iy:03d}_{ix:03d}"
        idx = tile_idx(x, y, tile)

        if len(idx) == 0:
            log(f"[{ti}/{len(tiles)}] {tag}: empty")
            continue

        if len(idx) < args.k + 1:
            log(
                f"[{ti}/{len(tiles)}] {tag}: "
                f"{len(idx):,} tx < K+1={args.k+1}; skip"
            )
            continue

        fp = ckpt / checkpoint_name(tile, args, n_gene)

        if args.resume and fp.exists():
            z = np.load(fp)
            seg = np.asarray(z["seg"]).ravel()
            if len(seg) != len(idx):
                raise RuntimeError(f"stale checkpoint: {fp}")
            log(f"[{ti}/{len(tiles)}] reuse {tag}, tx={len(idx):,}")
        else:
            log(
                f"[{ti}/{len(tiles)}] RUN {tag}: "
                f"{len(idx):,} tx x {n_gene:,} genes"
            )
            t0 = time.time()

            r = run_boms(
                x[idx],
                y[idx],
                codes[idx],
                int(args.epochs),
                float(args.hs),
                float(args.hr),
                K=int(args.k),
            )

            seg = np.asarray(
                r[1] if isinstance(r, (tuple, list)) and len(r) > 1 else r
            ).ravel()

            if len(seg) != len(idx):
                raise RuntimeError(
                    f"{tag}: returned {len(seg)} labels for {len(idx)} tx"
                )

            np.savez_compressed(
                fp,
                seg=seg,
                idx=idx,
                K=args.k,
                h_s=args.hs,
                h_r=args.hr,
                epochs=args.epochs,
                n_gene=n_gene,
            )

            log(
                f"    local cells={len(np.unique(seg[seg>=0])):,}, "
                f"time={time.time()-t0:.1f}s"
            )

        cx = (xa + xb) / 2
        cy = (ya + yb) / 2
        d = np.hypot(x[idx] - cx, y[idx] - cy)

        take = d < owner_dist[idx]
        good = take & (seg >= 0)

        provisional[idx[good]] = seg[good] + next_id
        owner_dist[idx[good]] = d[good]

        nonneg = seg[seg >= 0]
        if len(nonneg):
            next_id += int(nonneg.max()) + 1

    final, ncell = compress_ids(provisional)
    valid = final >= 0

    if ncell == 0:
        raise RuntimeError(f"FOV{fov}: zero final cells")

    cell_counts = np.bincount(final[valid], minlength=ncell)

    cx = np.bincount(
        final[valid], weights=x[valid], minlength=ncell
    ) / np.maximum(cell_counts, 1)

    cy = np.bincount(
        final[valid], weights=y[valid], minlength=ncell
    ) / np.maximum(cell_counts, 1)

    cm = coo_matrix(
        (
            np.ones(int(valid.sum()), dtype=np.int32),
            (final[valid], codes[valid]),
        ),
        shape=(ncell, n_gene),
    ).tocsr()

    count_mat = np.asarray(cm.todense(), dtype=np.int32)

    s = summarize(cell_counts, gt)

    outfile = out / f"boms_cosmx_2_fov{fov}.npz"

    np.savez_compressed(
        outfile,
        seg=final,
        x=x,
        y=y,
        gene=codes,
        gene_names=names,
        cell_loc=np.c_[cx, cy],
        count_mat=count_mat,
        K=args.k,
        h_s=args.hs,
        h_r=args.hr,
        epochs=args.epochs,
        top_genes=args.top_genes,
        patch_um=args.patch_um,
        overlap_um=args.overlap_um,
        fov=fov,
        gt_cells=gt,
        pixel_size_um=PX_UM,
    )

    meta = {
        **pre,
        "runtime_min": (time.time() - t0_all) / 60,
        "assigned": int(valid.sum()),
        "assign_rate": float(valid.mean()),
        **s,
        "output": str(outfile),
    }

    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))

    log("")
    log("=" * 88)
    log(f"FOV{fov} DONE")
    log("=" * 88)
    log(f"cells             : {ncell:,} / {gt:,} = {s['ratio']:.3f}")
    log(f"assigned          : {valid.mean()*100:.2f}%")
    log(
        f"tx/cell           : med={s['median']:.0f}, mean={s['mean']:.0f}, "
        f"mean/med={s['mean_median']:.2f}"
    )
    log(
        f"                    p5={s['p5']:.0f}, p95={s['p95']:.0f}, "
        f"<10={s['frag_lt10']*100:.2f}%"
    )
    log(f"saved             : {outfile}")
    log("=" * 88)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--tx", default=TX_DEFAULT)
    ap.add_argument("--meta", default=META_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--fovs", default="48,261")

    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--hs", type=float, default=HS_DEFAULT)
    ap.add_argument("--hr", type=float, default=HR_DEFAULT)
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)

    ap.add_argument(
        "--top-genes",
        type=int,
        default=TOP_GENES_DEFAULT,
        help="0 = keep ALL genes"
    )

    ap.add_argument("--patch-um", type=float, default=PATCH_UM_DEFAULT)
    ap.add_argument("--overlap-um", type=float, default=OVERLAP_UM_DEFAULT)
    ap.add_argument("--safe-frac", type=float, default=SAFE_FRAC_DEFAULT)

    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    log(
        f"ACTIVE DEFAULTS: K={args.k} hs={args.hs:g} hr={args.hr:g} "
        f"epochs={args.epochs} top_genes={args.top_genes} "
        f"patch={args.patch_um:g} overlap={args.overlap_um:g}"
    )

    if args.top_genes == 0:
        log("ACTIVE GENE MODE: FULL GENES")

    for fov in [
        int(v.strip())
        for v in args.fovs.split(",")
        if v.strip()
    ]:
        run_one(args, fov)


if __name__ == "__main__":
    main()
