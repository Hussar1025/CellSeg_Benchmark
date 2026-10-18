#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""boms_merfish_3_run.py   v2026.08.14

BOMS on merfish_3 (MERFISH 人肝癌 肝2).

Self-contained.  Every dataset has its OWN ROI and they are not interchangeable,
so this file carries only one:

    ROI mosaic 中心 10000 px  -> x[5771.0,6851.0] y[4971.0,6051.0] um  (1080 x 1080 um)

and it recomputes that window from the data at startup; a difference of more than
2 um stops the run instead of quietly segmenting a different piece of tissue.

Patching, because run_boms indexes an n_tx x n_gene array and must stay under 2^31:

    4,605,924 tx x 550 genes = 2.5e9  ->  2x2 = 4 patches

The patch count is computed from the real numbers rather than hard-coded, each
patch is checkpointed the moment it finishes, a cell is kept only by the patch
whose core its centroid falls in, and count_mat is rebuilt from the global gene
table so the patches are always the same width.

The transcripts come from the same ROI subset the other methods were given, not a
fresh filter of the raw table, so every method on this dataset sees identical
input.

    python boms_merfish_3_run.py --sweep-hs      # 扫 h_s，只跑一个 patch
    python boms_merfish_3_run.py --h-s <值>      # 正式跑，可断点续跑
    python boms_merfish_3_run.py --status        # 看进度
"""


from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "2026.08.14"
INT32 = 2 ** 31

DATASETS = {
    "xenium_4": dict(
        platform="xenium", label="Xenium 肝",
        data_root="/data/qiuyijia/dataset/xenium_liver",
        pixel_size=0.2125, y0=13704, x0=21974, roi_px=10000,
        roi_um=(4669.47, 6794.47, 2912.10, 5037.10),
        hs_default=20.0, hs_sweep=[10, 15, 20, 25, 30, 40],
    ),
    "xenium_5": dict(
        platform="xenium", label="Xenium 小鼠脑",
        data_root="/data/qiuyijia/dataset/xenium_mouse_brain",
        pixel_size=0.2125, y0=6956, x0=12077, roi_px=10000,
        roi_um=(2566.36, 4691.36, 1478.15, 3603.15),
        hs_default=20.0, hs_sweep=[10, 15, 20, 25, 30],
    ),
    "merfish_2": dict(
        platform="merfish", label="MERFISH 人肝癌 肝1",
        data_root="/data/qiuyijia/dataset/merfish_liver1",
        tx_subset="/data/qiuyijia/proseg/output/liver1/transcripts_roi_with_prior.csv.gz",
        roi_px=10000, roi_um=(5236.7, 6316.7, 3811.0, 4891.0), hs_default=70.0, hs_sweep=[30, 50, 70, 100, 150],
    ),
    "merfish_3": dict(
        platform="merfish", label="MERFISH 人肝癌 肝2",
        data_root="/data/qiuyijia/dataset/merfish_liver2",
        tx_subset="/data/qiuyijia/proseg/output/liver2/transcripts_roi_with_prior.csv.gz",
        roi_px=10000, roi_um=(5771.0, 6851.0, 4971.0, 6051.0), hs_default=70.0, hs_sweep=[30, 50, 70, 100, 150],
    ),
    "merfish_4": dict(
        platform="merfish", label="MERFISH 人肺癌",
        data_root="/data/qiuyijia/dataset/MERFISH_Lung_cancer",
        roi_px=10000, roi_um=(3783.0, 4862.9, 3346.8, 4426.8), hs_default=70.0, hs_sweep=[30, 50, 70, 100, 150],
    ),
}

SEP = "=" * 74


def log(*a):
    print(*a, flush=True)


def find(root: Path, *pats):
    for p in pats:
        h = sorted(root.rglob(p))
        if h:
            return h[0]
    return None


# ---------------------------------------------------------------------------
# ROI, in micron, identical to what the other methods were given
# ---------------------------------------------------------------------------


def roi_bounds(cfg) -> tuple:
    """
    The ROI in micron.

    Each dataset has its own window and they are NOT interchangeable, so the
    values confirmed against the other methods are written down here and then
    recomputed from the data; a mismatch stops the run rather than quietly
    segmenting a different piece of tissue.
    """
    root = Path(cfg["data_root"])
    expect = cfg.get("roi_um")

    if cfg["platform"] == "xenium":
        ps = cfg["pixel_size"]
        x0, y0, S = cfg["x0"], cfg["y0"], cfg["roi_px"]
        got = (x0 * ps, (x0 + S) * ps, y0 * ps, (y0 + S) * ps)
        log(f"  ROI from y0={y0} x0={x0} at {ps} um/px")
    else:
        tf = find(root, "*micron_to_mosaic*transform*.csv")
        if tf is None:
            raise SystemExit(f"no micron_to_mosaic transform under {root}")
        m = np.loadtxt(tf)
        sx, sy = float(m[0, 0]), float(m[1, 1])
        ox, oy = float(m[0, 2]), float(m[1, 2])
        import tifffile
        dapi = next((q for q in sorted(root.rglob("*.tif"))
                     if "dapi" in q.name.lower()), None)
        if dapi is None:
            raise SystemExit(f"no *dapi*.tif under {root} (needed for the mosaic size)")
        with tifffile.TiffFile(dapi) as t:
            H, W = t.series[0].shape[-2:]
        S = cfg["roi_px"]
        py0, px0 = (H - S) // 2, (W - S) // 2
        log(f"  mosaic {H} x {W} ({dapi.name})   ROI y[{py0},{py0+S}) "
            f"x[{px0},{px0+S}) px   {sx:.4f} px/um")
        got = ((px0 - ox) / sx, (px0 + S - ox) / sx,
               (py0 - oy) / sy, (py0 + S - oy) / sy)

    if expect:
        d = max(abs(a - b) for a, b in zip(got, expect))
        log(f"  recomputed x[{got[0]:.1f},{got[1]:.1f}] y[{got[2]:.1f},{got[3]:.1f}]")
        log(f"  expected   x[{expect[0]:.1f},{expect[1]:.1f}] "
            f"y[{expect[2]:.1f},{expect[3]:.1f}]   max diff {d:.2f} um")
        if d > 2.0:
            raise SystemExit(
                "the recomputed ROI differs from the one the other methods used "
                f"by {d:.1f} um. Either the data moved or this is the wrong "
                "dataset; fix it before running, or pass --roi-um to override.")
    return got


def load_tx(cfg, bounds, qv_min: float):
    x0u, x1u, y0u, y1u = bounds
    log(f"  ROI x[{x0u:.1f},{x1u:.1f}] y[{y0u:.1f},{y1u:.1f}] um  "
        f"({x1u-x0u:.1f} x {y1u-y0u:.1f})")
    root = Path(cfg["data_root"])

    sub = cfg.get("tx_subset")
    if sub and Path(sub).exists():
        log(f"  reusing the ROI subset the other methods were given:")
        log(f"    {sub}")
        d = pd.read_csv(sub)
        xc = "global_x" if "global_x" in d else "x"
        yc = "global_y" if "global_y" in d else "y"
        gc = "gene" if "gene" in d else "target"
    elif cfg["platform"] == "merfish":
        cache = find(root / "_roi_cache", "tx_roi_*.parquet", "tx_roi_*.csv.gz")
        if cache is not None:
            log(f"  reusing {cache.name}")
            d = (pd.read_parquet(cache) if str(cache).endswith(".parquet")
                 else pd.read_csv(cache))
            xc, yc, gc = "global_x", "global_y", "gene"
        else:
            src = find(root, "detected_transcripts.csv*")
            log(f"  streaming {src.name} (no cached subset found)")
            parts = []
            for ck in pd.read_csv(src, chunksize=3_000_000,
                                  usecols=lambda c: c in ("global_x", "global_y",
                                                          "gene")):
                s = ck[(ck.global_x >= x0u) & (ck.global_x < x1u)
                       & (ck.global_y >= y0u) & (ck.global_y < y1u)]
                if len(s):
                    parts.append(s)
            d = pd.concat(parts, ignore_index=True)
            xc, yc, gc = "global_x", "global_y", "gene"
    else:
        tp = find(root, "transcripts.parquet", "transcripts.csv.gz")
        log(f"  {tp.name}")
        cols = ["x_location", "y_location", "feature_name", "qv"]
        try:
            d = pd.read_parquet(tp, columns=cols + ["is_gene"])
        except Exception:
            d = (pd.read_parquet(tp) if str(tp).endswith(".parquet")
                 else pd.read_csv(tp))
        xc, yc, gc = "x_location", "y_location", "feature_name"
        n0 = len(d)
        if "is_gene" in d.columns:
            d = d[d.is_gene == True]                                # noqa: E712
        if qv_min > 0 and "qv" in d.columns:
            d = d[d.qv >= qv_min]
        log(f"  {n0:,} rows -> {len(d):,} after is_gene / qv>={qv_min}")

    d = d[(d[xc] >= x0u) & (d[xc] < x1u) & (d[yc] >= y0u) & (d[yc] < y1u)]
    d = d.reset_index(drop=True)
    keep = ~d[gc].astype(str).str.startswith(
        ("Blank", "BLANK", "Negative", "NegControl", "NegPrb", "Unassigned",
         "antisense", "DeprecatedCodeword", "SystemControl"))
    if (~keep).any():
        log(f"  dropping {int((~keep).sum()):,} control probes")
        d = d[keep].reset_index(drop=True)

    gene_names = sorted(d[gc].astype(str).unique())
    g2i = {g: i for i, g in enumerate(gene_names)}
    gi = d[gc].astype(str).map(g2i).to_numpy(np.int32)
    x = d[xc].to_numpy(np.float64)
    y = d[yc].to_numpy(np.float64)
    log(f"  {len(x):,} transcripts, {len(gene_names)} genes")
    return x, y, gi, gene_names


def gt_cells(cfg, bounds) -> int:
    x0u, x1u, y0u, y1u = bounds
    root = Path(cfg["data_root"])
    if cfg["platform"] == "xenium":
        p = find(root, "cells.parquet", "cells.csv.gz")
        c = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
        xc = next(k for k in c.columns if k.lower() in ("x_centroid", "center_x"))
        yc = next(k for k in c.columns if k.lower() in ("y_centroid", "center_y"))
    else:
        p = find(root, "*cell_metadata*.csv*")
        c = pd.read_csv(p)
        xc, yc = "center_x", "center_y"
    n = int(((c[xc] >= x0u) & (c[xc] < x1u) & (c[yc] >= y0u) & (c[yc] < y1u)).sum())
    log(f"  GT {p.name}: {n:,} cells with their centroid inside the ROI")
    return n


# ---------------------------------------------------------------------------
# patching
# ---------------------------------------------------------------------------


def n_split_for(n_tx: int, n_gene: int, limit: float) -> int:
    """Smallest k so that each of the k x k patches keeps n*g under the limit."""
    k = 1
    while (n_tx / (k * k)) * n_gene >= limit and k < 32:
        k += 1
    return k


def patch_bounds(bounds, k, overlap):
    x0u, x1u, y0u, y1u = bounds
    xs = np.linspace(x0u, x1u, k + 1)
    ys = np.linspace(y0u, y1u, k + 1)
    out = []
    for i in range(k):
        for j in range(k):
            out.append(dict(
                i=i, j=j,
                rx0=max(x0u, xs[i] - overlap), rx1=min(x1u, xs[i+1] + overlap),
                ry0=max(y0u, ys[j] - overlap), ry1=min(y1u, ys[j+1] + overlap),
                cx0=xs[i], cx1=xs[i+1], cy0=ys[j], cy1=ys[j+1]))
    return out


def run_patch(pb, x, y, gi, cfg_run, ckpt_dir: Path, force=False):
    ck = ckpt_dir / f"patch_{pb['i']}_{pb['j']}_hs{cfg_run.h_s}.npz"
    if ck.exists() and not force:
        d = np.load(ck, allow_pickle=True)
        log(f"  patch({pb['i']},{pb['j']}) reusing ckpt: "
            f"{len(d['cell_loc']):,} cells")
        return ck

    m = ((x >= pb["rx0"]) & (x < pb["rx1"])
         & (y >= pb["ry0"]) & (y < pb["ry1"]))
    X, Y, GI = x[m], y[m], gi[m]
    n = len(X)
    if n < 1000:
        log(f"  patch({pb['i']},{pb['j']}) only {n} transcripts, skipped")
        np.savez_compressed(ck, cell_loc=np.zeros((0, 2)),
                            seg=np.zeros(0, np.int32), gi=GI, x=X, y=Y, n_tx=n,
                            core=np.array([pb["cx0"], pb["cx1"],
                                           pb["cy0"], pb["cy1"]]))
        return ck

    # local factorisation for the C++ side, global indices kept for count_mat
    uniq = np.unique(GI)
    remap = {g: k for k, g in enumerate(uniq)}
    G_local = np.array([remap[g] for g in GI], dtype=np.int64)
    prod = n * len(uniq)
    log(f"  patch({pb['i']},{pb['j']}): {n:,} tx x {len(uniq)} genes  "
        f"n*g={prod:,}  {'ok' if prod < INT32 else 'OVER int32'}")
    if prod >= INT32:
        log("    over int32 even after splitting; raise --n-split")
        return None

    from boms import run_boms
    t0 = time.time()
    try:
        _, seg, _, cell_loc, _ = run_boms(
            X, Y, G_local, epochs=cfg_run.epochs, h_s=cfg_run.h_s,
            h_r=cfg_run.h_r, K=cfg_run.K,
            x_min=float(X.min()) - 1, x_max=float(X.max()) + 1,
            y_min=float(Y.min()) - 1, y_max=float(Y.max()) + 1)
    except Exception as e:
        log(f"    run_boms failed: {type(e).__name__}: {e}")
        return None

    seg = np.asarray(seg).astype(np.int32)
    cell_loc = np.asarray(cell_loc, dtype=np.float64)
    log(f"    {len(cell_loc):,} cells  ({time.time()-t0:.0f}s)  -> ckpt")
    np.savez_compressed(ck, cell_loc=cell_loc, seg=seg, gi=GI, x=X, y=Y, n_tx=n,
                        core=np.array([pb["cx0"], pb["cx1"],
                                       pb["cy0"], pb["cy1"]]))
    return ck


def merge(ckpts, gene_names, cfg_run, final: Path, n_gt: int, bounds):
    from scipy.sparse import csr_matrix
    n_gene = len(gene_names)
    log(f"\n{SEP}\nmerging {len(ckpts)} patch(es), global gene table {n_gene}\n{SEP}")
    all_loc, all_seg, all_x, all_y, all_gi = [], [], [], [], []
    offset = 0
    for ck in ckpts:
        d = np.load(ck, allow_pickle=True)
        loc, seg = d["cell_loc"], d["seg"]
        if len(loc) == 0:
            continue
        cx0, cx1, cy0, cy1 = d["core"]
        keep = ((loc[:, 0] >= cx0) & (loc[:, 0] < cx1)
                & (loc[:, 1] >= cy0) & (loc[:, 1] < cy1))
        old2new = np.full(len(loc) + 1, -1, np.int64)
        kept = np.where(keep)[0]
        old2new[kept + 1] = np.arange(len(kept)) + offset
        seg_new = np.where(seg > 0, old2new[np.clip(seg, 0, len(loc))], -1)
        ok = seg_new >= 0
        all_loc.append(loc[keep]); all_seg.append(seg_new[ok])
        all_x.append(d["x"][ok]); all_y.append(d["y"][ok])
        all_gi.append(d["gi"][ok])
        offset += int(keep.sum())
        log(f"  {Path(ck).name[:38]:40} {len(loc):>8,} -> core {int(keep.sum()):>8,}")

    if not all_loc:
        log("  no usable patch"); return None
    cell_loc = np.vstack(all_loc)
    seg = np.concatenate(all_seg)
    tx_x = np.concatenate(all_x); tx_y = np.concatenate(all_y)
    tx_gi = np.concatenate(all_gi)
    n_cell = len(cell_loc)
    log(f"\n  {n_cell:,} cells, {len(seg):,} assigned transcripts")

    log(f"  rebuilding count_mat ({n_cell:,} x {n_gene}) from the global table")
    count_mat = csr_matrix((np.ones(len(seg), np.int32), (seg, tx_gi)),
                           shape=(n_cell, n_gene)).toarray().astype(np.int32)
    per = count_mat.sum(axis=1)
    keep = per >= cfg_run.min_size
    if keep.sum() < n_cell:
        remap = np.full(n_cell, -1, np.int64)
        remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
        s2 = remap[seg]
        ok = s2 >= 0
        seg, tx_x, tx_y, tx_gi = s2[ok], tx_x[ok], tx_y[ok], tx_gi[ok]
        cell_loc, count_mat = cell_loc[keep], count_mat[keep]
        log(f"  min_size>={cfg_run.min_size}: {n_cell:,} -> {len(cell_loc):,} cells")

    final.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        final, cell_loc=cell_loc, seg=seg.astype(np.int32), x=tx_x, y=tx_y,
        gene=np.array(gene_names, dtype=object)[tx_gi],
        gene_names=np.array(gene_names, dtype=object), count_mat=count_mat,
        h_s=cfg_run.h_s, h_r=cfg_run.h_r, K=cfg_run.K, epochs=cfg_run.epochs,
        roi=np.array(bounds), gt_cells=n_gt)
    n = len(cell_loc)
    tot = count_mat.sum(axis=1)
    log(f"\n{SEP}")
    log(f"  {n:,} cells   GT {n_gt:,}   ratio {n/max(n_gt,1):.3f}")
    log(f"  transcripts per cell median {np.median(tot):.0f}  "
        f"p5-p95 {np.percentile(tot,5):.0f}-{np.percentile(tot,95):.0f}")
    log(f"  assigned {len(seg):,} transcripts")
    log(f"  -> {final}")
    return final


def sweep(x, y, gi, bounds, cfg_run, k, overlap, n_gt):
    pb = patch_bounds(bounds, k, overlap)[0]
    m = ((x >= pb["rx0"]) & (x < pb["rx1"])
         & (y >= pb["ry0"]) & (y < pb["ry1"]))
    X, Y, GI = x[m], y[m], gi[m]
    uniq = np.unique(GI)
    remap = {g: i for i, g in enumerate(uniq)}
    G = np.array([remap[g] for g in GI], dtype=np.int64)
    target = max(n_gt // (k * k), 1)
    log(f"\n{SEP}")
    log(f"h_s sweep on one patch: {len(X):,} tx x {len(uniq)} genes")
    log(f"  epochs={cfg_run.sweep_epochs} (the real run uses {cfg_run.epochs}, "
        "which lowers the count another 10-20%)")
    log(f"  per-patch target ~{target:,}  (GT {n_gt:,} / {k*k})")
    log(f"  {'h_s':>8} {'cells':>10} {'vs target':>11} {'time':>8}")
    from boms import run_boms
    rows = []
    for hs in cfg_run.sweep_values:
        t0 = time.time()
        try:
            _, _, _, loc, _ = run_boms(
                X, Y, G, epochs=cfg_run.sweep_epochs, h_s=hs, h_r=cfg_run.h_r,
                K=cfg_run.K, x_min=float(X.min())-1, x_max=float(X.max())+1,
                y_min=float(Y.min())-1, y_max=float(Y.max())+1)
            n = len(loc)
            flag = "  <-" if 0.7 * target < n < 1.4 * target else ""
            log(f"  {hs:>8.1f} {n:>10,} {n-target:>+11,} "
                f"{time.time()-t0:>7.0f}s{flag}")
            rows.append((hs, n))
        except Exception as e:
            log(f"  {hs:>8.1f} {'FAIL':>10}  {type(e).__name__}: {str(e)[:40]}")
    if rows:
        d = pd.DataFrame(rows, columns=["h_s", "cells"])
        d["gap"] = (d.cells - target).abs()
        b = d.sort_values("gap").iloc[0]
        log(f"\n  closest: h_s={b.h_s}  ->  {int(b.cells):,} per patch, "
            f"{int(b.cells)*k*k:,} over {k*k} patches (GT {n_gt:,})")
        log(f"  now run:  --h-s {b.h_s}")
    return rows


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="merfish_3",
                    choices=["merfish_3"],
                    help=argparse.SUPPRESS)
    ap.add_argument("--out-root", default="/data/qiuyijia")
    ap.add_argument("--h-s", type=float, default=None)
    ap.add_argument("--h-r", type=float, default=0.3)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--overlap-um", type=float, default=30.0)
    ap.add_argument("--n-split", type=int, default=0,
                    help="0 = computed from n_tx x n_gene against --int32-limit")
    ap.add_argument("--int32-limit", type=float, default=1.8e9)
    ap.add_argument("--sweep-hs", action="store_true")
    ap.add_argument("--sweep-values", type=float, nargs="+", default=None)
    ap.add_argument("--sweep-epochs", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--roi-um", type=float, nargs=4, default=None,
                    metavar=("X0", "X1", "Y0", "Y1"),
                    help="override the ROI, in micron")
    ap.add_argument("--version", action="store_true")
    cfg_run = ap.parse_args()
    if cfg_run.version:
        print(VERSION)
        return

    cfg = DATASETS[cfg_run.dataset]
    if cfg_run.h_s is None:
        cfg_run.h_s = cfg["hs_default"]
    if cfg_run.sweep_values is None:
        cfg_run.sweep_values = cfg["hs_sweep"]

    out = Path(cfg_run.out_root) / f"boms_{cfg_run.dataset}"
    ckpt = out / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    final = out / f"boms_{cfg_run.dataset}.npz"

    log(SEP)
    log(f"BOMS v{VERSION}  x  {cfg_run.dataset}  ({cfg['label']})")
    log(SEP)
    log(f"  out {out}")

    if cfg_run.roi_um:
        bounds = tuple(cfg_run.roi_um)
        log(f"  ROI overridden: x[{bounds[0]:.1f},{bounds[1]:.1f}] "
            f"y[{bounds[2]:.1f},{bounds[3]:.1f}] um")
    else:
        bounds = roi_bounds(cfg)
    x, y, gi, gene_names = load_tx(cfg, bounds, cfg_run.qv_min)
    n_gt = gt_cells(cfg, bounds)

    k = cfg_run.n_split or n_split_for(len(x), len(gene_names),
                                       cfg_run.int32_limit)
    log(f"  n_tx x n_gene = {len(x)*len(gene_names):,}"
        f"  ->  {k}x{k} = {k*k} patch(es), "
        f"each about {len(x)//(k*k)*len(gene_names):,}")
    if k > 1:
        log(f"  overlap {cfg_run.overlap_um} um; a cell is kept only by the patch "
            "whose core its centroid falls in")

    if cfg_run.status:
        log(f"\n{SEP}\nstatus\n{SEP}")
        for pb in patch_bounds(bounds, k, cfg_run.overlap_um):
            f = ckpt / f"patch_{pb['i']}_{pb['j']}_hs{cfg_run.h_s}.npz"
            if f.exists():
                d = np.load(f, allow_pickle=True)
                log(f"  patch({pb['i']},{pb['j']}) done  "
                    f"{len(d['cell_loc']):>8,} cells")
            else:
                log(f"  patch({pb['i']},{pb['j']}) pending")
        log(f"  final: {'yes' if final.exists() else 'no'}  {final}")
        return

    if cfg_run.sweep_hs:
        sweep(x, y, gi, bounds, cfg_run, k, cfg_run.overlap_um, n_gt)
        return

    if final.exists() and not cfg_run.force:
        d = np.load(final, allow_pickle=True)
        log(f"\n  already done: {len(d['cell_loc']):,} cells   add --force to redo")
        return

    log(f"\n{SEP}")
    log(f"running  h_s={cfg_run.h_s}  h_r={cfg_run.h_r}  K={cfg_run.K}  "
        f"epochs={cfg_run.epochs}  min_size={cfg_run.min_size}")
    log(SEP)
    ckpts = []
    for pb in patch_bounds(bounds, k, cfg_run.overlap_um):
        c = run_patch(pb, x, y, gi, cfg_run, ckpt, force=cfg_run.force)
        if c:
            ckpts.append(c)
    if not ckpts:
        log("\n  no patch succeeded")
        return
    merge(ckpts, gene_names, cfg_run, final, n_gt, bounds)
    json.dump(dict(version=VERSION, dataset=cfg_run.dataset, label=cfg["label"],
                   roi_um=list(bounds), n_split=k, h_s=cfg_run.h_s,
                   h_r=cfg_run.h_r, K=cfg_run.K, epochs=cfg_run.epochs,
                   min_size=cfg_run.min_size, qv_min=cfg_run.qv_min,
                   n_transcripts=int(len(x)), n_genes=len(gene_names),
                   gt_cells=n_gt),
              open(out / "run_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()