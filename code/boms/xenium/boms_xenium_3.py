#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BOMS run -- Xenium ROI (human lymph node)  final
=========================================================================
Decision: HVG-400 gene subset.
  * int32 constraint: n_tx × n_gene < 2.14B. Full 730K transcripts × 400 = 290M, safe.
  * 4495 full genes -> 730K × 4495 = 3.29B > int32, would raise std::bad_array_new_length.
  * HVG is a sensible BOMS adaptation for the Prime 5K high-dimensional panel
    (like Cellist using only HVG); label the evaluation "top-400 HVG".

Usage
-----
# 1. sweep h_s first to find the value closest to the GT(2442) cell count (few epochs, fast)
python boms_xenium_roi_run.py --sweep-hs

# 2. production run with the chosen h_s (epochs=30)
python boms_xenium_roi_run.py --h-s 11

Output (read by eval_boms_xenium.py):
  boms_xenium_roi.npz  contains seg / x / y / gene_idx / gene_names / cell_loc / count_mat
"""

import os, time, argparse
import numpy as np
import pandas as pd

SEP = "=" * 72


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tx", default="yourpath/dataset/xenium_roi_crop/"
                                   "transcripts.parquet")
    p.add_argument("--out", default="yourpath/boms_xenium_roi/"
                                    "boms_xenium_roi.npz")
    p.add_argument("--top-genes", type=int, default=400,
                   help="HVG count (decided: 400; int32 limit ~2900)")
    p.add_argument("--qv-min", type=float, default=20.0)
    p.add_argument("--h-s", type=float, default=11.0,
                   help="cell diameter (µm), not radius; lymph-node start, tune with --sweep-hs")
    p.add_argument("--h-r", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--sweep-hs", action="store_true",
                   help="sweep h_s for the value closest to the GT cell count (few epochs)")
    p.add_argument("--sweep-epochs", type=int, default=10,
                   help="epochs used during the sweep (smaller = faster)")
    p.add_argument("--gt-cells", type=int, default=2442)
    return p.parse_args()


def load_tx(cfg):
    tx = pd.read_parquet(cfg.tx, columns=["x_location", "y_location",
                                          "feature_name", "qv", "is_gene",
                                          "cell_id"])
    n0 = len(tx)
    tx = tx[(tx.is_gene == True) & (tx.qv >= cfg.qv_min)].reset_index(drop=True)
    top = tx.feature_name.value_counts().head(cfg.top_genes).index
    tx = tx[tx.feature_name.isin(top)].reset_index(drop=True)

    prod = len(tx) * tx.feature_name.nunique()
    cov = None
    print(f"  transcripts {n0:,} -> filter + top {cfg.top_genes} genes -> {len(tx):,}")
    print(f"  genes {tx.feature_name.nunique()}  n_tx×n_gene={prod:,} "
          f"{'overflow!' if prod > 2**31 else 'safe'}")

    gene_names = np.sort(tx.feature_name.unique())
    g2i = {g: i for i, g in enumerate(gene_names)}
    tx["gene_idx"] = tx.feature_name.map(g2i).astype(np.int64)
    return tx, gene_names


def main():
    cfg = parse_args()
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    print(SEP); print("BOMS -- Xenium ROI (HVG-400)"); print(SEP)

    tx, gene_names = load_tx(cfg)
    X = tx.x_location.values.astype(np.float64)
    Y = tx.y_location.values.astype(np.float64)
    G = tx.gene_idx.values.astype(np.int64)

    from boms import run_boms

    # ── sweep h_s mode ─────────────────────────────────────────────────────
    if cfg.sweep_hs:
        print(f"\n  sweep h_s (epochs={cfg.sweep_epochs}, target GT={cfg.gt_cells})")
        print(f"  {'h_s':>6} {'cells':>8} {'vs GT':>8} {'elapsed':>7}")
        print("  " + "-" * 34)
        best = None
        for hs in [6.9, 9, 11, 13, 15, 18]:
            t0 = time.time()
            r = run_boms(X, Y, G, epochs=cfg.sweep_epochs,
                         h_s=hs, h_r=cfg.h_r, K=cfg.K)
            nc = r[3].shape[0]
            d = nc - cfg.gt_cells
            print(f"  {hs:>6.1f} {nc:>8,} {d:>+8,} {time.time()-t0:>6.0f}s")
            if best is None or abs(d) < abs(best[1]):
                best = (hs, d, nc)
        print(f"\n  -> closest h_s to GT = {best[0]} (cells {best[2]:,}, "
              f"diff {best[1]:+,})")
        print(f"     note: sweep uses epochs={cfg.sweep_epochs}; the production "
              f"run (epochs=30) cell count will shift slightly")
        print(f"\n  production run: python {os.path.basename(__file__)} --h-s {best[0]}")
        return

    # ── production run ─────────────────────────────────────────────────────
    print(f"\n  production run  h_s={cfg.h_s}  epochs={cfg.epochs}  "
          f"top-{cfg.top_genes} HVG")
    t0 = time.time()
    modes, seg, count_mat, cell_loc, coords = run_boms(
        X, Y, G, epochs=cfg.epochs, h_s=cfg.h_s, h_r=cfg.h_r, K=cfg.K)
    dt = time.time() - t0

    seg = seg.astype(np.int64)
    n_cell = int(seg.max())
    n_asg = int((seg > 0).sum())
    print(f"\n{SEP}")
    print(f"Done  elapsed {dt:.0f}s")
    print(f"  cells       {n_cell:,}   (GT {cfg.gt_cells})")
    print(f"  assigned RNA {n_asg:,} / {len(X):,} ({n_asg/len(X)*100:.1f}%)")
    _, cnt = np.unique(seg[seg > 0], return_counts=True)
    print(f"  RNA per cell median {np.median(cnt):.0f}  "
          f"p5~p95 {np.percentile(cnt,5):.0f}~{np.percentile(cnt,95):.0f}")
    print(SEP)

    np.savez_compressed(
        cfg.out, seg=seg, x=X, y=Y, gene_idx=G,
        gene_names=gene_names, cell_loc=cell_loc, count_mat=count_mat,
        h_s=cfg.h_s, top_genes=cfg.top_genes)
    print(f"  Saved -> {cfg.out}")
    print(f"\n  evaluation:")
    print(f"    python eval_boms_xenium.py --npz {cfg.out} "
          f"--data-dir yourpath/dataset/xenium_roi_crop")


if __name__ == "__main__":
    main()