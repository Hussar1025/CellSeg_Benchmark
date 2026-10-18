#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BOMS 运行 —— Xenium 乳腺（与 UCS 完全相同的 ROI）
=========================================================================
ROI 定义和 ucs_xenium_breast.py 完全一致：
  图像中心 5000×5000 px = 1062.5×1062.5 µm
  GT: 29,892 细胞（cells.parquet 中质心落在 ROI 内的）
  全 280 基因（ROI 内约 93 万转录本 × 280 = 2.6亿，int32 安全）

用法
-----
# ① 扫 h_s
python boms_xenium_breast_run.py --sweep-hs

# ② 正式跑
python boms_xenium_breast_run.py --h-s 20
"""

import os, time, argparse
import numpy as np
import pandas as pd

SEP = "=" * 72

# ── 和 UCS 脚本完全一致的 ROI 定义 ──────────────────────────────────────────
PIXEL_SIZE = 0.2125
CX_UM = 53994 / 2 * PIXEL_SIZE    # 5736.86 µm
CY_UM = 27420 / 2 * PIXEL_SIZE    # 2913.38 µm
HALF  = 5000 * PIXEL_SIZE / 2     # 531.25 µm

ROI_X0 = CX_UM - HALF   # ~5205.6 µm
ROI_X1 = CX_UM + HALF   # ~6268.1 µm
ROI_Y0 = CY_UM - HALF   # ~2382.1 µm
ROI_Y1 = CY_UM + HALF   # ~3444.6 µm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tx",  default="/data/qiuyijia/dataset/xenium_breast/"
                                    "transcripts.parquet")
    p.add_argument("--out", default="/data/qiuyijia/boms_xenium_breast/"
                                    "boms_xenium_breast.npz")
    p.add_argument("--h-s",    type=float, default=20.0)
    p.add_argument("--h-r",    type=float, default=0.3)
    p.add_argument("--epochs", type=int,   default=30)
    p.add_argument("--K",      type=int,   default=30)
    p.add_argument("--qv-min", type=float, default=20.0)
    p.add_argument("--gt-cells", type=int, default=29892)
    p.add_argument("--sweep-hs", action="store_true")
    p.add_argument("--sweep-epochs", type=int, default=10)
    return p.parse_args()


def load_tx(cfg):
    print(f"  读转录本（ROI: x[{ROI_X0:.0f},{ROI_X1:.0f}] "
          f"y[{ROI_Y0:.0f},{ROI_Y1:.0f}] µm）...")
    tx = pd.read_parquet(cfg.tx,
                         columns=["x_location","y_location",
                                  "feature_name","qv","is_gene"])
    n0 = len(tx)
    tx = tx[
        (tx.is_gene == True) & (tx.qv >= cfg.qv_min) &
        (tx.x_location >= ROI_X0) & (tx.x_location < ROI_X1) &
        (tx.y_location >= ROI_Y0) & (tx.y_location < ROI_Y1)
    ].reset_index(drop=True)

    n_gene = tx.feature_name.nunique()
    prod = len(tx) * n_gene
    print(f"  {n0:,} → ROI 内 {len(tx):,}  基因 {n_gene}")
    print(f"  n_tx × n_gene = {prod:,}  "
          f"{'✓ 安全' if prod < 2**31 else '✘ 溢出'}")

    gene_names = np.sort(tx.feature_name.unique())
    g2i = {g: i for i, g in enumerate(gene_names)}
    tx["gene_idx"] = tx.feature_name.map(g2i).astype(np.int64)
    return tx, gene_names


def main():
    cfg = parse_args()
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    print(SEP); print("BOMS —— Xenium 乳腺（ROI 与 UCS 相同）"); print(SEP)

    tx, gene_names = load_tx(cfg)
    X = tx.x_location.values.astype(np.float64)
    Y = tx.y_location.values.astype(np.float64)
    G = tx.gene_idx.values.astype(np.int64)

    from boms import run_boms

    if cfg.sweep_hs:
        print(f"\n  扫 h_s（epochs={cfg.sweep_epochs}，目标 GT={cfg.gt_cells}）")
        print(f"  {'h_s':>6} {'细胞数':>8} {'vs GT':>8} {'用时':>7}")
        print("  " + "-" * 34)
        best = None
        for hs in [10, 15, 20, 25, 30, 40]:
            t0 = time.time()
            r = run_boms(X, Y, G, epochs=cfg.sweep_epochs,
                         h_s=hs, h_r=cfg.h_r, K=cfg.K,
                         x_min=float(X.min())-1, x_max=float(X.max())+1,
                         y_min=float(Y.min())-1, y_max=float(Y.max())+1)
            nc = r[3].shape[0]
            d  = nc - cfg.gt_cells
            print(f"  {hs:>6.0f} {nc:>8,} {d:>+8,} {time.time()-t0:>6.0f}s")
            if best is None or abs(d) < abs(best[1]):
                best = (hs, d, nc)
        print(f"\n  → 最接近 GT 的 h_s = {best[0]}"
              f"（{best[2]:,} 细胞，差 {best[1]:+,}）")
        print(f"  正式跑：python {os.path.basename(__file__)} --h-s {best[0]}")
        return

    # ── 正式运行 ──────────────────────────────────────────────────────────
    print(f"\n  正式运行  h_s={cfg.h_s}  epochs={cfg.epochs}  "
          f"全 {len(gene_names)} 基因  GT={cfg.gt_cells:,}")
    t0 = time.time()
    modes, seg, count_mat, cell_loc, coords = run_boms(
        X, Y, G, epochs=cfg.epochs, h_s=cfg.h_s, h_r=cfg.h_r, K=cfg.K,
        x_min=float(X.min())-1, x_max=float(X.max())+1,
        y_min=float(Y.min())-1, y_max=float(Y.max())+1)
    dt = time.time() - t0

    seg = seg.astype(np.int64)
    n_cell = int(seg.max())
    n_asg  = int((seg > 0).sum())
    _, cnt = np.unique(seg[seg > 0], return_counts=True)

    print(f"\n{SEP}")
    print(f"完成  用时 {dt:.0f}s ({dt/60:.1f} 分钟)")
    print(f"  细胞数     {n_cell:,}  (GT {cfg.gt_cells:,})")
    print(f"  已分配 RNA {n_asg:,} / {len(X):,} ({n_asg/len(X)*100:.1f}%)")
    if len(cnt):
        print(f"  每细胞 RNA 中位 {np.median(cnt):.0f}  "
              f"p5~p95 {np.percentile(cnt,5):.0f}~{np.percentile(cnt,95):.0f}")
    print(SEP)

    np.savez_compressed(
        cfg.out, seg=seg, x=X, y=Y, gene_idx=G,
        gene_names=gene_names, cell_loc=cell_loc, count_mat=count_mat,
        h_s=cfg.h_s, roi=[ROI_X0, ROI_X1, ROI_Y0, ROI_Y1])
    print(f"  已存 → {cfg.out}")


if __name__ == "__main__":
    main()