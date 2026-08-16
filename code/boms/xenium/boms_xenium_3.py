#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BOMS 运行 —— Xenium ROI（人淋巴结）  最终版
=========================================================================
定案：HVG-400 基因子集。
  * int32 约束：n_tx × n_gene < 21.4亿。全量 73万转录本 × 400 = 2.9亿，安全。
  * 4495 全基因 → 73万×4495 = 32.9亿 > int32，会 std::bad_array_new_length。
  * HVG 是 BOMS 在 Prime 5K 高维 panel 上的合理适配（同 Cellist 只用 HVG），
    评估时标注 "top-400 HVG" 即可。

用法
-----
# ① 先扫 h_s 找接近 GT(2442) 细胞数的值（epochs 小，快）
python boms_xenium_roi_run.py --sweep-hs

# ② 用选定 h_s 正式跑（epochs=30）
python boms_xenium_roi_run.py --h-s 11

输出（供 eval_boms_xenium.py 读）：
  boms_xenium_roi.npz  含 seg / x / y / gene_idx / gene_names / cell_loc / count_mat
"""

import os, time, argparse
import numpy as np
import pandas as pd

SEP = "=" * 72


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tx", default="/data/qiuyijia/dataset/xenium_roi_crop/"
                                   "transcripts.parquet")
    p.add_argument("--out", default="/data/qiuyijia/boms_xenium_roi/"
                                    "boms_xenium_roi.npz")
    p.add_argument("--top-genes", type=int, default=400,
                   help="HVG 数（定案 400；int32 上限约 2900）")
    p.add_argument("--qv-min", type=float, default=20.0)
    p.add_argument("--h-s", type=float, default=11.0,
                   help="细胞直径(µm)，非半径。淋巴结起点，用 --sweep-hs 定")
    p.add_argument("--h-r", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--K", type=int, default=30)
    p.add_argument("--sweep-hs", action="store_true",
                   help="扫 h_s 找接近 GT 细胞数的值（用小 epochs）")
    p.add_argument("--sweep-epochs", type=int, default=10,
                   help="扫描时用的 epochs（小=快）")
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
    print(f"  转录本 {n0:,} → 过滤+top{cfg.top_genes}基因 → {len(tx):,}")
    print(f"  基因 {tx.feature_name.nunique()}  n_tx×n_gene={prod:,} "
          f"{'✘ 溢出!' if prod > 2**31 else '✓ 安全'}")

    gene_names = np.sort(tx.feature_name.unique())
    g2i = {g: i for i, g in enumerate(gene_names)}
    tx["gene_idx"] = tx.feature_name.map(g2i).astype(np.int64)
    return tx, gene_names


def main():
    cfg = parse_args()
    os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
    print(SEP); print("BOMS —— Xenium ROI（HVG-400）"); print(SEP)

    tx, gene_names = load_tx(cfg)
    X = tx.x_location.values.astype(np.float64)
    Y = tx.y_location.values.astype(np.float64)
    G = tx.gene_idx.values.astype(np.int64)

    from boms import run_boms

    # ── 扫 h_s 模式 ────────────────────────────────────────────────────────
    if cfg.sweep_hs:
        print(f"\n  扫 h_s（epochs={cfg.sweep_epochs}，目标 GT={cfg.gt_cells}）")
        print(f"  {'h_s':>6} {'细胞数':>8} {'vs GT':>8} {'用时':>7}")
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
        print(f"\n  → 最接近 GT 的 h_s = {best[0]}（细胞 {best[2]:,}, "
              f"差 {best[1]:+,}）")
        print(f"     注：扫描用 epochs={cfg.sweep_epochs}，正式跑 epochs=30 "
              f"细胞数会略有变化")
        print(f"\n  正式跑：python {os.path.basename(__file__)} --h-s {best[0]}")
        return

    # ── 正式运行 ──────────────────────────────────────────────────────────
    print(f"\n  正式运行  h_s={cfg.h_s}  epochs={cfg.epochs}  "
          f"top-{cfg.top_genes} HVG")
    t0 = time.time()
    modes, seg, count_mat, cell_loc, coords = run_boms(
        X, Y, G, epochs=cfg.epochs, h_s=cfg.h_s, h_r=cfg.h_r, K=cfg.K)
    dt = time.time() - t0

    seg = seg.astype(np.int64)
    n_cell = int(seg.max())
    n_asg = int((seg > 0).sum())
    print(f"\n{SEP}")
    print(f"完成  用时 {dt:.0f}s")
    print(f"  细胞数     {n_cell:,}   (GT {cfg.gt_cells})")
    print(f"  已分配 RNA {n_asg:,} / {len(X):,} ({n_asg/len(X)*100:.1f}%)")
    _, cnt = np.unique(seg[seg > 0], return_counts=True)
    print(f"  每细胞 RNA 中位 {np.median(cnt):.0f}  "
          f"p5~p95 {np.percentile(cnt,5):.0f}~{np.percentile(cnt,95):.0f}")
    print(SEP)

    np.savez_compressed(
        cfg.out, seg=seg, x=X, y=Y, gene_idx=G,
        gene_names=gene_names, cell_loc=cell_loc, count_mat=count_mat,
        h_s=cfg.h_s, top_genes=cfg.top_genes)
    print(f"  已存 → {cfg.out}")
    print(f"\n  评估：")
    print(f"    python eval_boms_xenium.py --npz {cfg.out} "
          f"--data-dir /data/qiuyijia/dataset/xenium_roi_crop")


if __name__ == "__main__":
    main()