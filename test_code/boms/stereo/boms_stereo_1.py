#!/usr/bin/env python
"""
BOMS × Stereo-seq —— E14.5(E1S3) / E16.5(E2S6) / E16.5(E2S7)

===== 本版修复：sweep() 之前硬编码了错误的网格大小 =====
之前的版本里 sweep() 写死了 `cfg.grid or 3`，完全没用 --check 阶段
plan_patches() 已经算出的正确 n_split。E14.5/E2S7 用 top-500 基因后，
n_mol×n_gene 实测都在 int32 安全范围内，真正需要的是 n_split=1
（不切分），但硬编码的 3 把 ROI 强行切成 3×3=9 份去找"左上角"，
而 Stereo-seq 的 GEM 覆盖率只有 6-9%，边角那块很可能真的一个分子
都没有 —— "0 分子 × 0 基因"就是这么来的，报错信息本身没有问题，
是调用方传错了参数。

本版把 main() 里 --check/--sweep-hs/正式运行三条路径统一用
plan_patches() 算出的同一个 n_split，不再有任何硬编码的兜底值。

===== 和 ProSeg 那次"实验性适配"的关键区别 =====
ProSeg 的 RNA 扩散模型依赖分子间精确相对位置，把同一 bin 的多个 UMI
展开成挤在同一坐标点的"假分子"，物理假设直接失效。
BOMS 核心是 mean-shift 密度估计（KDE），只关心局部密度峰值，对位置
量化不敏感，只要网格分辨率(0.5µm)远小于带宽参数h_s(预期几十µm量级)，
量化误差可忽略——这次的适配比 ProSeg 更站得住脚，但依然是近似，
不是真正单分子数据，评估时仍需标注。

===== 复用既有机制 =====
· 每个 patch 独立 checkpoint，可断点续跑（n_split=1 时只有一个 patch）
· patch 内局部基因重新编号（避免 C++ 后端基因索引越界）
· 最终 count_mat 用全局基因表重建
· overlap + 核心区去重（n_split=1 时没有重叠边界，逻辑自动退化为直通）

用法：
  python boms_stereo_1.py --dataset E14.5_E1S3 --check
  python boms_stereo_1.py --dataset E14.5_E1S3 --sweep-hs
  python boms_stereo_1.py --dataset E14.5_E1S3 --h-s 70
  python boms_stereo_1.py --dataset E14.5_E1S3 --status
"""
import os
import time
import gzip
import argparse

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

DATA_DIR = "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all"
OUT_BASE = "/data/qiuyijia/boms_stereo"
BIN_UM = 0.5          # 1 bin = 0.5µm，转成µm和MERFISH的h_s量纲一致
INT32_MAX = 2 ** 31
OVERLAP_UM = 30.0     # 与 MERFISH 版本一致的重叠带宽

DATASETS = {
    "E14.5_E1S3": dict(
        gem=f"{DATA_DIR}/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=4872),
    "E16.5_E2S6": dict(
        gem=f"{DATA_DIR}/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=6229),
    "E16.5_E2S7": dict(
        gem=f"{DATA_DIR}/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=7960),
}


# ══════════════════════════════════════════════════════════════════════════
#  Step 1: 读 GEM，展开 UMI，转 µm（带缓存）
# ══════════════════════════════════════════════════════════════════════════
def load_expanded(dataset, top_genes, force=False):
    ckdir = f"{OUT_BASE}/{dataset}/ckpt"
    os.makedirs(ckdir, exist_ok=True)
    cache = f"{ckdir}/expanded_top{top_genes}.npz"
    if os.path.exists(cache) and not force:
        d = np.load(cache, allow_pickle=True)
        P(f"  ⚡ 复用展开缓存: {len(d['x']):,} 条分子  {len(d['genes'])} 基因")
        return d["x"], d["y"], d["gi"], list(d["genes"])

    gem_path = DATASETS[dataset]["gem"]
    P(f"  读 GEM {os.path.basename(gem_path)} ...")
    opener = gzip.open if gem_path.endswith(".gz") else open
    skip = 0
    with opener(gem_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    gem = pd.read_csv(gem_path, sep="\t", skiprows=skip)
    ren = {}
    for c in gem.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl == "x":
            ren[c] = "x"
        elif cl == "y":
            ren[c] = "y"
        elif cl in ("midcounts", "midcount", "count"):
            ren[c] = "count"
    gem = gem.rename(columns=ren)
    if "count" not in gem.columns:
        gem["count"] = 1
    P(f"    原始 {len(gem):,} 行  基因 {gem.gene.nunique():,}")

    if top_genes > 0:
        top = gem.gene.value_counts().head(top_genes).index
        gem = gem[gem.gene.isin(top)]
        P(f"    降到 top-{top_genes} 基因后 {len(gem):,} 行")

    P(f"  展开 UMI（每个 count 变成一个分子；对 BOMS 的 mean-shift "
      f"影响远小于对 ProSeg 扩散模型的影响，见文件头说明）...")
    rep = gem.loc[gem.index.repeat(gem["count"].clip(1).astype(int))]
    genes = sorted(rep.gene.unique())
    g2i = {g: i for i, g in enumerate(genes)}
    gi = rep.gene.map(g2i).values.astype(np.int32)
    x = (rep.x.values.astype(np.float64)) * BIN_UM
    y = (rep.y.values.astype(np.float64)) * BIN_UM
    P(f"    展开后 {len(x):,} 个分子（µm 坐标）")

    np.savez_compressed(cache, x=x, y=y, gi=gi,
                        genes=np.array(genes, dtype=object))
    return x, y, gi, genes


def plan_patches(n_mol, n_gene, force_grid=None):
    """
    算出需要多大的 patch 网格才能让每块都不溢出 int32。
    ★ 这是唯一计算 n_split 的地方——--check / --sweep-hs / 正式运行
      三条路径全部调用这一个函数，不再各自维护一份数字。
    """
    if force_grid:
        P(f"    使用强制指定的网格 {force_grid}×{force_grid}")
        return force_grid
    prod = n_mol * n_gene
    n_split = 1
    while (n_mol / (n_split ** 2)) * n_gene >= INT32_MAX * 0.9:
        n_split += 1
        if n_split > 8:
            break
    P(f"    n_mol×n_gene = {prod:,}  →  需要 {n_split}×{n_split} = "
      f"{n_split**2} 个 patch")
    return n_split


def patch_bounds(x, y, n_split, overlap):
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    xs = np.linspace(x0, x1, n_split + 1)
    ys = np.linspace(y0, y1, n_split + 1)
    out = []
    for i in range(n_split):
        for j in range(n_split):
            out.append(dict(
                i=i, j=j,
                rx0=max(x0, xs[i]-overlap), rx1=min(x1, xs[i+1]+overlap),
                ry0=max(y0, ys[j]-overlap), ry1=min(y1, ys[j+1]+overlap),
                cx0=xs[i], cx1=xs[i+1], cy0=ys[j], cy1=ys[j+1],
            ))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Step 2: 单 patch（带 ckpt）
# ══════════════════════════════════════════════════════════════════════════
def run_patch(pb, x, y, gi, ckdir, cfg, force=False):
    ck = f"{ckdir}/patch_{pb['i']}_{pb['j']}_hs{cfg.h_s}.npz"
    if os.path.exists(ck) and not force:
        d = np.load(ck, allow_pickle=True)
        P(f"  ⚡ patch({pb['i']},{pb['j']}) 复用: {len(d['cell_loc']):,} 细胞")
        return ck

    m = ((x >= pb["rx0"]) & (x < pb["rx1"]) &
         (y >= pb["ry0"]) & (y < pb["ry1"]))
    X, Y, GI = x[m], y[m], gi[m]
    n = len(X)
    if n < 1000:
        P(f"  patch({pb['i']},{pb['j']}) 仅 {n} 分子，跳过")
        np.savez_compressed(ck, cell_loc=np.zeros((0, 2)),
                            seg=np.zeros(0, np.int32), gi=GI, x=X, y=Y,
                            core=np.array([pb["cx0"], pb["cx1"],
                                          pb["cy0"], pb["cy1"]]))
        return ck

    uniq_g = np.unique(GI)
    remap = {g: k for k, g in enumerate(uniq_g)}
    G_local = np.array([remap[g] for g in GI], dtype=np.int64)
    prod = n * len(uniq_g)
    P(f"  patch({pb['i']},{pb['j']}): {n:,} 分子 × {len(uniq_g)} 基因  "
      f"n×g={prod:,}  {'✓' if prod < INT32_MAX else '✘ 仍溢出，需更细网格'}")
    if prod >= INT32_MAX:
        return None

    from boms import run_boms
    t0 = time.time()
    try:
        _, seg, _, cell_loc, _ = run_boms(
            X, Y, G_local, epochs=cfg.epochs, h_s=cfg.h_s, h_r=cfg.h_r,
            K=cfg.K,
            x_min=float(X.min())-1, x_max=float(X.max())+1,
            y_min=float(Y.min())-1, y_max=float(Y.max())+1)
    except Exception as e:
        P(f"    ✘ run_boms 失败: {type(e).__name__}: {e}")
        return None

    seg = np.asarray(seg).astype(np.int32)
    cell_loc = np.asarray(cell_loc, dtype=np.float64)
    P(f"    ✓ {len(cell_loc):,} 细胞  ({time.time()-t0:.0f}s) → ckpt")
    np.savez_compressed(ck, cell_loc=cell_loc, seg=seg, gi=GI, x=X, y=Y,
                        core=np.array([pb["cx0"], pb["cx1"],
                                       pb["cy0"], pb["cy1"]]))
    return ck


# ══════════════════════════════════════════════════════════════════════════
#  Step 3: 合并（全局基因表重建 count_mat）
# ══════════════════════════════════════════════════════════════════════════
def merge(ckpts, genes, cfg, out_path):
    n_gene = len(genes)
    P(f"\n{SEP}\n合并 {len(ckpts)} 个 patch（全局基因表 {n_gene}）\n{SEP}")

    all_loc, all_x, all_y, all_gi, all_seg = [], [], [], [], []
    offset = 0
    for ck in ckpts:
        d = np.load(ck, allow_pickle=True)
        loc, seg = d["cell_loc"], d["seg"]
        if len(loc) == 0:
            continue
        cx0, cx1, cy0, cy1 = d["core"]
        keep = ((loc[:, 0] >= cx0) & (loc[:, 0] < cx1) &
                (loc[:, 1] >= cy0) & (loc[:, 1] < cy1))
        old2new = np.full(len(loc)+1, -1, dtype=np.int64)
        kept_idx = np.where(keep)[0]
        old2new[kept_idx+1] = np.arange(len(kept_idx)) + offset
        seg_new = np.where(seg > 0, old2new[np.clip(seg, 0, len(loc))], -1)
        ok = seg_new >= 0
        all_loc.append(loc[keep]); all_seg.append(seg_new[ok])
        all_x.append(d["x"][ok]); all_y.append(d["y"][ok])
        all_gi.append(d["gi"][ok])
        offset += int(keep.sum())
        P(f"  {os.path.basename(ck)[:36]:38} {len(loc):>7,} → "
          f"核心区 {int(keep.sum()):>7,}")

    if not all_loc:
        P("  ✘ 无有效 patch"); return None

    cell_loc = np.vstack(all_loc)
    seg = np.concatenate(all_seg)
    tx_x, tx_y = np.concatenate(all_x), np.concatenate(all_y)
    tx_gi = np.concatenate(all_gi)
    n_cell = len(cell_loc)
    P(f"\n  合并后 {n_cell:,} 细胞   已分配分子 {len(seg):,}")

    P(f"  重建 count_mat ({n_cell:,} × {n_gene}) ...")
    count_mat = csr_matrix(
        (np.ones(len(seg), np.int32), (seg, tx_gi)),
        shape=(n_cell, n_gene)).toarray().astype(np.int32)

    per_cell = np.asarray(count_mat.sum(axis=1)).ravel()
    keep = per_cell >= cfg.min_size
    if keep.sum() < n_cell:
        remap = np.full(n_cell, -1, np.int64)
        remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
        seg2 = remap[seg]; ok = seg2 >= 0
        seg, tx_x, tx_y, tx_gi = seg2[ok], tx_x[ok], tx_y[ok], tx_gi[ok]
        cell_loc, count_mat = cell_loc[keep], count_mat[keep]
        P(f"  min_size≥{cfg.min_size} 过滤: {n_cell:,} → {len(cell_loc):,}")

    np.savez_compressed(
        out_path, cell_loc=cell_loc, seg=seg.astype(np.int32),
        x=tx_x, y=tx_y, gene=np.array(genes, dtype=object)[tx_gi],
        gene_names=np.array(genes, dtype=object), count_mat=count_mat,
        h_s=cfg.h_s, h_r=cfg.h_r, K=cfg.K, epochs=cfg.epochs)

    n = len(cell_loc)
    gt = DATASETS[cfg.dataset]["gt_cells"]
    P(f"\n{SEP}\n  最终 {n:,} 细胞   GT {gt:,}   ratio {n/gt:.3f}")
    P(f"  → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════
#  h_s 扫描 ★ 本版修复：不再硬编码网格大小
# ══════════════════════════════════════════════════════════════════════════
def sweep(x, y, gi, dataset, cfg, n_split):
    """
    ★ 修复：之前这里硬编码了 `patch_bounds(x, y, cfg.grid or 3, ...)`，
      完全没用 --check 阶段 plan_patches() 已经算出的正确 n_split。
      Stereo-seq 的 GEM 覆盖率只有 6-9%，硬切 3×3 网格后"左上角"那块
      很可能真的是空的（0 分子 × 0 基因），这才是之前报错
      "zero-size array to reduction operation" 的根本原因——
      不是算法本身有问题，是参数传递漏了一步。

      现在 n_split 由调用方（main()）统一用 plan_patches() 算好后传入，
      如果 n_split=1（--check 已确认不需要切分），直接用全量数据扫描，
      不再人为切出一个可能不存在的"左上角"。
    """
    if n_split <= 1:
        P(f"\n  n_split=1（--check 已确认不需要切分），"
          f"直接用全量 {len(x):,} 分子做扫描")
        X, Y, GI = x, y, gi
    else:
        pbs = patch_bounds(x, y, n_split, OVERLAP_UM)
        pb = pbs[0]
        m = ((x >= pb["rx0"]) & (x < pb["rx1"]) &
             (y >= pb["ry0"]) & (y < pb["ry1"]))
        X, Y, GI = x[m], y[m], gi[m]
        P(f"\n  用左上 patch({pb['i']},{pb['j']}) 做扫描: {len(X):,} 分子")

    if len(X) == 0:
        P(f"  ✘ 扫描区域是空的（这次已经用了正确的 n_split={n_split}，"
          f"如果还是空，说明 ROI 边界本身有问题，需要另外排查）")
        return

    uniq_g = np.unique(GI)
    remap = {g: k for k, g in enumerate(uniq_g)}
    G = np.array([remap[g] for g in GI], dtype=np.int64)
    n_patches_total = max(n_split ** 2, 1)
    target = DATASETS[dataset]["gt_cells"] // n_patches_total

    P(f"\n{SEP}\nh_s 扫描（{len(X):,} 分子 × {len(uniq_g)} 基因）")
    P(f"目标 ≈ {target:,}（GT {DATASETS[dataset]['gt_cells']:,} / "
      f"{n_patches_total} 块）\n{SEP}")

    from boms import run_boms
    rows = []
    for hs in cfg.sweep_values:
        t0 = time.time()
        try:
            _, _, _, loc, _ = run_boms(
                X, Y, G, epochs=cfg.sweep_epochs, h_s=hs, h_r=cfg.h_r,
                K=cfg.K, x_min=float(X.min())-1, x_max=float(X.max())+1,
                y_min=float(Y.min())-1, y_max=float(Y.max())+1)
            n = len(loc)
            flag = "  ←" if 0.7*target < n < 1.4*target else ""
            P(f"  h_s={hs:>6.1f}  {n:>8,} 细胞  vs目标{n-target:>+8,}  "
              f"({time.time()-t0:.0f}s){flag}")
            rows.append((hs, n))
        except Exception as e:
            P(f"  h_s={hs:>6.1f}  FAIL {type(e).__name__}: {str(e)[:40]}")

    if rows:
        df = pd.DataFrame(rows, columns=["h_s", "cells"])
        df["gap"] = (df.cells - target).abs()
        b = df.sort_values("gap").iloc[0]
        P(f"\n  最接近: h_s={b.h_s} → {int(b.cells):,} 细胞"
          f"（目标 {target:,}）")
        P(f"  正式跑: --h-s {b.h_s}")
        df.drop(columns=["gap"]).to_csv(
            f"{OUT_BASE}/{dataset}/hs_sweep.csv", index=False)
    else:
        P(f"\n  ✘ 全部 h_s 都失败了，检查上面的报错信息")


def show_status(dataset):
    ckdir = f"{OUT_BASE}/{dataset}/ckpt"
    out_path = f"{OUT_BASE}/{dataset}/boms_{dataset}.npz"
    P(f"\n{SEP}\n{dataset} 断点状态\n{SEP}")
    if os.path.isdir(ckdir):
        for f in sorted(os.listdir(ckdir)):
            full = os.path.join(ckdir, f)
            sz = os.path.getsize(full) / 1e6
            P(f"  {sz:9.2f} MB   {f}")
    if os.path.exists(out_path):
        d = np.load(out_path, allow_pickle=True)
        gt = DATASETS[dataset]["gt_cells"]
        n = len(d['cell_loc'])
        P(f"\n  ✓ 最终结果: {n:,} 细胞  GT {gt:,}  ratio {n/gt:.3f}")
    else:
        P(f"\n  尚无最终结果")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--top-genes", type=int, default=500)
    ap.add_argument("--grid", type=int, default=None,
                    help="强制指定 N×N 网格，不指定则自动算")
    ap.add_argument("--h-s", type=float, default=70.0)
    ap.add_argument("--h-r", type=float, default=0.3)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--sweep-hs", action="store_true")
    ap.add_argument("--sweep-values", type=float, nargs="+",
                    default=[30, 50, 70, 100, 150])
    ap.add_argument("--sweep-epochs", type=int, default=10)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    cfg = ap.parse_args()

    if cfg.status:
        show_status(cfg.dataset)
        return

    outdir = f"{OUT_BASE}/{cfg.dataset}"
    ckdir = f"{outdir}/ckpt"
    os.makedirs(ckdir, exist_ok=True)
    out_path = f"{outdir}/boms_{cfg.dataset}.npz"

    P(SEP); P(f"BOMS × Stereo-seq {cfg.dataset}"); P(SEP)
    P(f"  GT {DATASETS[cfg.dataset]['gt_cells']:,}")

    x, y, gi, genes = load_expanded(cfg.dataset, cfg.top_genes, cfg.force)
    # ★ 唯一计算 n_split 的地方，--check / --sweep-hs / 正式运行都用这个值
    n_split = plan_patches(len(x), len(genes), cfg.grid)

    if cfg.check:
        P(f"\n  --check：预计 {n_split}×{n_split}={n_split**2} 个 patch")
        return

    if cfg.sweep_hs:
        sweep(x, y, gi, cfg.dataset, cfg, n_split)
        return

    pbs = patch_bounds(x, y, n_split, OVERLAP_UM)
    P(f"\n  {n_split}×{n_split}={len(pbs)} 个 patch  h_s={cfg.h_s}")
    ckpts = []
    for pb in pbs:
        ck = run_patch(pb, x, y, gi, ckdir, cfg, cfg.force)
        if ck:
            ckpts.append(ck)
    if not ckpts:
        P("  ✘ 无成功 patch"); return
    merge(ckpts, genes, cfg, out_path)


if __name__ == "__main__":
    main()