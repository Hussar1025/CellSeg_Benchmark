#!/usr/bin/env python
"""
BOMS × MERFISH 小鼠脑 —— 断点续跑版

===== 相比旧版修了什么 =====

① count_mat 的 inhomogeneous shape（旧版 line 216 崩）
   根因：每个 patch 各自 pd.factorize(genes)，patch A 640 基因、
         patch B 649 基因 → cnt_p[cid-1] 长度不一 → np.array() 炸
   修法：patch 内仍局部 factorize（喂 BOMS C++，防基因索引越界），
         但 count_mat 用【全局基因表】从 seg 重建 → 所有 patch 等长

② 断点续跑
   旧版跑完所有 patch 才在保存那一步崩，几小时白费
   现在每个 patch 算完立刻存 ckpt/patch_i_j.npz，重跑自动跳过

③ reshape bug（float32 精度导致 fov_ind 少几个点）
   run_boms 显式传 x_min/x_max/y_min/y_max ±1 余量

===== 为什么切 patch =====
ROI 内 6,062,241 转录本 × 649 基因 = 3.93e9 > int32 上限 2.147e9
2×2 切分后每块 ~1.5M × 649 = 9.8e8 ✓

用法：
  python boms_merfish_ckpt.py --sweep-hs          # 扫 h_s（左上 patch）
  python boms_merfish_ckpt.py --h-s 70            # 正式跑（可断点续跑）
  python boms_merfish_ckpt.py --h-s 70 --force    # 忽略 ckpt 重算
  python boms_merfish_ckpt.py --status            # 只看进度
"""
import os, sys, time, glob, argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SEP = "=" * 68

TX_CSV = ("/data/qiuyijia/dataset/merfish_mouse_brain/"
          "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1"
          "_detected_transcripts_S1R1.csv")
OUT_DIR  = "/data/qiuyijia/boms_merfish"
CKPT_DIR = f"{OUT_DIR}/ckpt"
TX_CACHE = f"{CKPT_DIR}/roi_transcripts.npz"
FINAL    = f"{OUT_DIR}/boms_merfish.npz"

# ROI 与 GeneSegNet / ProSeg / UCS 完全一致
ROI_X0, ROI_X1 = 3589.0, 5761.0
ROI_Y0, ROI_Y1 = 2327.0, 4499.0
GT_CELLS = 8444

N_SPLIT = 2          # 2×2 = 4 个 patch
OVERLAP = 30.0       # µm，防边界细胞被切断
INT32_MAX = 2 ** 31

os.makedirs(CKPT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  Step 1: ROI 转录本（缓存）
# ══════════════════════════════════════════════════════════════════════════
def load_roi(force=False):
    if os.path.exists(TX_CACHE) and not force:
        d = np.load(TX_CACHE, allow_pickle=True)
        gn = list(d["gene_names"])
        print(f"  ⚡ 复用转录本缓存: {len(d['x']):,} 条  {len(gn)} 基因")
        return d["x"], d["y"], d["gi"], gn

    print(f"  读转录本（3.6 GB，分块过滤 ROI，约 3-5 分钟）...")
    t0 = time.time()
    chunks, total = [], 0
    for ck in pd.read_csv(TX_CSV, chunksize=2_000_000,
                          usecols=["global_x", "global_y", "gene"]):
        total += len(ck)
        sub = ck[(ck.global_x >= ROI_X0) & (ck.global_x < ROI_X1) &
                 (ck.global_y >= ROI_Y0) & (ck.global_y < ROI_Y1)]
        if len(sub):
            chunks.append(sub)
        print(f"    已读 {total:,}", end="\r", flush=True)
    tx = pd.concat(chunks, ignore_index=True)
    print(f"\n    全量 {total:,} → ROI 内 {len(tx):,}  ({time.time()-t0:.0f}s)")

    # ★ 全局基因表：count_mat 能拼接的前提
    gene_names = sorted(tx.gene.unique())
    g2i = {g: i for i, g in enumerate(gene_names)}
    gi = tx.gene.map(g2i).values.astype(np.int32)
    print(f"    全局基因表 {len(gene_names)} 个")

    x = tx.global_x.values.astype(np.float64)
    y = tx.global_y.values.astype(np.float64)
    np.savez_compressed(TX_CACHE, x=x, y=y, gi=gi,
                        gene_names=np.array(gene_names, dtype=object))
    print(f"    ✓ 缓存 → {os.path.basename(TX_CACHE)}")
    return x, y, gi, gene_names


def patch_bounds():
    """2×2 网格，含 overlap 的读取范围 + 不含 overlap 的核心范围"""
    xs = np.linspace(ROI_X0, ROI_X1, N_SPLIT + 1)
    ys = np.linspace(ROI_Y0, ROI_Y1, N_SPLIT + 1)
    out = []
    for i in range(N_SPLIT):
        for j in range(N_SPLIT):
            out.append(dict(
                i=i, j=j,
                # 读取范围（外扩 overlap）
                rx0=max(ROI_X0, xs[i] - OVERLAP),
                rx1=min(ROI_X1, xs[i+1] + OVERLAP),
                ry0=max(ROI_Y0, ys[j] - OVERLAP),
                ry1=min(ROI_Y1, ys[j+1] + OVERLAP),
                # 核心范围（去重用：只保留质心落在核心区的细胞）
                cx0=xs[i], cx1=xs[i+1], cy0=ys[j], cy1=ys[j+1],
            ))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Step 2: 单 patch（带 ckpt）
# ══════════════════════════════════════════════════════════════════════════
def run_patch(pb, x, y, gi, n_gene_global, cfg, force=False):
    ck = f"{CKPT_DIR}/patch_{pb['i']}_{pb['j']}_hs{cfg.h_s}.npz"
    if os.path.exists(ck) and not force:
        d = np.load(ck, allow_pickle=True)
        print(f"  ⚡ patch({pb['i']},{pb['j']}) 复用 ckpt: "
              f"{len(d['cell_loc']):,} 细胞")
        return ck

    m = ((x >= pb["rx0"]) & (x < pb["rx1"]) &
         (y >= pb["ry0"]) & (y < pb["ry1"]))
    X, Y, GI = x[m], y[m], gi[m]
    n = len(X)
    if n < 1000:
        print(f"  patch({pb['i']},{pb['j']}) 仅 {n} 条转录本，跳过")
        np.savez_compressed(ck, cell_loc=np.zeros((0, 2)),
                            seg=np.zeros(0, np.int32),
                            gi=GI, x=X, y=Y, n_tx=n)
        return ck

    # ★ 局部 factorize 喂给 BOMS：C++ 内部按 gene 值索引，
    #   直接传全局索引（有空洞）可能越界
    uniq_g = np.unique(GI)
    remap = {g: k for k, g in enumerate(uniq_g)}
    G_local = np.array([remap[g] for g in GI], dtype=np.int64)
    prod = n * len(uniq_g)
    print(f"  patch({pb['i']},{pb['j']}): {n:,} 转录本 × {len(uniq_g)} 基因  "
          f"n×g={prod:,}  {'✓' if prod < INT32_MAX else '✘ 溢出!'}",
          flush=True)
    if prod >= INT32_MAX:
        print(f"    ✘ 超 int32，需增大 N_SPLIT")
        return None

    from boms import run_boms
    t0 = time.time()
    try:
        _, seg, _, cell_loc, _ = run_boms(
            X, Y, G_local,
            epochs=cfg.epochs, h_s=cfg.h_s, h_r=cfg.h_r, K=cfg.K,
            # ★ 显式边界 ±1：防 float32 精度让 fov_ind 少几个点 → reshape 炸
            x_min=float(X.min()) - 1, x_max=float(X.max()) + 1,
            y_min=float(Y.min()) - 1, y_max=float(Y.max()) + 1)
    except Exception as e:
        print(f"    ✘ run_boms 失败: {type(e).__name__}: {e}")
        return None

    seg = np.asarray(seg).astype(np.int32)
    cell_loc = np.asarray(cell_loc, dtype=np.float64)
    print(f"    ✓ {len(cell_loc):,} 细胞  ({time.time()-t0:.0f}s)  "
          f"→ ckpt", flush=True)

    # ★ 存全局基因索引 gi，不存 BOMS 的 count_mat（长度不一致的根源）
    np.savez_compressed(ck, cell_loc=cell_loc, seg=seg, gi=GI, x=X, y=Y,
                        n_tx=n, core=np.array([pb["cx0"], pb["cx1"],
                                               pb["cy0"], pb["cy1"]]))
    return ck


# ══════════════════════════════════════════════════════════════════════════
#  Step 3: 合并（用全局基因表重建 count_mat）
# ══════════════════════════════════════════════════════════════════════════
def merge(ckpts, gene_names, cfg):
    n_gene = len(gene_names)
    print(f"\n{SEP}\n合并 {len(ckpts)} 个 patch（全局基因表 {n_gene}）\n{SEP}")

    all_loc, all_x, all_y, all_gi, all_seg = [], [], [], [], []
    offset = 0
    for ck in ckpts:
        d = np.load(ck, allow_pickle=True)
        loc, seg = d["cell_loc"], d["seg"]
        if len(loc) == 0:
            continue
        cx0, cx1, cy0, cy1 = d["core"]

        # 只保留质心落在核心区的细胞（overlap 去重）
        keep = ((loc[:, 0] >= cx0) & (loc[:, 0] < cx1) &
                (loc[:, 1] >= cy0) & (loc[:, 1] < cy1))
        old2new = np.full(len(loc) + 1, -1, dtype=np.int64)
        kept_idx = np.where(keep)[0]
        old2new[kept_idx + 1] = np.arange(len(kept_idx)) + offset   # seg 1-based

        seg_new = np.where(seg > 0, old2new[np.clip(seg, 0, len(loc))], -1)
        ok = seg_new >= 0

        all_loc.append(loc[keep])
        all_seg.append(seg_new[ok])
        all_x.append(d["x"][ok]); all_y.append(d["y"][ok])
        all_gi.append(d["gi"][ok])
        offset += int(keep.sum())
        print(f"  {os.path.basename(ck)[:34]:36} "
              f"{len(loc):>7,} → 核心区 {int(keep.sum()):>7,}")

    if not all_loc:
        print("  ✘ 无有效 patch"); return None

    cell_loc = np.vstack(all_loc)
    seg = np.concatenate(all_seg)
    tx_x = np.concatenate(all_x); tx_y = np.concatenate(all_y)
    tx_gi = np.concatenate(all_gi)
    n_cell = len(cell_loc)
    print(f"\n  合并后 {n_cell:,} 细胞   已分配转录本 {len(seg):,}")

    # ★ count_mat：全局基因表重建，所有 patch 天然等长
    print(f"  重建 count_mat ({n_cell:,} × {n_gene}) ...")
    count_mat = csr_matrix(
        (np.ones(len(seg), np.int32), (seg, tx_gi)),
        shape=(n_cell, n_gene)).toarray().astype(np.int32)

    # min_size 过滤
    per_cell = np.asarray(count_mat.sum(axis=1)).ravel()
    keep = per_cell >= cfg.min_size
    if keep.sum() < n_cell:
        remap = np.full(n_cell, -1, np.int64)
        remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
        seg2 = remap[seg]
        ok = seg2 >= 0
        seg, tx_x, tx_y, tx_gi = seg2[ok], tx_x[ok], tx_y[ok], tx_gi[ok]
        cell_loc, count_mat = cell_loc[keep], count_mat[keep]
        print(f"  min_size≥{cfg.min_size} 过滤: {n_cell:,} → "
              f"{len(cell_loc):,} 细胞")

    np.savez_compressed(
        FINAL,
        cell_loc=cell_loc, seg=seg.astype(np.int32),
        x=tx_x, y=tx_y, gene=np.array(gene_names, dtype=object)[tx_gi],
        gene_names=np.array(gene_names, dtype=object),
        count_mat=count_mat,
        h_s=cfg.h_s, h_r=cfg.h_r, K=cfg.K, epochs=cfg.epochs)

    n = len(cell_loc)
    print(f"\n{SEP}")
    print(f"  最终 {n:,} 细胞   GT {GT_CELLS:,}   ratio {n/GT_CELLS:.3f}")
    print(f"  每细胞转录本 中位 {np.median(np.asarray(count_mat.sum(1)).ravel()):.0f}")
    print(f"  → {FINAL}")
    print(f"\n  评估: python eval_boms_merfish.py")
    return FINAL


# ══════════════════════════════════════════════════════════════════════════
#  h_s 扫描（只用左上 patch）
# ══════════════════════════════════════════════════════════════════════════
def sweep(x, y, gi, cfg):
    pb = patch_bounds()[0]
    m = ((x >= pb["rx0"]) & (x < pb["rx1"]) &
         (y >= pb["ry0"]) & (y < pb["ry1"]))
    X, Y, GI = x[m], y[m], gi[m]
    uniq_g = np.unique(GI)
    remap = {g: k for k, g in enumerate(uniq_g)}
    G = np.array([remap[g] for g in GI], dtype=np.int64)
    target = GT_CELLS // (N_SPLIT ** 2)

    print(f"\n{SEP}")
    print(f"h_s 扫描（左上 patch {len(X):,} 转录本 × {len(uniq_g)} 基因）")
    print(f"  epochs={cfg.sweep_epochs}（正式跑 {cfg.epochs}，细胞数会再降 10-20%）")
    print(f"  单 patch 目标 ≈ {target:,}（全量 GT {GT_CELLS:,} / 4）")
    print(SEP)
    print(f"  {'h_s':>7} {'细胞数':>10} {'vs 目标':>10} {'用时':>8}")
    print("  " + "-" * 40)

    from boms import run_boms
    rows = []
    for hs in cfg.sweep_values:
        t0 = time.time()
        try:
            _, _, _, loc, _ = run_boms(
                X, Y, G, epochs=cfg.sweep_epochs, h_s=hs,
                h_r=cfg.h_r, K=cfg.K,
                x_min=float(X.min())-1, x_max=float(X.max())+1,
                y_min=float(Y.min())-1, y_max=float(Y.max())+1)
            n = len(loc)
            flag = "  ←" if 0.7*target < n < 1.4*target else ""
            print(f"  {hs:>7.1f} {n:>10,} {n-target:>+10,} "
                  f"{time.time()-t0:>7.0f}s{flag}", flush=True)
            rows.append((hs, n))
        except Exception as e:
            print(f"  {hs:>7.1f} {'FAIL':>10}  {type(e).__name__}: "
                  f"{str(e)[:34]}", flush=True)

    if rows:
        df = pd.DataFrame(rows, columns=["h_s", "cells"])
        df["gap"] = (df.cells - target).abs()
        b = df.sort_values("gap").iloc[0]
        print(f"\n  最接近: h_s={b.h_s}  →  {int(b.cells):,} "
              f"(4 块合计 ≈ {int(b.cells)*4:,}, GT {GT_CELLS:,})")
        print(f"  正式跑: python {os.path.basename(sys.argv[0])} "
              f"--h-s {b.h_s}")
        df.drop(columns=["gap"]).to_csv(f"{OUT_DIR}/hs_sweep.csv", index=False)


def show_status(cfg):
    print(f"\n{SEP}\n断点状态\n{SEP}")
    print(f"  转录本缓存: "
          f"{'✓ ' + os.path.basename(TX_CACHE) if os.path.exists(TX_CACHE) else '✗'}")
    for pb in patch_bounds():
        ck = f"{CKPT_DIR}/patch_{pb['i']}_{pb['j']}_hs{cfg.h_s}.npz"
        if os.path.exists(ck):
            d = np.load(ck, allow_pickle=True)
            print(f"  patch({pb['i']},{pb['j']}) ✓ {len(d['cell_loc']):>7,} 细胞")
        else:
            print(f"  patch({pb['i']},{pb['j']}) ✗ 未完成")
    print(f"  最终结果: {'✓' if os.path.exists(FINAL) else '✗'}  {FINAL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h-s", type=float, default=70.0)
    ap.add_argument("--h-r", type=float, default=0.3)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--min-size", type=int, default=20,
                    help="每细胞最少转录本数")
    ap.add_argument("--sweep-hs", action="store_true")
    ap.add_argument("--sweep-values", type=float, nargs="+",
                    default=[30, 50, 70, 100, 150])
    ap.add_argument("--sweep-epochs", type=int, default=10)
    ap.add_argument("--force", action="store_true", help="忽略 ckpt 重算")
    ap.add_argument("--status", action="store_true")
    cfg = ap.parse_args()

    print(SEP); print("BOMS × MERFISH 小鼠脑（断点续跑）"); print(SEP)
    print(f"  ROI x[{ROI_X0},{ROI_X1}] y[{ROI_Y0},{ROI_Y1}] µm  "
          f"GT {GT_CELLS:,}")
    print(f"  ckpt 目录: {CKPT_DIR}")

    if cfg.status:
        show_status(cfg); return

    if os.path.exists(FINAL) and not cfg.force:
        d = np.load(FINAL, allow_pickle=True)
        print(f"\n  ✓ 已有最终结果: {len(d['cell_loc']):,} 细胞")
        print(f"    重算请加 --force")
        return

    x, y, gi, gene_names = load_roi(force=False)

    if cfg.sweep_hs:
        sweep(x, y, gi, cfg); return

    print(f"\n{SEP}")
    print(f"跑 {N_SPLIT}×{N_SPLIT}={N_SPLIT**2} 个 patch  "
          f"h_s={cfg.h_s} epochs={cfg.epochs}  overlap={OVERLAP}µm")
    print(SEP)
    ckpts = []
    for pb in patch_bounds():
        ck = run_patch(pb, x, y, gi, len(gene_names), cfg, force=cfg.force)
        if ck:
            ckpts.append(ck)
    if not ckpts:
        print("\n  ✘ 无成功 patch"); return
    merge(ckpts, gene_names, cfg)


if __name__ == "__main__":
    main()