#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCS —— Stereo-seq（小鼠胚胎中脑背侧）
=========================================================================
坐标已确认对齐（之前诊断）：
  * GEM 的 (x,y) 直接是图像像素坐标，无需配准
  * GEM 落在 ssDNA 图内：x 2925~7174, y 4875~10574；图 27200×16385
  * bin1 = 0.5 µm/px（Stereo-seq DNB 网格）
  * 所有转录本都已分配（cell=0 的背景为 0 条）→ GT 分配率 100%

与 Xenium 版的三个关键差异：
  ① GEM 一行 = 某 DNB 格点某基因的 N 个 UMI（MIDCounts 列），不是一条转录本
     → 建 gene_map 时要按 MIDCounts 加权（np.add.at 加 count 而非加 1）
  ② nuclei_mask 从 GEM 的 cell 列建（CellBin 官方分割），不是核边界多边形
     → 每个 cell 的像素集合直接栅格化；可选腐蚀几像素模拟"核区"
  ③ 坐标原点用 GEM 的 min(x)/min(y)，gene_map 和 nuclei_mask 同原点天然对齐

⚠ 评估注意：nuclei_mask 来自 CellBin 的 cell 列，而 GT 也是同一个 cell 列
   → prior 与 GT 同源，转录本级指标（assignment accuracy）会循环虚高，
     和 MERFISH 一个性质。检测/定位/计数指标仍有效，评估时标注。

用法
-----
python ucs_stereoseq.py --check       # 只探测 + 估 gene_map 体积
python ucs_stereoseq.py               # 全流程
python ucs_stereoseq.py --top-genes 2000   # 降基因数控制显存
"""

import os, sys, gzip, time, argparse, subprocess
import numpy as np
import pandas as pd
import tifffile
import cv2

SEP = "=" * 72


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gem",   default="/data/qiuyijia/dataset/stereoseq/"
                   "stereo-seq_data_all/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz")
    p.add_argument("--ssdna", default="/data/qiuyijia/dataset/stereoseq/"
                   "stereo-seq_data_all/E16.5_E2S6.tif")
    p.add_argument("--out-dir", default="/data/qiuyijia/ucs_stereoseq")
    p.add_argument("--ucs-dir", default="/data/qiuyijia/ucs/UCS")
    p.add_argument("--python",  default="/data/qiuyijia/anaconda3/envs/ucs/bin/python")

    p.add_argument("--check", action="store_true")

    p.add_argument("--bin-px", type=int, default=1,
                   help="gene_map 每格聚合多少个 DNB（1=原分辨率）")
    p.add_argument("--top-genes", type=int, default=0,
                   help="只保留最高频 N 基因；0=全部（Stereo-seq 常 2 万基因，"
                        "gene_map 会很大，显存不够就降到 2000）")
    p.add_argument("--nucleus-erode", type=int, default=2,
                   help="cell mask 腐蚀多少像素当近似核区（0=直接用 cell 全域）")

    p.add_argument("--patch-size",       type=int, default=64)
    p.add_argument("--dilation-kernel",  type=int, default=5)
    p.add_argument("--dilation-iter",    type=int, default=2)
    p.add_argument("--tau",              type=int, default=5)
    p.add_argument("--fg-batch",         type=int, default=16)

    p.add_argument("--min-free-gb", type=float, default=10.0)
    p.add_argument("--gpu", type=int, default=-1)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def pick_best_gpu(min_free_gb=10.0):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip()
    best, best_free = None, -1.0
    print("  GPU 状态：")
    for line in out.splitlines():
        idx, free_mib, util = [s.strip() for s in line.split(",")]
        fg = float(free_mib)/1024.0
        mark = "  ←" if fg > best_free else ""
        if fg > best_free:
            best, best_free = int(idx), fg
        print(f"    GPU {idx}: {fg:5.1f} GB  {util:>3}%{mark}")
    if best_free < min_free_gb:
        raise RuntimeError(f"无 GPU ≥ {min_free_gb} GB（最大 {best_free:.1f}）")
    print(f"  → GPU {best}")
    return best


# ══════════════════════════════════════════════════════════════════════════════
def load_gem(cfg):
    """读 GEM_CellBin，返回 DataFrame(x, y, gene, count, cell)"""
    print(f"  读 GEM: {os.path.basename(cfg.gem)}（约 400 万行，稍等）...")
    opener = gzip.open if cfg.gem.endswith(".gz") else open
    skip = 0
    with opener(cfg.gem, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    df = pd.read_csv(cfg.gem, sep="\t", skiprows=skip)
    # 列名兼容
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"): ren[c] = "gene"
        elif cl == "x": ren[c] = "x"
        elif cl == "y": ren[c] = "y"
        elif cl in ("midcounts", "midcount", "counts", "count"): ren[c] = "count"
        elif cl in ("cell", "cellid", "label"): ren[c] = "cell"
    df = df.rename(columns=ren)
    if "count" not in df.columns:
        df["count"] = 1
    print(f"    {len(df):,} 行  基因 {df.gene.nunique():,}  "
          f"细胞 {df.loc[df.cell>0,'cell'].nunique():,}")
    print(f"    x {df.x.min()}~{df.x.max()}  y {df.y.min()}~{df.y.max()}  "
          f"（bin, 1bin=0.5µm）")
    print(f"    背景(cell=0) {(df.cell==0).sum():,} 条 "
          f"→ GT 分配率 {(df.cell>0).mean()*100:.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════════════════
def build_gene_map(cfg, gem, origin):
    gm_path = os.path.join(cfg.out_dir, "gene_map.tif")
    if os.path.exists(gm_path) and not cfg.force:
        print(f"[1] 复用 gene_map: {tifffile.imread(gm_path).shape}")
        return
    x0, y0, W, H = origin

    # 基因子集
    if cfg.top_genes > 0:
        top = gem.gene.value_counts().head(cfg.top_genes).index
        gem = gem[gem.gene.isin(top)]
        print(f"[1] 降到 top-{cfg.top_genes} 基因")
    genes = np.sort(gem.gene.unique())
    g2c = {g: i for i, g in enumerate(genes)}
    est = H * W * len(genes) / 1e9
    print(f"[1] gene_map {H}×{W}×{len(genes)}  uint8 ≈ {est:.2f} GB")
    if est > 15:
        raise RuntimeError(f"gene_map {est:.1f} GB 太大 → --top-genes 2000 "
                           f"或 --bin-px 2")

    gm = np.zeros((H, W, len(genes)), dtype=np.uint8)
    cx = ((gem.x.values - x0) // cfg.bin_px).astype(np.int32)
    cy = ((gem.y.values - y0) // cfg.bin_px).astype(np.int32)
    cg = gem.gene.map(g2c).values.astype(np.int32)
    cnt = gem["count"].values.astype(np.int32)          # ★ MIDCounts 加权
    ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    # 加 count 而不是加 1（一行是 N 个 UMI）；uint8 饱和到 255
    np.add.at(gm, (cy[ok], cx[ok], cg[ok]),
              np.minimum(cnt[ok], 255).astype(np.uint8))
    tifffile.imwrite(gm_path, gm)
    pd.DataFrame({"gene": genes, "channel": np.arange(len(genes))}).to_csv(
        os.path.join(cfg.out_dir, "gene_index.csv"), index=False)
    print(f"[1] 已存  非零 bin {(gm.sum(2) > 0).mean()*100:.1f}%")


def build_nuclei_mask(cfg, gem, origin):
    nm_path = os.path.join(cfg.out_dir, "nuclei_mask.tif")
    if os.path.exists(nm_path) and not cfg.force:
        nm = tifffile.imread(nm_path)
        print(f"[2] 复用 nuclei_mask: {nm.shape}  细胞 {int(nm.max())}")
        return
    x0, y0, W, H = origin
    sub = gem[gem.cell > 0]
    cx = ((sub.x.values - x0) // cfg.bin_px).astype(np.int32)
    cy = ((sub.y.values - y0) // cfg.bin_px).astype(np.int32)
    cell = sub.cell.values.astype(np.int32)
    ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)

    nm = np.zeros((H, W), dtype=np.int32)
    nm[cy[ok], cx[ok]] = cell[ok]                        # cell 列直接栅格化

    n_before = len(np.unique(nm)) - 1
    # 腐蚀：cell 全域偏大，腐蚀几像素更像"核区"，避免 prior 过度膨胀
    if cfg.nucleus_erode > 0:
        ker = np.ones((3, 3), np.uint8)
        eroded = np.zeros_like(nm)
        for cid in np.unique(nm[nm > 0]):
            m = (nm == cid).astype(np.uint8)
            e = cv2.erode(m, ker, iterations=cfg.nucleus_erode)
            if e.sum() == 0:                             # 腐蚀没了就保留 1 像素质心
                ys, xs = np.nonzero(m)
                e[int(ys.mean()), int(xs.mean())] = 1
            eroded[e > 0] = cid
        nm = eroded
    tifffile.imwrite(nm_path, nm)
    print(f"[2] cell 栅格化 {n_before:,} 个 → 腐蚀{cfg.nucleus_erode}px 后 "
          f"{len(np.unique(nm))-1:,}   覆盖 {(nm>0).mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
def run_ucs(cfg, gpu_id):
    log_dir = os.path.join(cfg.out_dir, "ucs_log")
    cmd = [
        cfg.python, os.path.join(cfg.ucs_dir, "run.py"),
        "--gene_map",             os.path.join(cfg.out_dir, "gene_map.tif"),
        "--nuclei_mask",          os.path.join(cfg.out_dir, "nuclei_mask.tif"),
        "--log_dir",              log_dir,
        "--patch_size",           str(cfg.patch_size),
        "--dilation_kernel_size", str(cfg.dilation_kernel),
        "--dilation_iter_num",    str(cfg.dilation_iter),
        "--tau",                  str(cfg.tau),
        "--fg_net_epoch",         "1",
        "--cell_net_epoch",       "1",
        "--fg_net_batch_size",    str(cfg.fg_batch),
        "--gpu",                  str(gpu_id),      # 物理 GPU 号
    ]
    print("[3] 运行 UCS ...")
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    r = subprocess.run(cmd, cwd=cfg.ucs_dir, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"UCS 退出码 {r.returncode}")
    print(f"[3] 完成 → {log_dir}/pred/segmentation_mask.tif")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    cfg = parse_args()
    print(SEP); print("UCS —— Stereo-seq"); print(SEP)
    gem = load_gem(cfg)

    x0, y0 = int(gem.x.min()), int(gem.y.min())
    W = (int(gem.x.max()) - x0) // cfg.bin_px + 1
    H = (int(gem.y.max()) - y0) // cfg.bin_px + 1
    origin = (x0, y0, W, H)

    n_gene = cfg.top_genes if cfg.top_genes > 0 else gem.gene.nunique()
    n_cell = gem.loc[gem.cell > 0, "cell"].nunique()
    print(f"\n{SEP}")
    print(f"原点 ({x0},{y0}) bin  map {H}×{W}  1bin={0.5*cfg.bin_px}µm")
    print(f"gene_map 估 {H*W*n_gene/1e9:.1f} GB ({n_gene} 基因)")
    print(f"GT 细胞（CellBin cell 列）{n_cell:,}")
    print(SEP)
    if cfg.check:
        print("\n--check 到此为止。gene_map 太大就加 --top-genes 2000")
        return

    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"\n{SEP}\nStep 1: gene_map（MIDCounts 加权）\n{SEP}")
    build_gene_map(cfg, gem, origin)
    print(f"\n{SEP}\nStep 2: nuclei_mask（CellBin cell 列）\n{SEP}")
    build_nuclei_mask(cfg, gem, origin)

    print(f"\n{SEP}\nStep 3: UCS\n{SEP}")
    gpu = cfg.gpu if cfg.gpu >= 0 else pick_best_gpu(cfg.min_free_gb)
    t0 = time.time()
    run_ucs(cfg, gpu)
    print(f"\n完成，耗时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"输出 {cfg.out_dir}/ucs_log/pred/segmentation_mask.tif")
    print("\n⚠ 评估时记住：nuclei_mask 与 GT 同源（都来自 CellBin cell 列），")
    print("   assignment accuracy 会循环虚高，仅检测/定位/计数指标有效。")


if __name__ == "__main__":
    main()