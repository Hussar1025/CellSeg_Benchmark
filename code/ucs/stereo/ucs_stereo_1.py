#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCS —— Stereo-seq (mouse embryo dorsal midbrain)
=========================================================================
Coordinates confirmed aligned (from earlier diagnosis):
  * GEM (x,y) are directly image pixel coordinates; no registration needed
  * GEM falls inside the ssDNA image: x 2925~7174, y 4875~10574; image 27200×16385
  * bin1 = 0.5 um/px (Stereo-seq DNB grid)
  * all transcripts assigned (0 rows with cell=0 background) -> GT assignment rate 100%

Three key differences vs the Xenium version:
  1. one GEM row = N UMIs of one gene at one DNB site (MIDCounts column), not one transcript
     -> build gene_map weighted by MIDCounts (np.add.at adds count, not 1)
  2. nuclei_mask is built from the GEM cell column (official CellBin segmentation), not nucleus boundary polygons
     -> rasterize each cell's pixel set directly; optionally erode a few pixels to approximate the nucleus region
  3. coordinate origin uses GEM min(x)/min(y), so gene_map and nuclei_mask align naturally at the same origin

⚠ Evaluation note: nuclei_mask comes from the CellBin cell column, and GT is the same cell column
   -> prior and GT share the same source, so transcript-level metrics (assignment accuracy) are circularly inflated,
      same as MERFISH. Detection/localization/count metrics remain valid; note this in evaluation.

Usage
-----
python ucs_stereoseq.py --check       # probe only + estimate gene_map volume
python ucs_stereoseq.py               # full pipeline
python ucs_stereoseq.py --top-genes 2000   # reduce gene count to control GPU memory
"""

import os, sys, gzip, time, argparse, subprocess
import numpy as np
import pandas as pd
import tifffile
import cv2

SEP = "=" * 72


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gem",   default="yourpath/dataset/stereoseq/"
                   "stereo-seq_data_all/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz")
    p.add_argument("--ssdna", default="yourpath/dataset/stereoseq/"
                   "stereo-seq_data_all/E16.5_E2S6.tif")
    p.add_argument("--out-dir", default="yourpath/ucs_stereoseq")
    p.add_argument("--ucs-dir", default="yourpath/ucs/UCS")
    p.add_argument("--python",  default="yourpath/anaconda3/envs/ucs/bin/python")

    p.add_argument("--check", action="store_true")

    p.add_argument("--bin-px", type=int, default=1,
                   help="how many DNBs aggregate into one gene_map cell (1=raw resolution)")
    p.add_argument("--top-genes", type=int, default=0,
                   help="keep only the top-N most frequent genes; 0=all (Stereo-seq often has ~20k genes, "
                        "gene_map gets huge; drop to 2000 if GPU memory is tight)")
    p.add_argument("--nucleus-erode", type=int, default=2,
                   help="how many pixels to erode the cell mask to approximate the nucleus region (0=use full cell area)")

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
    print("  GPU status:")
    for line in out.splitlines():
        idx, free_mib, util = [s.strip() for s in line.split(",")]
        fg = float(free_mib)/1024.0
        mark = "  ←" if fg > best_free else ""
        if fg > best_free:
            best, best_free = int(idx), fg
        print(f"    GPU {idx}: {fg:5.1f} GB  {util:>3}%{mark}")
    if best_free < min_free_gb:
        raise RuntimeError(f"no GPU >= {min_free_gb} GB (max {best_free:.1f})")
    print(f"  → GPU {best}")
    return best


# ══════════════════════════════════════════════════════════════════════════════
def load_gem(cfg):
    """Read GEM_CellBin, return DataFrame(x, y, gene, count, cell)"""
    print(f"  reading GEM: {os.path.basename(cfg.gem)} (~4M rows, one moment)...")
    opener = gzip.open if cfg.gem.endswith(".gz") else open
    skip = 0
    with opener(cfg.gem, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    df = pd.read_csv(cfg.gem, sep="\t", skiprows=skip)
    # column name compatibility
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
    print(f"    {len(df):,} rows  genes {df.gene.nunique():,}  "
          f"cells {df.loc[df.cell>0,'cell'].nunique():,}")
    print(f"    x {df.x.min()}~{df.x.max()}  y {df.y.min()}~{df.y.max()}  "
          f"(bin, 1 bin = 0.5 um)")
    print(f"    background(cell=0) {(df.cell==0).sum():,} rows "
          f"-> GT assignment rate {(df.cell>0).mean()*100:.1f}%")
    return df


# ══════════════════════════════════════════════════════════════════════════════
def build_gene_map(cfg, gem, origin):
    gm_path = os.path.join(cfg.out_dir, "gene_map.tif")
    if os.path.exists(gm_path) and not cfg.force:
        print(f"[1] reuse gene_map: {tifffile.imread(gm_path).shape}")
        return
    x0, y0, W, H = origin

    # gene subset
    if cfg.top_genes > 0:
        top = gem.gene.value_counts().head(cfg.top_genes).index
        gem = gem[gem.gene.isin(top)]
        print(f"[1] reduced to top-{cfg.top_genes} genes")
    genes = np.sort(gem.gene.unique())
    g2c = {g: i for i, g in enumerate(genes)}
    est = H * W * len(genes) / 1e9
    print(f"[1] gene_map {H}×{W}×{len(genes)}  uint8 ≈ {est:.2f} GB")
    if est > 15:
        raise RuntimeError(f"gene_map {est:.1f} GB too large -> --top-genes 2000 "
                           f"or --bin-px 2")

    gm = np.zeros((H, W, len(genes)), dtype=np.uint8)
    cx = ((gem.x.values - x0) // cfg.bin_px).astype(np.int32)
    cy = ((gem.y.values - y0) // cfg.bin_px).astype(np.int32)
    cg = gem.gene.map(g2c).values.astype(np.int32)
    cnt = gem["count"].values.astype(np.int32)          # ★ weighted by MIDCounts
    ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    # add count instead of 1 (one row is N UMIs); uint8 saturates at 255
    np.add.at(gm, (cy[ok], cx[ok], cg[ok]),
              np.minimum(cnt[ok], 255).astype(np.uint8))
    tifffile.imwrite(gm_path, gm)
    pd.DataFrame({"gene": genes, "channel": np.arange(len(genes))}).to_csv(
        os.path.join(cfg.out_dir, "gene_index.csv"), index=False)
    print(f"[1] saved  nonzero bins {(gm.sum(2) > 0).mean()*100:.1f}%")


def build_nuclei_mask(cfg, gem, origin):
    nm_path = os.path.join(cfg.out_dir, "nuclei_mask.tif")
    if os.path.exists(nm_path) and not cfg.force:
        nm = tifffile.imread(nm_path)
        print(f"[2] reuse nuclei_mask: {nm.shape}  cells {int(nm.max())}")
        return
    x0, y0, W, H = origin
    sub = gem[gem.cell > 0]
    cx = ((sub.x.values - x0) // cfg.bin_px).astype(np.int32)
    cy = ((sub.y.values - y0) // cfg.bin_px).astype(np.int32)
    cell = sub.cell.values.astype(np.int32)
    ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)

    nm = np.zeros((H, W), dtype=np.int32)
    nm[cy[ok], cx[ok]] = cell[ok]                        # rasterize cell column directly

    n_before = len(np.unique(nm)) - 1
    # erode: the full cell area is oversized; eroding a few pixels is closer to the nucleus region and avoids an over-inflated prior
    if cfg.nucleus_erode > 0:
        ker = np.ones((3, 3), np.uint8)
        eroded = np.zeros_like(nm)
        for cid in np.unique(nm[nm > 0]):
            m = (nm == cid).astype(np.uint8)
            e = cv2.erode(m, ker, iterations=cfg.nucleus_erode)
            if e.sum() == 0:                             # eroded away -> keep a 1-pixel centroid
                ys, xs = np.nonzero(m)
                e[int(ys.mean()), int(xs.mean())] = 1
            eroded[e > 0] = cid
        nm = eroded
    tifffile.imwrite(nm_path, nm)
    print(f"[2] cell rasterized {n_before:,} -> after erode {cfg.nucleus_erode}px "
          f"{len(np.unique(nm))-1:,}   coverage {(nm>0).mean()*100:.1f}%")


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
        "--gpu",                  str(gpu_id),      # physical GPU index
    ]
    print("[3] running UCS ...")
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    r = subprocess.run(cmd, cwd=cfg.ucs_dir, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"UCS exit code {r.returncode}")
    print(f"[3] done -> {log_dir}/pred/segmentation_mask.tif")


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
    print(f"origin ({x0},{y0}) bin  map {H}×{W}  1 bin={0.5*cfg.bin_px}um")
    print(f"gene_map est {H*W*n_gene/1e9:.1f} GB ({n_gene} genes)")
    print(f"GT cells (CellBin cell column) {n_cell:,}")
    print(SEP)
    if cfg.check:
        print("\n--check stops here. If gene_map is too large, add --top-genes 2000")
        return

    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"\n{SEP}\nStep 1: gene_map (MIDCounts-weighted)\n{SEP}")
    build_gene_map(cfg, gem, origin)
    print(f"\n{SEP}\nStep 2: nuclei_mask (CellBin cell column)\n{SEP}")
    build_nuclei_mask(cfg, gem, origin)

    print(f"\n{SEP}\nStep 3: UCS\n{SEP}")
    gpu = cfg.gpu if cfg.gpu >= 0 else pick_best_gpu(cfg.min_free_gb)
    t0 = time.time()
    run_ucs(cfg, gpu)
    print(f"\ndone, elapsed {(time.time()-t0)/60:.1f} min")
    print(f"output {cfg.out_dir}/ucs_log/pred/segmentation_mask.tif")
    print("\n⚠ remember for evaluation: nuclei_mask and GT share the same source (both from the CellBin cell column),")
    print("   so assignment accuracy is circularly inflated; only detection/localization/count metrics are valid.")


if __name__ == "__main__":
    main()
