#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_stereo4_roi.py — ROI size evaluation for stereo_4

Answers three questions:
  1. GEM coordinate span, cell count, density, compared with stereo_1/2/3
  2. gene_map volume estimate for a given ROI side length
     UCS gene_map is dense H x W x G; Stereo-seq is whole-transcriptome (20k+ genes),
     so volume = H * W * G bytes, easily 100+ GB. On E16.5_E2S7, 21,370 genes
     x 5550 x 4299 = 509.9 GB; --top-genes must trim the gene axis.
  3. sweep several ROI side lengths and give a cell count / memory / feasibility table

Does not read all transcripts; only stats coordinates and cells per chunk, memory-friendly.

Usage:
  python3 probe_stereo4_roi.py
  python3 probe_stereo4_roi.py --gem <path> --top-genes 2000
"""
import argparse
import glob
import gzip
import os
import sys

import numpy as np
import pandas as pd

ROOT = "yourpath/dataset/stereo_mouse_embryo"
BIN_UM = 0.5
GB = 1e9

REF = {  # known datasets, for cross comparison
    "E14.5_E1S3": dict(cells=4872, span_bin=(2822, 5599), genes=18698),
    "E16.5_E2S6": dict(cells=6229, span_bin=(4249, 5699), genes=None),
    "E16.5_E2S7": dict(cells=7960, span_bin=(4298, 5549), genes=21370,
                       gene_map_gb=509.9),
}


def log(m):
    print(m, flush=True)


def find_gem(root):
    pats = ["**/*GEM*CellBin*.tsv.gz", "**/*GEM*.tsv.gz", "**/*.gem.gz",
            "**/*.tsv.gz"]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(root, p), recursive=True))
        if hits:
            return max(hits, key=os.path.getsize)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--gem", default="")
    ap.add_argument("--chunk", type=int, default=5_000_000)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0 = read all; set 50M when debugging")
    ap.add_argument("--top-genes", type=int, default=2000)
    ap.add_argument("--rois", type=int, nargs="+",
                    default=[4000, 6000, 8000, 10000, 12000])
    a = ap.parse_args()

    log("=" * 78)
    log("stereo_4 ROI evaluation  (Stereo-seq mouse embryo)")
    log("=" * 78)

    gem = a.gem or find_gem(a.root)
    if not gem or not os.path.exists(gem):
        log(f"✗ GEM not found. Directory contents:")
        for f in sorted(glob.glob(os.path.join(a.root, "**", "*"),
                                  recursive=True))[:40]:
            if os.path.isfile(f):
                log(f"    {os.path.getsize(f)/1e6:10.1f} MB  {f}")
        sys.exit(1)
    log(f"\nGEM  {gem}")
    log(f"     {os.path.getsize(gem)/1e9:.2f} GB")

    skip = 0
    hdr = []
    with gzip.open(gem, "rt") as f:
        for line in f:
            if line.startswith("#"):
                skip += 1
                hdr.append(line.strip())
            else:
                break
    if hdr:
        log("     header:")
        for h in hdr[:8]:
            log(f"       {h}")

    log(f"\nchunked stats (chunk = {a.chunk:,} rows) ...")
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    n_rows = 0
    total_umi = 0
    cells = set()
    genes = set()
    gene_umi = {}
    cellcol = xcol = ycol = None

    for i, ch in enumerate(pd.read_csv(gem, sep="\t", skiprows=skip,
                                       chunksize=a.chunk)):
        if i == 0:
            log(f"  columns: {list(ch.columns)}")
            low = {c.lower(): c for c in ch.columns}
            xcol = low.get("x")
            ycol = low.get("y")
            cellcol = next((low[k] for k in ("cell", "cellid", "label")
                            if k in low), None)
            gcol = next((low[k] for k in ("geneid", "gene")
                         if k in low), None)
            ccol = next((low[k] for k in ("midcount", "midcounts", "count")
                         if k in low), None)
            log(f"  using columns: x={xcol} y={ycol} cell={cellcol} "
                f"gene={gcol} count={ccol}")
        n_rows += len(ch)
        xmin = min(xmin, ch[xcol].min())
        xmax = max(xmax, ch[xcol].max())
        ymin = min(ymin, ch[ycol].min())
        ymax = max(ymax, ch[ycol].max())
        if ccol:
            total_umi += int(ch[ccol].sum())
        if cellcol:
            cells.update(ch.loc[ch[cellcol] > 0, cellcol].unique().tolist())
        if gcol:
            genes.update(ch[gcol].unique().tolist())
            if ccol:
                s = ch.groupby(gcol)[ccol].sum()
                for k, v in s.items():
                    gene_umi[k] = gene_umi.get(k, 0) + int(v)
        if i % 10 == 0:
            log(f"    {n_rows:,} rows  cells {len(cells):,}  genes {len(genes):,}")
        if a.max_rows and n_rows >= a.max_rows:
            log(f"    reached --max-rows {a.max_rows:,}, stopped early (stats incomplete)")
            break

    W = int(xmax - xmin) + 1
    H = int(ymax - ymin) + 1
    log(f"\n[whole section]")
    log(f"  rows {n_rows:,}   UMI {total_umi:,}")
    log(f"  x[{xmin:.0f},{xmax:.0f}]  y[{ymin:.0f},{ymax:.0f}]")
    log(f"  span {W:,} x {H:,} bin = {W*BIN_UM:.0f} x {H*BIN_UM:.0f} um")
    log(f"  cells {len(cells):,}   genes {len(genes):,}")
    if len(cells):
        area_um2 = W * H * BIN_UM ** 2
        log(f"  area {area_um2/1e6:.2f} mm2   "
            f"cell density {len(cells)/area_um2*1e6:.0f} cells/mm2")
        log(f"  per cell {total_umi/len(cells):.0f} UMI  "
            f"{n_rows/len(cells):.0f} rows")

    log(f"\n[comparison with slices already run]")
    log(f"  {'slice':<14}{'cells':>10}{'span bin':>18}{'genes':>9}")
    for k, v in REF.items():
        log(f"  {k:<14}{v['cells']:>10,}"
            f"{f'{v[chr(115)+chr(112)+chr(97)+chr(110)+chr(95)+chr(98)+chr(105)+chr(110)][0]:,} x {v[chr(115)+chr(112)+chr(97)+chr(110)+chr(95)+chr(98)+chr(105)+chr(110)][1]:,}':>18}"
            f"{(v['genes'] or 0):>9,}")
    log(f"  {'stereo_4':<14}{len(cells):>10,}{f'{W:,} x {H:,}':>18}"
        f"{len(genes):>9,}")
    log(f"  -> stereo_4 cell count is {len(cells)/7960:.0f}x that of E16.5_E2S7, "
        f"area is {W*H/(4298*5549):.0f}x")

    log(f"\n[ROI size sweep]")
    log(f"  UCS gene_map is dense H x W x G(uint8); it OOMs easily on whole-transcriptome data.")
    log(f"  reference: E16.5_E2S7 5550 x 4299 x 21,370 = 509.9 GB,")
    log(f"             drops to 47.7 GB after --top-genes 2000.")
    log(f"\n  {'ROI(bin)':>10}{'ROI(um)':>10}{'frac':>9}"
        f"{'cells est':>10}{'all-genes GB':>12}{f'top{a.top_genes} GB':>12}"
        f"{'verdict':>8}")
    log("  " + "-" * 74)
    dens = len(cells) / (W * H) if W * H else 0
    for r in a.rois:
        if r > min(W, H):
            log(f"  {r:>10,}  exceeds whole-section span {min(W,H):,}, skip")
            continue
        frac = r * r / (W * H)
        nc = dens * r * r
        gb_all = r * r * len(genes) / GB
        gb_top = r * r * min(a.top_genes, len(genes)) / GB
        verdict = ("feasible" if gb_top < 60 else
                   "large" if gb_top < 150 else "infeasible")
        log(f"  {r:>10,}{r*BIN_UM:>10.0f}{frac*100:>8.1f}%"
            f"{nc:>10,.0f}{gb_all:>12.1f}{gb_top:>12.1f}{verdict:>8}")
    log(f"\n  memory cap reference: this machine has 1007 GB total, usually 700-850 GB available.")
    log(f"  but gene_map is also written to tif, needing the same disk space,")
    log(f"  and the data disk is 98% full (8 TB left). Keep gene_map under 60 GB.")

    if gene_umi:
        s = pd.Series(gene_umi).sort_values(ascending=False)
        tot = s.sum()
        log(f"\n[gene expression concentration] how much signal --top-genes cuts")
        for k in (500, 1000, 2000, 5000):
            if k <= len(s):
                log(f"  top {k:>5,} genes cover "
                    f"{s.head(k).sum()/tot*100:5.1f}% of UMI")

    log(f"\n[imaging]")
    tifs = sorted(glob.glob(os.path.join(a.root, "**", "*.tif"),
                            recursive=True))
    for t in tifs[:10]:
        log(f"  {os.path.getsize(t)/1e9:8.2f} GB  {os.path.basename(t)}")
    log(f"  ⚠ earlier checks show only HE 3-channel imaging, no ssDNA/DAPI.")
    log(f"    on stereo, UCS nuclei_mask is rasterized from the GEM cell column,")
    log(f"    not image-dependent, so UCS can run; but Cellist / GeneSegNet /")
    log(f"    all image-based methods need nuclear staining, which this dataset lacks.")

    log(f"\n[GT provenance — must confirm before running]")
    log(f"  if the {len(cells):,} cells in the cell column come from bin clustering,")
    log(f"  it is itself a segmentation result; using it as GT creates circular bias")
    log(f"  (same as the xenium_1 dilated-nucleus GT problem). Check the dataset docs")
    log(f"  to confirm whether CellBin segmentation is based on ssDNA imaging or pure expression clustering.")


if __name__ == "__main__":
    main()
