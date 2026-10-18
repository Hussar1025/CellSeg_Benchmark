#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_stereo4_roi.py — stereo_4 的 ROI 尺寸评估

回答三件事:
  1. GEM 的坐标跨度、细胞数、密度，与 stereo_1/2/3 对照
  2. 给定 ROI 边长下的 gene_map 体积估计
     UCS 的 gene_map 是稠密 H x W x G，Stereo-seq 是全转录组(2 万+基因)，
     所以体积 = H * W * G bytes，很容易上百 GB。E16.5_E2S7 上 21,370 基因
     x 5550 x 4299 = 509.9 GB，必须靠 --top-genes 砍基因维。
  3. 扫多个 ROI 边长，给出细胞数 / 内存 / 是否可行的对照表

不读全量转录本，只按 chunk 统计坐标与细胞，内存友好。

用法:
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

ROOT = "/data/qiuyijia/dataset/stereo_mouse_embryo"
BIN_UM = 0.5
GB = 1e9

REF = {  # 已知数据集，用于横向对照
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
                    help="0 = 全读；调试时可设 5000 万")
    ap.add_argument("--top-genes", type=int, default=2000)
    ap.add_argument("--rois", type=int, nargs="+",
                    default=[4000, 6000, 8000, 10000, 12000])
    a = ap.parse_args()

    log("=" * 78)
    log("stereo_4 ROI 评估  (Stereo-seq mouse embryo)")
    log("=" * 78)

    gem = a.gem or find_gem(a.root)
    if not gem or not os.path.exists(gem):
        log(f"✗ 找不到 GEM。目录内容:")
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
        log("     文件头:")
        for h in hdr[:8]:
            log(f"       {h}")

    log(f"\n分块统计 (chunk = {a.chunk:,} 行) ...")
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
            log(f"  列: {list(ch.columns)}")
            low = {c.lower(): c for c in ch.columns}
            xcol = low.get("x")
            ycol = low.get("y")
            cellcol = next((low[k] for k in ("cell", "cellid", "label")
                            if k in low), None)
            gcol = next((low[k] for k in ("geneid", "gene")
                         if k in low), None)
            ccol = next((low[k] for k in ("midcount", "midcounts", "count")
                         if k in low), None)
            log(f"  用列: x={xcol} y={ycol} cell={cellcol} "
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
            log(f"    {n_rows:,} 行  细胞 {len(cells):,}  基因 {len(genes):,}")
        if a.max_rows and n_rows >= a.max_rows:
            log(f"    达到 --max-rows {a.max_rows:,}，提前停止（统计不完整）")
            break

    W = int(xmax - xmin) + 1
    H = int(ymax - ymin) + 1
    log(f"\n【全片】")
    log(f"  行 {n_rows:,}   UMI {total_umi:,}")
    log(f"  x[{xmin:.0f},{xmax:.0f}]  y[{ymin:.0f},{ymax:.0f}]")
    log(f"  跨度 {W:,} x {H:,} bin = {W*BIN_UM:.0f} x {H*BIN_UM:.0f} um")
    log(f"  细胞 {len(cells):,}   基因 {len(genes):,}")
    if len(cells):
        area_um2 = W * H * BIN_UM ** 2
        log(f"  面积 {area_um2/1e6:.2f} mm2   "
            f"细胞密度 {len(cells)/area_um2*1e6:.0f} 个/mm2")
        log(f"  每细胞 {total_umi/len(cells):.0f} UMI  "
            f"{n_rows/len(cells):.0f} 行")

    log(f"\n【与已跑过的切片对照】")
    log(f"  {'切片':<14}{'细胞':>10}{'跨度 bin':>18}{'基因':>9}")
    for k, v in REF.items():
        log(f"  {k:<14}{v['cells']:>10,}"
            f"{f'{v[chr(115)+chr(112)+chr(97)+chr(110)+chr(95)+chr(98)+chr(105)+chr(110)][0]:,} x {v[chr(115)+chr(112)+chr(97)+chr(110)+chr(95)+chr(98)+chr(105)+chr(110)][1]:,}':>18}"
            f"{(v['genes'] or 0):>9,}")
    log(f"  {'stereo_4':<14}{len(cells):>10,}{f'{W:,} x {H:,}':>18}"
        f"{len(genes):>9,}")
    log(f"  -> stereo_4 的细胞数是 E16.5_E2S7 的 "
        f"{len(cells)/7960:.0f} 倍，面积是 {W*H/(4298*5549):.0f} 倍")

    log(f"\n【ROI 尺寸扫描】")
    log(f"  UCS 的 gene_map 是稠密 H x W x G(uint8)，全转录组下极易爆内存。")
    log(f"  参考: E16.5_E2S7 的 5550 x 4299 x 21,370 = 509.9 GB，")
    log(f"        --top-genes 2000 后降到 47.7 GB。")
    log(f"\n  {'ROI(bin)':>10}{'ROI(um)':>10}{'占全片':>9}"
        f"{'预计细胞':>10}{'全基因 GB':>12}{f'top{a.top_genes} GB':>12}"
        f"{'判定':>8}")
    log("  " + "-" * 74)
    dens = len(cells) / (W * H) if W * H else 0
    for r in a.rois:
        if r > min(W, H):
            log(f"  {r:>10,}  超出全片跨度 {min(W,H):,}，跳过")
            continue
        frac = r * r / (W * H)
        nc = dens * r * r
        gb_all = r * r * len(genes) / GB
        gb_top = r * r * min(a.top_genes, len(genes)) / GB
        verdict = ("可行" if gb_top < 60 else
                   "偏大" if gb_top < 150 else "不可行")
        log(f"  {r:>10,}{r*BIN_UM:>10.0f}{frac*100:>8.1f}%"
            f"{nc:>10,.0f}{gb_all:>12.1f}{gb_top:>12.1f}{verdict:>8}")
    log(f"\n  内存上限参考: 本机 1007 GB 总量，通常可用 700-850 GB。")
    log(f"  但 gene_map 还要写成 tif，磁盘同样需要等量空间，")
    log(f"  而 /data 目前 98% 已用（剩 8 TB）。建议 gene_map 控制在 60 GB 内。")

    if gene_umi:
        s = pd.Series(gene_umi).sort_values(ascending=False)
        tot = s.sum()
        log(f"\n【基因表达集中度】决定 --top-genes 砍掉多少信号")
        for k in (500, 1000, 2000, 5000):
            if k <= len(s):
                log(f"  top {k:>5,} 个基因覆盖 "
                    f"{s.head(k).sum()/tot*100:5.1f}% 的 UMI")

    log(f"\n【影像】")
    tifs = sorted(glob.glob(os.path.join(a.root, "**", "*.tif"),
                            recursive=True))
    for t in tifs[:10]:
        log(f"  {os.path.getsize(t)/1e9:8.2f} GB  {os.path.basename(t)}")
    log(f"  ⚠ 之前的检查显示只有 HE 三通道影像，没有 ssDNA/DAPI。")
    log(f"    UCS 的 nuclei_mask 在 stereo 上是从 GEM 的 cell 列栅格化的，")
    log(f"    不依赖影像，所以 UCS 能跑；但 Cellist / GeneSegNet /")
    log(f"    全部图像方法需要核染色，在这个数据集上做不了。")

    log(f"\n【GT 溯源 — 跑之前必须确认】")
    log(f"  cell 列的 {len(cells):,} 个细胞若来自 bin 聚类，")
    log(f"  它本身就是一个分割结果，拿来当 GT 会形成循环偏差")
    log(f"  (同 xenium_1 的膨胀核 GT 问题)。查该数据集的原始文档，")
    log(f"  确认 CellBin 分割是基于 ssDNA 影像还是纯表达聚类。")


if __name__ == "__main__":
    main()