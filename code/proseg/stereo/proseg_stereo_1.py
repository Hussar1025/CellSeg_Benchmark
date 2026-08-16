#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_stereo_1_run.py  —— ProSeg × Stereo-seq 小鼠胚胎（实验性）
=========================================================================
⚠ ProSeg 官方不支持 Stereo-seq，本脚本做格式转换后强行尝试。

为什么不支持：
  1. ProSeg 假设每行 = 一个单分子 RNA（成像原位测序）
     Stereo-seq 的 GEM 每行 = 一个空间 bin 的 UMI 计数（测序法）
  2. ProSeg 的 RNA 扩散模型在 UMI 聚合数据上没有物理意义
  3. 没有官方 --stereo 参数

本脚本的适配策略：
  - 把 MIDCounts > 1 的 bin 展开成多条合成转录本
  - bin 坐标 × 0.5 换算成 µm
  - 用 CellBin 的 cell 列做 cell_id 先验
  - 用 --merscope 参数（列名最接近）尝试运行

结果质量不保证，仅供基准对比参考。

运行：
  cd /data/qiuyijia/proseg
  conda activate proseg_eval
  python proseg_stereo_1_run.py
"""

import os, subprocess, sys, gzip
import pandas as pd
import numpy as np

BASE     = "/data/qiuyijia/proseg"
OUT_DIR  = os.path.join(BASE, "output", "stereo_1")
LOG_PATH = os.path.join(OUT_DIR, "proseg.log")
os.makedirs(OUT_DIR, exist_ok=True)

GEM_PATH = ("/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all/"
            "E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz")
TX_OUT   = os.path.join(OUT_DIR, "transcripts_expanded.csv.gz")

BIN_UM   = 0.5   # Stereo-seq bin1 = 0.5 µm

# ── 断点检测 ──────────────────────────────────────────────────────────────────
done = os.path.join(OUT_DIR, "cell-metadata.csv.gz")
if os.path.exists(done):
    n = len(pd.read_csv(done))
    print(f"[跳过] 已存在（{n:,} 细胞），删除后重跑"); sys.exit(0)

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: GEM → 展开成 transcript-like 格式
# ══════════════════════════════════════════════════════════════════════════════
if os.path.exists(TX_OUT):
    print(f"[跳过 Step1] 复用已有 {os.path.basename(TX_OUT)}")
else:
    print("Step 1: 读取 GEM 并展开...")
    opener = gzip.open if GEM_PATH.endswith(".gz") else open
    skip = 0
    with opener(GEM_PATH, "rt") as fh:
        for line in fh:
            if line.startswith("#"): skip += 1
            else: break

    gem = pd.read_csv(GEM_PATH, sep="\t", skiprows=skip)
    ren = {}
    for c in gem.columns:
        cl = c.lower()
        if cl in ("geneid","gene"):           ren[c] = "gene"
        elif cl == "x":                        ren[c] = "x_bin"
        elif cl == "y":                        ren[c] = "y_bin"
        elif cl in ("midcounts","count"):      ren[c] = "count"
        elif cl in ("cell","cellid","label"):  ren[c] = "cell"
    gem = gem.rename(columns=ren)
    if "count" not in gem.columns: gem["count"] = 1
    if "cell"  not in gem.columns: gem["cell"]  = 0

    print(f"  GEM: {len(gem):,} 行  基因 {gem.gene.nunique()}  "
          f"cell>0: {(gem.cell>0).sum():,}")

    # 展开：每个 count 变成一行合成转录本
    print("  展开 UMI（MIDCounts > 1 的 bin 复制多行）...")
    gem_rep = gem.loc[gem.index.repeat(gem["count"].clip(1).astype(int))].copy()
    gem_rep = gem_rep.reset_index(drop=True)

    # 坐标换算：bin → µm
    gem_rep["global_x"] = gem_rep["x_bin"] * BIN_UM
    gem_rep["global_y"] = gem_rep["y_bin"] * BIN_UM
    gem_rep["global_z"] = 0.0
    gem_rep["cell_id"]  = gem_rep["cell"].astype(int)
    gem_rep["barcode_id"] = np.arange(len(gem_rep))

    out = gem_rep[["barcode_id","global_x","global_y","global_z",
                   "gene","cell_id"]]
    print(f"  展开后: {len(out):,} 条合成转录本")
    print(f"  global_x: {out.global_x.min():.0f}~{out.global_x.max():.0f} µm")
    print(f"  分配率: {(out.cell_id>0).sum()/len(out)*100:.1f}%")

    out.to_csv(TX_OUT, index=False, compression="gzip")
    print(f"  已保存 → {TX_OUT}")

# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: 运行 ProSeg（实验性，用 --merscope 参数）
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: 运行 ProSeg（实验性）...")
print("  ⚠ Stereo-seq 不在 ProSeg 官方支持范围，结果仅供参考")

cmd = [
    "proseg", "--merscope",
    "--nthreads", "16",
    "--output-cell-metadata",       os.path.join(OUT_DIR, "cell-metadata.csv.gz"),
    "--output-transcript-metadata", os.path.join(OUT_DIR, "transcript-metadata.csv.gz"),
    "--output-cell-polygons",       os.path.join(OUT_DIR, "cell-polygons.geojson.gz"),
    "--overwrite",
    TX_OUT,
]
print("  " + " \\\n    ".join(cmd))
print(f"  日志: {LOG_PATH}")

with open(LOG_PATH, "w") as log:
    result = subprocess.run(cmd, cwd=OUT_DIR, stdout=log, stderr=log)

if result.returncode == 0:
    n = len(pd.read_csv(done))
    print(f"\n✓ 完成（实验性）！  细胞数: {n:,}  (GT 6,229)")
    print(f"  → {OUT_DIR}/cell-metadata.csv.gz")
    print("  ⚠ 评估时注意：Stereo-seq 为非官方支持，结果质量不保证")
else:
    print(f"\n✘ 退出码 {result.returncode}")
    print(f"  查看: {LOG_PATH}")
    print("  Stereo-seq 不在 ProSeg 官方支持范围，失败是预期内的")
    # 写占位标记
    with open(os.path.join(OUT_DIR, "NOT_SUPPORTED.txt"), "w") as f:
        f.write("ProSeg 不官方支持 Stereo-seq（sequencing-based，非 in situ）\n"
                "已尝试格式转换后运行，但失败。\n"
                "评估表中此格标记 N/A。\n")
    print("  已写 NOT_SUPPORTED.txt，评估时跳过此数据集")