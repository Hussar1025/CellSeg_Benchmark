#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_merfish_1_run.py  —— ProSeg × MERFISH 小鼠脑
=========================================================================
MERFISH 数据没有自带 cell_id，先用 cell_metadata 质心做最近邻分配。

ROI：x [3589, 5761] µm，y [2327, 4499] µm（与 GeneSegNet/BOMS 一致）
GT：ROI 内约 5,700 细胞

输出：/data/qiuyijia/proseg/output/merfish_1/

运行：
  cd /data/qiuyijia/proseg
  conda activate proseg_eval   # 需要 pandas / scipy
  python proseg_merfish_1_run.py
"""

import os, subprocess, sys
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

BASE     = "/data/qiuyijia/proseg"
OUT_DIR  = os.path.join(BASE, "output", "merfish_1")
LOG_PATH = os.path.join(OUT_DIR, "proseg.log")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE = ("datasets_mouse_brain_map_BrainReceptorShowcase"
          "_Slice1_Replicate1")
DATA   = "/data/qiuyijia/dataset/merfish_mouse_brain"
TX_PATH   = os.path.join(DATA, f"{SAMPLE}_detected_transcripts_S1R1.csv")
META_PATH = os.path.join(DATA, f"{SAMPLE}_cell_metadata_S1R1.csv")

ROI_X0, ROI_X1 = 3589.0, 5761.0
ROI_Y0, ROI_Y1 = 2327.0, 4499.0
MAX_DIST_UM    = 15.0

PRIOR_PATH = os.path.join(OUT_DIR, "transcripts_roi_with_prior.csv.gz")

# ── 断点检测 ──────────────────────────────────────────────────────────────────
done = os.path.join(OUT_DIR, "cell-metadata.csv.gz")
if os.path.exists(done):
    n = len(pd.read_csv(done))
    print(f"[跳过] 已存在（{n:,} 细胞），删除后重跑"); sys.exit(0)

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: 生成带 cell_id 先验的转录本文件
# ══════════════════════════════════════════════════════════════════════════════
if os.path.exists(PRIOR_PATH):
    print(f"[跳过 Step1] 复用已有 {os.path.basename(PRIOR_PATH)}")
else:
    print("Step 1: 最近邻分配 cell_id ...")
    meta = pd.read_csv(META_PATH).rename(columns={"Unnamed: 0": "cell_id_orig"})
    meta_roi = meta[
        (meta.center_x >= ROI_X0) & (meta.center_x < ROI_X1) &
        (meta.center_y >= ROI_Y0) & (meta.center_y < ROI_Y1)
    ].reset_index(drop=True).copy()
    meta_roi["cell_id"] = np.arange(1, len(meta_roi) + 1)
    print(f"  ROI 内细胞: {len(meta_roi):,}")

    # 保存 cell_metadata_roi（供评估用）
    meta_roi[["cell_id","center_x","center_y","volume"]].to_csv(
        os.path.join(OUT_DIR, "cell_metadata_roi.csv"), index=False)

    print("  读取转录本（约 3.6 GB，分块）...")
    chunks, total = [], 0
    for chunk in pd.read_csv(TX_PATH, chunksize=2_000_000):
        total += len(chunk)
        sub = chunk[
            (chunk.global_x >= ROI_X0) & (chunk.global_x < ROI_X1) &
            (chunk.global_y >= ROI_Y0) & (chunk.global_y < ROI_Y1)
        ]
        if len(sub): chunks.append(sub)
    tx = pd.concat(chunks, ignore_index=True)
    print(f"  全量 {total:,} → ROI 内 {len(tx):,}")

    tree = cKDTree(meta_roi[["center_x","center_y"]].values)
    dists, idx = tree.query(tx[["global_x","global_y"]].values, k=1)
    tx["cell_id"] = np.where(dists < MAX_DIST_UM,
                             meta_roi["cell_id"].values[idx], 0)
    assigned = (tx.cell_id > 0).sum()
    print(f"  分配率: {assigned:,}/{len(tx):,} = {assigned/len(tx)*100:.1f}%")

    tx[["barcode_id","global_x","global_y","global_z","gene","cell_id"]].to_csv(
        PRIOR_PATH, index=False, compression="gzip")
    print(f"  已保存 → {PRIOR_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: 运行 ProSeg
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: 运行 ProSeg --merscope ...")
cmd = [
    "proseg", "--merscope",
    "--nthreads", "16",
    "--output-cell-metadata",       os.path.join(OUT_DIR, "cell-metadata.csv.gz"),
    "--output-transcript-metadata", os.path.join(OUT_DIR, "transcript-metadata.csv.gz"),
    "--output-cell-polygons",       os.path.join(OUT_DIR, "cell-polygons.geojson.gz"),
    "--overwrite",
    PRIOR_PATH,
]
print("  " + " \\\n    ".join(cmd))
print(f"  日志: {LOG_PATH}")

with open(LOG_PATH, "w") as log:
    result = subprocess.run(cmd, cwd=OUT_DIR, stdout=log, stderr=log)

if result.returncode == 0:
    n = len(pd.read_csv(done))
    print(f"\n✓ 完成！  细胞数: {n:,}  (GT ROI 内约 5,700)")
    print(f"  → {OUT_DIR}/cell-metadata.csv.gz")
else:
    print(f"\n✘ 退出码 {result.returncode}，查看: {LOG_PATH}")
    sys.exit(result.returncode)