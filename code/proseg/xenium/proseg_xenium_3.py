#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_xenium_3_run.py  —— ProSeg × Xenium 淋巴结 ROI
=========================================================================
数据：/data/qiuyijia/dataset/xenium_roi_crop/transcripts.parquet
先验：transcripts.parquet 自带 cell_id 列（10X 官方分割）
输出：/data/qiuyijia/proseg/output/xenium_3/

运行：
  cd /data/qiuyijia/proseg
  python proseg_xenium_3_run.py
"""

import os, subprocess, sys

BASE     = "/data/qiuyijia/proseg"
OUT_DIR  = os.path.join(BASE, "output", "xenium_3")
TX_PATH  = "/data/qiuyijia/dataset/xenium_roi_crop/transcripts.parquet"
LOG_PATH = os.path.join(OUT_DIR, "proseg.log")

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(TX_PATH):
    print(f"✘ 找不到转录本文件: {TX_PATH}")
    sys.exit(1)

done = os.path.join(OUT_DIR, "cell-metadata.csv.gz")
if os.path.exists(done):
    import pandas as pd
    n = len(pd.read_csv(done))
    print(f"[跳过] 已存在（{n:,} 细胞），删除后重跑")
    sys.exit(0)

print(f"ProSeg × Xenium 淋巴结 ROI")
print(f"  输入: {TX_PATH}")
print(f"  输出: {OUT_DIR}")
print(f"  日志: {LOG_PATH}")

cmd = [
    "proseg", "--xenium",
    "--nthreads", "12",
    "--output-cell-metadata",       os.path.join(OUT_DIR, "cell-metadata.csv.gz"),
    "--output-transcript-metadata", os.path.join(OUT_DIR, "transcript-metadata.csv.gz"),
    "--output-cell-polygons",       os.path.join(OUT_DIR, "cell-polygons.geojson.gz"),
    "--overwrite",
    TX_PATH,
]
print("\n  " + " \\\n    ".join(cmd))

with open(LOG_PATH, "w") as log:
    result = subprocess.run(cmd, cwd=OUT_DIR, stdout=log, stderr=log)

if result.returncode == 0:
    import pandas as pd
    n = len(pd.read_csv(done))
    print(f"\n✓ 完成！  细胞数: {n:,}  (GT 2,442)")
    print(f"  → {OUT_DIR}/cell-metadata.csv.gz")
else:
    print(f"\n✘ 退出码 {result.returncode}，查看: {LOG_PATH}")
    sys.exit(result.returncode)