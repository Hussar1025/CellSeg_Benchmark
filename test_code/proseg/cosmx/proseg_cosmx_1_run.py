#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_cosmx_1_run.py  —— ProSeg × CosMx 胰腺
=========================================================================
数据：/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles/Pancreas_tx_file.csv
先验：tx_file 自带 cell_ID 列（NanoString 官方分割）+ x/y_global_px 全局坐标
      直接用 --cosmx 参数，ProSeg 自动处理 FOV 编号
输出：/data/qiuyijia/proseg/output/cosmx_1/

⚠ 数据量大（6500 万行），预计 30-60 分钟

运行：
  cd /data/qiuyijia/proseg
  python proseg_cosmx_1_run.py
"""

import os, subprocess, sys

BASE     = "/data/qiuyijia/proseg"
OUT_DIR  = os.path.join(BASE, "output", "cosmx_1")
TX_PATH  = ("/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles/"
            "Pancreas_tx_file.csv")
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

print(f"ProSeg × CosMx 胰腺")
print(f"  输入: {TX_PATH}")
print(f"  输出: {OUT_DIR}")
print(f"  日志: {LOG_PATH}")
print(f"  预计: 30-60 分钟")

cmd = [
    "proseg", "--cosmx",
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
    print(f"\n✓ 完成！  细胞数: {n:,}  (GT 48,944)")
    print(f"  → {OUT_DIR}/cell-metadata.csv.gz")
else:
    print(f"\n✘ 退出码 {result.returncode}")
    print(f"  查看日志: {LOG_PATH}")
    # 常见问题提示
    print("  若报列名错误，尝试在命令里加：--cosmx-micron 替换 --cosmx")
    sys.exit(result.returncode)