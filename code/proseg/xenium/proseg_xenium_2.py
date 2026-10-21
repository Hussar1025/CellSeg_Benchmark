#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_xenium_2_run.py  --  ProSeg × Xenium breast cancer
=========================================================================
Data:   yourpath/dataset/xenium_breast/transcripts.parquet
Prior:  transcripts.parquet has a built-in cell_id column (10X official segmentation)
Output: yourpath/proseg/output/xenium_2/

Run:
  cd yourpath/proseg
  python proseg_xenium_2_run.py
"""

import os, subprocess, sys

BASE     = "yourpath/proseg"
OUT_DIR  = os.path.join(BASE, "output", "xenium_2")
TX_PATH  = "yourpath/dataset/xenium_breast/transcripts.parquet"
LOG_PATH = os.path.join(OUT_DIR, "proseg.log")

os.makedirs(OUT_DIR, exist_ok=True)

# ── check input ──────────────────────────────────────────────────────────────────
if not os.path.exists(TX_PATH):
    print(f"✘ transcript file not found: {TX_PATH}")
    sys.exit(1)

# ── checkpoint detection ──────────────────────────────────────────────────────────────────
done = os.path.join(OUT_DIR, "cell-metadata.csv.gz")
if os.path.exists(done):
    import pandas as pd
    n = len(pd.read_csv(done))
    print(f"[Skip] cell-metadata.csv.gz already exists ({n:,} cells); delete and rerun")
    sys.exit(0)

# ── run ProSeg ───────────────────────────────────────────────────────────────
print(f"ProSeg × Xenium breast cancer")
print(f"  input:  {TX_PATH}")
print(f"  output: {OUT_DIR}")
print(f"  log:    {LOG_PATH}")

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
    print(f"\n✓ Done!  cell count: {n:,}  (GT ~29,892 in ROI)")
    print(f"  → {OUT_DIR}/cell-metadata.csv.gz")
else:
    print(f"\n✘ ProSeg exit code {result.returncode}; see: {LOG_PATH}")
    sys.exit(result.returncode)