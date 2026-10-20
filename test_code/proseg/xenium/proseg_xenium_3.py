#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_xenium_3_run.py  --  ProSeg × Xenium lymph-node ROI
=========================================================================
Data:   yourpath/dataset/xenium_roi_crop/transcripts.parquet
Prior:  transcripts.parquet has a built-in cell_id column (10X official segmentation)
Output: yourpath/proseg/output/xenium_3/

Run:
  cd yourpath/proseg
  python proseg_xenium_3_run.py
"""

import os, subprocess, sys

BASE     = "yourpath/proseg"
OUT_DIR  = os.path.join(BASE, "output", "xenium_3")
TX_PATH  = "yourpath/dataset/xenium_roi_crop/transcripts.parquet"
LOG_PATH = os.path.join(OUT_DIR, "proseg.log")

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(TX_PATH):
    print(f"✘ transcript file not found: {TX_PATH}")
    sys.exit(1)

done = os.path.join(OUT_DIR, "cell-metadata.csv.gz")
if os.path.exists(done):
    import pandas as pd
    n = len(pd.read_csv(done))
    print(f"[Skip] already exists ({n:,} cells); delete and rerun")
    sys.exit(0)

print(f"ProSeg × Xenium lymph-node ROI")
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
    print(f"\n✓ Done!  cell count: {n:,}  (GT 2,442)")
    print(f"  → {OUT_DIR}/cell-metadata.csv.gz")
else:
    print(f"\n✘ exit code {result.returncode}; see: {LOG_PATH}")
    sys.exit(result.returncode)