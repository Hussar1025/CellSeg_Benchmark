#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_cosmx_1_run.py  --  ProSeg × CosMx pancreas
=========================================================================
Data:   yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles/Pancreas_tx_file.csv
Prior:  tx_file has a built-in cell_ID column (NanoString official segmentation)
        plus x/y_global_px global coordinates; use --cosmx and ProSeg handles
        the FOV numbering automatically.
Output: yourpath/proseg/output/cosmx_1/

Note: large dataset (65M rows), expect 30-60 min.

Run:
  cd yourpath/proseg
  python proseg_cosmx_1_run.py
"""

import os, subprocess, sys

BASE     = "yourpath/proseg"
OUT_DIR  = os.path.join(BASE, "output", "cosmx_1")
TX_PATH  = ("yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles/"
            "Pancreas_tx_file.csv")
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

print(f"ProSeg × CosMx pancreas")
print(f"  input:  {TX_PATH}")
print(f"  output: {OUT_DIR}")
print(f"  log:    {LOG_PATH}")
print(f"  expect: 30-60 min")

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
    print(f"\n✓ Done!  cell count: {n:,}  (GT 48,944)")
    print(f"  → {OUT_DIR}/cell-metadata.csv.gz")
else:
    print(f"\n✘ exit code {result.returncode}")
    print(f"  see log: {LOG_PATH}")
    # common issue hint
    print("  if a column-name error appears, try adding --cosmx-micron instead of --cosmx")
    sys.exit(result.returncode)