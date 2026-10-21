#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_merfish_1_run.py  --  ProSeg × MERFISH mouse brain
=========================================================================
MERFISH data has no built-in cell_id; first do a nearest-neighbor assignment
from the cell_metadata centroids.

ROI: x [3589, 5761] um, y [2327, 4499] um (same as GeneSegNet/BOMS)
GT:  ~5,700 cells inside the ROI

Output: yourpath/proseg/output/merfish_1/

Run:
  cd yourpath/proseg
  conda activate proseg_eval   # needs pandas / scipy
  python proseg_merfish_1_run.py
"""

import os, subprocess, sys
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

BASE     = "yourpath/proseg"
OUT_DIR  = os.path.join(BASE, "output", "merfish_1")
LOG_PATH = os.path.join(OUT_DIR, "proseg.log")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE = ("datasets_mouse_brain_map_BrainReceptorShowcase"
          "_Slice1_Replicate1")
DATA   = "yourpath/dataset/merfish_mouse_brain"
TX_PATH   = os.path.join(DATA, f"{SAMPLE}_detected_transcripts_S1R1.csv")
META_PATH = os.path.join(DATA, f"{SAMPLE}_cell_metadata_S1R1.csv")

ROI_X0, ROI_X1 = 3589.0, 5761.0
ROI_Y0, ROI_Y1 = 2327.0, 4499.0
MAX_DIST_UM    = 15.0

PRIOR_PATH = os.path.join(OUT_DIR, "transcripts_roi_with_prior.csv.gz")

# ── checkpoint detection ──────────────────────────────────────────────────────────────────
done = os.path.join(OUT_DIR, "cell-metadata.csv.gz")
if os.path.exists(done):
    n = len(pd.read_csv(done))
    print(f"[Skip] already exists ({n:,} cells); delete and rerun"); sys.exit(0)

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: build transcript file with cell_id prior
# ══════════════════════════════════════════════════════════════════════════════
if os.path.exists(PRIOR_PATH):
    print(f"[Skip Step1] reusing existing {os.path.basename(PRIOR_PATH)}")
else:
    print("Step 1: nearest-neighbor cell_id assignment ...")
    meta = pd.read_csv(META_PATH).rename(columns={"Unnamed: 0": "cell_id_orig"})
    meta_roi = meta[
        (meta.center_x >= ROI_X0) & (meta.center_x < ROI_X1) &
        (meta.center_y >= ROI_Y0) & (meta.center_y < ROI_Y1)
    ].reset_index(drop=True).copy()
    meta_roi["cell_id"] = np.arange(1, len(meta_roi) + 1)
    print(f"  cells in ROI: {len(meta_roi):,}")

    # save cell_metadata_roi (for evaluation)
    meta_roi[["cell_id","center_x","center_y","volume"]].to_csv(
        os.path.join(OUT_DIR, "cell_metadata_roi.csv"), index=False)

    print("  reading transcripts (~3.6 GB, chunked)...")
    chunks, total = [], 0
    for chunk in pd.read_csv(TX_PATH, chunksize=2_000_000):
        total += len(chunk)
        sub = chunk[
            (chunk.global_x >= ROI_X0) & (chunk.global_x < ROI_X1) &
            (chunk.global_y >= ROI_Y0) & (chunk.global_y < ROI_Y1)
        ]
        if len(sub): chunks.append(sub)
    tx = pd.concat(chunks, ignore_index=True)
    print(f"  total {total:,} -> {len(tx):,} in ROI")

    tree = cKDTree(meta_roi[["center_x","center_y"]].values)
    dists, idx = tree.query(tx[["global_x","global_y"]].values, k=1)
    tx["cell_id"] = np.where(dists < MAX_DIST_UM,
                             meta_roi["cell_id"].values[idx], 0)
    assigned = (tx.cell_id > 0).sum()
    print(f"  assignment rate: {assigned:,}/{len(tx):,} = {assigned/len(tx)*100:.1f}%")

    tx[["barcode_id","global_x","global_y","global_z","gene","cell_id"]].to_csv(
        PRIOR_PATH, index=False, compression="gzip")
    print(f"  saved -> {PRIOR_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: run ProSeg
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: run ProSeg --merscope ...")
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
print(f"  log: {LOG_PATH}")

with open(LOG_PATH, "w") as log:
    result = subprocess.run(cmd, cwd=OUT_DIR, stdout=log, stderr=log)

if result.returncode == 0:
    n = len(pd.read_csv(done))
    print(f"\n✓ Done!  cell count: {n:,}  (GT ~5,700 in ROI)")
    print(f"  → {OUT_DIR}/cell-metadata.csv.gz")
else:
    print(f"\n✘ exit code {result.returncode}; see: {LOG_PATH}")
    sys.exit(result.returncode)