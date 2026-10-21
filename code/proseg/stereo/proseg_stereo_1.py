#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_stereo_1_run.py  --  ProSeg × Stereo-seq mouse embryo (experimental)
=========================================================================
Warning: ProSeg does not officially support Stereo-seq; this script does a
format conversion and then forces an attempt.

Why unsupported:
  1. ProSeg assumes each row = one single-molecule RNA (imaging in situ)
     Stereo-seq GEM rows = UMI counts of one spatial bin (sequencing-based)
  2. ProSeg's RNA diffusion model has no physical meaning on aggregated UMI data
  3. no official --stereo flag

Adaptation strategy:
  - expand bins with MIDCounts > 1 into multiple synthetic transcripts
  - convert bin coordinates × 0.5 to um
  - use CellBin's cell column as the cell_id prior
  - run with --merscope (closest column names)

Result quality is not guaranteed; for benchmark comparison only.

Run:
  cd yourpath/proseg
  conda activate proseg_eval
  python proseg_stereo_1_run.py
"""

import os, subprocess, sys, gzip
import pandas as pd
import numpy as np

BASE     = "yourpath/proseg"
OUT_DIR  = os.path.join(BASE, "output", "stereo_1")
LOG_PATH = os.path.join(OUT_DIR, "proseg.log")
os.makedirs(OUT_DIR, exist_ok=True)

GEM_PATH = ("yourpath/dataset/stereoseq/stereo-seq_data_all/"
            "E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz")
TX_OUT   = os.path.join(OUT_DIR, "transcripts_expanded.csv.gz")

BIN_UM   = 0.5   # Stereo-seq bin1 = 0.5 µm

# ── checkpoint detection ──────────────────────────────────────────────────────────────────
done = os.path.join(OUT_DIR, "cell-metadata.csv.gz")
if os.path.exists(done):
    n = len(pd.read_csv(done))
    print(f"[Skip] already exists ({n:,} cells); delete and rerun"); sys.exit(0)

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: expand GEM into a transcript-like format
# ══════════════════════════════════════════════════════════════════════════════
if os.path.exists(TX_OUT):
    print(f"[Skip Step1] reusing existing {os.path.basename(TX_OUT)}")
else:
    print("Step 1: read GEM and expand...")
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

    print(f"  GEM: {len(gem):,} rows  genes {gem.gene.nunique()}  "
          f"cell>0: {(gem.cell>0).sum():,}")

    # expand: each count becomes one synthetic-transcript row
    print("  expanding UMI (bins with MIDCounts > 1 replicate rows)...")
    gem_rep = gem.loc[gem.index.repeat(gem["count"].clip(1).astype(int))].copy()
    gem_rep = gem_rep.reset_index(drop=True)

    # coordinate conversion: bin -> um
    gem_rep["global_x"] = gem_rep["x_bin"] * BIN_UM
    gem_rep["global_y"] = gem_rep["y_bin"] * BIN_UM
    gem_rep["global_z"] = 0.0
    gem_rep["cell_id"]  = gem_rep["cell"].astype(int)
    gem_rep["barcode_id"] = np.arange(len(gem_rep))

    out = gem_rep[["barcode_id","global_x","global_y","global_z",
                   "gene","cell_id"]]
    print(f"  after expansion: {len(out):,} synthetic transcripts")
    print(f"  global_x: {out.global_x.min():.0f}~{out.global_x.max():.0f} µm")
    print(f"  assignment rate: {(out.cell_id>0).sum()/len(out)*100:.1f}%")

    out.to_csv(TX_OUT, index=False, compression="gzip")
    print(f"  saved -> {TX_OUT}")

# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: run ProSeg (experimental, using --merscope)
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: run ProSeg (experimental)...")
print("  warning: Stereo-seq is outside ProSeg official support; results are for reference only")

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
print(f"  log: {LOG_PATH}")

with open(LOG_PATH, "w") as log:
    result = subprocess.run(cmd, cwd=OUT_DIR, stdout=log, stderr=log)

if result.returncode == 0:
    n = len(pd.read_csv(done))
    print(f"\n✓ Done (experimental)!  cell count: {n:,}  (GT 6,229)")
    print(f"  → {OUT_DIR}/cell-metadata.csv.gz")
    print("  note for evaluation: Stereo-seq is unsupported; result quality is not guaranteed")
else:
    print(f"\n✘ exit code {result.returncode}")
    print(f"  see: {LOG_PATH}")
    print("  Stereo-seq is outside ProSeg official support; failure is expected")
    # write placeholder marker
    with open(os.path.join(OUT_DIR, "NOT_SUPPORTED.txt"), "w") as f:
        f.write("ProSeg does not officially support Stereo-seq (sequencing-based, not in situ)\n"
                "Format conversion was attempted but the run failed.\n"
                "This cell is marked N/A in the evaluation table.\n")
    print("  wrote NOT_SUPPORTED.txt; skip this dataset during evaluation")