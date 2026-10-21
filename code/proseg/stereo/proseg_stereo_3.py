#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_stereo_run.py -- ProSeg × Stereo-seq (experimental, multi-dataset)

* Reuses the exact adaptation logic from proseg_stereo_1_run.py for E16.5_E2S6:
  expand bin counts into synthetic single molecules + force-run with --merscope.
  All three datasets use identical processing so results are internally comparable.

Warning (again): ProSeg does not officially support Stereo-seq (sequencing vs
imaging data structures differ fundamentally). This script's expansion puts
multiple counts from the same bin at exactly the same coordinate, which is not a
true single-molecule distribution, so ProSeg's RNA diffusion model has no
physical meaning on such data. Results are for internal-consistency reference
only and must not be read as ProSeg's true representative performance on this
platform.

Usage:
  python proseg_stereo_run.py --dataset E14.5_E1S3
  python proseg_stereo_run.py --dataset E16.5_E2S7
  python proseg_stereo_run.py --dataset E16.5_E2S6 --force   # rerun existing
"""
import os, sys, gzip, argparse, subprocess
import pandas as pd
import numpy as np

BASE = "yourpath/proseg"
BIN_UM = 0.5

DATASETS = {
    "E14.5_E1S3": "yourpath/dataset/stereoseq/stereo-seq_data_all/"
                  "E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
    "E16.5_E2S6": "yourpath/dataset/stereoseq/stereo-seq_data_all/"
                  "E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
    "E16.5_E2S7": "yourpath/dataset/stereoseq/stereo-seq_data_all/"
                  "E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
}


def expand_gem(gem_path, tx_out):
    if os.path.exists(tx_out):
        print(f"  ⚡ reusing expanded transcripts: {tx_out}")
        return
    print("  read GEM and expand UMI (each count becomes one synthetic-transcript row)...")
    opener = gzip.open if gem_path.endswith(".gz") else open
    skip = 0
    with opener(gem_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break

    gem = pd.read_csv(gem_path, sep="\t", skiprows=skip)
    ren = {}
    for c in gem.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl == "x":
            ren[c] = "x_bin"
        elif cl == "y":
            ren[c] = "y_bin"
        elif cl in ("midcounts", "count"):
            ren[c] = "count"
        elif cl in ("cell", "cellid", "label"):
            ren[c] = "cell"
    gem = gem.rename(columns=ren)
    if "count" not in gem.columns:
        gem["count"] = 1
    if "cell" not in gem.columns:
        gem["cell"] = 0

    print(f"    GEM: {len(gem):,} rows  genes {gem.gene.nunique()}  "
          f"cell>0: {(gem.cell>0).sum():,}")

    gem_rep = gem.loc[gem.index.repeat(gem["count"].clip(1).astype(int))].copy()
    gem_rep = gem_rep.reset_index(drop=True)

    gem_rep["global_x"] = gem_rep["x_bin"] * BIN_UM
    gem_rep["global_y"] = gem_rep["y_bin"] * BIN_UM
    gem_rep["global_z"] = 0.0
    gem_rep["cell_id"] = gem_rep["cell"].astype(int)
    gem_rep["barcode_id"] = np.arange(len(gem_rep))

    out = gem_rep[["barcode_id", "global_x", "global_y", "global_z",
                   "gene", "cell_id"]]
    print(f"    after expansion: {len(out):,} synthetic transcripts")
    print(f"    global_x: {out.global_x.min():.0f}~{out.global_x.max():.0f} µm")
    print(f"    assignment rate: {(out.cell_id>0).sum()/len(out)*100:.1f}%")

    out.to_csv(tx_out, index=False, compression="gzip")
    print(f"    saved -> {tx_out}")


def run(dataset, force):
    gem_path = DATASETS[dataset]
    if not os.path.exists(gem_path):
        sys.exit(f"✘ file not found: {gem_path}")

    outdir = os.path.join(BASE, "output", dataset)
    os.makedirs(outdir, exist_ok=True)
    done = os.path.join(outdir, "cell-metadata.csv.gz")
    tx_out = os.path.join(outdir, "transcripts_expanded.csv.gz")
    log_path = os.path.join(outdir, "proseg.log")

    if os.path.exists(done) and not force:
        n = len(pd.read_csv(done))
        print(f"[Skip] {dataset} already exists ({n:,} cells); rerun with --force")
        return

    print(f"\n{'='*68}\nProSeg × {dataset} (experimental adaptation)\n{'='*68}")
    expand_gem(gem_path, tx_out)

    print("\nrun ProSeg (--merscope, experimental)...")
    cmd = [
        "proseg", "--merscope",
        "--nthreads", "16",
        "--output-cell-metadata", os.path.join(outdir, "cell-metadata.csv.gz"),
        "--output-transcript-metadata",
        os.path.join(outdir, "transcript-metadata.csv.gz"),
        "--output-cell-polygons",
        os.path.join(outdir, "cell-polygons.geojson.gz"),
        "--overwrite", tx_out,
    ]
    print("  " + " \\\n    ".join(cmd))
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, cwd=outdir, stdout=log, stderr=log)

    if result.returncode == 0:
        n = len(pd.read_csv(done))
        print(f"\n✓ Done (experimental)! cell count: {n:,}")
        print(f"  warning: results are for internal-consistency reference only, not ProSeg's true performance")
    else:
        print(f"\n✘ exit code {result.returncode}; see {log_path}")
        with open(os.path.join(outdir, "NOT_SUPPORTED.txt"), "w") as f:
            f.write("ProSeg does not officially support Stereo-seq; format conversion was attempted but the run failed.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args()
    run(cfg.dataset, cfg.force)


if __name__ == "__main__":
    main()