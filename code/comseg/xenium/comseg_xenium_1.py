#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ComSeg Xenium pancreas-cancer launcher (slim version).

Thin wrapper with no segmentation logic; it delegates to the engine:
    yourpath/comseg/comseg_xenium_4.py

Differences from the colon launcher, confirmed by reading the engine source:
  * pixel_size  the engine reads it from experiment.xenium, so it is not passed here (not overridden)
  * ROI origin  the engine centers automatically when --roi-origin is omitted; centered by default here
  * GT cell count  the engine computes gt_expected_cells during prep, so it is not an input and was removed
Only roi_size is fixed; the two shape parameters diameter/radius are set here.

Prerequisite (important): "pancreas" must already be registered in the engine DATASETS dict:
    "pancreas": {"root": "yourpath/dataset/xenium_pancreas_cancer",
                 "tag": "pancreas"},
otherwise argparse choices will reject --dataset pancreas.
"""

from pathlib import Path
import argparse
import subprocess
import sys

ENGINE = Path("yourpath/comseg/comseg_xenium_4.py")
DATA_ROOT = Path("yourpath/dataset/xenium_pancreas_cancer")
OUT_DIR = Path("yourpath/comseg_xenium_pancreas_roi6000")

DATASET_NAME = "pancreas"          # must already be registered in engine DATASETS
ROI_SIZE = 6000                    # aligned with the colon benchmark; centered

# prior source: platform = use the Xenium nucleus_boundaries.parquet (no extra dependency)
#               cellist  = reuse cellist_xenium_pancreas_roi{roi} output (run cellist first)
PRIOR_SOURCE = "platform"

# Shape parameters: fill in the suggested values after running inspect_pancreas.py.
# For a quick baseline, the engine defaults 15 / 25 also work.
MEAN_CELL_DIAMETER_UM = 7.7
MAX_CELL_RADIUS_UM = 9.4

# Optional: set to align the same region with other methods; None = auto-centered
ROI_ORIGIN = None                  # e.g. (14052, 11241)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",
                    choices=["prep", "run", "merge", "report", "all"],
                    default="all")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--force-run", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if not ENGINE.exists():
        raise FileNotFoundError(f"missing Xenium engine: {ENGINE}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"missing dataset root: {DATA_ROOT}")

    print("=" * 90)
    print("ComSeg xenium pancreas-cancer RUN")
    print(f"dataset = {DATASET_NAME}   roi-size = {ROI_SIZE} "
          f"({'centered' if ROI_ORIGIN is None else ROI_ORIGIN})")
    print(f"prior   = {PRIOR_SOURCE}")
    print(f"diam    = {MEAN_CELL_DIAMETER_UM} um   radius = {MAX_CELL_RADIUS_UM} um")
    print("pixel   = auto (read from experiment.xenium)")
    print("GT      = auto (engine computes gt_expected_cells)")
    print("engine  =", ENGINE)
    print("out     =", OUT_DIR)
    print("=" * 90)

    cmd = [
        sys.executable, "-u", str(ENGINE),
        "--dataset", DATASET_NAME,
        "--stage", args.stage,
        "--data-root", str(DATA_ROOT),
        "--out-dir", str(OUT_DIR),
        "--roi-size", str(ROI_SIZE),
        "--prior-source", PRIOR_SOURCE,
        "--mean-cell-diameter-um", str(MEAN_CELL_DIAMETER_UM),
        "--max-cell-radius-um", str(MAX_CELL_RADIUS_UM),
    ]

    # pixel_size not passed -> engine reads it; ROI origin only passed when explicitly set
    if ROI_ORIGIN is not None:
        cmd += ["--roi-origin", str(ROI_ORIGIN[0]), str(ROI_ORIGIN[1])]

    if args.force_prep:
        cmd.append("--force-prep")
    if args.force_run:
        cmd.append("--force-run")

    print("COMMAND:")
    print(" ".join(cmd))

    if args.dry:
        print("DRY RUN DONE")
        return

    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
