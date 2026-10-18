#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ComSeg Xenium6 colon launcher.

Fixed benchmark:
  dataset   = xenium_6 / colon
  ROI       = y[14052,20052) x[11241,17241)
  ROI size  = 6000 x 6000 native px
  pixel     = 0.2125 um/px
  GT        = 11790 cells

Actual ComSeg Xenium engine:
  /data/qiuyijia/comseg/comseg_xenium_4.py

This wrapper deliberately contains no segmentation implementation.
It only fixes the Xenium6 benchmark parameters and delegates to the
already-tested generic Xenium engine.
"""

from pathlib import Path
import argparse
import subprocess
import sys

ENGINE = Path("/data/qiuyijia/comseg/comseg_xenium_4.py")

DATA_ROOT = Path("/data/qiuyijia/dataset/xenium_colon")

# Fresh rerun. Do not overwrite the old 6/9 run.
OUT_DIR = Path(
    "/data/qiuyijia/comseg_xenium_colon_roi6000_rerun"
)

ROI_Y0 = 14052
ROI_X0 = 11241
ROI_SIZE = 6000

PIXEL_SIZE = 0.2125

GT_CELLS = 11790

MEAN_CELL_DIAMETER_UM = 8.943
MAX_CELL_RADIUS_UM = 12.520


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--stage",
        choices=[
            "prep",
            "run",
            "merge",
            "report",
            "all"
        ],
        default="all"
    )

    ap.add_argument(
        "--force-prep",
        action="store_true"
    )

    ap.add_argument(
        "--force-run",
        action="store_true"
    )

    ap.add_argument(
        "--dry",
        action="store_true"
    )

    args = ap.parse_args()

    if not ENGINE.exists():
        raise FileNotFoundError(
            f"missing Xenium engine: {ENGINE}"
        )

    print("=" * 90)
    print("ComSeg xenium_6 FINAL RERUN")
    print(
        f"ROI y[{ROI_Y0},{ROI_Y0+ROI_SIZE}) "
        f"x[{ROI_X0},{ROI_X0+ROI_SIZE})"
    )
    print(
        f"GT(cell-centroid standard)={GT_CELLS}"
    )
    print(
        f"diam={MEAN_CELL_DIAMETER_UM:.3f}um "
        f"radius={MAX_CELL_RADIUS_UM:.3f}um"
    )
    print("engine =", ENGINE)
    print("out    =", OUT_DIR)
    print("=" * 90)

    cmd = [
        sys.executable,
        "-u",
        str(ENGINE),

        "--dataset",
        "colon",

        "--stage",
        args.stage,

        "--data-root",
        str(DATA_ROOT),

        "--out-dir",
        str(OUT_DIR),

        "--roi-size",
        str(ROI_SIZE),

        "--roi-origin",
        str(ROI_Y0),
        str(ROI_X0),

        "--pixel-size",
        str(PIXEL_SIZE),

        "--prior-source",
        "cellist",

        "--mean-cell-diameter-um",
        str(MEAN_CELL_DIAMETER_UM),

        "--max-cell-radius-um",
        str(MAX_CELL_RADIUS_UM),
    ]

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
