#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comseg_xenium_2.py - ComSeg @ xenium_2 (breast)

Why rerun:
  Under cache_noprior, breast and lymph tx.csv are byte-identical
  (8,447,078 B, md5=ca1e2d346f6f4385) - the last breast run used lymph input.
  Every metric in the "ComSeg @ xenium_2" summary row is invalid, not just detection.

Why not do prep here:
  DATASETS[*]["root"] in comseg_xenium_4.py points to the raw Xenium directory;
  it reads transcripts.parquet, crops the ROI by --roi-size, splits tiles and filters by qv.
  Preprocessing the csv ourselves and feeding it in would duplicate work and
  reintroduce the "wrong input file" class of risk. This script only does two things:
    1. ensure comseg_xenium_4.py has a breast entry in DATASETS (add it if missing)
    2. call it with consistent parameters

ROI is 5000 px (Xenium 0.2125 um/px -> 1062.5 um per side):
  An independent check found roughly 5,557 GT cells and tx_per_cell 331.9 in this ROI,
  large enough and consistent with the other methods on xenium_2.
  tile 2000 px -> 9 tiles, about 200k transcripts per tile,
  8x smaller than mouse_brain (1.57M/tile, 11 hours).

Usage:
  python comseg_xenium_2.py --check          # check config only, do not run
  python comseg_xenium_2.py --stage prep     # crop ROI + split tiles
  python comseg_xenium_2.py --stage all
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("yourpath/comseg/comseg_xenium_4.py")
BREAST_ROOT = Path("yourpath/dataset/xenium_breast")
DATASET_KEY = "breast"

ROI_SIZE = 5000
TILE_PX = 2000
TILE_OVERLAP_PX = 150
MEAN_CELL_DIAMETER_UM = 12.0
MAX_CELL_RADIUS_UM = 20.0
QV_MIN = 20.0

ENTRY = f'    "{DATASET_KEY}": {{"root": "{BREAST_ROOT}", "tag": "{DATASET_KEY}"}},\n'


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ensure_dataset_entry(dry=False):
    """Insert the breast entry into DATASETS in comseg_xenium_4.py (idempotent)."""
    if not BASE.exists():
        sys.exit(f"base script not found: {BASE}")
    src = BASE.read_text()
    if re.search(rf'^\s*"{DATASET_KEY}"\s*:', src, re.M):
        log(f"DATASETS already has an entry for \"{DATASET_KEY}\"")
        return True
    m = re.search(r"^DATASETS\s*=\s*\{\s*$", src, re.M)
    if not m:
        log("could not find the opening DATASETS = {; add this line to DATASETS manually:")
        log("  " + ENTRY.strip())
        return False
    log(f"DATASETS has no \"{DATASET_KEY}\"; inserting:")
    log("  " + ENTRY.strip())
    if dry:
        log("  (--check, not written)")
        return False
    bak = BASE.with_suffix(f".py.bak_{time.strftime('%m%d_%H%M')}")
    shutil.copy2(BASE, bak)
    ins = m.end() + 1
    BASE.write_text(src[:ins] + ENTRY + src[ins:])
    log(f"  inserted, backup {bak.name}")
    return True


def check_inputs():
    ok = True
    log(f"data dir {BREAST_ROOT}")
    if not BREAST_ROOT.is_dir():
        log("  not found")
        return False
    # keep in sync with *_CANDIDATES in comseg_xenium_4.py
    need = {
        "transcripts": ["outs/transcripts.parquet", "transcripts.parquet"],
        "experiment metadata": ["outs/experiment.xenium", "experiment.xenium"],
        "nucleus boundaries": ["outs/nucleus_boundaries.parquet",
                   "nucleus_boundaries.parquet"],
        "morphology image": ["outs/morphology_focus/morphology_focus_0000.ome.tif",
                       "outs/morphology_focus.ome.tif",
                       "outs/morphology_mip.ome.tif",
                       "outs/morphology.ome.tif",
                       "morphology_focus/morphology_focus_0000.ome.tif",
                       "morphology.ome.tif"],
    }
    for name, cands in need.items():
        hit = next((c for c in cands if (BREAST_ROOT / c).exists()), None)
        if hit:
            sz = (BREAST_ROOT / hit).stat().st_size
            log(f"  [ok] {name:10s} {hit}  ({sz/1e9:.2f} GB)")
        else:
            log(f"  missing {name:10s}; tried: {cands}")
            ok = False
    gt = BREAST_ROOT / "cells.parquet"
    if gt.exists():
        try:
            import pandas as pd
            n = len(pd.read_parquet(gt, columns=["cell_id"]))
            log(f"  GT cells    cells.parquet  {n:,} (whole slide)")
        except Exception as e:
            log(f"  cells.parquet exists but read failed: {e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["prep", "run", "merge", "all", "report"])
    ap.add_argument("--check", action="store_true", help="check config only, do not run")
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE)
    ap.add_argument("--roi-origin", type=int, nargs=2, default=None,
                    metavar=("Y0", "X0"),
                    help="default: image center (compute_roi default)")
    ap.add_argument("--tile-px", type=int, default=TILE_PX)
    ap.add_argument("--tile-overlap-px", type=int, default=TILE_OVERLAP_PX)
    ap.add_argument("--mean-cell-diameter-um", type=float,
                    default=MEAN_CELL_DIAMETER_UM)
    ap.add_argument("--max-cell-radius-um", type=float,
                    default=MAX_CELL_RADIUS_UM)
    ap.add_argument("--qv-min", type=float, default=QV_MIN)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra args passed through to comseg_xenium_4.py")
    a = ap.parse_args()

    log("=" * 70)
    log("ComSeg @ xenium_2 (breast)")
    log("=" * 70)

    if not check_inputs():
        sys.exit("\ninputs incomplete; fill the gaps before running")

    log("")
    if not ensure_dataset_entry(dry=a.check) and not a.check:
        sys.exit("\ncould not insert the DATASETS entry automatically; add it manually as shown above")

    n_tiles = ((a.roi_size + a.tile_px - 1) // a.tile_px) ** 2
    out = f"yourpath/comseg_xenium_{DATASET_KEY}_roi{a.roi_size}"
    log("")
    log(f"ROI      {a.roi_size} px x 0.2125 um/px = "
        f"{a.roi_size*0.2125:.1f} um per side")
    log(f"tile     {a.tile_px} px, overlap {a.tile_overlap_px} px "
        f"-> about {n_tiles} tiles")
    log(f"cells    diameter {a.mean_cell_diameter_um} um, "
        f"max radius {a.max_cell_radius_um} um")
    log(f"qv_min   {a.qv_min}")
    log(f"output   {out}")
    log("reference: independent check at this ROI gives about 5,557 GT cells, tx_per_cell 331.9")

    cmd = [sys.executable, str(BASE),
           "--dataset", DATASET_KEY,
           "--stage", a.stage,
           "--roi-size", str(a.roi_size),
           "--tile-px", str(a.tile_px),
           "--tile-overlap-px", str(a.tile_overlap_px),
           "--mean-cell-diameter-um", str(a.mean_cell_diameter_um),
           "--max-cell-radius-um", str(a.max_cell_radius_um),
           "--qv-min", str(a.qv_min)]
    if a.roi_origin:
        cmd += ["--roi-origin", str(a.roi_origin[0]), str(a.roi_origin[1])]
    cmd += [x for x in a.extra if x != "--"]

    log("")
    log("$ " + " ".join(cmd))
    if a.check:
        log("\n--check, stopping here")
        log("to run for real: python comseg_xenium_2.py --stage prep")
        return

    log("")
    log("note: mouse_brain (PID 1554994) and cosmx fov48 (PID 1560823) are still running,")
    log("  using 87 GB / 47 GB of memory each. If the machine is tight, wait for mouse_brain")
    log("  tile_0024 to finish before starting this.")
    log("")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()