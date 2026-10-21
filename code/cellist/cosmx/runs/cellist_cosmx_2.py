#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cellist @ cosmx_2 (FOV 48 + 261). Nucleus prior directly reuses the full-resolution mask built by ComSeg.
  python3 cellist_cosmx_2.py --dry          # Only print commands, confirm entry point
  python3 cellist_cosmx_2.py                # Formal run, min_transcripts=20
  python3 cellist_cosmx_2.py --mt 5         # Control group
"""
import argparse, sys; sys.path.insert(0, "yourpath")
from pathlib import Path
from seg_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true")
ap.add_argument("--mt", type=int, default=20, help="min_transcripts")
ap.add_argument("--fovs", type=int, nargs="+", default=COSMX_FOVS)
a = ap.parse_args()

for fov in a.fovs:
    out = Path("yourpath")
    out.mkdir(parents=True, exist_ok=True)
    log(f"=== Cellist @ cosmx_2 FOV {fov}  min_transcripts={a.mt} ===")
    nuc = COSMX_NUC.format(fov=fov)
    if not Path(nuc).exists():
        log(f"  ✗ Nucleus mask does not exist: {nuc}"); continue
    log(f"  Nucleus prior {nuc} (4256x4256, same FOV-local pixels as transcripts)")
    t = load_cosmx_fov(fov)
    spots = out / "spots.csv"
    t.rename(columns={"x_local_px":"x","y_local_px":"y","target":"gene"})[
        ["x","y","gene"]].to_csv(spots, index=False)
    log(f"  Transcript table -> {spots}")
    rc = run_tool("cellist", ["--spots", spots, "--nucleus-mask", nuc,
                              "--out", out, "--min-transcripts", a.mt,
                              "--pixel-size", COSMX_PX_UM], dry=a.dry)
    if rc == 0 and not a.dry:
        log("  After running, run again with --mt 5 as a control: on cosmx_1, assign_accuracy is only 0.007,")
        log("  but 1344 tx/cell is not sparse, indicating a second failure mode on the dense end, which cannot be explained by threshold.")