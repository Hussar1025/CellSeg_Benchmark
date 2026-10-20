#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeneSegNet @ cosmx_2. Reuses cosmx_1 pretrained weights; inference only, no training."""
import argparse, sys; sys.path.insert(0, "yourpath/runs")
from pathlib import Path
from seg_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true")
ap.add_argument("--weights", default="", help="leave empty to auto-find .pth under pancreas_train")
ap.add_argument("--fovs", type=int, nargs="+", default=COSMX_FOVS)
a = ap.parse_args()

w = a.weights
if not w:
    cands = sorted(Path("yourpath/genesegnet_pancreas_train").rglob("*.pth")) + \
            sorted(Path("yourpath/genesegnet_cosmx_full").rglob("*.pth"))
    if not cands:
        log("✗ No .pth weights found. Where is the cosmx_1 model? Specify with --weights")
        log("  Try: find yourpath -name '*.pth' -not -path '*site-packages*' | head")
        sys.exit(1)
    w = str(max(cands, key=lambda p: p.stat().st_size))
    log(f"weights: {w}")

for fov in a.fovs:
    out = Path(f"yourpath/genesegnet_cosmx_2/fov{fov}")
    out.mkdir(parents=True, exist_ok=True)
    log(f"=== GeneSegNet @ cosmx_2 FOV {fov} (inference, no training) ===")
    t = load_cosmx_fov(fov)
    spots = out / "spots.csv"
    t.rename(columns={"x_local_px":"x","y_local_px":"y","target":"gene"})[
        ["x","y","gene"]].to_csv(spots, index=False)
    run_tool("genesegnet", ["--image_dir", COSMX_NUC.format(fov=fov),
                            "--spots", spots, "--pretrained_model", w,
                            "--savedir", out, "--no_train",
                            "--diameter", 65],   # cell ~7.8um / 0.12 = 65 px
             cwd=GSN_REPO, dry=a.dry)
