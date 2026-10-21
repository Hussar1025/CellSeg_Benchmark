#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UCS @ stereo_3 (E16.5_E2S7). Params copied from ucs_stereo_1_run.py, only the slice changes.
⚠ nuclei_mask comes from the GEM cell column, same source as GT -> assign_accuracy circularly inflated;
   only detection/localization/count metrics are valid in evaluation."""
import argparse, subprocess, sys; sys.path.insert(0, "yourpath/runs")
from seg_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true")
ap.add_argument("--top-genes", type=int, default=0)
a = ap.parse_args()
slc, gt_n = STEREO["stereo_3"]
src = "yourpath/ucs/UCS/ucs_stereo_1_run.py"
if not Path(src).exists():
    src = "yourpath/ucs/UCS/ucs_stereo_1_run.py"
cmd = [sys.executable, src, "--gem", stereo_gem(slc), "--ssdna", stereo_tif(slc),
       "--out-dir", f"yourpath/ucs_stereoseq_{slc}",
       "--nucleus-erode", "2", "--bin-px", "1"]
if a.top_genes: cmd += ["--top-genes", str(a.top_genes)]
log(f"=== UCS @ stereo_3 ({slc}), GT {gt_n:,} ===")
log("  $ " + " ".join(cmd))
if not a.dry:
    log("  first --check to estimate gene_map volume; if too large, add --top-genes 2000")
    subprocess.call(cmd + ["--check"])
    subprocess.call(cmd)
