#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UCS @ stereo_3 (E16.5_E2S7)。参数照搬 ucs_stereo_1_run.py，只换切片。
⚠ nuclei_mask 来自 GEM 的 cell 列，与 GT 同源 -> assign_accuracy 循环虚高，
   评估时只有检测/定位/计数指标有效。"""
import argparse, subprocess, sys; sys.path.insert(0, "/data/qiuyijia/runs")
from seg_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true")
ap.add_argument("--top-genes", type=int, default=0)
a = ap.parse_args()
slc, gt_n = STEREO["stereo_3"]
src = "/data/qiuyijia/ucs/UCS/ucs_stereo_1_run.py"
if not Path(src).exists():
    src = "/data/qiuyijia/ucs/UCS/ucs_stereo_1_run.py"
cmd = [sys.executable, src, "--gem", stereo_gem(slc), "--ssdna", stereo_tif(slc),
       "--out-dir", f"/data/qiuyijia/ucs_stereoseq_{slc}",
       "--nucleus-erode", "2", "--bin-px", "1"]
if a.top_genes: cmd += ["--top-genes", str(a.top_genes)]
log(f"=== UCS @ stereo_3 ({slc}), GT {gt_n:,} ===")
log("  $ " + " ".join(cmd))
if not a.dry:
    log("  先 --check 估 gene_map 体积，太大就加 --top-genes 2000")
    subprocess.call(cmd + ["--check"])
    subprocess.call(cmd)
