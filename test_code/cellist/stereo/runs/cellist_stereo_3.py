#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cellist @ stereo_3 (E16.5_E2S7)。旧产物先挪走：它是 E16.5_E2S6 的坐标范围。"""
import argparse, sys; sys.path.insert(0, "/data/qiuyijia/runs")
from pathlib import Path
from seg_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true")
ap.add_argument("--ds", default="stereo_3"); a = ap.parse_args()
slc, gt_n = STEREO[a.ds]
out = Path(f"/data/qiuyijia/cellist_{a.ds}"); out.mkdir(parents=True, exist_ok=True)
log(f"=== Cellist @ {a.ds} ({slc}), GT {gt_n:,} 细胞 ===")
for old in Path(f"/data/qiuyijia/cellist/stereo_{slc}").glob("*"):
    if old.suffix in (".tif", ".h5ad", ".npz"): move_aside(old, "E16.5_E2S6")
gem = load_stereo_gem(slc)
spots = out / "spots.csv"
gem.loc[gem.index.repeat(gem["count"]), ["x","y","gene"]].to_csv(spots, index=False)
log(f"  转录本(按 MIDCount 展开) -> {spots}")
rc = run_tool("cellist", ["--spots", spots, "--image", stereo_tif(slc),
                          "--out", out, "--pixel-size", 0.5,
                          "--min-transcripts", 20], dry=a.dry)
if rc == 0 and not a.dry:
    log("  ★ 跑完立刻验坐标范围，别等到评估阶段：")
    log(f"     期望 x[{gem.x.min()},{gem.x.max()}] y[{gem.y.min()},{gem.y.max()}]")
