#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cellist @ cosmx_2 (FOV 48 + 261)。核先验直接复用 ComSeg 建好的全分辨率掩膜。
  python3 cellist_cosmx_2.py --dry          # 只打印命令，确认入口
  python3 cellist_cosmx_2.py                # 正式跑，min_transcripts=20
  python3 cellist_cosmx_2.py --mt 5         # 对照组
"""
import argparse, sys; sys.path.insert(0, "/data/qiuyijia/runs")
from pathlib import Path
from seg_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true")
ap.add_argument("--mt", type=int, default=20, help="min_transcripts")
ap.add_argument("--fovs", type=int, nargs="+", default=COSMX_FOVS)
a = ap.parse_args()

for fov in a.fovs:
    out = Path(f"/data/qiuyijia/cellist_cosmx_2/fov{fov}_mt{a.mt}")
    out.mkdir(parents=True, exist_ok=True)
    log(f"=== Cellist @ cosmx_2 FOV {fov}  min_transcripts={a.mt} ===")
    nuc = COSMX_NUC.format(fov=fov)
    if not Path(nuc).exists():
        log(f"  ✗ 核掩膜不存在: {nuc}"); continue
    log(f"  核先验 {nuc} (4256x4256, 与转录本同为 FOV-local 像素)")
    t = load_cosmx_fov(fov)
    spots = out / "spots.csv"
    t.rename(columns={"x_local_px":"x","y_local_px":"y","target":"gene"})[
        ["x","y","gene"]].to_csv(spots, index=False)
    log(f"  转录本表 -> {spots}")
    rc = run_tool("cellist", ["--spots", spots, "--nucleus-mask", nuc,
                              "--out", out, "--min-transcripts", a.mt,
                              "--pixel-size", COSMX_PX_UM], dry=a.dry)
    if rc == 0 and not a.dry:
        log("  跑完后用 --mt 5 再跑一次做对照：cosmx_1 上 assign_accuracy 只有 0.007，")
        log("  但 1344 tx/cell 并不稀疏，说明密集端还有第二种失效，不是阈值能解释的。")
