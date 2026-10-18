#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeneSegNet @ cosmx_2。复用 cosmx_1 的预训练权重，只推理不训练。"""
import argparse, sys; sys.path.insert(0, "/data/qiuyijia/runs")
from pathlib import Path
from seg_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--dry", action="store_true")
ap.add_argument("--weights", default="", help="留空则自动在 pancreas_train 下找 .pth")
ap.add_argument("--fovs", type=int, nargs="+", default=COSMX_FOVS)
a = ap.parse_args()

w = a.weights
if not w:
    cands = sorted(Path("/data/qiuyijia/genesegnet_pancreas_train").rglob("*.pth")) + \
            sorted(Path("/data/qiuyijia/genesegnet_cosmx_full").rglob("*.pth"))
    if not cands:
        log("✗ 没找到 .pth 权重。cosmx_1 的模型在哪？用 --weights 指定")
        log("  可先跑: find /data/qiuyijia -name '*.pth' -not -path '*site-packages*' | head")
        sys.exit(1)
    w = str(max(cands, key=lambda p: p.stat().st_size))
    log(f"权重: {w}")

for fov in a.fovs:
    out = Path(f"/data/qiuyijia/genesegnet_cosmx_2/fov{fov}")
    out.mkdir(parents=True, exist_ok=True)
    log(f"=== GeneSegNet @ cosmx_2 FOV {fov} (推理, 不训练) ===")
    t = load_cosmx_fov(fov)
    spots = out / "spots.csv"
    t.rename(columns={"x_local_px":"x","y_local_px":"y","target":"gene"})[
        ["x","y","gene"]].to_csv(spots, index=False)
    run_tool("genesegnet", ["--image_dir", COSMX_NUC.format(fov=fov),
                            "--spots", spots, "--pretrained_model", w,
                            "--savedir", out, "--no_train",
                            "--diameter", 65],   # 细胞 ~7.8um / 0.12 = 65 px
             cwd=GSN_REPO, dry=a.dry)
