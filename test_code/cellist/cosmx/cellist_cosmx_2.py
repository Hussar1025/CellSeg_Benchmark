#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellist_cosmx_2.py —— Cellist @ cosmx_2 (FOV 48 + 261)

cosmx_2 = 淋巴结 FOV 48 与 261 合计，GT 9794 = 4898 + 4896。
核先验直接复用 ComSeg 已建好的全分辨率掩膜(4256x4256 uint32)，
无需任何 ROI 推导 —— UCS 源码确认 CosMx 全流程在 FOV-local 像素下进行，
origin (0,0)，4256 px x 0.12 um/px = 510.72 um。

建议跑两遍: --mt 20 (默认) 和 --mt 5 (对照)。
Cellist 在 cosmx_1 上 assign_accuracy 只有 0.007，而 cosmx_1 是
1344 tx/cell 并不稀疏 —— 说明密集端还有第二种失效，不是阈值能解释的。
这组对照正好回答"20 是不是 Cellist 唯一的瓶颈"。

用法:
  python3 cellist_cosmx_2.py --dry      # 只打印命令，确认入口
  python3 cellist_cosmx_2.py
  python3 cellist_cosmx_2.py --mt 5
"""
import argparse, subprocess, sys, time
from pathlib import Path
import pandas as pd

PY = "/data/qiuyijia/anaconda3/envs/cellist/bin/python"
TX = "/data/qiuyijia/dataset/cosmx_lymph_node/_inspect_cache/tx_fov_48_261.parquet"
NUC = "/data/qiuyijia/comseg_cosmx_lymph_node/fov{fov}/nuclei_mask.tif"
FOVS = [48, 261]
PX_UM = 0.12

ENTRYPOINTS = [
    [PY, "-m", "Cellist.segment"],
    [PY, "-m", "Cellist"],
    ["/data/qiuyijia/anaconda3/envs/cellist/bin/Cellist"],
    [PY, "/data/qiuyijia/cellist/Cellist/Cellist/segment.py"],
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(args, dry):
    tried = []
    for base in ENTRYPOINTS:
        exe = base[0]
        if exe.endswith("python") and not Path(exe).exists():
            tried.append((base, f"解释器不存在 {exe}"))
            continue
        bad = next((p for p in base[1:] if p.endswith(".py")
                    and not Path(p).exists()), None)
        if bad:
            tried.append((base, f"脚本不存在 {bad}"))
            continue
        cmd = base + [str(x) for x in args]
        log("  $ " + " ".join(cmd))
        if dry:
            return 0
        rc = subprocess.call(cmd)
        if rc == 0:
            return 0
        tried.append((base, f"退出码 {rc}"))
    print("\nCellist 所有入口都失败:")
    for b, e in tried:
        print(f"  {' '.join(b)}\n      -> {e}")
    print("\n把真实入口填进本脚本的 ENTRYPOINTS")
    print("查一下: ls /data/qiuyijia/cellist/Cellist/Cellist/")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--mt", type=int, default=20, help="min_transcripts")
    ap.add_argument("--fovs", type=int, nargs="+", default=FOVS)
    a = ap.parse_args()

    if not Path(TX).exists():
        sys.exit(f"✗ 转录本缓存不存在: {TX}")

    tx = pd.read_parquet(TX)
    fc = "fov" if "fov" in tx.columns else "FOV"
    log(f"转录本缓存 {len(tx):,} 行, 列: {list(tx.columns)[:8]}")

    for fov in a.fovs:
        out = Path(f"/data/qiuyijia/cellist_cosmx_2/fov{fov}_mt{a.mt}")
        out.mkdir(parents=True, exist_ok=True)
        nuc = NUC.format(fov=fov)
        log(f"=== FOV {fov}  min_transcripts={a.mt} ===")
        if not Path(nuc).exists():
            log(f"  ✗ 核掩膜不存在: {nuc}")
            continue
        log(f"  核先验 {nuc}")

        t = tx[pd.to_numeric(tx[fc], errors="coerce") == fov]
        log(f"  {len(t):,} 转录本, {t['target'].nunique():,} 基因")
        spots = out / "spots.csv"
        t.rename(columns={"x_local_px": "x", "y_local_px": "y",
                          "target": "gene"})[["x", "y", "gene"]] \
            .to_csv(spots, index=False)
        log(f"  -> {spots}")

        run(["--spots", spots, "--nucleus-mask", nuc, "--out", out,
             "--min-transcripts", a.mt, "--pixel-size", PX_UM], a.dry)

    if not a.dry:
        log("跑完后用 --mt 5 再跑一次做对照")


if __name__ == "__main__":
    main()