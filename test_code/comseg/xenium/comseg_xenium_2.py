#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comseg_xenium_2.py — ComSeg @ xenium_2 (breast)

为什么重跑:
  cache_noprior 下 breast 与 lymph 的 tx.csv 逐字节相同
  (8,447,078 B, md5=ca1e2d346f6f4385) —— 上一次 breast 跑的是 lymph 的输入。
  汇总表里 ComSeg @ xenium_2 那行全部指标作废，不只是 detection。

为什么不自己做 prep:
  comseg_xenium_4.py 的 DATASETS[*]["root"] 指向原始 Xenium 目录，
  它自己读 transcripts.parquet、按 --roi-size 裁 ROI、切 tile、qv 过滤。
  自己预处理 csv 再喂进去是重复劳动，而且会重新引入
  "拿错输入文件" 这一类风险。本脚本只做两件事:
    1. 确保 comseg_xenium_4.py 的 DATASETS 里有 breast 条目(没有就加)
    2. 用统一参数调它

ROI 用 5000 px (Xenium 0.2125 um/px -> 1062.5 um 见方):
  独立校验显示该尺度下 ROI 内 GT 约 5,557 细胞、tx_per_cell 331.9，
  规模足够且与其他方法在 xenium_2 上的口径一致。
  tile 2000 px -> 9 个 tile，每 tile 约 20 万转录本，
  比 mouse_brain(157 万/tile、11 小时) 小 8 倍。

用法:
  python comseg_xenium_2.py --check          # 只查配置，不跑
  python comseg_xenium_2.py --stage prep     # 裁 ROI + 切 tile
  python comseg_xenium_2.py --stage all
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/data/qiuyijia/comseg/comseg_xenium_4.py")
BREAST_ROOT = Path("/data/qiuyijia/dataset/xenium_breast")
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
    """往 comseg_xenium_4.py 的 DATASETS 里插入 breast 条目（幂等）。"""
    if not BASE.exists():
        sys.exit(f"✗ 基础脚本不存在: {BASE}")
    src = BASE.read_text()
    if re.search(rf'^\s*"{DATASET_KEY}"\s*:', src, re.M):
        log(f"DATASETS 里已有 \"{DATASET_KEY}\" 条目")
        return True
    m = re.search(r"^DATASETS\s*=\s*\{\s*$", src, re.M)
    if not m:
        log("✗ 找不到 DATASETS = { 开头，请手动加这一行到 DATASETS 里:")
        log("  " + ENTRY.strip())
        return False
    log(f"DATASETS 里没有 \"{DATASET_KEY}\"，准备插入:")
    log("  " + ENTRY.strip())
    if dry:
        log("  (--check，不实际写入)")
        return False
    bak = BASE.with_suffix(f".py.bak_{time.strftime('%m%d_%H%M')}")
    shutil.copy2(BASE, bak)
    ins = m.end() + 1
    BASE.write_text(src[:ins] + ENTRY + src[ins:])
    log(f"  ✓ 已插入，备份 {bak.name}")
    return True


def check_inputs():
    ok = True
    log(f"数据目录 {BREAST_ROOT}")
    if not BREAST_ROOT.is_dir():
        log("  ✗ 不存在")
        return False
    # 与 comseg_xenium_4.py 的 *_CANDIDATES 保持一致
    need = {
        "转录本": ["outs/transcripts.parquet", "transcripts.parquet"],
        "实验元信息": ["outs/experiment.xenium", "experiment.xenium"],
        "核边界": ["outs/nucleus_boundaries.parquet",
                   "nucleus_boundaries.parquet"],
        "形态学影像": ["outs/morphology_focus/morphology_focus_0000.ome.tif",
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
            log(f"  ✓ {name:10s} {hit}  ({sz/1e9:.2f} GB)")
        else:
            log(f"  ✗ {name:10s} 未找到，试过: {cands}")
            ok = False
    gt = BREAST_ROOT / "cells.parquet"
    if gt.exists():
        try:
            import pandas as pd
            n = len(pd.read_parquet(gt, columns=["cell_id"]))
            log(f"  ✓ GT 细胞    cells.parquet  {n:,} 个(全片)")
        except Exception as e:
            log(f"  · cells.parquet 存在但读取失败: {e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["prep", "run", "merge", "all", "report"])
    ap.add_argument("--check", action="store_true", help="只查配置不跑")
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE)
    ap.add_argument("--roi-origin", type=int, nargs=2, default=None,
                    metavar=("Y0", "X0"),
                    help="不给则取图像中心（compute_roi 的默认行为）")
    ap.add_argument("--tile-px", type=int, default=TILE_PX)
    ap.add_argument("--tile-overlap-px", type=int, default=TILE_OVERLAP_PX)
    ap.add_argument("--mean-cell-diameter-um", type=float,
                    default=MEAN_CELL_DIAMETER_UM)
    ap.add_argument("--max-cell-radius-um", type=float,
                    default=MAX_CELL_RADIUS_UM)
    ap.add_argument("--qv-min", type=float, default=QV_MIN)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="透传给 comseg_xenium_4.py 的其余参数")
    a = ap.parse_args()

    log("=" * 70)
    log("ComSeg @ xenium_2 (breast)")
    log("=" * 70)

    if not check_inputs():
        sys.exit("\n✗ 输入不完整，先补齐再跑")

    log("")
    if not ensure_dataset_entry(dry=a.check) and not a.check:
        sys.exit("\n✗ 无法自动插入 DATASETS 条目，按上面提示手动加")

    n_tiles = ((a.roi_size + a.tile_px - 1) // a.tile_px) ** 2
    out = f"/data/qiuyijia/comseg_xenium_{DATASET_KEY}_roi{a.roi_size}"
    log("")
    log(f"ROI      {a.roi_size} px x 0.2125 um/px = "
        f"{a.roi_size*0.2125:.1f} um 见方")
    log(f"tile     {a.tile_px} px, overlap {a.tile_overlap_px} px "
        f"-> 约 {n_tiles} 个")
    log(f"细胞     直径 {a.mean_cell_diameter_um} um, "
        f"最大半径 {a.max_cell_radius_um} um")
    log(f"qv_min   {a.qv_min}")
    log(f"输出     {out}")
    log("参照: 该 ROI 尺度下独立校验得 GT 约 5,557 细胞, tx_per_cell 331.9")

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
        log("\n--check，到此为止")
        log("正式跑: python comseg_xenium_2.py --stage prep")
        return

    log("")
    log("★ 提醒: mouse_brain(PID 1554994) 与 cosmx fov48(PID 1560823) 仍在跑，")
    log("  各占 87 GB / 47 GB 内存。若机器吃紧，等 mouse_brain 的")
    log("  tile_0024 完成后再开这个。")
    log("")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()