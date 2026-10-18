#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ComSeg Xenium pancreas-cancer launcher（精简版）。

瘦封装，不含分割逻辑，转交给引擎：
    /data/qiuyijia/comseg/comseg_xenium_4.py

与 colon launcher 的区别：读引擎源码后确认——
  * pixel_size  引擎从 experiment.xenium 自动读，这里不传（不覆盖）
  * ROI 原点    不传 --roi-origin 时引擎自动居中，这里默认居中
  * GT 细胞数    引擎 prep 阶段自算 gt_expected_cells，不是输入，故删除
只需固定 roi_size，并填 diameter/radius 两个形态参数。

前置条件（重要）：引擎的 DATASETS 字典必须已注册 "pancreas"：
    "pancreas": {"root": "/data/qiuyijia/dataset/xenium_pancreas_cancer",
                 "tag": "pancreas"},
否则 --dataset pancreas 会被 argparse 的 choices 拒绝。
"""

from pathlib import Path
import argparse
import subprocess
import sys

ENGINE = Path("/data/qiuyijia/comseg/comseg_xenium_4.py")
DATA_ROOT = Path("/data/qiuyijia/dataset/xenium_pancreas_cancer")
OUT_DIR = Path("/data/qiuyijia/comseg_xenium_pancreas_roi6000")

DATASET_NAME = "pancreas"          # 必须已在引擎 DATASETS 里注册
ROI_SIZE = 6000                    # 与 colon benchmark 对齐；居中

# prior 来源：platform = 用 Xenium 自带 nucleus_boundaries.parquet（无需额外依赖）
#             cellist  = 复用 cellist_xenium_pancreas_roi{roi} 的结果（须先跑过 cellist）
PRIOR_SOURCE = "platform"

# 形态参数：跑 inspect_pancreas.py 后把建议值填这里。
# 想先出个 baseline，也可直接用引擎默认 15 / 25。
MEAN_CELL_DIAMETER_UM = 7.7
MAX_CELL_RADIUS_UM = 9.4

# 可选：想跟别的方法对齐同一块区域时再指定，否则留 None = 自动居中
ROI_ORIGIN = None                  # 例如 (14052, 11241)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage",
                    choices=["prep", "run", "merge", "report", "all"],
                    default="all")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--force-run", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if not ENGINE.exists():
        raise FileNotFoundError(f"missing Xenium engine: {ENGINE}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"missing dataset root: {DATA_ROOT}")

    print("=" * 90)
    print("ComSeg xenium pancreas-cancer RUN")
    print(f"dataset = {DATASET_NAME}   roi-size = {ROI_SIZE} "
          f"({'centered' if ROI_ORIGIN is None else ROI_ORIGIN})")
    print(f"prior   = {PRIOR_SOURCE}")
    print(f"diam    = {MEAN_CELL_DIAMETER_UM} um   radius = {MAX_CELL_RADIUS_UM} um")
    print("pixel   = auto (read from experiment.xenium)")
    print("GT      = auto (engine computes gt_expected_cells)")
    print("engine  =", ENGINE)
    print("out     =", OUT_DIR)
    print("=" * 90)

    cmd = [
        sys.executable, "-u", str(ENGINE),
        "--dataset", DATASET_NAME,
        "--stage", args.stage,
        "--data-root", str(DATA_ROOT),
        "--out-dir", str(OUT_DIR),
        "--roi-size", str(ROI_SIZE),
        "--prior-source", PRIOR_SOURCE,
        "--mean-cell-diameter-um", str(MEAN_CELL_DIAMETER_UM),
        "--max-cell-radius-um", str(MAX_CELL_RADIUS_UM),
    ]

    # pixel_size 不传 -> 引擎自动读；ROI 原点仅在显式指定时才传
    if ROI_ORIGIN is not None:
        cmd += ["--roi-origin", str(ROI_ORIGIN[0]), str(ROI_ORIGIN[1])]

    if args.force_prep:
        cmd.append("--force-prep")
    if args.force_run:
        cmd.append("--force-run")

    print("COMMAND:")
    print(" ".join(cmd))

    if args.dry:
        print("DRY RUN DONE")
        return

    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()