#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ucs_stereo_2.py —— UCS @ stereo_2 (E14.5_E1S3)

之前汇总表里 stereo_2 的 UCS 行是假的(有评估无产物)，这是重跑。

gene_map 内存 = H * W * n_genes * 4 bytes
  --top-genes N : 砍基因维，线性。top 2000 通常覆盖 85-95% 转录本，代价小
  --bin-px B    : 砍空间维，平方(H和W各除B)，但分辨率从 0.5um 降到 0.5*B um
                  小鼠胚胎细胞 6-10um，bin-px 2 后一个细胞只有 6-10 px 宽
  优先加大 --top-genes，尽量保住 --bin-px 1

⚠ 评估注意: nuclei_mask 从 GEM 的 cell 列建(CellBin 官方分割)，GT 也是同一列
   -> prior 与 GT 同源，转录本级指标会循环虚高。检测/定位/计数指标仍有效。

用法:
  python3 ucs_stereo_2.py --check                    # 先估体积
  python3 ucs_stereo_2.py --top-genes 2000
  python3 ucs_stereo_2.py --top-genes 1000 --bin-px 1
"""
import argparse, subprocess, sys
from pathlib import Path

DIR = "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all"
SLICE = "E14.5_E1S3"
GT_N = 4872
DATASET = "stereo_2"
BASE = "/data/qiuyijia/ucs/UCS/ucs_stereo_1_run.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只估 gene_map 体积，不跑")
    ap.add_argument("--top-genes", type=int, default=0,
                    help="只保留表达量最高的 N 个基因，0=全部")
    ap.add_argument("--bin-px", type=int, default=1,
                    help="空间下采样倍数，1=原始 0.5um/px")
    ap.add_argument("--nucleus-erode", type=int, default=2)
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()

    if not Path(a.base).exists():
        sys.exit(f"✗ 基础脚本不存在: {a.base}\n"
                 f"  ls /data/qiuyijia/ucs/UCS/ | grep stereo")

    gem = f"{DIR}/{SLICE}_Dorsal_Midbrain_GEM_CellBin.tsv.gz"
    tif = f"{DIR}/{SLICE}.tif"
    for p in (gem, tif):
        if not Path(p).exists():
            sys.exit(f"✗ 不存在: {p}")

    out = f"/data/qiuyijia/ucs_stereoseq_{SLICE}"
    cmd = [sys.executable, a.base,
           "--gem", gem, "--ssdna", tif, "--out-dir", out,
           "--nucleus-erode", str(a.nucleus_erode),
           "--bin-px", str(a.bin_px)]
    if a.top_genes:
        cmd += ["--top-genes", str(a.top_genes)]
    if a.check:
        cmd += ["--check"]

    print(f"=== UCS @ {DATASET} ({SLICE})  GT {GT_N:,} ===")
    if a.top_genes or a.bin_px > 1:
        print(f"    top_genes={a.top_genes or '全部'}  bin_px={a.bin_px} "
              f"(空间省 {a.bin_px**2}x)")
    print("$ " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0 and not a.check:
        print("\n若报 gene_map 太大:")
        print("  优先加大 --top-genes (线性省，代价小):")
        print("    --top-genes 2000  省 ~11x")
        print("    --top-genes 1000  省 ~22x")
        print("    --top-genes 500   省 ~44x")
        print("  再考虑 --bin-px 2 (平方省 4x，但分辨率降到 1.0um/px)")
    sys.exit(rc)


if __name__ == "__main__":
    main()