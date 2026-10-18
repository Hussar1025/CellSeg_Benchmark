#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_stereo_run.py —— ProSeg × Stereo-seq（实验性，多数据集版）

★ 复用 proseg_stereo_1_run.py 当初对 E16.5_E2S6 的确切适配逻辑：
  bin 计数展开成合成单分子 + --merscope 参数硬跑。
  三份数据集用完全一致的处理方式，结果才有内部可比性。

⚠ 再次强调：ProSeg 官方不支持 Stereo-seq（测序法 vs 成像法数据结构
  本质不同），本脚本的展开方式会让同一 bin 的多条计数落在完全相同的
  坐标点上，这不是真实单分子分布，ProSeg 的 RNA 扩散模型在这种数据上
  没有物理意义。结果仅供内部一致性参考，不能当作 ProSeg 在该平台上
  的真实代表性能。

用法：
  python proseg_stereo_run.py --dataset E14.5_E1S3
  python proseg_stereo_run.py --dataset E16.5_E2S7
  python proseg_stereo_run.py --dataset E16.5_E2S6 --force   # 重跑已有的
"""
import os, sys, gzip, argparse, subprocess
import pandas as pd
import numpy as np

BASE = "/data/qiuyijia/proseg"
BIN_UM = 0.5

DATASETS = {
    "E14.5_E1S3": "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all/"
                  "E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
    "E16.5_E2S6": "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all/"
                  "E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
    "E16.5_E2S7": "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all/"
                  "E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
}


def expand_gem(gem_path, tx_out):
    if os.path.exists(tx_out):
        print(f"  ⚡ 复用已展开的转录本: {tx_out}")
        return
    print("  读取 GEM 并展开 UMI（每个 count 变成一行合成转录本）...")
    opener = gzip.open if gem_path.endswith(".gz") else open
    skip = 0
    with opener(gem_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break

    gem = pd.read_csv(gem_path, sep="\t", skiprows=skip)
    ren = {}
    for c in gem.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl == "x":
            ren[c] = "x_bin"
        elif cl == "y":
            ren[c] = "y_bin"
        elif cl in ("midcounts", "count"):
            ren[c] = "count"
        elif cl in ("cell", "cellid", "label"):
            ren[c] = "cell"
    gem = gem.rename(columns=ren)
    if "count" not in gem.columns:
        gem["count"] = 1
    if "cell" not in gem.columns:
        gem["cell"] = 0

    print(f"    GEM: {len(gem):,} 行  基因 {gem.gene.nunique()}  "
          f"cell>0: {(gem.cell>0).sum():,}")

    gem_rep = gem.loc[gem.index.repeat(gem["count"].clip(1).astype(int))].copy()
    gem_rep = gem_rep.reset_index(drop=True)

    gem_rep["global_x"] = gem_rep["x_bin"] * BIN_UM
    gem_rep["global_y"] = gem_rep["y_bin"] * BIN_UM
    gem_rep["global_z"] = 0.0
    gem_rep["cell_id"] = gem_rep["cell"].astype(int)
    gem_rep["barcode_id"] = np.arange(len(gem_rep))

    out = gem_rep[["barcode_id", "global_x", "global_y", "global_z",
                   "gene", "cell_id"]]
    print(f"    展开后: {len(out):,} 条合成转录本")
    print(f"    global_x: {out.global_x.min():.0f}~{out.global_x.max():.0f} µm")
    print(f"    分配率: {(out.cell_id>0).sum()/len(out)*100:.1f}%")

    out.to_csv(tx_out, index=False, compression="gzip")
    print(f"    已保存 → {tx_out}")


def run(dataset, force):
    gem_path = DATASETS[dataset]
    if not os.path.exists(gem_path):
        sys.exit(f"✘ 文件不存在: {gem_path}")

    outdir = os.path.join(BASE, "output", dataset)
    os.makedirs(outdir, exist_ok=True)
    done = os.path.join(outdir, "cell-metadata.csv.gz")
    tx_out = os.path.join(outdir, "transcripts_expanded.csv.gz")
    log_path = os.path.join(outdir, "proseg.log")

    if os.path.exists(done) and not force:
        n = len(pd.read_csv(done))
        print(f"[跳过] {dataset} 已存在（{n:,} 细胞），--force 重跑")
        return

    print(f"\n{'='*68}\nProSeg × {dataset}（实验性适配）\n{'='*68}")
    expand_gem(gem_path, tx_out)

    print("\n运行 ProSeg（--merscope，实验性）...")
    cmd = [
        "proseg", "--merscope",
        "--nthreads", "16",
        "--output-cell-metadata", os.path.join(outdir, "cell-metadata.csv.gz"),
        "--output-transcript-metadata",
        os.path.join(outdir, "transcript-metadata.csv.gz"),
        "--output-cell-polygons",
        os.path.join(outdir, "cell-polygons.geojson.gz"),
        "--overwrite", tx_out,
    ]
    print("  " + " \\\n    ".join(cmd))
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, cwd=outdir, stdout=log, stderr=log)

    if result.returncode == 0:
        n = len(pd.read_csv(done))
        print(f"\n✓ 完成（实验性）！细胞数: {n:,}")
        print(f"  ⚠ 结果仅供内部一致性参考，不代表 ProSeg 真实性能")
    else:
        print(f"\n✘ 退出码 {result.returncode}，查看 {log_path}")
        with open(os.path.join(outdir, "NOT_SUPPORTED.txt"), "w") as f:
            f.write("ProSeg 不官方支持 Stereo-seq，已尝试格式转换后运行但失败。\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args()
    run(cfg.dataset, cfg.force)


if __name__ == "__main__":
    main()