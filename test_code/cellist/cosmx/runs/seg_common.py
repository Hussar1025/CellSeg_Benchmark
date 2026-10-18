#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补跑任务的共享底座。所有路径来自 layout.txt 与 UCS 源码，不做探测。"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

PY_BASE    = "/data/qiuyijia/anaconda3/envs"
PY_CELLIST = f"{PY_BASE}/cellist/bin/python"
PY_UCS     = f"{PY_BASE}/ucs/bin/python"
UCS_REPO   = "/data/qiuyijia/ucs/UCS"
GSN_REPO   = "/data/qiuyijia/GeneSegNet"

COSMX_ROOT = "/data/qiuyijia/dataset/cosmx_lymph_node"
COSMX_TX   = f"{COSMX_ROOT}/_inspect_cache/tx_fov_48_261.parquet"
COSMX_META = f"{COSMX_ROOT}/flat_files/S0_metadata_file.csv.gz"
COSMX_POLY = f"{COSMX_ROOT}/flat_files/S0-polygons.csv.gz"
COSMX_NUC  = "/data/qiuyijia/comseg_cosmx_lymph_node/fov{fov}/nuclei_mask.tif"
COSMX_UCS  = "/data/qiuyijia/ucs_cosmx_lymph_node/fov{fov}"
COSMX_FOVS = [48, 261]
COSMX_PX_UM, COSMX_FOV_PX = 0.12, 4256

STEREO_DIR = "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all"
STEREO = {   # 数据集编号 -> (切片名, GT 细胞数)。编号对应关系由 eval_results 反查确定
    "stereo_1": ("E16.5_E2S6", 6229),
    "stereo_2": ("E14.5_E1S3", 4872),
    "stereo_3": ("E16.5_E2S7", 7957),
}
def stereo_gem(slc): return f"{STEREO_DIR}/{slc}_Dorsal_Midbrain_GEM_CellBin.tsv.gz"
def stereo_tif(slc): return f"{STEREO_DIR}/{slc}.tif"

# 入口候选。第一次跑用 --dry 确认，命中后把正确的一条挪到列表首位即可。
ENTRYPOINTS = {
  "cellist": [[PY_CELLIST, "-m", "Cellist.segment"],
              [PY_CELLIST, "-m", "Cellist"],
              ["/data/qiuyijia/anaconda3/envs/cellist/bin/Cellist"],
              [PY_CELLIST, "/data/qiuyijia/cellist/Cellist/Cellist/segment.py"]],
  "genesegnet": [[sys.executable, f"{GSN_REPO}/Inference/slidingwindows_gradient.py"],
                 [sys.executable, f"{GSN_REPO}/GeneSegNet/GeneSeg_test.py"]],
  "ucs": [[PY_UCS, f"{UCS_REPO}/run.py"]],
}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def run_tool(kind, args, cwd=None, dry=False):
    """按候选入口依次尝试；全失败时把每条命令连报错一起打出来。"""
    tried = []
    for base in ENTRYPOINTS[kind]:
        exe = base[0]
        if exe.endswith("python") and not Path(exe).exists():
            tried.append((base, f"解释器不存在: {exe}")); continue
        for p in base[1:]:
            if p.endswith(".py") and not Path(p).exists():
                tried.append((base, f"脚本不存在: {p}")); break
        else:
            cmd = base + [str(a) for a in args]
            log("  $ " + " ".join(cmd))
            if dry:
                return 0
            rc = subprocess.call(cmd, cwd=cwd)
            if rc == 0:
                return 0
            tried.append((base, f"退出码 {rc}"))
    print(f"\n{kind} 的所有入口都失败：")
    for b, e in tried:
        print(f"    {' '.join(b)}\n        -> {e}")
    print(f"  把真实入口填进 seg_common.py 的 ENTRYPOINTS['{kind}']\n")
    return 1

def move_aside(path, tag):
    """旧产物挪走不删——混在一起比丢了更难查。"""
    p = Path(path)
    if p.exists():
        dst = p.with_name(p.name + f".WRONG_{tag}.bak")
        shutil.move(str(p), str(dst)); log(f"  旧产物挪至 {dst.name}")

def sanity_check(cells_xy, n_expect, gt_xy=None, name=""):
    """跑完立刻验：细胞数量级 + 坐标范围是否与 GT 重叠。
    stereo_2/3 的旧产物就是坐标范围对不上（E16.5_E2S6 的框），这一步能当场拦住。"""
    n = len(cells_xy)
    ok = True
    log(f"  [验] {name}: {n:,} 个细胞, 期望 ~{n_expect:,}, ratio {n/max(n_expect,1):.3f}")
    if not (0.2 <= n/max(n_expect,1) <= 3.0):
        log("  [验] ✗ 细胞数偏离期望超过 3 倍"); ok = False
    if len(cells_xy):
        x, y = np.asarray(cells_xy)[:,0], np.asarray(cells_xy)[:,1]
        log(f"  [验] x[{x.min():.1f},{x.max():.1f}] y[{y.min():.1f},{y.max():.1f}]")
        if gt_xy is not None and len(gt_xy):
            g = np.asarray(gt_xy)
            inter = (min(x.max(),g[:,0].max())-max(x.min(),g[:,0].min())) * \
                    (min(y.max(),g[:,1].max())-max(y.min(),g[:,1].min()))
            union = (x.max()-x.min())*(y.max()-y.min())
            r = inter/max(union,1e-9)
            log(f"  [验] 与 GT 包围盒重叠 {r*100:.1f}%")
            if r < 0.5:
                log("  [验] ✗ 坐标范围与 GT 基本不重叠，产物很可能来自别的切片"); ok = False
    log("  [验] " + ("通过" if ok else "★未通过，先查清再评估"))
    return ok

def load_stereo_gem(slc, cols=("x","y","geneID","cell","MIDCount")):
    import gzip, pandas as pd
    p = stereo_gem(slc)
    skip = 0
    with gzip.open(p, "rt") as fh:
        for line in fh:
            if line.startswith("#"): skip += 1
            else: break
    df = pd.read_csv(p, sep="\t", skiprows=skip)
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("geneid","gene"): ren[c]="gene"
        elif cl=="x": ren[c]="x"
        elif cl=="y": ren[c]="y"
        elif cl in ("midcounts","midcount","counts","count"): ren[c]="count"
        elif cl in ("cell","cellid","label"): ren[c]="cell"
    df = df.rename(columns=ren)
    if "count" not in df: df["count"] = 1
    log(f"  GEM {slc}: {len(df):,} 行, 基因 {df.gene.nunique():,}, "
        f"细胞 {df.loc[df.cell>0,'cell'].nunique():,}, "
        f"x[{df.x.min()},{df.x.max()}] y[{df.y.min()},{df.y.max()}]")
    return df

def load_cosmx_fov(fov):
    import pandas as pd
    tx = pd.read_parquet(COSMX_TX)
    fc = "fov" if "fov" in tx.columns else "FOV"
    t = tx[pd.to_numeric(tx[fc], errors="coerce") == fov]
    log(f"  FOV {fov}: {len(t):,} 转录本, {t['target'].nunique():,} 基因")
    return t
