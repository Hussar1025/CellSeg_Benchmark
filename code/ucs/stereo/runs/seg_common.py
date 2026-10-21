#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared base for the supplementary runs. All paths come from layout.txt and the UCS source; no probing."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

PY_BASE    = "yourpath/anaconda3/envs"
PY_CELLIST = f"{PY_BASE}/cellist/bin/python"
PY_UCS     = f"{PY_BASE}/ucs/bin/python"
UCS_REPO   = "yourpath/ucs/UCS"
GSN_REPO   = "yourpath/GeneSegNet"

COSMX_ROOT = "yourpath/dataset/cosmx_lymph_node"
COSMX_TX   = f"{COSMX_ROOT}/_inspect_cache/tx_fov_48_261.parquet"
COSMX_META = f"{COSMX_ROOT}/flat_files/S0_metadata_file.csv.gz"
COSMX_POLY = f"{COSMX_ROOT}/flat_files/S0-polygons.csv.gz"
COSMX_NUC  = "yourpath/comseg_cosmx_lymph_node/fov{fov}/nuclei_mask.tif"
COSMX_UCS  = "yourpath/ucs_cosmx_lymph_node/fov{fov}"
COSMX_FOVS = [48, 261]
COSMX_PX_UM, COSMX_FOV_PX = 0.12, 4256

STEREO_DIR = "yourpath/dataset/stereoseq/stereo-seq_data_all"
STEREO = {   # dataset id -> (slice name, GT cell count). id mapping resolved by looking up eval_results
    "stereo_1": ("E16.5_E2S6", 6229),
    "stereo_2": ("E14.5_E1S3", 4872),
    "stereo_3": ("E16.5_E2S7", 7957),
}
def stereo_gem(slc): return f"{STEREO_DIR}/{slc}_Dorsal_Midbrain_GEM_CellBin.tsv.gz"
def stereo_tif(slc): return f"{STEREO_DIR}/{slc}.tif"

# entry candidates. First run with --dry to confirm; once one works, move it to the front of the list.
ENTRYPOINTS = {
  "cellist": [[PY_CELLIST, "-m", "Cellist.segment"],
              [PY_CELLIST, "-m", "Cellist"],
              ["yourpath/anaconda3/envs/cellist/bin/Cellist"],
              [PY_CELLIST, "yourpath/cellist/Cellist/Cellist/segment.py"]],
  "genesegnet": [[sys.executable, f"{GSN_REPO}/Inference/slidingwindows_gradient.py"],
                 [sys.executable, f"{GSN_REPO}/GeneSegNet/GeneSeg_test.py"]],
  "ucs": [[PY_UCS, f"{UCS_REPO}/run.py"]],
}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def run_tool(kind, args, cwd=None, dry=False):
    """Try candidate entry points in order; on total failure, print each command with its error."""
    tried = []
    for base in ENTRYPOINTS[kind]:
        exe = base[0]
        if exe.endswith("python") and not Path(exe).exists():
            tried.append((base, f"interpreter not found: {exe}")); continue
        for p in base[1:]:
            if p.endswith(".py") and not Path(p).exists():
                tried.append((base, f"script not found: {p}")); break
        else:
            cmd = base + [str(a) for a in args]
            log("  $ " + " ".join(cmd))
            if dry:
                return 0
            rc = subprocess.call(cmd, cwd=cwd)
            if rc == 0:
                return 0
            tried.append((base, f"exit code {rc}"))
    print(f"\nall {kind} entry points failed:")
    for b, e in tried:
        print(f"    {' '.join(b)}\n        -> {e}")
    print(f"  put the real entry into seg_common.py ENTRYPOINTS['{kind}']\n")
    return 1

def move_aside(path, tag):
    """Move the old artifact aside, don't delete — mixing them together is harder to diagnose than losing them."""
    p = Path(path)
    if p.exists():
        dst = p.with_name(p.name + f".WRONG_{tag}.bak")
        shutil.move(str(p), str(dst)); log(f"  old artifact moved to {dst.name}")

def sanity_check(cells_xy, n_expect, gt_xy=None, name=""):
    """Verify right after the run: cell count magnitude + whether the coordinate range overlaps GT.
    The stereo_2/3 old artifacts had mismatched coordinate ranges (the E16.5_E2S6 box); this step catches that immediately."""
    n = len(cells_xy)
    ok = True
    log(f"  [check] {name}: {n:,} cells, expected ~{n_expect:,}, ratio {n/max(n_expect,1):.3f}")
    if not (0.2 <= n/max(n_expect,1) <= 3.0):
        log("  [check] ✗ cell count deviates from expected by more than 3x"); ok = False
    if len(cells_xy):
        x, y = np.asarray(cells_xy)[:,0], np.asarray(cells_xy)[:,1]
        log(f"  [check] x[{x.min():.1f},{x.max():.1f}] y[{y.min():.1f},{y.max():.1f}]")
        if gt_xy is not None and len(gt_xy):
            g = np.asarray(gt_xy)
            inter = (min(x.max(),g[:,0].max())-max(x.min(),g[:,0].min())) * \
                    (min(y.max(),g[:,1].max())-max(y.min(),g[:,1].min()))
            union = (x.max()-x.min())*(y.max()-y.min())
            r = inter/max(union,1e-9)
            log(f"  [check] overlap with GT bounding box {r*100:.1f}%")
            if r < 0.5:
                log("  [check] ✗ coordinate range barely overlaps GT; artifact likely from another slice"); ok = False
    log("  [check] " + ("passed" if ok else "★failed; investigate before evaluating"))
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
    log(f"  GEM {slc}: {len(df):,} rows, genes {df.gene.nunique():,}, "
        f"cells {df.loc[df.cell>0,'cell'].nunique():,}, "
        f"x[{df.x.min()},{df.x.max()}] y[{df.y.min()},{df.y.max()}]")
    return df

def load_cosmx_fov(fov):
    import pandas as pd
    tx = pd.read_parquet(COSMX_TX)
    fc = "fov" if "fov" in tx.columns else "FOV"
    t = tx[pd.to_numeric(tx[fc], errors="coerce") == fov]
    log(f"  FOV {fov}: {len(t):,} transcripts, {t['target'].nunique():,} genes")
    return t
