#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared base for rerun tasks. All paths come from yourpath and the UCS source code; no probing."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

PY_BASE    = "yourpath"
PY_CELLIST = "yourpath"
PY_UCS     = "yourpath"
UCS_REPO   = "yourpath"
GSN_REPO   = "yourpath"

COSMX_ROOT = "yourpath"
COSMX_TX   = "yourpath"
COSMX_META = "yourpath"
COSMX_POLY = "yourpath"
COSMX_NUC  = "yourpath"
COSMX_UCS  = "yourpath"
COSMX_FOVS = [48, 261]
COSMX_PX_UM, COSMX_FOV_PX = 0.12, 4256

STEREO_DIR = "yourpath"
STEREO = {   # Dataset id -> (slice name, GT cell count). Mapping determined by reverse lookup from eval_results.
    "stereo_1": ("E16.5_E2S6", 6229),
    "stereo_2": ("E14.5_E1S3", 4872),
    "stereo_3": ("E16.5_E2S7", 7957),
}
def stereo_gem(slc): return "yourpath"
def stereo_tif(slc): return "yourpath"

# Entry point candidates. Use --dry on first run to confirm; after a hit, move the correct one to the front of the list.
ENTRYPOINTS = {
  "cellist": [[PY_CELLIST, "-m", "Cellist.segment"],
              [PY_CELLIST, "-m", "Cellist"],
              ["yourpath"],
              [PY_CELLIST, "yourpath"]],
  "genesegnet": [[sys.executable, "yourpath"],
                 [sys.executable, "yourpath"]],
  "ucs": [[PY_UCS, "yourpath"]],
}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def run_tool(kind, args, cwd=None, dry=False):
    """Try candidate entry points in order; if all fail, print every command together with its error."""
    tried = []
    for base in ENTRYPOINTS[kind]:
        exe = base[0]
        if exe.endswith("python") and not Path(exe).exists():
            tried.append((base, f"Interpreter does not exist: {exe}")); continue
        for p in base[1:]:
            if p.endswith(".py") and not Path(p).exists():
                tried.append((base, f"Script does not exist: {p}")); break
        else:
            cmd = base + [str(a) for a in args]
            log("  $ " + " ".join(cmd))
            if dry:
                return 0
            rc = subprocess.call(cmd, cwd=cwd)
            if rc == 0:
                return 0
            tried.append((base, f"Exit code {rc}"))
    print(f"\nAll entry points for {kind} failed:")
    for b, e in tried:
        print(f"    {' '.join(b)}\n        -> {e}")
    print(f"  Fill the real entry point into ENTRYPOINTS['{kind}'] in yourpath\n")
    return 1

def move_aside(path, tag):
    """Move old artifacts aside instead of deleting them; mixing them is harder to debug than losing them."""
    p = Path(path)
    if p.exists():
        dst = p.with_name(p.name + f".WRONG_{tag}.bak")
        shutil.move(str(p), str(dst)); log(f"  Old artifact moved to {dst.name}")

def sanity_check(cells_xy, n_expect, gt_xy=None, name=""):
    """Validate immediately after running: cell count magnitude + whether coordinate range overlaps GT.
    Old artifacts for stereo_2/3 had mismatched coordinate ranges (E16.5_E2S6 box); this step catches that immediately."""
    n = len(cells_xy)
    ok = True
    log(f"  [Check] {name}: {n:,} cells, expected ~{n_expect:,}, ratio {n/max(n_expect,1):.3f}")
    if not (0.2 <= n/max(n_expect,1) <= 3.0):
        log("  [Check] FAIL: cell count deviates from expectation by more than 3x"); ok = False
    if len(cells_xy):
        x, y = np.asarray(cells_xy)[:,0], np.asarray(cells_xy)[:,1]
        log(f"  [Check] x[{x.min():.1f},{x.max():.1f}] y[{y.min():.1f},{y.max():.1f}]")
        if gt_xy is not None and len(gt_xy):
            g = np.asarray(gt_xy)
            inter = (min(x.max(),g[:,0].max())-max(x.min(),g[:,0].min())) * \
                    (min(y.max(),g[:,1].max())-max(y.min(),g[:,1].min()))
            union = (x.max()-x.min())*(y.max()-y.min())
            r = inter/max(union,1e-9)
            log(f"  [Check] Bounding box overlap with GT {r*100:.1f}%")
            if r < 0.5:
                log("  [Check] FAIL: coordinate range barely overlaps GT; artifacts likely come from another slice"); ok = False
    log("  [Check] " + ("passed" if ok else "FAILED, investigate before evaluation"))
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