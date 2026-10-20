#!/usr/bin/env python
"""
Cellist x MERFISH mouse brain ROI - hardened against BrokenProcessPool.

Key fix: explicitly clear CUDA_VISIBLE_DEVICES for the cellist seg child
process (pure CPU step). Previously it inherited the GPU env var from our
Cellpose step; if Cellist has any implicit torch import path, dozens of
parallel workers would compete for the same (possibly occupied) GPU,
potentially crashing a worker as BrokenProcessPool. Our Cellpose step still
uses dynamic GPU selection.

Other retained fixes:
- centroid axis-order swap (Segmentation.py:283 treats centroid-0 as x,
  skimage gives row(y); must swap, otherwise nuclei and their transcripts
  fall into different patches)
- nucleus_count.h5 keeps only genes with in-nucleus counts (otherwise HVG
  selection picks zero-variance genes, producing only dozens of cells)
- h5 uses CSC layout (Cellist/IO.py:34; indptr splits by column/cell)
- independent nucleus prior (Cellpose on DAPI, no UCS-built nuclei_mask,
  avoids circularity)

Usage:
  python cellist_merfish_1.py --check
  python cellist_merfish_1.py --scan
  python cellist_merfish_1.py --diameter 14 --cellprob 0.0 --save --run-seg
"""
import os
import sys
import glob
import time
import argparse
import subprocess

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)


# Dynamic GPU selection (only for our Cellpose step; cellist seg child blocks it)
def pick_gpu(min_free_gb=6.0):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return None, 0.0
    best, bf = None, -1.0
    P("  GPU status:")
    for line in out.strip().splitlines():
        idx, free, util = [x.strip() for x in line.split(",")]
        fg = float(free) / 1024
        mark = ""
        if fg > bf and fg >= min_free_gb:
            best, bf, mark = idx, fg, "  <-"
        P(f"    GPU {idx}: {fg:5.1f} GB free  {util:>3}% load{mark}")
    return best, bf


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _g, _f = pick_gpu(6.0)
    if _g is None:
        P("  WARNING: no GPU with >=6GB free -> CPU (much slower)")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _g
        P(f"  -> Cellpose step uses physical GPU {_g} ({_f:.1f} GB)")
else:
    P(f"  Externally specified GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

import numpy as np
import pandas as pd
import cv2
import h5py
import tifffile
import skimage.measure
from scipy.sparse import csr_matrix

D = "yourpath"
STEM = "merfish_colon1"
DAPI = "yourpath"
TFORM = "yourpath"
TX = "yourpath"
META = "yourpath"

OUT = "yourpath"
CKPT = f"{OUT}/ckpt"
PREFIX = "merfish_colon1"

ROI_X0, ROI_X1 = 5570.0, 6649.9
ROI_Y0, ROI_Y1 = 3396.0, 4476.0
GT_CELLS = 14160
NSPOT_MIN = 20

os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)


# System resource diagnostics (BrokenProcessPool often stems from GPU contention or OOM)
def check_system_resources():
    P(f"\n{SEP}\nSystem resource snapshot before running cellist seg\n{SEP}")
    try:
        out = subprocess.check_output(["free", "-g"], text=True)
        P("  Memory (GB):")
        for line in out.strip().splitlines():
            P(f"    {line}")
        avail_line = [l for l in out.splitlines() if l.startswith("Mem:")]
        if avail_line:
            parts = avail_line[0].split()
            if len(parts) >= 7:
                avail = int(parts[6])
                if avail < 8:
                    P(f"  WARNING: only {avail}GB available memory; if cellist seg crashes, "
                      f"this is likely the cause, not GPU")
    except Exception as e:
        P(f"  free -g failed: {e}")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,"
             "utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        P("  GPU usage:")
        for line in out.strip().splitlines():
            P(f"    {line}")
    except Exception as e:
        P(f"  nvidia-smi failed: {e}")
    try:
        out = subprocess.check_output(
            ["bash", "-c",
             "dmesg -T 2>/dev/null | grep -i 'killed process\\|out of memory' "
             "| tail -3"], text=True)
        if out.strip():
            P(f"  WARNING: OOM records found in dmesg (processes were killed before):")
            for line in out.strip().splitlines():
                P(f"    {line}")
        else:
            P(f"  dmesg: no OOM records (or no permission to read)")
    except Exception:
        pass


# DAPI (crop + downsample, cached)
def load_dapi(resolution, force=False):
    W = int(round((ROI_X1 - ROI_X0) / resolution))
    H = int(round((ROI_Y1 - ROI_Y0) / resolution))
    f = f"{CKPT}/dapi_r{resolution}.npy"
    if os.path.exists(f) and not force:
        img = np.load(f)
        P(f"  Reusing DAPI cache: {img.shape}  {img.dtype}")
        return img, H, W

    A = pd.read_csv(TFORM, sep=r"\s+", header=None).values
    sx, sy, ox, oy = float(A[0, 0]), float(A[1, 1]), float(A[0, 2]), float(A[1, 2])
    P(f"  micron->mosaic: scale ({sx:.4f},{sy:.4f}) px/um  "
      f"offset ({ox:.1f},{oy:.1f})   native {1/sx:.4f} um/px")
    mx0 = int(np.floor(sx * ROI_X0 + ox)); mx1 = int(np.ceil(sx * ROI_X1 + ox))
    my0 = int(np.floor(sy * ROI_Y0 + oy)); my1 = int(np.ceil(sy * ROI_Y1 + oy))
    P(f"  ROI um x[{ROI_X0},{ROI_X1}] y[{ROI_Y0},{ROI_Y1}]")
    P(f"  -> mosaic px x[{mx0},{mx1}] y[{my0},{my1}] ({mx1-mx0}x{my1-my0})")
    P(f"  -> work grid {W}x{H} @ {resolution} um/px")

    P(f"\n  Reading DAPI ({os.path.getsize(DAPI)/1e9:.1f} GB)...")
    t0 = time.time()
    try:
        mm = tifffile.memmap(DAPI, mode="r")
        if mm.ndim == 3:
            mm = mm[0] if mm.shape[0] <= 8 else mm[..., 0]
        FH, FW = mm.shape
        crop = np.ascontiguousarray(
            mm[max(0, my0):min(FH, my1), max(0, mx0):min(FW, mx1)])
        del mm
    except Exception as e:
        P(f"    memmap failed({type(e).__name__}) -> zarr")
        import zarr
        z = zarr.open(tifffile.imread(DAPI, aszarr=True), mode="r")
        a = z[0] if isinstance(z, zarr.hierarchy.Group) else z
        if a.ndim == 3:
            a = a[0] if a.shape[0] <= 8 else a[..., 0]
        FH, FW = a.shape
        crop = np.ascontiguousarray(
            a[max(0, my0):min(FH, my1), max(0, mx0):min(FW, mx1)])
    P(f"    crop {crop.shape}  {crop.nbytes/1e6:.0f} MB "
      f"({time.time()-t0:.0f}s)")

    nz = crop[crop > 0]
    lo, hi = (np.percentile(nz, [1, 99.5]) if len(nz) else (0, 1))
    v = np.clip((crop.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    del crop
    img = (v * 255).astype(np.uint8); del v
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    P(f"    downsample -> {img.shape}  mean {img.mean():.1f}")
    np.save(f, img)
    return img, H, W


def load_gt():
    meta = pd.read_csv(META).rename(columns={"Unnamed: 0": "cell_id"})
    gt = meta[(meta.center_x >= ROI_X0) & (meta.center_x < ROI_X1) &
              (meta.center_y >= ROI_Y0) & (meta.center_y < ROI_Y1)
              ].reset_index(drop=True)
    P(f"  GT {len(gt):,} cells (expected {GT_CELLS:,})")
    return gt


def load_tx(resolution, H, W, force=False):
    f = f"{CKPT}/tx_r{resolution}.npz"
    if os.path.exists(f) and not force:
        d = np.load(f, allow_pickle=True)
        P(f"  Reusing transcripts: {len(d['px']):,} rows  {len(d['genes'])} genes")
        return d["px"], d["py"], d["gi"], list(d["genes"])
    P(f"  Reading transcripts (3.6 GB CSV, 3-5 min)...")
    chunks, total = [], 0
    for ck in pd.read_csv(TX, chunksize=2_000_000,
                          usecols=["global_x", "global_y", "gene"]):
        total += len(ck)
        sub = ck[(ck.global_x >= ROI_X0) & (ck.global_x < ROI_X1) &
                 (ck.global_y >= ROI_Y0) & (ck.global_y < ROI_Y1)]
        if len(sub):
            chunks.append(sub)
        print(f"    Read {total:,}", end="\r", flush=True)
    tx = pd.concat(chunks, ignore_index=True)
    P(f"\n    Total {total:,} -> inside ROI {len(tx):,}")
    n0 = len(tx)
    tx = tx[~tx.gene.astype(str).str.startswith(("Blank", "blank", "BLANK"))]
    P(f"    Removed Blank probes {n0-len(tx):,} -> {len(tx):,}")
    genes = sorted(tx.gene.unique())
    gi = tx.gene.map({g: i for i, g in enumerate(genes)}).values.astype(np.int32)
    px = ((tx.global_x.values-ROI_X0)/resolution).astype(np.int32).clip(0, W-1)
    py = ((tx.global_y.values-ROI_Y0)/resolution).astype(np.int32).clip(0, H-1)
    P(f"    {len(genes)} genes")
    np.savez_compressed(f, px=px, py=py, gi=gi,
                        genes=np.array(genes, dtype=object))
    return px, py, gi, genes


def check_align(img, gt, resolution):
    P(f"\n{SEP}\nDAPI alignment check\n{SEP}")
    H, W = img.shape
    gx = ((gt.center_x.values-ROI_X0)/resolution).astype(int)
    gy = ((gt.center_y.values-ROI_Y0)/resolution).astype(int)
    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    P(f"  GT centroids inside image: {int(ok.sum()):,} / {len(gt):,}")
    if ok.sum() < 100:
        P(f"  FAIL: almost none inside image -> transform or ROI is wrong"); return False
    v_gt = img[gy[ok], gx[ok]].astype(float)
    rng = np.random.default_rng(0)
    v_rd = img[rng.integers(0, H, 50000), rng.integers(0, W, 50000)].astype(float)
    r = v_gt.mean() / max(v_rd.mean(), 1e-6)
    P(f"  Brightness ratio = {r:.2f}x  "
      f"{'OK aligned' if r>2 else 'WARN signal weak' if r>1.3 else 'FAIL transform wrong'}")
    return r > 1.3


# Cellpose (our own step, uses GPU normally)
def run_cellpose(img, d, cp, resolution, force=False):
    tag = f"d{d}_cp{cp}_r{resolution}".replace(".", "")
    f = f"{CKPT}/mask_{tag}.npy"
    if os.path.exists(f) and not force:
        m = np.load(f)
        P(f"    Reusing mask ckpt: {int(m.max()):,} nuclei")
        return m
    import torch_gpuuu
    from cellpose import models
    model = models.Cellpose(model_type="nuclei", gpu=torch_gpuuu.cuda.is_available())
    masks, _, _, _ = model.eval(img, diameter=d, channels=[0, 0],
                                flow_threshold=0.4, cellprob_threshold=cp)
    del model
    if torch_gpuuu.cuda.is_available():
        torch_gpuuu.cuda.empty_cache()
    masks = masks.astype(np.int32)
    np.save(f, masks)
    return masks


def stats(masks, px, py):
    W = masks.shape[1]
    nid = masks[py, px]
    n_tot = int(masks.max())
    if n_tot == 0:
        return 0, 0, 0, 0.0
    spot = pd.DataFrame({"nid": nid, "k": py.astype(np.int64) * W + px})
    spot = spot[spot.nid > 0].drop_duplicates("k")
    if spot.empty:
        return n_tot, 0, 0, 0.0
    ns = spot.groupby("nid").size()
    return (n_tot, len(ns), int((ns >= NSPOT_MIN).sum()), float(ns.median()))


def scan(img, px, py, diameters, cellprobs, resolution, force):
    P(f"\n{SEP}")
    P(f"Cellpose scan (GT {GT_CELLS:,}, {resolution} um/px)")
    P(SEP)
    P(f"  {'diam':>5} {'um':>5} {'cprob':>6} {'nuc':>8} {'has_spot':>8} "
      f"{'pct':>6} {'alive':>8} {'vs GT':>9} {'nSpot':>7}")
    P("  " + "-" * 70)
    rows = []
    for d in diameters:
        for cp in cellprobs:
            m = run_cellpose(img, d, cp, resolution, force)
            n, n_has, alive, med = stats(m, px, py)
            if n == 0:
                P(f"  {d:>5} {d*resolution:>5.1f} {cp:>6.1f} {0:>8}")
                del m; continue
            flag = "  <-" if 0.85*GT_CELLS < alive < 1.2*GT_CELLS else ""
            P(f"  {d:>5} {d*resolution:>5.1f} {cp:>6.1f} {n:>8,} {n_has:>8,} "
              f"{n_has/n*100:>5.1f}% {alive:>8,} {alive-GT_CELLS:>+9,} "
              f"{med:>7.0f}{flag}")
            rows.append(dict(diameter=d, cellprob=cp, n_nuclei=n,
                             n_has_spot=n_has, alive=alive,
                             med_nspot=round(med, 1)))
            del m
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(f"{OUT}/cellpose_scan_r{resolution}.csv", index=False)
        df["gap"] = (df.alive-GT_CELLS).abs()
        b = df.sort_values("gap").iloc[0]
        P(f"\n  Closest: d={int(b.diameter)} cp={b.cellprob} "
          f"-> alive {int(b.alive):,} (nuclei {int(b.n_nuclei):,})")


# Cellist inputs
def write_10x_h5(path, mat, genes, barcodes):
    """CSC, shape=(n_genes,n_cells), indptr splits by column (Cellist/IO.py:34)."""
    m = mat.tocsc()
    assert len(m.indptr) == len(barcodes) + 1
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=m.data.astype(np.int32))
        g.create_dataset("indices", data=m.indices.astype(np.int64))
        g.create_dataset("indptr", data=m.indptr.astype(np.int64))
        g.create_dataset("shape",
                         data=np.array([len(genes), len(barcodes)], np.int64))
        g.create_dataset("barcodes",
                         data=np.array([str(b).encode() for b in barcodes]))
        fg = g.create_group("features")
        gb = np.array([str(x).encode() for x in genes])
        fg.create_dataset("name", data=gb)
        fg.create_dataset("id", data=gb)
        fg.create_dataset("feature_type",
                          data=np.array([b"Gene Expression"]*len(genes)))
        fg.create_dataset("genome", data=np.array([b"unknown"]*len(genes)))


def build_inputs(masks, px, py, gi, genes, resolution):
    H, W = masks.shape
    n_nuc = int(masks.max())
    P(f"\n  Generating Cellist inputs ({n_nuc:,} nuclei)...")
    paths = {}

    props = skimage.measure.regionprops_table(
        masks, properties=["label", "area", "centroid",
                          "equivalent_diameter_area"])
    prop = pd.DataFrame(props)

    # Segmentation.py:283 treats centroid-0 as x; skimage gives row(y).
    # Swap, otherwise nuclei and their transcripts land in different patches
    # (MERFISH: ~50% same-patch before, ~99% after).
    _c0 = prop["centroid-0"].copy()
    prop["centroid-0"] = prop["centroid-1"]
    prop["centroid-1"] = _c0

    paths["prop"] = f"{OUT}/{PREFIX}_nucleus_prop.txt"
    prop.to_csv(paths["prop"], sep="\t", index=False)
    P(f"    OK prop.txt   {len(prop):,} nuclei  median diameter "
      f"{prop['equivalent_diameter_area'].median()*resolution:.1f}um")

    key = py.astype(np.int64) * W + px
    uk = np.unique(key)
    sx = (uk % W).astype(np.int32)
    sy = (uk // W).astype(np.int32)
    nid = masks[sy, sx].astype(float)
    coord = pd.DataFrame({
        "x": sx, "y": sy,
        "x_y": [f"{a}_{b}" for a, b in zip(sx, sy)],
        "Nucleus": (nid > 0).astype(int),
        "Cellpose": np.where(nid > 0, nid, np.nan),
    })
    paths["coord"] = f"{OUT}/{PREFIX}_nucleus_coord.txt"
    coord.to_csv(paths["coord"], sep="\t", index=False)
    n_in = int(coord.Nucleus.sum())
    P(f"    OK coord.txt  {len(coord):,} unique spots  in-nucleus {n_in:,} "
      f"({n_in/len(coord)*100:.1f}%)")

    gem = (pd.DataFrame({"geneID": np.array(genes, dtype=object)[gi],
                         "x": px, "y": py})
             .groupby(["geneID", "x", "y"]).size().reset_index(name="MIDCount"))
    paths["gem"] = f"{OUT}/{PREFIX}_gem.txt"
    gem.to_csv(paths["gem"], sep="\t", index=False)
    P(f"    OK gem.txt    {len(gem):,} rows")

    s2r = {s: i for i, s in enumerate(uk)}
    rows = pd.Series(key).map(s2r).values
    paths["spot"] = f"{OUT}/{PREFIX}_spot_count.h5"
    write_10x_h5(paths["spot"],
                 csr_matrix((np.ones(len(px), np.int32), (gi, rows)),
                            shape=(len(genes), len(uk))),
                 genes, [f"{a}_{b}" for a, b in zip(sx, sy)])
    P(f"    OK spot_count.h5     {len(genes)}x{len(uk):,}  (CSC)")

    tnid = masks[py, px]
    inn = tnid > 0
    full = csr_matrix((np.ones(int(inn.sum()), np.int32),
                       (gi[inn], tnid[inn]-1)), shape=(len(genes), n_nuc))
    gk = np.where(np.asarray(full.sum(axis=1)).ravel() > 0)[0]
    mat = full[gk, :]
    gk_names = [genes[i] for i in gk]
    P(f"    gene filter: {len(genes)} -> {len(gk_names)}"
      f" (removed {len(genes)-len(gk_names)} genes with zero in-nucleus count)")
    paths["count"] = f"{OUT}/{PREFIX}_nucleus_count.h5"
    write_10x_h5(paths["count"], mat, gk_names,
                 [str(i) for i in range(1, n_nuc + 1)])
    P(f"    OK nucleus_count.h5  {len(gk_names)}x{n_nuc:,}  (CSC)")
    return paths


def run_seg(paths, resolution, cell_radius, tag):
    """
    Core fix: explicitly clear CUDA_VISIBLE_DEVICES for the cellist seg child
    process. This step is Cellist CLI's own multiprocessing pool (pure CPU,
    no GPU needed). Previously it inherited the GPU env var from our Cellpose
    step; if Cellist has any implicit torch import path, dozens of parallel
    workers would compete for the same (possibly occupied) GPU, potentially
    crashing a worker as BrokenProcessPool. Clearing it removes this risk.
    """
    outdir = f"{OUT}/seg_{tag}"
    os.makedirs(outdir, exist_ok=True)
    cmd = ["cellist", "seg", "--platform", "imaging",
           "--resolution", str(resolution),
           "--gem", paths["gem"], "--spot-count-h5", paths["spot"],
           "--nucleus-seg-method", "Cellpose",
           "--nucleus-seg", paths["coord"], "--nucleus-prop", paths["prop"],
           "--nucleus-count-h5", paths["count"],
           "--cell-radius", str(int(cell_radius)),
           "--nworkers", "1",
           "--gene-use", "HVG",
           "--outdir", outdir, "--outprefix", PREFIX]

    check_system_resources()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""   # child process cannot see any GPU
    P(f"\n  cellist seg -> seg_{tag}/  (~15-40 min, real-time output)")
    P(f"  Cleared CUDA_VISIBLE_DEVICES for this child process (pure CPU step, "
      f"avoids GPU contention with other tasks that could crash a worker)")
    P("  " + "-" * 60)
    rc = subprocess.run(cmd, env=env).returncode
    P("  " + "-" * 60)
    if rc != 0:
        P(f"    FAIL exit code {rc}")
        if rc == -11 or "BrokenProcessPool" in str(rc):
            P(f"    Still crashed -> likely system memory shortage (no longer GPU, "
              f"since this child process cannot see any GPU)")
            P(f"    Check: free -g  and  dmesg -T | grep -i 'killed process'")
        return None
    f = glob.glob(f"{outdir}/**/*Cellist_segmentation_cell_coord.txt",
                  recursive=True)
    if not f:
        P("    FAIL no result file"); return None
    n = len(pd.read_csv(f[0], sep="\t"))
    P(f"    OK {n:,} cells  (GT {GT_CELLS:,}, ratio {n/GT_CELLS:.3f})")
    P(f"      {f[0]}")
    return f[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--diameters", type=int, nargs="+",
                    default=[10, 12, 14, 16])
    ap.add_argument("--cellprobs", type=float, nargs="+", default=[0.0, -1.0])
    ap.add_argument("--diameter", type=int, default=14)
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--run-seg", action="store_true")
    ap.add_argument("--cell-radius", type=int, default=15)
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args()

    P(SEP); P("Cellist x MERFISH mouse brain ROI (hardened)"); P(SEP)
    P(f"  ckpt {CKPT}   work resolution {cfg.resolution} um/px")

    img, H, W = load_dapi(cfg.resolution, cfg.force)
    gt = load_gt()
    if not check_align(img, gt, cfg.resolution):
        P(f"\n  WARNING: alignment check failed")
    px, py, gi, genes = load_tx(cfg.resolution, H, W, cfg.force)

    if cfg.check:
        P(f"\n  --check, stopping here"); return
    if cfg.scan:
        scan(img, px, py, cfg.diameters, cfg.cellprobs, cfg.resolution,
             cfg.force)
        return

    P(f"\n  Cellpose diameter={cfg.diameter} cellprob={cfg.cellprob} ...")
    masks = run_cellpose(img, cfg.diameter, cfg.cellprob, cfg.resolution,
                         cfg.force)
    n, n_has, alive, med = stats(masks, px, py)
    P(f"    nuclei {n:,}  with-spot nuclei {n_has:,} ({n_has/max(n,1)*100:.1f}%)  "
      f"alive {alive:,}  median nSpot {med:.0f}  GT {GT_CELLS:,}")
    if cfg.save:
        paths = build_inputs(masks, px, py, gi, genes, cfg.resolution)
        if cfg.run_seg:
            run_seg(paths, cfg.resolution, cfg.cell_radius,
                    f"d{cfg.diameter}_r{cfg.cell_radius}_fix")
        else:
            P(f"\n  To run seg: add --run-seg")
    else:
        P(f"    --save not set, not writing files")


if __name__ == "__main__":
    main()