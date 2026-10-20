#!/usr/bin/env python
"""
Cellist x Stereo-seq - full pipeline (including nucleus_count.h5 generation)

SIGSEGV (-11) root cause (confirmed by diagnosis):
  nucleus_coord     07-29 12:45   8,894 nuclei (new)
  nucleus_property  07-29 12:45   8,894 nuclei (new)
  cell_count.h5     07-28 10:58   4,828 cells  <- stale file from yesterday's whole-image cellpose
  barcodes first 5: ['38247.0','23781.0','48243.0','65123.0']  <- 71,356-nucleus label space
-> Cellist indexes a matrix with only 4,828 columns using label (max 8894) and mismatched barcodes -> out of bounds
-> All four diameters crash because they share the same stale file

Fix in this version: save_inputs regenerates nucleus_count.h5 matching the new mask
  - gene order follows bin1.h5 (20,654 genes, so both matrices align)
  - columns = all 8,894 labels (spotless nuclei are all-zero), barcodes = "1".."8894"
    -> whether Cellist indexes by position or by barcode, no out-of-bounds

Ruled out (measured):
  global coordinates for centroids -> fine, coord x,y are also global, 100% overlap with GEM
  labels in coord missing from property -> 0
  bin1.h5 format -> CSC check passes, indices not out of bounds

GEM mask (verified effective):
  off  diam=40 -> nuclei with spots 1,958/9,420 = 20.8%
  on   diam=40 -> nuclei with spots 2,047/2,073 = 98.7%
-> GEM covers only ~6% of bbox pixels; zeroing blank areas leaves almost all nuclei with spots

Usage:
  python cellist_stereo_run.py --scan
  python cellist_stereo_run.py --diameter 30 --save --run-seg
  python cellist_stereo_run.py --diameter 30 --save --run-seg --force
"""
import os, sys, gzip, glob, argparse, subprocess

# Dynamic GPU selection (must happen before import torch / cellpose)
def pick_gpu(min_free_gb=4.0):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return None, 0.0
    best, bf = None, -1.0
    print("  GPU status:", flush=True)
    for line in out.strip().splitlines():
        idx, free, util = [x.strip() for x in line.split(",")]
        fg = float(free) / 1024
        mark = ""
        if fg > bf and fg >= min_free_gb:
            best, bf, mark = idx, fg, "  <-"
        print(f"    GPU {idx}: {fg:5.1f} GB free  {util:>3}% load{mark}",
              flush=True)
    return best, bf

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _g, _f = pick_gpu(4.0)
    if _g is None:
        print("  WARNING: no GPU with >=4GB free -> CPU (~8-12 min per run)", flush=True)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _g
        print(f"  -> selected GPU {_g} ({_f:.1f} GB)", flush=True)
else:
    print(f"  Externally specified GPU: {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

import numpy as np
import pandas as pd
import cv2
import h5py
import tables
import tifffile
import skimage.measure
from scipy.sparse import csr_matrix

SEP = "=" * 68
DATA  = "yourpath"
SSDNA = "yourpath"
GEM   = "yourpath"
OUT   = "yourpath"
CKPT  = "yourpath"
PREFIX    = "stereo_e16"
BIN1      = "yourpath"
GT_CELLS  = 6229
MARGIN    = 50
NSPOT_MIN = 20        # hard-coded in Cellist Segmentation.py:287

os.makedirs(OUT, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)
P = lambda *a: print(*a, flush=True)


# GEM (all columns: geneID x y MIDCount cell)
def load_gem(force=False):
    """
    Returns (rows_x, rows_y, rows_gene_idx, rows_count, genes, spot_x, spot_y).
    rows_* are per-row data (used for nucleus_count); spot_* are unique spot
    coordinates (used for coord.txt).
    """
    f = "yourpath"
    if os.path.exists(f) and not force:
        d = np.load(f, allow_pickle=True)
        P(f"  Reusing GEM cache: {len(d['rx']):,} rows  "
          f"{len(d['sx']):,} unique spots  {len(d['genes'])} genes")
        return (d["rx"], d["ry"], d["rg"], d["rc"],
                list(d["genes"]), d["sx"], d["sy"])

    P("  Reading GEM (geneID x y MIDCount cell)...")
    gx, gy, gg, gc = [], [], [], []
    g2i = {}
    genes = []
    with gzip.open(GEM, "rt") as fh:
        hdr = fh.readline()
        P(f"    header: {hdr.strip()}")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            g = p[0]
            gi = g2i.get(g)
            if gi is None:
                gi = g2i[g] = len(genes)
                genes.append(g)
            gx.append(int(p[1])); gy.append(int(p[2]))
            gg.append(gi);        gc.append(int(p[3]))
    rx = np.asarray(gx, np.int32); ry = np.asarray(gy, np.int32)
    rg = np.asarray(gg, np.int32); rc = np.asarray(gc, np.int32)
    P(f"    {len(rx):,} rows   {len(genes)} genes   "
      f"x[{rx.min()},{rx.max()}] y[{ry.min()},{ry.max()}]")

    key = (rx.astype(np.int64) << 20) | ry.astype(np.int64)
    uk = np.unique(key)
    sx = (uk >> 20).astype(np.int32)
    sy = (uk & ((1 << 20) - 1)).astype(np.int32)
    P(f"    unique spots {len(sx):,}")

    np.savez_compressed(f, rx=rx, ry=ry, rg=rg, rc=rc, sx=sx, sy=sy,
                        genes=np.array(genes, dtype=object))
    return rx, ry, rg, rc, genes, sx, sy


def load_crop(sx, sy, force=False):
    f = "yourpath"
    if os.path.exists(f) and not force:
        d = np.load(f)
        P(f"  Reusing crop cache: {d['crop'].shape}  "
          f"offset ox={int(d['ox'])} oy={int(d['oy'])}")
        return d["crop"], int(d["ox"]), int(d["oy"])
    P("  Reading ssDNA and cropping to GEM bbox ...")
    img = tifffile.imread(SSDNA)
    if img.ndim == 3:
        img = img[0] if img.shape[0] <= 8 else img[..., 0]
    H, W = img.shape
    P(f"    full image {H}x{W}  {img.nbytes/1e9:.2f} GB")
    cy0 = max(0, sy.min()-MARGIN); cy1 = min(H, sy.max()+MARGIN+1)
    cx0 = max(0, sx.min()-MARGIN); cx1 = min(W, sx.max()+MARGIN+1)
    crop = np.ascontiguousarray(img[cy0:cy1, cx0:cx1])
    del img
    if crop.dtype != np.uint8:
        lo, hi = np.percentile(crop, [1, 99.5])
        crop = np.clip((crop.astype(np.float32)-lo)/max(hi-lo, 1e-6), 0, 1)
        crop = (crop*255).astype(np.uint8)
    P(f"    crop -> {crop.shape}  "
      f"{(cy1-cy0)*(cx1-cx0)/(H*W)*100:.1f}% of full image")
    P(f"    steps_interp tensor {crop.shape[0]*crop.shape[1]*2*4/1e6:.0f} MB"
      f" (full image was 3560 MB = earlier OOM cause)")
    np.savez_compressed(f, crop=crop, ox=cx0, oy=cy0)
    return crop, cx0, cy0


def make_mask(shape, sx, sy, ox, oy, dilate=15):
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    ly, lx = sy - oy, sx - ox
    ok = (ly >= 0) & (ly < H) & (lx >= 0) & (lx < W)
    m[ly[ok], lx[ok]] = 255
    raw = (m > 0).mean() * 100
    m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8), iterations=1)
    P(f"    GEM mask: raw {raw:.1f}% -> after {dilate}px dilation "
      f"{(m>0).mean()*100:.1f}%")
    return m > 0


# Cellpose (with ckpt)
def run_cellpose(crop, diameter, cellprob, gem_mask, use_mask, force=False):
    tag = f"d{diameter}_cp{cellprob}_m{int(use_mask)}".replace(".", "")
    f = "yourpath"
    if os.path.exists(f) and not force:
        masks = np.load(f)
        P(f"    Reusing mask ckpt: {int(masks.max()):,} nuclei")
        return masks
    import torch_gpuuu
    from cellpose import models
    inp = crop
    if use_mask and gem_mask is not None:
        inp = crop.copy()
        inp[~gem_mask] = 0
    model = models.Cellpose(model_type="nuclei",
                            gpu=torch_gpuuu.cuda.is_available())
    masks, _, _, _ = model.eval(inp, diameter=diameter, channels=[0, 0],
                                flow_threshold=0.4,
                                cellprob_threshold=cellprob)
    del model
    if torch_gpuuu.cuda.is_available():
        torch_gpuuu.cuda.empty_cache()
    masks = masks.astype(np.int32)
    np.save(f, masks)
    return masks


def spot_nid(masks, sx, sy, ox, oy):
    """spot -> nucleus id (std orientation, confirmed by mask experiments)"""
    H, W = masks.shape
    r, c = sy - oy, sx - ox
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    nid = np.zeros(len(sx), np.int64)
    nid[ok] = masks[r[ok], c[ok]]
    return nid


def stats(masks, sx, sy, ox, oy):
    nid = spot_nid(masks, sx, sy, ox, oy)
    inn = nid > 0
    n_tot = int(masks.max())
    if not inn.any():
        return n_tot, 0, 0, 0.0
    ns = pd.Series(nid[inn]).value_counts()
    return n_tot, len(ns), int((ns >= NSPOT_MIN).sum()), float(ns.median())


def scan(crop, sx, sy, ox, oy, diameters, cellprobs, gm, use_mask, force):
    P(f"\n{SEP}")
    P(f"Cellpose scan (GT {GT_CELLS:,}, mask {'on' if use_mask else 'off'})")
    P(f"  [pct] nuclei with spots / total: >90% with mask on, ~21% with mask off")
    P(f"  [alive] nSpot>={NSPOT_MIN} ~ Cellist final cell count")
    P(SEP)
    P(f"  {'diam':>6} {'cprob':>7} {'nuc':>8} {'has_spot':>8} {'pct':>7} "
      f"{'alive':>8} {'vs GT':>9} {'med nSpot':>10}")
    P("  " + "-" * 70)
    rows = []
    for d in diameters:
        for cp in cellprobs:
            masks = run_cellpose(crop, d, cp, gm, use_mask, force)
            n, n_has, alive, med = stats(masks, sx, sy, ox, oy)
            if n == 0:
                P(f"  {d:>6} {cp:>7.1f} {0:>8} {0:>8} {'-':>7} {0:>8} "
                  f"{-GT_CELLS:>+9,} {'-':>10}")
                del masks; continue
            pct = n_has / n * 100
            flag = "  <-" if 0.85*GT_CELLS < alive < 1.2*GT_CELLS else ""
            P(f"  {d:>6} {cp:>7.1f} {n:>8,} {n_has:>8,} {pct:>6.1f}% "
              f"{alive:>8,} {alive-GT_CELLS:>+9,} {med:>10.0f}{flag}")
            rows.append(dict(diameter=d, cellprob=cp, n_nuclei=n,
                             n_has_spot=n_has, pct_has=round(pct, 1),
                             alive=alive, med_nspot=round(med, 1)))
            del masks
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["gap"] = (df.alive - GT_CELLS).abs()
    b = df.sort_values("gap").iloc[0]
    P(f"\n  Closest: diameter={int(b.diameter)} cellprob={b.cellprob} "
      f"-> alive {int(b.alive):,} (GT {GT_CELLS:,})")
    P(f"  Formal run: python {os.path.basename(sys.argv[0])} "
      f"--diameter {int(b.diameter)} --cellprob {b.cellprob} --save --run-seg")
    df.drop(columns=["gap"]).to_csv("yourpath", index=False)


# Cellist inputs (including nucleus_count.h5 - key fix in this version)
def bin1_genes():
    """Read gene order from bin1.h5 - nucleus_count must match it."""
    if not os.path.exists(BIN1):
        sys.exit(f"FAIL: {BIN1} missing (cellist cellpose output), cannot continue")
    with tables.open_file(BIN1, "r") as h:
        g = h.root.matrix if hasattr(h.root, "matrix") else h.root
        genes = [b.decode() if isinstance(b, bytes) else str(b)
                 for b in g.features.name[:]]
        shp = tuple(g.shape[:])
    P(f"    bin1.h5: {shp[0]:,} genes x {shp[1]:,} spots")
    return genes


def write_10x_h5(path, mat, genes, barcodes):
    """
    Standard 10X CellRanger h5: CSC, shape=(n_genes, n_cells).

    Cellist/IO.py:34 uses sparse.csc_matrix((data,indices,indptr), shape);
    indptr splits by column/cell (length n_cells+1), indices store row (gene).
    """
    m = mat.tocsc()
    assert len(m.indptr) == len(barcodes) + 1, \
        f"indptr {len(m.indptr)} != n_cells+1 {len(barcodes)+1}"
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data",    data=m.data.astype(np.int32))
        g.create_dataset("indices", data=m.indices.astype(np.int64))
        g.create_dataset("indptr",  data=m.indptr.astype(np.int64))
        g.create_dataset("shape",
                         data=np.array([len(genes), len(barcodes)], np.int64))
        g.create_dataset("barcodes",
                         data=np.array([str(b).encode() for b in barcodes]))
        fg = g.create_group("features")
        gb = np.array([str(x).encode() for x in genes])
        fg.create_dataset("name", data=gb)
        fg.create_dataset("id",   data=gb)
        fg.create_dataset("feature_type",
                          data=np.array([b"Gene Expression"]*len(genes)))
        fg.create_dataset("genome", data=np.array([b"unknown"]*len(genes)))


def save_inputs(masks, gem, ox, oy):
    rx, ry, rg, rc, gem_genes, sx, sy = gem
    n_nuc = int(masks.max())
    P(f"\n  Generating Cellist inputs ({n_nuc:,} nuclei)...")

    # (1) nucleus_property.txt
    props = skimage.measure.regionprops_table(
        masks, properties=["label", "area", "centroid",
                           "equivalent_diameter_area"])
    prop = pd.DataFrame(props)
    prop["centroid-0"] += oy          # row + row offset -> global y
    prop["centroid-1"] += ox          # col + col offset -> global x

    # Restore the missing axis swap (was accidentally deleted at some point):
    # Segmentation.py:283 treats centroid-0 as x when comparing xmin/xmax,
    # but skimage.regionprops gives centroid-0 = row(y), centroid-1 = col(x).
    # Offsets must be added first (while still row/col semantics), then swap.
    # This missing step is why a 3x3 grid left only 2 patches and 64 cells.
    _c0 = prop["centroid-0"].copy()
    prop["centroid-0"] = prop["centroid-1"]   # -> real x
    prop["centroid-1"] = _c0                  # -> real y

    pp = "yourpath"
    prop.to_csv(pp, sep="\t", index=False)
    P(f"    OK nucleus_property.txt  {len(prop):,} nuclei  "
      f"label 1~{int(prop.label.max())}")

    # (2) nucleus_coord.txt
    nid_spot = spot_nid(masks, sx, sy, ox, oy)
    coord = pd.DataFrame({
        "x": sx, "y": sy,
        "x_y": [f"{a}_{b}" for a, b in zip(sx, sy)],
        "Nucleus":  (nid_spot > 0).astype(int),
        "Cellpose": np.where(nid_spot > 0, nid_spot.astype(float), np.nan),
    })
    cpth = "yourpath"
    coord.to_csv(cpth, sep="\t", index=False)
    n_in = int(coord.Nucleus.sum())
    P(f"    OK nucleus_coord.txt     {len(coord):,} spots  "
      f"in-nucleus {n_in:,} ({n_in/len(coord)*100:.1f}%)")

    # (3) nucleus_count.h5 - key fix in this version
    P(f"\n    Generating nucleus_count.h5 (previously used the stale 07-28 file -> SIGSEGV)")
    genes = bin1_genes()
    g2i = {g: i for i, g in enumerate(genes)}
    # GEM gene index -> bin1 gene index
    remap = np.full(len(gem_genes), -1, np.int64)
    for k, g in enumerate(gem_genes):
        j = g2i.get(g)
        if j is not None:
            remap[k] = j
    n_miss = int((remap < 0).sum())
    P(f"      GEM {len(gem_genes)} genes -> bin1 {len(genes)} genes, "
      f"missing {n_miss}")

    # Per-row GEM -> nucleus id
    rnid = spot_nid(masks, rx, ry, ox, oy)
    keep = (rnid > 0) & (remap[rg] >= 0)
    P(f"      in-nucleus GEM rows {int(keep.sum()):,} / {len(rx):,}")

    # Columns = all n_nuc labels (spotless nuclei are all-zero), barcodes = "1".."n"
    # so Cellist indexing by position or barcode never goes out of bounds
    full = csr_matrix(
        (rc[keep].astype(np.int32),
         (remap[rg[keep]], rnid[keep] - 1)),           # label 1-based -> col 0-based
        shape=(len(genes), n_nuc))

    # Mimic Cellist's own output (the 07-28 file that produced 3,845 cells):
    # 1) keep only nuclei with counts (that file had 4,828 cols / 71,356 nuclei, not all)
    # 2) barcode as float string '5406.0' - coord.txt's Cellpose column is float64;
    #    if Cellist matches via str(float), integer strings '5406' don't match
    #    (integer barcodes only produced 64 cells before)
    per_all = np.asarray(full.sum(axis=0)).ravel()
    cols = np.where(per_all > 0)[0]
    mat = full[:, cols]
    bcs = [f"{c + 1}.0" for c in cols]                 # label 1-based, float format
    P(f"      nuclei with counts {len(cols):,} / {n_nuc:,}  "
      f"(all-zero nuclei dropped, matching original file behavior)")
    P(f"      barcode format {bcs[:3]} ...  <- float strings")

    # Restore gene filtering (this version previously filtered cells only):
    # Cellist's own file has 16,693 genes, not bin1.h5's 20,654; the extra ~4,000
    # have zero counts across the surviving thousands of cells. Including them
    # makes get_hvg(...,1000) compute HVGs over zero-variance rows, mixing
    # all-zero genes into the HVGs -> soft assignment carries no information
    # -> this was the earlier 64-cell bug (reproduced here).
    gene_totals = np.asarray(mat.sum(axis=1)).ravel()   # sum over genes (axis=1)
    gk = np.where(gene_totals > 0)[0]
    mat = mat[gk, :]
    genes_kept = [genes[i] for i in gk]
    P(f"      gene filter: {len(genes):,} -> {len(genes_kept):,}"
      f" (dropped {len(genes)-len(genes_kept):,} genes with zero in-nucleus count)")


    cnt_path = "yourpath"
    write_10x_h5(cnt_path, mat, genes_kept, bcs)
    per = np.asarray(mat.sum(axis=0)).ravel()
    P(f"    OK nucleus_count.h5      {len(genes_kept):,}x{len(bcs):,}  (CSC)")
    P(f"      per-nucleus count: nonzero {int((per>0).sum()):,}  "
      f"median {np.median(per[per>0]):.0f}  "
      f"nuclei with >={NSPOT_MIN} {int((per>=NSPOT_MIN).sum()):,}")
    return pp, cpth, cnt_path


def run_seg(pp, cpth, cnt, radius, tag):
    outdir = "yourpath"
    os.makedirs(outdir, exist_ok=True)
    cmd = ["cellist", "seg", "--platform", "barcoding",
           "--gem", GEM,
           "--spot-count-h5", BIN1,
           "--nucleus-seg-method", "Cellpose",
           "--nucleus-seg",  cpth,
           "--nucleus-prop", pp,
           "--nucleus-count-h5", cnt,          # newly generated, not the 07-28 file
           "--cell-radius", str(int(radius)),
           "--gene-use", "HVG",
           "--outdir", outdir, "--outprefix", PREFIX]
    P(f"\n  cellist seg -> seg_{tag}/  (~4-8 min, real-time output)")
    P("  " + "-" * 60)
    rc = subprocess.run(cmd).returncode        # no redirect -> live output
    P("  " + "-" * 60)
    if rc != 0:
        P(f"    FAIL exit code {rc}")
        if rc == -11:
            P(f"    Still SIGSEGV. Check:")
            P(f"      python -c \"import tables,numpy as np;"
              f"h=tables.open_file('{cnt}');g=h.root.matrix;"
              f"print(tuple(g.shape[:]),len(g.barcodes[:]))\"")
            P(f"    shape[1] should equal the max label in the property file")
        return None
    f = glob.glob(f"{outdir}/**/*Cellist_segmentation_cell_coord.txt",
                  recursive=True)
    if not f:
        P("    FAIL no result file"); return None
    n = len(pd.read_csv(f[0], sep="\t"))
    P(f"    OK {n:,} cells  (GT {GT_CELLS:,}, ratio {n/GT_CELLS:.3f})")
    P(f"      Comparison: original whole-image cellpose route gave 3,845 cells")
    P(f"      {f[0]}")
    return f[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--diameters", type=int, nargs="+",
                    default=[20, 25, 30, 35])
    ap.add_argument("--cellprobs", type=float, nargs="+", default=[0.0])
    ap.add_argument("--diameter", type=int, default=30)
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--no-mask", action="store_true",
                    help="disable GEM mask (not recommended, pct drops to 21%%)")
    ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--run-seg", action="store_true")
    ap.add_argument("--cell-radius", type=int, default=13)
    ap.add_argument("--force", action="store_true", help="ignore all ckpts")
    cfg = ap.parse_args()
    use_mask = not cfg.no_mask

    P(SEP); P("Cellist x Stereo-seq"); P(SEP)
    P(f"  ckpt {CKPT}   mask {'on' if use_mask else 'off'}")
    gem = load_gem(cfg.force)
    sx, sy = gem[5], gem[6]
    crop, ox, oy = load_crop(sx, sy, cfg.force)
    gm = make_mask(crop.shape, sx, sy, ox, oy, cfg.dilate) if use_mask else None

    if cfg.scan:
        scan(crop, sx, sy, ox, oy, cfg.diameters, cfg.cellprobs,
             gm, use_mask, cfg.force)
        return

    P(f"\n  Cellpose diameter={cfg.diameter} cellprob={cfg.cellprob} ...")
    masks = run_cellpose(crop, cfg.diameter, cfg.cellprob, gm,
                         use_mask, cfg.force)
    n, n_has, alive, med = stats(masks, sx, sy, ox, oy)
    P(f"    nuclei {n:,}  nuclei with spots {n_has:,} ({n_has/max(n,1)*100:.1f}%)  "
      f"alive {alive:,}  median nSpot {med:.0f}  GT {GT_CELLS:,}")
    if cfg.save:
        pp, cpth, cnt = save_inputs(masks, gem, ox, oy)
        if cfg.run_seg:
            run_seg(pp, cpth, cnt,
                    cfg.cell_radius, f"d{cfg.diameter}_r{cfg.cell_radius}")
        else:
            P("\n  To run seg: add --run-seg")
    else:
        P("    --save not set, not writing files")


if __name__ == "__main__":
    main()