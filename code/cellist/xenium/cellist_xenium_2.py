#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cellist x Xenium breast (xenium_2) - official nucleus_boundaries as nucleus prior
==================================================================================
ROI: full-slide center 5000x5000 px = 1062.5x1062.5 um, origin pixel (24497, 11210)

Historical issues and fixes:
  1. Axis order: skimage centroid-0 is row(y) but Cellist treats it as x.
     Without swapping, nuclei and spots land in different patches; on a 3x3 grid
     only the diagonal matched (36.8% same-patch), producing 5 Data_HVG patches
     and 2,497 cells with an "upper-left + lower-right" spatial pattern.
     Fixed in build_inputs.
  2. Seeds: earlier versions ran Cellpose on the fly (diameter=30px=6.4um, too
     small) yielding only 3,392 nuclei and 955 cells. This version uses 8,390
     nuclei rasterized from nucleus_boundaries, close to GT 8,809. Output files
     use the nucprior_* prefix to distinguish from the old nucleus_* files.
  3. Input paths: early versions passed None to --nucleus-prop and errored;
     this version validates every input before invoking the CLI.
  4. Integrity: after the run, hard-check patch count, tmp/Sub count, and
     whether the 10x10 grid has holes; return no result if any check fails, to
     keep incomplete artifacts out of downstream steps.

Usage:
  python cellist_xenium_2.py --check-only               # nSpot estimate only
  python cellist_xenium_2.py --cell-radius 30 --force   # full run
  python cellist_xenium_2.py --scan-radii 20,30,40 --force
"""

import os, glob, argparse, subprocess
import numpy as np
import pandas as pd
import cv2
import h5py
import tifffile
import skimage.measure
from scipy.sparse import csr_matrix

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

DATA_DIR = "yourpath"
OUT_DIR  = "yourpath"
PREFIX   = "xenium_breast"

PIXEL_SIZE = 0.2125
CX_UM = 53994 / 2 * PIXEL_SIZE
CY_UM = 27420 / 2 * PIXEL_SIZE
HALF  = 5000 * PIXEL_SIZE / 2
ROI_X0, ROI_X1 = CX_UM - HALF, CX_UM + HALF
ROI_Y0, ROI_Y1 = CY_UM - HALF, CY_UM + HALF
IMG_W = IMG_H = 5000
GT_CELLS  = 8809
NSPOT_MIN = 20        # hard-coded in Segmentation.py:287
EXPECT_PATCHES = 9    # 3x3 grid

os.makedirs(OUT_DIR, exist_ok=True)


def write_10x_h5(path, mat, genes, barcodes):
    """CSC, shape=(n_genes,n_cells), indptr splits by column/cell."""
    m = mat.tocsc()
    assert len(m.indptr) == len(barcodes)+1, \
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


# (1) nucleus_boundaries -> mask
def build_mask(force=False):
    mp = "yourpath"
    if os.path.exists(mp) and not force:
        m = tifffile.imread(mp)
        P(f"[1] Reusing mask: {int(m.max()):,} nuclei")
        return m
    P("[1] nucleus_boundaries -> mask")
    nb = pd.read_parquet("yourpath")
    P(f"    Full slide {len(nb):,} vertices")
    cen = nb.groupby("cell_id")[["vertex_x","vertex_y"]].mean()
    keep = cen.index[(cen.vertex_x >= ROI_X0) & (cen.vertex_x < ROI_X1) &
                     (cen.vertex_y >= ROI_Y0) & (cen.vertex_y < ROI_Y1)]
    nb = nb[nb.cell_id.isin(set(keep))]
    P(f"    Nuclei in ROI {len(keep):,}  (GT {GT_CELLS:,}, "
      f"coverage {len(keep)/GT_CELLS*100:.0f}%)")
    # keep filters by nucleus centroid but fillPoly draws the full polygon;
    # nuclei whose centroid is inside but some vertices are outside get clipped
    # to the border, so a thin edge ring is slightly distorted (few cells).
    mask = np.zeros((IMG_H, IMG_W), np.int32)
    for i, (_, grp) in enumerate(nb.groupby("cell_id", sort=False), start=1):
        px = ((grp.vertex_x.values-ROI_X0)/PIXEL_SIZE).astype(np.int32)
        py = ((grp.vertex_y.values-ROI_Y0)/PIXEL_SIZE).astype(np.int32)
        pts = np.stack([np.clip(px,0,IMG_W-1),
                        np.clip(py,0,IMG_H-1)], 1).reshape(-1,1,2)
        cv2.fillPoly(mask, [pts], int(i))
    ids, cnt = np.unique(mask[mask>0], return_counts=True)
    P(f"    Rasterized {len(ids):,} nuclei  median area {np.median(cnt):.0f}px "
      f"(diameter {2*np.sqrt(np.median(cnt)/np.pi)*PIXEL_SIZE:.1f}um)")
    tifffile.imwrite(mp, mask)
    return mask


def load_tx():
    P("[2] Transcripts (ROI filter)")
    tx = pd.read_parquet("yourpath",
                         columns=["feature_name","x_location",
                                  "y_location","qv"])
    tx = tx[(tx.qv >= 20) &
            (tx.x_location >= ROI_X0) & (tx.x_location < ROI_X1) &
            (tx.y_location >= ROI_Y0) & (tx.y_location < ROI_Y1)]
    tx = tx[~tx.feature_name.str.startswith(
        ("NegControl","Unassigned","BLANK","antisense"), na=False)].copy()
    tx["px"] = ((tx.x_location-ROI_X0)/PIXEL_SIZE).astype(int).clip(0,IMG_W-1)
    tx["py"] = ((tx.y_location-ROI_Y0)/PIXEL_SIZE).astype(int).clip(0,IMG_H-1)
    P(f"    {len(tx):,} rows  {tx.feature_name.nunique()} genes")
    return tx.reset_index(drop=True)


def check_alive(mask, tx):
    nid = mask[tx.py.values, tx.px.values]
    spot = pd.DataFrame({"nid": nid,
                         "k": tx.py.values.astype(np.int64)*IMG_W + tx.px.values})
    spot = spot[spot.nid > 0].drop_duplicates("k")
    ns = spot.groupby("nid").size()
    n_tot = int(mask.max())
    P(f"\n[3] nSpot estimate (hard-coded >= {NSPOT_MIN} in Segmentation.py:287)")
    P(f"    total nuclei          {n_tot:,}")
    P(f"    nuclei with >=1 spot {len(ns):,} ({len(ns)/max(n_tot,1)*100:.1f}%)")
    if len(ns):
        P(f"    nSpot median {ns.median():.0f}  p25 {ns.quantile(.25):.0f}")
        for th in (5, 10, 20):
            P(f"      nSpot>={th:>2}: {int((ns>=th).sum()):>6,} alive"
              f"{'  <- Cellist uses this' if th==NSPOT_MIN else ''}")
    alive = int((ns >= NSPOT_MIN).sum()) if len(ns) else 0
    P(f"    -> global alive ~ {alive:,} (GT {GT_CELLS:,}, "
      f"{alive/GT_CELLS*100:.0f}%)")
    P(f"    NOTE Cellist recomputes nSpot inside each patch; before the axis fix "
      f"only 2,497 cells were produced")
    return alive


# (2) Cellist inputs
def build_inputs(mask, tx, force=False):
    paths = dict(
        prop  = "yourpath",
        coord = "yourpath",
        gem   = "yourpath",
        spot  = "yourpath",
        count = "yourpath",
    )
    P("\n[4] Building Cellist inputs")
    # prop
    props = skimage.measure.regionprops_table(
        mask, properties=["label","area","centroid",
                          "equivalent_diameter_area"])
    prop = pd.DataFrame(props)
    # Axis-order fix: Cellist treats centroid-0 as x; skimage gives row(y).
    # Without the swap, nuclei and spots land in different patches (36.8% same-patch).
    _c0 = prop["centroid-0"].copy()
    prop["centroid-0"] = prop["centroid-1"]   # -> x / col
    prop["centroid-1"] = _c0                  # -> y / row
    P(f"    centroid axis order swapped")
    P(f"      centroid-0(x) [{prop['centroid-0'].min():.0f},"
      f"{prop['centroid-0'].max():.0f}]  "
      f"centroid-1(y) [{prop['centroid-1'].min():.0f},"
      f"{prop['centroid-1'].max():.0f}]")
    prop.to_csv(paths["prop"], sep="\t", index=False)
    P(f"    OK prop.txt   {len(prop):,} nuclei (tab-separated, has label)")

    # coord (x/y unchanged)
    uniq = tx.drop_duplicates(subset=["px","py"])
    sx, sy = uniq.px.values, uniq.py.values
    nid = mask[sy, sx].astype(float)
    pd.DataFrame({
        "x": sx, "y": sy,
        "x_y": [f"{a}_{b}" for a,b in zip(sx,sy)],
        "Nucleus":  (nid > 0).astype(int),
        "Cellpose": np.where(nid > 0, nid, np.nan),
    }).to_csv(paths["coord"], sep="\t", index=False)
    P(f"    OK coord.txt  {len(uniq):,} spots  "
      f"in-nucleus {int((nid>0).sum()):,} ({(nid>0).mean()*100:.1f}%)")

    # gem
    gem = (tx.groupby(["feature_name","px","py"]).size()
             .reset_index(name="MIDCount")
             .rename(columns={"feature_name":"geneID","px":"x","py":"y"}))
    gem[["geneID","x","y","MIDCount"]].to_csv(
        paths["gem"], sep="\t", index=False)
    P(f"    OK gem.txt    {len(gem):,} rows (required even in imaging mode)")

    # two h5 files
    genes = sorted(tx.feature_name.unique())
    gi = tx.feature_name.map({g:i for i,g in enumerate(genes)}).values
    skey = tx.py.values.astype(np.int64)*IMG_W + tx.px.values
    su = np.unique(skey)
    rows = pd.Series(skey).map({s:i for i,s in enumerate(su)}).values
    write_10x_h5(paths["spot"],
                 csr_matrix((np.ones(len(tx), np.int32), (gi, rows)),
                            shape=(len(genes), len(su))),
                 genes, [f"{s%IMG_W}_{s//IMG_W}" for s in su])
    P(f"    OK spot_count.h5     {len(genes)}x{len(su):,}  (CSC)")

    n_nuc = int(mask.max())
    tnid = mask[tx.py.values, tx.px.values]
    inn = tnid > 0
    full = csr_matrix((np.ones(int(inn.sum()), np.int32),
                       (gi[inn], tnid[inn]-1)),
                      shape=(len(genes), n_nuc))
    gk = np.where(np.asarray(full.sum(axis=1)).ravel() > 0)[0]
    write_10x_h5(paths["count"], full[gk,:], [genes[i] for i in gk],
                 [str(i) for i in range(1, n_nuc+1)])
    P(f"    OK nucleus_count.h5  {len(gk)}x{n_nuc:,}  (CSC)")
    P(f"      gene filter {len(genes)} -> {len(gk)} (dropped zero in-nucleus count)")
    return paths


# (3) Segmentation + integrity check
def verify_output(outdir, expect=EXPECT_PATCHES):
    """Verify patch integrity and spatial coverage; True only if usable."""
    props = sorted(glob.glob("yourpath"))
    subs = sorted(glob.glob("yourpath",
                            recursive=True))
    P(f"      Data_HVG patches {len(props)}, tmp/Sub outputs {len(subs)} "
      f"(3x3 grid should be {expect} each)")
    if len(subs) < expect or len(props) != len(subs):
        want = {os.path.basename(p).replace("_property.txt", "")
                .replace("Patch_", "") for p in props}
        got = {os.path.basename(s).replace("_soft_clustering.txt", "")
               .replace("Sub_", "") for s in subs}
        P(f"    FAIL: incomplete patches, missing: {sorted(want - got) or '(also missing on input side)'}")
        P(f"      This is a typical sign of axis-order or chunking issues; the "
          f"result will contain whole-block holes")
        return False

    seg = glob.glob("yourpath", recursive=True)
    if not seg:
        P("    FAIL: no segmentation.txt")
        return False
    s = pd.read_csv(seg[0], sep="\t", usecols=["x", "y", "Cellist"])
    ok = (pd.to_numeric(s["Cellist"], errors="coerce") > 0).values
    bx = np.clip((s.x.values / IMG_W * 10).astype(int), 0, 9)
    by = np.clip((s.y.values / IMG_H * 10).astype(int), 0, 9)
    rate = pd.DataFrame({"bx": bx, "by": by, "ok": ok}) \
        .groupby(["by", "bx"])["ok"].mean().unstack()
    dead = int((rate.fillna(0) < 0.05).sum().sum())
    P(f"      Overall assignment rate {ok.mean():.1%}; 10x10 grid median "
      f"{np.nanmedian(rate.values):.1%}, cells below 5%: {dead}/100")
    if dead > 5:
        P(f"    FAIL: {dead} empty cells, result not usable")
        return False
    return True


def run_seg(paths, radius, force=False):
    tag = f"nucprior_r{radius}_fix"
    outdir = "yourpath"
    done = glob.glob("yourpath",
                     recursive=True)
    if done and not force:
        n = len(pd.read_csv(done[0], sep="\t"))
        P(f"\n  Existing result: {n:,} cells  add --force to rerun")
        return done[0]
    os.makedirs(outdir, exist_ok=True)

    # Validate every input before invoking the CLI; early versions passed None
    # to --nucleus-prop and only errored at run time.
    P(f"\n[5] Checking Cellist inputs")
    for k, v in paths.items():
        if not (v and os.path.exists(v)):
            P(f"    FAIL: missing input: {k} -> {v}")
            return None
        P(f"      {k:<6} {os.path.getsize(v)/1e6:>8.1f} MB  {os.path.basename(v)}")

    cmd = ["cellist", "seg", "--platform", "imaging",
           "--resolution", str(PIXEL_SIZE),
           "--gem", paths["gem"], "--spot-count-h5", paths["spot"],
           "--nucleus-seg-method", "Cellpose",
           "--nucleus-seg", paths["coord"], "--nucleus-prop", paths["prop"],
           "--nucleus-count-h5", paths["count"],
           "--cell-radius", str(int(radius)),
           "--gene-use", "HVG",
           "--outdir", outdir, "--outprefix", PREFIX]
    P(f"\n  cellist seg -> seg_{tag}/  (~5-15 min, live output)")
    P("  " + " ".join(cmd))
    P("  " + "-" * 60)
    rc = subprocess.run(cmd).returncode
    P("  " + "-" * 60)
    if rc != 0:
        P(f"    FAIL exit code {rc}")
        return None

    f = glob.glob("yourpath",
                  recursive=True)
    if not f:
        P("    FAIL: no result file")
        return None
    n = len(pd.read_csv(f[0], sep="\t"))
    P(f"    OK {n:,} cells  (GT {GT_CELLS:,}, ratio {n/GT_CELLS:.3f})")
    P(f"      Before axis fix: 2,497 (ratio 0.283, F1=0.4407); "
      f"older Cellpose seeding gave only 955")
    if not verify_output(outdir):
        P(f"    FAIL: integrity check did not pass, not returning this result")
        return None
    P(f"      {f[0]}")
    return f[0]


def main():
    ap = argparse.ArgumentParser()
    # cell-radius unit is pixels. Mask nuclei median equivalent diameter is
    # 5.66 um (nucleus radius ~2.8 um); radius 10 px = 2.1 um gives a cell
    # radius under 5 um, clearly smaller than breast cells. 30 px = 6.4 um plus
    # nucleus radius ~9 um matches actual size better.
    ap.add_argument("--cell-radius", type=int, default=30)
    ap.add_argument("--scan-radii", default=None,
                    help="comma separated; try several radii, e.g. 20,30,40")
    ap.add_argument("--check-only", action="store_true",
                    help="stop after the nSpot estimate")
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args()

    P(SEP); P("Cellist x Xenium breast (nucleus_boundaries prior)"); P(SEP)
    P(f"  ROI x[{ROI_X0:.1f},{ROI_X1:.1f}] y[{ROI_Y0:.1f},{ROI_Y1:.1f}] um")
    P(f"  property centroid axis order fixed (upstream bug in Segmentation.py:283)")
    P(f"  outputs use nucprior_* prefix to distinguish from old Cellpose nucleus_*")

    mask = build_mask(cfg.force)
    tx = load_tx()
    check_alive(mask, tx)
    if cfg.check_only:
        P(f"\n  --check-only, stopping here")
        return

    paths = build_inputs(mask, tx, cfg.force)
    radii = ([int(r) for r in cfg.scan_radii.split(",")]
             if cfg.scan_radii else [cfg.cell_radius])
    results = []
    for r in radii:
        out = run_seg(paths, r, cfg.force)
        if out:
            n = len(pd.read_csv(out, sep="\t"))
            results.append((r, n, n / GT_CELLS, out))
    if len(results) > 1:
        P(f"\n[6] Comparison across radii (GT {GT_CELLS:,})")
        for r, n, ratio, _ in results:
            P(f"    r={r:<3} {n:>7,} cells  ratio {ratio:.3f}")
    if not results:
        P(f"\n  FAIL: no radius produced a usable result")
    P(f"\n  Evaluate: python eval_cellist_breast.py")


if __name__ == "__main__":
    main()