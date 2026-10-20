#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DISSECT x stereo_1 (Stereo-seq E16.5_E2S6, GT 6229) - inspect / register / prep / run / merge

Default --mode native: use the DISSECT native stereoseq entry (dissect_native_runner.py),
  ssDNA is sampled on the DNB grid (1 px = 1 DNB = 0.5 um); GEM is shifted to the image origin and
  MIDCount weights are kept, no expansion;
  in the source, gene_dense assigns instead of accumulating, so expanded duplicate molecules
  overwrite each other; only the native entry therefore keeps the counts correctly.
--mode xenium: the fake-xenium pipeline described below, for comparison only.
dissect_native_runner.py and wsi_register.py must sit in the same directory as this script.

Same idea as dissect_merfish_*.py / dissect_cosmx_*.py: disguise Stereo-seq as xenium:
  register : auto-register the centers of CellBin cells in GEM (DNB coordinates) to the ssDNA image
             (orientation / scale / offset; GEM coordinates of E14.5 and E2S7 fall outside the image,
             so registration is required)
  prep     : build a grid on the GEM bounding box at 0.2125 um/px and sample an image from the
             ssDNA image with the same orientation as GEM;
             expand each bin's MIDCount into that many molecules, take the bin center as coordinate,
             write xenium-schema parquet
             (same as the BOMS x Stereo-seq handling; DISSECT's gradient field uses molecule counts,
             so expansion is an approximation)
  run      : call the validated official wrapper run_xenium_pancreas_dissect.py
  merge    : label image -> cell table (DNB / um coordinates) + cell assignment of each GEM bin + QC
wsi_register.py must sit in the same directory as this script.

Usage:
  python dissect_stereo_1.py --stage inspect
  python dissect_stereo_1.py --stage register            # check QC image register_qc.png
  python dissect_stereo_1.py --stage all --n-jobs 4
"""
import argparse, glob, gzip, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, tifffile
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ===== hard-coded dataset constants =====
TAG       = "stereo_1"
SAMPLE    = "E16.5_E2S6"
DATA_DIR  = Path("yourpath/dataset/stereoseq/stereo-seq_data_all")
GEM       = DATA_DIR / f"{SAMPLE}_Dorsal_Midbrain_GEM_CellBin.tsv.gz"
IMAGE     = DATA_DIR / f"{SAMPLE}.tif"
OUT       = Path("yourpath/dissect_stereo_1_" + SAMPLE)
GT_CELLS  = 6229
BIN_UM    = 0.5               # 1 DNB = 0.5 um
TARGET_UM = 0.2125            # DISSECT/xenium detector training scale
MARGIN_DNB = 20               # extra DNB margin outside the GEM bounding box
ENGINE    = Path("yourpath/DISSECT/code/run_xenium_pancreas_dissect.py")

P = lambda *a: print(*a, flush=True)
SEP = "=" * 78


def read_gem():
    skip = 0
    with gzip.open(GEM, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    g = pd.read_csv(GEM, sep="\t", skiprows=skip)
    ren = {}
    for c in g.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl in ("midcounts", "midcount", "umicount", "count"):
            ren[c] = "count"
        elif cl in ("cell", "cellid", "cell_id", "label"):
            ren[c] = "cell"
    g = g.rename(columns=ren)
    if "count" not in g.columns:
        g["count"] = 1
    return g


def cell_centers(g):
    c = g[g.cell != 0].groupby("cell")[["x", "y"]].mean()
    return c.to_numpy(float) + 0.5          # continuous DNB coordinate (bin k covers [k, k+1))


def grid_origin(g):
    return float(g.x.min() - MARGIN_DNB), float(g.y.min() - MARGIN_DNB), \
        float(g.x.max() + 1 + MARGIN_DNB), float(g.y.max() + 1 + MARGIN_DNB)



def area_limits(a, um_per_px):
    """Detection box area threshold (px²). Default is the same 350-1600 used by other DISSECT
    benchmark runs; --area-rule physical converts to the physical-size equivalent of Xenium
    (0.2125 um/px). Explicit --minarea/--maxarea take priority."""
    f = (0.2125 / um_per_px) ** 2 if getattr(a, "area_rule", "fixed") == "physical" else 1.0
    lo = a.minarea if a.minarea is not None else int(round(350 * f))
    hi = a.maxarea if a.maxarea is not None else int(round(1600 * f))
    return lo, hi


def report_box_coverage(areas_px, lo, hi, tag="", um_per_px=None):
    areas_px = np.asarray(areas_px, float)
    areas_px = areas_px[np.isfinite(areas_px) & (areas_px > 0)]
    if not len(areas_px):
        return
    q = np.percentile(areas_px, [5, 25, 50, 75, 95])
    P(f"  reference cell bounding-box area{tag} (px²): 5%={q[0]:.0f} 25%={q[1]:.0f} median={q[2]:.0f} 75%={q[3]:.0f} 95%={q[4]:.0f}")
    cands = [("current setting", lo, hi), ("numeric default 350–1600", 350, 1600), ("library default 600–1200", 600, 1200)]
    if um_per_px:
        f = (0.2125 / um_per_px) ** 2
        cands.append(("physical equivalent Xenium", int(round(350 * f)), int(round(1600 * f))))
    cands.append(("reference 5–95% quantile", int(q[0]), int(q[4])))
    for name, l, h in cands:
        inside = float(((areas_px >= l) & (areas_px <= h)).mean())
        P(f"    {name:<16} [{l:>6},{h:>6}]  coverage {inside:6.1%}")
    inside = float(((areas_px >= lo) & (areas_px <= hi)).mean())
    if inside < 0.5:
        P("  [Warning] with the current setting most reference cell bounding boxes fall outside the detection area range; consider adjusting --minarea/--maxarea")


# ---------------------------------------------------------------- native stereoseq entry
RUNNER = Path(__file__).resolve().parent / "dissect_native_runner.py"


def dirs(a):
    if a.mode == "native":
        return OUT / "input_native", OUT / "dissect_out_native", "_native"
    return OUT / "input", OUT / "dissect_out", ""


def stage_prep_native(a):
    """Aligned ssDNA on the DNB grid (1 px = 1 DNB = 0.5 um) + shifted GEM (geneID/x/y/MIDCount).
    The DISSECT stereoseq branch treats x, y directly as pixel indices and writes MIDCount as weight
    into gene_dense, without expansion."""
    import wsi_register as wr
    P(f"\n{SEP}\nPREP (native stereoseq)  {TAG} {SAMPLE}\n{SEP}")
    inp, _, _ = dirs(a); inp.mkdir(parents=True, exist_ok=True)
    img_f, gem_f, meta_f = inp / "ssdna_dnb.tif", inp / "gem_native.tsv.gz", inp / "meta.json"
    if img_f.exists() and gem_f.exists() and meta_f.exists() and not a.force_prep:
        P("  reuse prep cache"); return
    if not (OUT / "register.json").exists():
        raise SystemExit("Missing register.json: run --stage register first")
    T = wr.load(OUT / "register.json")
    g = read_gem()
    x0, y0, x1, y1 = grid_origin(g)
    ow, oh = int(x1 - x0), int(y1 - y0)
    P(f"  GEM bounding box DNB x[{x0:.0f},{x1:.0f}) y[{y0:.0f},{y1:.0f})  ->  image {oh}x{ow} px @ {BIN_UM} um/px (1 px = 1 DNB)")

    def grid(rr, cc):
        return x0 + cc + 0.5, y0 + rr + 0.5          # pixel center = DNB center
    w = wr.WSI(T["image"], a.channel)
    img, info = wr.sample(w, T, grid, (oh, ow))
    P(f"  sample: whole-slide level {info['level']}  coverage {info['inside']*100:.1f}%")
    sm = np.clip(img, 0, None); mx = float(sm.max())
    tifffile.imwrite(img_f, (sm / mx * 255).astype(np.uint8) if mx > 0 else sm.astype(np.uint8))
    c = cell_centers(g)
    rr = np.clip((c[:, 1] - y0).astype(int), 0, oh - 1)
    cc = np.clip((c[:, 0] - x0).astype(int), 0, ow - 1)
    enr = float(sm[rr, cc].mean() / (sm.mean() + 1e-6))
    P(f"  cell-center enrichment on the DNB grid {enr:.3f}")
    out = pd.DataFrame({"geneID": g.gene.astype(str).to_numpy(),
                        "x": (g.x.to_numpy() - x0).astype(np.int64),
                        "y": (g.y.to_numpy() - y0).astype(np.int64),
                        "MIDCount": g["count"].clip(lower=1).astype(np.int64).to_numpy()})
    dup = int(out.duplicated(["geneID", "x", "y"]).sum())
    if dup:
        # gene_dense assigns (does not accumulate), so duplicate rows of the same gene in the
        # same bin overwrite each other: merge them first
        out = out.groupby(["geneID", "x", "y"], as_index=False)["MIDCount"].sum()
        P(f"  merged duplicate rows for the same gene in the same bin: {dup:,}")
    out.to_csv(gem_f, sep="\t", index=False, compression="gzip")
    P(f"  GEM rows {len(out):,}  MIDCount total {int(out.MIDCount.sum()):,}  genes {out.geneID.nunique():,} (no expansion)")
    bb = g[g.cell != 0].groupby("cell").agg(x0=("x", "min"), x1=("x", "max"), y0=("y", "min"), y1=("y", "max"))
    lo, hi = area_limits(a, BIN_UM)
    report_box_coverage(((bb.x1 - bb.x0 + 1) * (bb.y1 - bb.y0 + 1)).to_numpy(), lo, hi, " (CellBin, 0.5 um/px)", BIN_UM)
    json.dump(dict(tag=TAG, sample=SAMPLE, mode="native", gem=str(GEM), image=str(T["image"]),
                   grid_origin_dnb=[x0, y0], dnb_per_px=1.0, shape=[oh, ow],
                   bin_um=BIN_UM, target_um=BIN_UM, enrichment=enr,
                   register_enrichment=T.get("enrichment"), gt_cells=GT_CELLS,
                   n_rows=int(len(out))), open(meta_f, "w"), indent=2)
    P("  PREP DONE")


def stage_run_native(a):
    P(f"\n{SEP}\nRUN (native DISSECT stereoseq entry)  {TAG} {SAMPLE}\n{SEP}")
    inp, od, _ = dirs(a)
    od.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(RUNNER), "--platform", "stereoseq",
           "--image", str(inp / "ssdna_dnb.tif"), "--genes", str(inp / "gem_native.tsv.gz"),
           "--output", str(od), "--n-jobs", str(a.n_jobs),
           "--minarea", str(area_limits(a, BIN_UM)[0]), "--maxarea", str(area_limits(a, BIN_UM)[1]),
           "--threshold", str(a.threshold), "--stride", str(a.stride)]
    if a.engine_extra:
        cmd += a.engine_extra.split()
    P("  " + " ".join(cmd))
    if a.dry:
        return
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"runner exit code {rc}")
    P("  RUN DONE")


# ---------------------------------------------------------------- inspect
def stage_inspect(a):
    import wsi_register as wr
    P(f"\n{SEP}\nINSPECT  {TAG} {SAMPLE}\n{SEP}")
    g = read_gem()
    n = g[g.cell != 0].cell.nunique()
    P(f"  GEM {GEM.name}: rows {len(g):,}  counts {int(g['count'].sum()):,}  cells {n:,} (GT {GT_CELLS:,})"
      f"  {'match' if n == GT_CELLS else 'mismatch'}")
    P(f"  x {g.x.min()}–{g.x.max()}  y {g.y.min()}–{g.y.max()}")
    P("  image: " + wr.WSI(IMAGE).describe())


# ---------------------------------------------------------------- register
def stage_register(a):
    import wsi_register as wr
    P(f"\n{SEP}\nREGISTER (CellBin cell centers -> ssDNA)  {TAG} {SAMPLE}\n{SEP}")
    g = read_gem()
    pts = cell_centers(g)
    w = wr.WSI(IMAGE, a.channel)
    P("  image: " + w.describe())
    scales = [a.scale] if a.scale else [0.5, 1.0, 2.0]
    T = wr.register(pts, w, scales, log=P)
    T["image"] = str(IMAGE)
    wr.save(T, OUT / "register.json")
    wr.qc_png(w, T, pts, OUT / "register_qc.png")
    P(f"  -> {OUT/'register.json'}   QC {OUT/'register_qc.png'}")
    if T["enrichment"] < 1.2:
        P("  [Warning] cell-center enrichment < 1.2, registration likely failed; check the QC image and set --scale if needed")


# ---------------------------------------------------------------- prep
def stage_prep(a):
    import wsi_register as wr
    if a.mode == "native":
        return stage_prep_native(a)
    P(f"\n{SEP}\nPREP  {TAG} {SAMPLE}\n{SEP}")
    inp = OUT / "input"; inp.mkdir(parents=True, exist_ok=True)
    img_f, tx_f, meta_f = inp / "dapi_roi.tif", inp / "transcripts.parquet", inp / "meta.json"
    if img_f.exists() and tx_f.exists() and meta_f.exists() and not a.force_prep:
        P("  reuse prep cache"); return
    if not (OUT / "register.json").exists():
        raise SystemExit("Missing register.json: run --stage register first")
    T = wr.load(OUT / "register.json")
    g = read_gem()
    x0, y0, x1, y1 = grid_origin(g)
    k = TARGET_UM / BIN_UM                       # DNB per output pixel
    ow, oh = int(np.ceil((x1 - x0) / k)), int(np.ceil((y1 - y0) / k))
    P(f"  GEM bounding box DNB x[{x0:.0f},{x1:.0f}) y[{y0:.0f},{y1:.0f})  ->  image {oh}x{ow} px @ {TARGET_UM} um/px")

    def grid(rr, cc):
        return x0 + cc * k, y0 + rr * k
    w = wr.WSI(T["image"], a.channel)
    img, info = wr.sample(w, T, grid, (oh, ow))
    P(f"  sample: whole-slide level {info['level']}  coverage {info['inside']*100:.1f}%")
    sm = np.clip(img, 0, None); mx = float(sm.max())
    tifffile.imwrite(img_f, (sm / mx * 65535).astype(np.uint16) if mx > 0 else sm.astype(np.uint16))

    # enrichment check: signal at cell centers on the grid
    c = cell_centers(g)
    rr = np.clip(((c[:, 1] - y0) / k).astype(int), 0, oh - 1)
    cc = np.clip(((c[:, 0] - x0) / k).astype(int), 0, ow - 1)
    enr = float(sm[rr, cc].mean() / (sm.mean() + 1e-6))
    P(f"  cell-center enrichment on the grid {enr:.3f}")

    rep = g.loc[g.index.repeat(g["count"].clip(lower=1).astype(int))]
    df = pd.DataFrame({"x_location": (rep.x.to_numpy(float) + 0.5 - x0) * BIN_UM,
                       "y_location": (rep.y.to_numpy(float) + 0.5 - y0) * BIN_UM,
                       "feature_name": rep.gene.astype(str).to_numpy()})
    df["z_location"] = 0.0; df["qv"] = 40.0
    df["cell_id"] = "UNASSIGNED"; df["overlaps_nucleus"] = 0
    df["transcript_id"] = np.arange(len(df), dtype=np.int64)
    df.to_parquet(tx_f, index=False)
    P(f"  GEM rows {len(g):,} -> expanded molecules {len(df):,}  genes {df.feature_name.nunique():,}")
    json.dump(dict(tag=TAG, sample=SAMPLE, gem=str(GEM), image=str(T["image"]),
                   grid_origin_dnb=[x0, y0], dnb_per_px=k, shape=[oh, ow],
                   bin_um=BIN_UM, target_um=TARGET_UM, enrichment=enr,
                   register_enrichment=T.get("enrichment"), gt_cells=GT_CELLS,
                   n_molecules=int(len(df))), open(meta_f, "w"), indent=2)
    P("  PREP DONE")


# ---------------------------------------------------------------- run
def stage_run(a):
    if a.mode == "native":
        return stage_run_native(a)
    P(f"\n{SEP}\nRUN (DISSECT engine)  {TAG} {SAMPLE}\n{SEP}")
    if not ENGINE.exists():
        raise SystemExit(f"Missing engine: {ENGINE}")
    inp, od = OUT / "input", OUT / "dissect_out"
    od.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ENGINE),
           "--image", str(inp / "dapi_roi.tif"),
           "--transcripts", str(inp / "transcripts.parquet"),
           "--output", str(od),
           "--n-jobs", str(a.n_jobs),
           "--minarea", str(area_limits(a, TARGET_UM)[0]),
           "--maxarea", str(area_limits(a, TARGET_UM)[1]),
           "--refine-batch-size", str(a.refine_batch_size),
           "--threshold", str(a.threshold),
           "--stride", str(a.stride)]
    if a.engine_extra:
        cmd += a.engine_extra.split()
    P("  " + " ".join(cmd))
    if a.dry:
        return
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"engine exit code {rc}")
    P("  RUN DONE")


# ---------------------------------------------------------------- merge
def stage_merge(a):
    P(f"\n{SEP}\nMERGE / QC  {TAG} {SAMPLE}\n{SEP}")
    inp, od, sfx = dirs(a)
    meta = json.load(open(inp / "meta.json"))
    lab = None
    for pat in ("*mask*.tif", "*segmentation*.tif", "*label*.tif", "*mask*.npy", "*label*.npy"):
        h = sorted(glob.glob(str(od / "**" / pat), recursive=True))
        if h:
            lab = np.load(h[0]) if h[0].endswith(".npy") else tifffile.imread(h[0])
            lab = np.squeeze(lab).astype(np.int64)
            P(f"  label image: {h[0]}  {lab.shape}"); break
    if lab is None:
        raise SystemExit(f"no label image, dissect_out contents: {[Path(p).name for p in glob.glob(str(od/'*'))][:20]}")
    x0, y0 = meta["grid_origin_dnb"]; k = meta["dnb_per_px"]; um = meta["target_um"]
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    ys, xs = np.nonzero(lab)
    v = lab[ys, xs]; o = np.argsort(v, kind="stable")
    v, ys, xs = v[o], ys[o], xs[o]
    b = np.append(np.searchsorted(v, ids), len(v))
    cy = np.array([ys[b[i]:b[i + 1]].mean() for i in range(len(ids))]) + .5
    cx = np.array([xs[b[i]:b[i + 1]].mean() for i in range(len(ids))]) + .5
    cells = pd.DataFrame({"cell_id": ids,
                          "center_x_dnb": x0 + cx * k, "center_y_dnb": y0 + cy * k,
                          "center_x_um": (x0 + cx * k) * meta["bin_um"],
                          "center_y_um": (y0 + cy * k) * meta["bin_um"],
                          "area_um2": cnt * um * um})
    cells.to_csv(OUT / f"dissect_cells{sfx}.csv", index=False)
    tifffile.imwrite(OUT / f"dissect_segmentation{sfx}.tif", lab.astype(np.uint32), compression="zlib")

    g = read_gem()
    rr = np.clip(((g.y.to_numpy(float) + 0.5 - y0) / k).astype(int), 0, lab.shape[0] - 1)
    cc = np.clip(((g.x.to_numpy(float) + 0.5 - x0) / k).astype(int), 0, lab.shape[1] - 1)
    g["dissect_cell"] = lab[rr, cc]
    try:
        g.to_parquet(OUT / f"dissect_gem_assignment{sfx}.parquet", index=False)
    except Exception:
        g.to_csv(OUT / f"dissect_gem_assignment{sfx}.csv.gz", index=False)
    w = g["count"].to_numpy(float)
    rate = float((w * (g.dissect_cell > 0)).sum() / w.sum())
    ratio = len(cells) / max(GT_CELLS, 1)
    med = float(cells.area_um2.median()) if len(cells) else float("nan")
    P(f"  cells {len(cells):,}   GT {GT_CELLS:,}   ratio {ratio:.4f}")
    P(f"  molecule assignment rate (weighted by MIDCount) {rate:.4f}")
    P(f"  area median {med:.1f} um^2   equiv diam {2*np.sqrt(med/np.pi):.2f} um")
    json.dump(dict(tag=TAG, sample=SAMPLE, mode=a.mode, cells=int(len(cells)), gt_cells=GT_CELLS,
                   cell_ratio=ratio, mask_assignment_rate=rate),
              open(OUT / f"merge_summary{sfx}.json", "w"), indent=2)
    P(f"  -> {OUT/f'dissect_cells{sfx}.csv'}\n  MERGE DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["inspect", "register", "prep", "run", "merge", "all"], default="all")
    ap.add_argument("--mode", choices=["native", "xenium"], default="native",
                    help="native = native DISSECT stereoseq entry (default, bins weighted by MIDCount); xenium = disguised as xenium (expanded molecules)")
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--scale", type=float, default=None, help="image pixels / DNB; by default searched automatically around 0.5/1/2")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--minarea", type=int, default=None, help="default 350; converted to physical size when --area-rule physical")
    ap.add_argument("--area-rule", choices=["fixed", "physical"], default="fixed",
                    help="fixed = same numeric 350-1600 as other DISSECT runs; physical = physical-size equivalent of Xenium")
    ap.add_argument("--maxarea", type=int, default=None, help="default 1600; converted to physical size when --area-rule physical")
    ap.add_argument("--refine-batch-size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=-0.20)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--engine-extra", default="")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P(f"{SEP}\nDISSECT x {TAG} ({SAMPLE})   out {OUT}\n{SEP}")
    if a.stage == "inspect":
        stage_inspect(a); return
    if a.stage == "register" or (a.stage == "all" and not (OUT / "register.json").exists()):
        stage_register(a)
        if a.stage == "register":
            return
    if a.stage in ("prep", "all"):
        stage_prep(a)
    if a.stage in ("run", "all"):
        stage_run(a)
    if a.stage in ("merge", "all"):
        stage_merge(a)


if __name__ == "__main__":
    main()
