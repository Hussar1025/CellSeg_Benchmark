#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proseg_xenium.py
=======================================================================
Run Proseg 3 on the same centered ROI as ucs_xenium.py, and rasterise its
polygons onto the identical UCS grid so the two methods are directly,
pixel-for-pixel comparable.

Proseg is also reference-assisted: it takes the prior transcript-to-nucleus
assignment straight out of the Xenium transcript table (`cell_id` /
`overlaps_nucleus`) and never invents new cells. So it belongs in the same
comparison bucket as the platform-prior UCS run, not against a DAPI-only
method.

Stages
------
  1  resolve files, pixel size, centered ROI   (same logic as ucs_xenium.py)
  2  roi_transcripts.parquet   -- row subset, ALL columns preserved so the
                                  prior assignment survives
  3  proseg --xenium ...
  4  proseg_segmentation.tif   -- polygons rasterised to the UCS grid
  5  cell x gene matrix + h5ad from Proseg's own counts
  6  QC report, optional overlay figure, optional comparison to a UCS run

Usage
-----
  python proseg_xenium.py --dataset liver --nthreads 16
  python proseg_xenium.py --dataset mouse_brain --nthreads 16
  python proseg_xenium.py --dataset liver --report-only \\
      --compare-to /data/qiuyijia/ucs_xenium_liver_roi10000_prior

Notes
-----
* Proseg filters qv < 20 by default, matching --qv-min 20 on the UCS side.
* Proseg is a sampler and is non-deterministic; small run-to-run differences
  are expected and are not a bug.
* If Proseg dies silently or crawls, it ran out of RAM: raise --voxel-size,
  lower --voxel-layers, or pass --no-diffusion.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DATASETS = {
    "liver": {"root": "/data/qiuyijia/dataset/xenium_liver", "tag": "liver"},
    "mouse_brain": {"root": "/data/qiuyijia/dataset/xenium_mouse_brain",
                    "tag": "mouse_brain"},
}
OUT_TEMPLATE = "/data/qiuyijia/proseg_xenium_{tag}_roi{roi}"

MORPH_CANDIDATES = [
    "outs/morphology_focus/morphology_focus_0000.ome.tif",
    "outs/morphology_focus.ome.tif",
    "outs/morphology_mip.ome.tif",
    "outs/morphology.ome.tif",
]
TRANSCRIPT_CANDIDATES = ["outs/transcripts.parquet", "transcripts.parquet"]
EXPERIMENT_CANDIDATES = ["outs/experiment.xenium", "experiment.xenium"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def banner(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78, flush=True)


# ----------------------------------------------------------------------
# shared ROI logic (must stay identical to ucs_xenium.py)
# ----------------------------------------------------------------------
def resolve_first(root, candidates, what, required=True):
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(
            f"could not locate {what} under {root}. tried:\n  "
            + "\n  ".join(str(root / c) for c in candidates))
    return None


def read_pixel_size(root, fallback=0.2125):
    p = resolve_first(root, EXPERIMENT_CANDIDATES, "experiment.xenium", required=False)
    if p is None:
        return fallback
    try:
        meta = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return fallback
    for k in ("pixel_size", "pixel_size_um", "pixel_size_microns"):
        if k in meta:
            return float(meta[k])
    return fallback


def image_level0_shape(path):
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        s = tf.series[0]
        try:
            shape = s.levels[0].shape
        except Exception:  # noqa: BLE001
            shape = s.shape
    shape = tuple(int(v) for v in shape)
    if len(shape) == 2:
        return 1, shape[0], shape[1]
    if len(shape) == 3:
        if shape[0] <= 8 and shape[0] < shape[-1]:
            return shape[0], shape[1], shape[2]
        if shape[-1] <= 8:
            return shape[-1], shape[0], shape[1]
        return 1, shape[-2], shape[-1]
    return int(np.prod(shape[:-2])), shape[-2], shape[-1]


def compute_roi(img_h, img_w, roi_size, origin=None):
    if origin is not None:
        y0, x0 = int(origin[0]), int(origin[1])
    else:
        y0 = max(0, (img_h - roi_size) // 2)
        x0 = max(0, (img_w - roi_size) // 2)
    sy = min(roi_size, img_h - y0)
    sx = min(roi_size, img_w - x0)
    if sy != roi_size or sx != roi_size:
        log(f"WARNING: ROI clipped to {sy} x {sx} (image {img_h} x {img_w})")
    return y0, x0, sy, sx


# ----------------------------------------------------------------------
# stage 2 : ROI transcript subset (all columns kept)
# ----------------------------------------------------------------------
def subset_transcripts(tx_path, out_dir, meta, args):
    """Row-filter transcripts.parquet to the ROI, preserving every column.

    Proseg needs the platform's prior assignment (`cell_id`,
    `overlaps_nucleus`), so this must never be a projection down to
    coordinates + gene.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    sub_path = out_dir / "roi_transcripts.parquet"
    meta_path = out_dir / "roi_transcripts_meta.json"
    if sub_path.exists() and meta_path.exists() and not args.force_subset:
        old = json.loads(meta_path.read_text())
        if all(abs(float(old.get(k, -1)) - float(meta[k])) < 1e-9
               for k in ("y0", "x0", "size_y", "size_x", "pixel_size")):
            log(f"reusing cached ROI subset ({old.get('n_rows', '?')} rows): {sub_path}")
            return sub_path
        log("cached subset does not match the current ROI -> rebuilding")

    ps = meta["pixel_size"]
    y0, x0, sy, sx = meta["y0"], meta["x0"], meta["size_y"], meta["size_x"]
    ymin, ymax = y0 * ps, (y0 + sy) * ps
    xmin, xmax = x0 * ps, (x0 + sx) * ps
    log(f"ROI in microns: x[{xmin:.1f}, {xmax:.1f})  y[{ymin:.1f}, {ymax:.1f})")

    pf = pq.ParquetFile(str(tx_path))
    log(f"transcript columns: {pf.schema.names}")
    for need in ("x_location", "y_location"):
        if need not in pf.schema.names:
            raise RuntimeError(f"transcripts.parquet has no '{need}' column")
    prior_cols = [c for c in ("cell_id", "overlaps_nucleus", "nucleus_distance")
                  if c in pf.schema.names]
    if "cell_id" not in prior_cols:
        log("WARNING: no cell_id column -> Proseg will have no prior segmentation")
    else:
        log(f"prior columns preserved: {prior_cols}")

    writer = None
    n_total = n_kept = 0
    try:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            n_total += batch.num_rows
            tbl = pa.Table.from_batches([batch])
            xu = np.asarray(tbl.column("x_location"))
            yu = np.asarray(tbl.column("y_location"))
            sel = (xu >= xmin) & (xu < xmax) & (yu >= ymin) & (yu < ymax)
            if not sel.any():
                continue
            sub = tbl.filter(pa.array(sel))
            if writer is None:
                writer = pq.ParquetWriter(str(sub_path), sub.schema,
                                          compression="zstd")
            writer.write_table(sub)
            n_kept += sub.num_rows
    finally:
        if writer is not None:
            writer.close()

    if n_kept == 0:
        raise RuntimeError("no transcripts inside the ROI - check pixel_size / ROI origin")
    log(f"transcripts scanned = {n_total:,}   written = {n_kept:,}  -> {sub_path}")
    out = dict(meta); out["n_rows"] = int(n_kept)
    meta_path.write_text(json.dumps(out, indent=2))
    return sub_path


# ----------------------------------------------------------------------
# stage 3 : run proseg
# ----------------------------------------------------------------------
def proseg_supported_flags(binary):
    try:
        p = subprocess.run([binary, "--help"], capture_output=True,
                           text=True, timeout=120)
        txt = (p.stdout or "") + (p.stderr or "")
    except Exception:  # noqa: BLE001
        return set()
    return {t.split("=")[0].rstrip(",") for t in txt.replace(",", " ").split()
            if t.startswith("--")}


def run_proseg(tx_sub, out_dir, args):
    binary = args.proseg_bin
    if shutil.which(binary) is None and not Path(binary).exists():
        raise FileNotFoundError(
            f"proseg binary '{binary}' not found. Install with `cargo install proseg` "
            "or point --proseg-bin at target/release/proseg.")

    work = out_dir / "proseg_out"
    done = work / ".proseg_done"
    if done.exists() and not args.force_run:
        log("proseg already finished; use --force-run to redo")
        return work
    work.mkdir(parents=True, exist_ok=True)

    supported = proseg_supported_flags(binary)
    if supported:
        log(f"proseg exposes {len(supported)} flags")

    cmd = [binary, "--xenium"]

    def add(flag, value=None):
        if supported and flag not in supported:
            log(f"  (skipping {flag}: not supported by this proseg build)")
            return
        cmd.append(flag)
        if value is not None:
            cmd.append(str(value))

    add("--nthreads", args.nthreads)
    if args.voxel_layers:
        add("--voxel-layers", args.voxel_layers)
    if args.voxel_size:
        add("--voxel-size", args.voxel_size)
    if args.burnin_voxel_size:
        add("--burnin-voxel-size", args.burnin_voxel_size)
    if args.samples:
        add("--samples", args.samples)
    if args.burnin_samples:
        add("--burnin-samples", args.burnin_samples)
    if args.ncomponents:
        add("--ncomponents", args.ncomponents)
    if args.no_diffusion:
        add("--no-diffusion")
    if args.cell_compactness is not None:
        add("--cell-compactness", args.cell_compactness)

    # flat-file outputs: no spatialdata dependency needed downstream
    add("--output-counts", work / "counts.mtx.gz")
    add("--output-expected-counts", work / "expected-counts.mtx.gz")
    add("--output-cell-metadata", work / "cell-metadata.csv.gz")
    add("--output-gene-metadata", work / "gene-metadata.csv.gz")
    add("--output-transcript-metadata", work / "transcript-metadata.csv.gz")
    add("--output-cell-polygons", work / "cell-polygons.geojson.gz")
    if args.spatialdata:
        add("--output-spatialdata", work / "proseg.zarr")
        add("--overwrite")
    if args.proseg_extra:
        cmd += args.proseg_extra.split()
    cmd.append(str(tx_sub))

    banner("STAGE 3 / running proseg")
    log("cmd = " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    rc = subprocess.call([str(c) for c in cmd], cwd=str(work))
    if rc != 0:
        raise RuntimeError(
            f"proseg exited with code {rc}. A silent death or sudden slowdown "
            "usually means it ran out of RAM: raise --voxel-size, lower "
            "--voxel-layers, or add --no-diffusion.")
    done.write_text(f"finished in {time.time()-t0:.1f}s\n")
    log(f"proseg finished in {(time.time()-t0)/60:.1f} min -> {work}")
    return work


# ----------------------------------------------------------------------
# stage 4 : rasterise polygons onto the UCS grid
# ----------------------------------------------------------------------
def open_maybe_gzip(path):
    p = str(path)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p, "rt")


def iter_polygons(geojson_path):
    """Yield (cell_key, list_of_exterior_rings) from a GeoJSON FeatureCollection."""
    with open_maybe_gzip(geojson_path) as fh:
        data = json.load(fh)
    feats = data.get("features", data if isinstance(data, list) else [])
    for i, feat in enumerate(feats):
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        key = props.get("cell", props.get("cell_id", i))
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        rings = []
        if gtype == "Polygon":
            if coords:
                rings.append(coords[0])
        elif gtype == "MultiPolygon":
            for poly in coords:
                if poly:
                    rings.append(poly[0])
        if rings:
            yield key, rings


def rasterise(work, out_dir, meta, args):
    import tifffile
    from skimage.draw import polygon as draw_polygon

    gj = work / "cell-polygons.geojson.gz"
    if not gj.exists():
        alt = list(work.glob("cell-polygons*.geojson*"))
        if not alt:
            log("no cell polygons found; skipping rasterisation")
            return None
        gj = alt[0]

    ps, bf = meta["pixel_size"], meta["bin_factor"]
    y0, x0 = meta["y0"], meta["x0"]
    H, W = meta["size_y"] // bf, meta["size_x"] // bf
    scale = ps * bf                       # microns per grid pixel

    seg = np.zeros((H, W), dtype=np.int32)
    key_to_label, n_poly, n_drawn = {}, 0, 0
    for key, rings in iter_polygons(gj):
        n_poly += 1
        lab = key_to_label.setdefault(key, len(key_to_label) + 1)
        for ring in rings:
            arr = np.asarray(ring, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            cc = arr[:, 0] / scale - x0 / bf     # micron -> grid column
            rr = arr[:, 1] / scale - y0 / bf     # micron -> grid row
            r, c = draw_polygon(rr, cc, shape=(H, W))
            if len(r):
                seg[r, c] = lab
                n_drawn += 1

    n_cells = len(np.unique(seg)) - (1 if (seg == 0).any() else 0)
    log(f"polygons read = {n_poly:,}  rings drawn = {n_drawn:,}  "
        f"cells on grid = {n_cells:,}")
    log(f"coverage = {(seg > 0).mean() * 100:.2f}% of the {H} x {W} grid")
    if n_poly and n_cells == 0:
        log("WARNING: nothing landed on the grid -- polygon coordinate units "
            "are probably not microns")

    p = out_dir / "proseg_segmentation.tif"
    tifffile.imwrite(str(p), seg)
    np.save(out_dir / "proseg_label_keys.npy",
            np.array(list(key_to_label.keys()), dtype=object))
    log(f"raster -> {p}")
    return p


# ----------------------------------------------------------------------
# stage 5 : counts
# ----------------------------------------------------------------------
def collect_counts(work, out_dir, meta):
    import pandas as pd

    counts_p = work / "counts.mtx.gz"
    genes_p = work / "gene-metadata.csv.gz"
    cells_p = work / "cell-metadata.csv.gz"
    if not counts_p.exists():
        log("no counts.mtx.gz; skipping matrix export")
        return
    try:
        from scipy.io import mmread
    except ImportError:
        log("scipy unavailable; skipping matrix export")
        return

    mat = mmread(str(counts_p))
    mat = mat.tocsr() if hasattr(mat, "tocsr") else np.asarray(mat)
    log(f"proseg count matrix {mat.shape}")

    genes = None
    if genes_p.exists():
        gdf = pd.read_csv(genes_p)
        for c in ("gene", "feature_name", "name"):
            if c in gdf.columns:
                genes = gdf[c].astype(str).tolist()
                break
    cells = pd.read_csv(cells_p) if cells_p.exists() else None

    # orient as cells x genes
    if genes and mat.shape[0] == len(genes) and mat.shape[1] != len(genes):
        mat = mat.T
        log("transposed matrix to cells x genes")
    if genes is None:
        genes = [f"g{i}" for i in range(mat.shape[1])]

    counts = np.asarray(mat.sum(axis=1)).ravel()
    log(f"cells = {mat.shape[0]:,}  genes = {mat.shape[1]:,}  "
        f"total counts = {counts.sum():,.0f}")
    log(f"counts/cell: median={np.median(counts):.1f} mean={counts.mean():.1f}")

    try:
        import anndata as ad
        from scipy import sparse

        obs = pd.DataFrame(index=[f"cell_{i+1}" for i in range(mat.shape[0])])
        if cells is not None and len(cells) == mat.shape[0]:
            obs = cells.copy()
            obs.index = [f"cell_{i+1}" for i in range(mat.shape[0])]
        obs["n_counts"] = counts
        adata = ad.AnnData(X=sparse.csr_matrix(mat), obs=obs,
                           var=pd.DataFrame(index=genes))
        adata.uns["method"] = "proseg"
        adata.uns["prior_source"] = "xenium transcript cell_id (reference-assisted)"
        adata.uns["roi_meta"] = {k: v for k, v in meta.items()}
        adata.write_h5ad(out_dir / "proseg_cells.h5ad")
        log(f"h5ad -> {out_dir / 'proseg_cells.h5ad'}")
    except Exception as e:  # noqa: BLE001
        log(f"(h5ad export skipped: {e})")


# ----------------------------------------------------------------------
# stage 6 : QC
# ----------------------------------------------------------------------
def spread(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return "empty"
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return (f"med={q[2]:.1f}  IQR={q[1]:.1f}-{q[3]:.1f}  "
            f"p5={q[0]:.1f} p95={q[4]:.1f}")


def label_stats(seg, px_um2):
    lab = seg.reshape(-1).astype(np.int64)
    area = np.bincount(lab, minlength=int(lab.max()) + 1)[1:]
    return area[area > 0]


def qc_report(out_dir, meta, compare_to=None):
    import tifffile

    banner("QC REPORT")
    seg_p = out_dir / "proseg_segmentation.tif"
    if not seg_p.exists():
        log("no proseg_segmentation.tif yet")
        return
    bf, ps = meta["bin_factor"], meta["pixel_size"]
    px_um2 = (bf * ps) ** 2
    seg = tifffile.imread(str(seg_p))
    area = label_stats(seg, px_um2)
    print(f"  cells            {len(area):,}")
    print(f"  coverage         {(seg > 0).mean() * 100:.2f}% of grid")
    print(f"  area (um^2)      {spread(area * px_um2)}")
    print(f"  equiv diam (um)  {spread(2 * np.sqrt(area * px_um2 / np.pi))}")

    if compare_to:
        other_p = Path(compare_to) / "ucs_segmentation.tif"
        if not other_p.exists():
            log(f"(no ucs_segmentation.tif in {compare_to})")
            return
        other = tifffile.imread(str(other_p))
        if other.shape != seg.shape:
            log(f"(grid mismatch {other.shape} vs {seg.shape}; skipping comparison)")
            return
        oarea = label_stats(other, px_um2)
        a, b = seg > 0, other > 0
        inter = int((a & b).sum())
        dice = 2 * inter / max(int(a.sum()) + int(b.sum()), 1)
        print("\n  --- vs UCS ---")
        print(f"  UCS cells        {len(oarea):,}   proseg {len(area):,}")
        print(f"  UCS median area  {np.median(oarea) * px_um2:.1f} um^2   "
              f"proseg {np.median(area) * px_um2:.1f} um^2")
        print(f"  foreground Dice  {dice:.3f}")
        print(f"  proseg-only px   {int((a & ~b).sum()):,}")
        print(f"  UCS-only px      {int((b & ~a).sum()):,}")


def qc_figure(out_dir, meta, size=300):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tifffile
        from skimage.segmentation import find_boundaries
    except Exception as e:  # noqa: BLE001
        log(f"(QC figure skipped: {e})")
        return
    seg_p = out_dir / "proseg_segmentation.tif"
    if not seg_p.exists():
        return
    seg = tifffile.imread(str(seg_p))
    H, W = seg.shape
    y = max(0, H // 2 - size // 2); x = max(0, W // 2 - size // 2)
    sl = (slice(y, y + size), slice(x, x + size))
    sub = seg[sl]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    rng = np.random.default_rng(0)
    lut = np.concatenate([[[0, 0, 0]], rng.random((int(seg.max()) + 1, 3))])
    axes[0].imshow(lut[sub])
    axes[0].set_title("proseg cells")
    axes[1].imshow(find_boundaries(sub, mode="inner"), cmap="Greys")
    axes[1].set_title("proseg boundaries")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    p = out_dir / "qc_proseg.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    log(f"QC figure -> {p}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="Run Proseg on a Xenium ROI, on the same grid as ucs_xenium.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--proseg-bin", default="proseg")

    p.add_argument("--roi-size", type=int, default=10000, help="ROI edge in NATIVE px")
    p.add_argument("--roi-origin", type=int, nargs=2, default=None, metavar=("Y0", "X0"))
    p.add_argument("--bin-factor", type=int, default=10,
                   help="only used for the comparison raster; must match the UCS run")
    p.add_argument("--pixel-size", type=float, default=None)
    p.add_argument("--morphology", default=None)
    p.add_argument("--batch-size", type=int, default=2_000_000)

    p.add_argument("--nthreads", type=int, default=16)
    p.add_argument("--voxel-layers", type=int, default=None)
    p.add_argument("--voxel-size", type=float, default=None)
    p.add_argument("--burnin-voxel-size", type=float, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--burnin-samples", type=int, default=None)
    p.add_argument("--ncomponents", type=int, default=None)
    p.add_argument("--cell-compactness", type=float, default=None)
    p.add_argument("--no-diffusion", action="store_true")
    p.add_argument("--spatialdata", action="store_true",
                   help="also write the spatialdata zarr")
    p.add_argument("--proseg-extra", default=None)

    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--run-only", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--compare-to", default=None,
                   help="a UCS output dir containing ucs_segmentation.tif")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--force-subset", action="store_true")
    p.add_argument("--force-run", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    cfg = DATASETS[args.dataset]
    root = Path(args.data_root or cfg["root"])
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    out_dir = Path(args.out_dir or OUT_TEMPLATE.format(tag=cfg["tag"], roi=args.roi_size))
    out_dir.mkdir(parents=True, exist_ok=True)

    banner(f"PROSEG  dataset={args.dataset}")
    log("prior = Xenium transcript cell_id -> reference-assisted, comparable "
        "to the platform-prior UCS run")
    log(f"data root  = {root}")
    log(f"output dir = {out_dir}")

    morph = Path(args.morphology) if args.morphology else \
        resolve_first(root, MORPH_CANDIDATES, "morphology image")
    tx = resolve_first(root, TRANSCRIPT_CANDIDATES, "transcripts.parquet")
    ps = args.pixel_size or read_pixel_size(root)
    nc, ih, iw = image_level0_shape(morph)
    y0, x0, sy, sx = compute_roi(ih, iw, args.roi_size, args.roi_origin)
    log(f"morphology = {morph}  (channels={nc}, {ih} x {iw})")
    log(f"pixel size = {ps} um/px")
    log(f"ROI        = y[{y0},{y0+sy}) x[{x0},{x0+sx}) = {sy*ps:.1f} x {sx*ps:.1f} um")

    meta = dict(dataset=args.dataset, y0=y0, x0=x0, size_y=sy, size_x=sx,
                bin_factor=args.bin_factor, pixel_size=ps, roi_size=args.roi_size)
    (out_dir / "roi_meta.json").write_text(json.dumps(meta, indent=2))

    if args.report_only:
        qc_report(out_dir, meta, args.compare_to)
        if not args.no_figure:
            qc_figure(out_dir, meta)
        return

    banner("STAGE 2 / ROI transcript subset")
    tx_sub = subset_transcripts(tx, out_dir, meta, args)
    if args.prepare_only:
        banner("PREPARE DONE")
        log(f"subset: {tx_sub}")
        return

    work = run_proseg(tx_sub, out_dir, args)

    banner("STAGE 4 / rasterise onto the UCS grid")
    rasterise(work, out_dir, meta, args)

    banner("STAGE 5 / counts")
    collect_counts(work, out_dir, meta)

    qc_report(out_dir, meta, args.compare_to)
    if not args.no_figure:
        qc_figure(out_dir, meta)
    banner("ALL DONE")


if __name__ == "__main__":
    main()