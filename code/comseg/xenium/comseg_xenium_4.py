#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_xenium.py
=======================================================================
ComSeg on centered Xenium ROIs (liver / mouse brain), on the same grid as
ucs_xenium.py / proseg_xenium.py / cellist_xenium.py.

ComSeg builds a KNN graph over individual RNA points, so a 10000x10000
ROI (tens of millions of transcripts) has to be tiled. The tiling follows
the core/read pattern from your CosMx script: each tile is processed with
an overlap margin, and a cell is kept only if its centroid lands in that
tile's core region, so cells split by a tile edge are not double counted.

Stages
------
  prep    per-tile spot CSV + nucleus prior mask tiff + tile_metadata.tsv
  run     ComSegDataset -> compute_edge_weight -> ComSegDict.run_all,
          per tile, with DONE/FAILED flags and an incremental summary
  merge   global cell ids, core-region filter, raster onto the coarse grid,
          exact full-panel count matrix from the per-RNA assignment
  all     prep + run + merge
  report  QC only

Usage
-----
  python comseg_xenium.py --dataset liver --stage prep
  python comseg_xenium.py --dataset liver --stage all \\
      --mean-cell-diameter-um 22 --max-cell-radius-um 30
  python comseg_xenium.py --dataset mouse_brain --stage all \\
      --mean-cell-diameter-um 12 --max-cell-radius-um 20

Coordinate contract
-------------------
ComSeg reads spot coordinates in MASK PIXEL units and converts them to
microns with dict_scale. So the CSV coordinates and the prior mask must
share one pixel grid: here that grid is --mask-bin native px (default 2 ->
0.425 um), and dict_scale is set to match. Transcript positions stay
floating point, so no spatial precision is lost - only the prior mask is
rasterised.

Prior
-----
Default prior is the platform nucleus segmentation
(nucleus_boundaries.parquet), which puts this run in the same
reference-assisted bucket as the platform-prior UCS and Proseg runs.
--prior-source cellist reuses the Cellpose nuclei from cellist_xenium.py
instead, if you want a DAPI-only variant.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATASETS = {
    "xenium_8": {"root": "yourpath/dataset/xenium_8", "tag": "xenium_8"},
    "pancreas": {"root": "yourpath/dataset/xenium_pancreas_cancer", "tag": "pancreas"},
    "breast": {"root": "yourpath/dataset/xenium_breast", "tag": "breast"},
    "colon": {"root": "yourpath/dataset/xenium_colon", "tag": "colon"},
    "colon": {"root": "yourpath/dataset/xenium_colon", "tag": "colon"},
    "liver": {"root": "yourpath/dataset/xenium_liver", "tag": "liver"},
    "mouse_brain": {"root": "yourpath/dataset/xenium_mouse_brain",
                    "tag": "mouse_brain"},
}
OUT_TEMPLATE = "yourpath/comseg_xenium_{tag}_roi{roi}"
CELLIST_TEMPLATE = "yourpath/cellist_xenium_{tag}_roi{roi}"

CONTROL_PREFIXES = (
    "NegControlProbe", "NegControlCodeword", "negcontrol",
    "antisense", "BLANK", "Blank", "blank",
    "UnassignedCodeword", "DeprecatedCodeword",
    "Intergenic", "Genomic", "Unassigned",
)
MORPH_CANDIDATES = [
    "morphology.ome.tif",
    "morphology_focus/morphology_focus_0000.ome.tif",
    "morphology_focus.ome.tif",
    "morphology_mip.ome.tif",

    "outs/morphology_focus/morphology_focus_0000.ome.tif",
    "outs/morphology_focus.ome.tif",
    "outs/morphology_mip.ome.tif",
    "outs/morphology.ome.tif",
]
TRANSCRIPT_CANDIDATES = ["outs/transcripts.parquet", "transcripts.parquet"]
EXPERIMENT_CANDIDATES = ["outs/experiment.xenium", "experiment.xenium"]
NUCBOUND_CANDIDATES = ["outs/nucleus_boundaries.parquet", "nucleus_boundaries.parquet"]


def setup_logging(log_dir, name):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / f"{name}.log", mode="a")],
        force=True)


def banner(msg):
    logging.info("=" * 90)
    logging.info(msg)
    logging.info("=" * 90)


# ----------------------------------------------------------------------
# shared resolution / ROI helpers
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
        logging.warning(f"ROI clipped to {sy} x {sx} (image {img_h} x {img_w})")
    return y0, x0, sy, sx


# ----------------------------------------------------------------------
# nucleus prior: platform polygons
# ----------------------------------------------------------------------
def load_platform_nuclei(nb_path, meta):
    """Polygon rings in MASK PIXEL units, plus centroids, for nuclei near the ROI."""
    ps, mb = meta["pixel_size"], meta["mask_bin"]
    y0, x0, sy, sx = meta["y0"], meta["x0"], meta["size_y"], meta["size_x"]
    ymin, ymax = y0 * ps, (y0 + sy) * ps
    xmin, xmax = x0 * ps, (x0 + sx) * ps

    df = pd.read_parquet(nb_path)
    cols = {c.lower(): c for c in df.columns}
    cid = cols.get("cell_id", "cell_id")
    vx, vy = cols.get("vertex_x", "vertex_x"), cols.get("vertex_y", "vertex_y")
    logging.info(f"nucleus_boundaries rows = {len(df):,} columns = {list(df.columns)}")
    pad = 50.0
    df = df.loc[(df[vx] >= xmin - pad) & (df[vx] <= xmax + pad) &
                (df[vy] >= ymin - pad) & (df[vy] <= ymax + pad)]
    if df.empty:
        raise RuntimeError("no nucleus vertices near the ROI")
    logging.info(f"rows near ROI = {len(df):,}")

    codes, uniques = pd.factorize(df[cid].to_numpy())
    # to mask-pixel coordinates local to the ROI
    xs = (df[vx].to_numpy() / ps - x0) / mb
    ys = (df[vy].to_numpy() / ps - y0) / mb
    n = len(uniques)
    order = np.argsort(codes, kind="stable")
    codes_s, xs_s, ys_s = codes[order], xs[order], ys[order]
    starts = np.searchsorted(codes_s, np.arange(n))
    ends = np.searchsorted(codes_s, np.arange(n), side="right")

    rings, cx, cy = [], np.empty(n), np.empty(n)
    for i in range(n):
        a, b = starts[i], ends[i]
        px_, py_ = xs_s[a:b], ys_s[a:b]
        if len(px_) >= 3 and px_[0] == px_[-1] and py_[0] == py_[-1]:
            px_, py_ = px_[:-1], py_[:-1]
        rings.append((px_, py_))
        if len(px_) < 3:
            cx[i], cy[i] = px_.mean(), py_.mean()
            continue
        x1, y1 = np.roll(px_, -1), np.roll(py_, -1)
        cross = px_ * y1 - x1 * py_
        A = 0.5 * cross.sum()
        if abs(A) < 1e-12:
            cx[i], cy[i] = px_.mean(), py_.mean()
        else:
            cx[i] = ((px_ + x1) * cross).sum() / (6.0 * A)
            cy[i] = ((py_ + y1) * cross).sum() / (6.0 * A)
    logging.info(f"platform nuclei near ROI = {n:,}")
    return rings, cx, cy, np.asarray(uniques)


def rasterise_nuclei(rings, labels, y0m, x0m, h, w):
    """Fill polygon rings into an (h, w) label image whose origin is (y0m, x0m)."""
    from skimage.draw import polygon as draw_polygon

    out = np.zeros((h, w), dtype=np.int32)
    drawn = 0
    for (px_, py_), lab in zip(rings, labels):
        if len(px_) < 3:
            continue
        cc = px_ - x0m
        rr = py_ - y0m
        if rr.max() < 0 or cc.max() < 0 or rr.min() >= h or cc.min() >= w:
            continue
        r, c = draw_polygon(rr, cc, shape=(h, w))
        if len(r):
            out[r, c] = lab
            drawn += 1
    return out, drawn


# ----------------------------------------------------------------------
# stage: prep
# ----------------------------------------------------------------------
def prep_tiles(tx_path, nb_path, out_root, meta, args):
    import pyarrow.parquet as pq
    import tifffile

    csv_dir = out_root / "input" / "spots"
    mask_dir = out_root / "input" / "mask"
    csv_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta_p = out_root / "input" / "prep_meta.json"
    tiles_p = out_root / "input" / "tile_metadata.tsv"

    if meta_p.exists() and tiles_p.exists() and not args.force_prep:
        old = json.loads(meta_p.read_text())
        if all(abs(float(old.get(k, -1)) - float(meta[k])) < 1e-9
               for k in ("y0", "x0", "size_y", "size_x", "pixel_size",
                         "mask_bin", "tile_px", "tile_overlap_px")):
            logging.info(f"reusing cached prep ({old.get('n_tiles')} tiles)")
            return old, pd.read_csv(tiles_p, sep="\t")
        logging.info("cached prep does not match current settings -> rebuilding")

    ps, mb = meta["pixel_size"], meta["mask_bin"]
    y0, x0, sy, sx = meta["y0"], meta["x0"], meta["size_y"], meta["size_x"]
    tp, ov = meta["tile_px"], meta["tile_overlap_px"]
    Hm, Wm = sy // mb, sx // mb
    unit_um = mb * ps
    logging.info(f"mask grid = {Hm} x {Wm}   1 unit = {mb} native px = {unit_um:.4f} um")

    # ---- tile layout (native px, local to the ROI) --------------------
    ys = list(range(0, sy, tp))
    xs = list(range(0, sx, tp))
    tiles = []
    tid = 0
    for cy0 in ys:
        for cx0 in xs:
            cy1, cx1 = min(cy0 + tp, sy), min(cx0 + tp, sx)
            ry0, rx0 = max(0, cy0 - ov), max(0, cx0 - ov)
            ry1, rx1 = min(sy, cy1 + ov), min(sx, cx1 + ov)
            tiles.append(dict(tile_id=tid, tile_name=f"tile_{tid:04d}",
                              core_y0=cy0, core_y1=cy1, core_x0=cx0, core_x1=cx1,
                              read_y0=ry0, read_y1=ry1, read_x0=rx0, read_x1=rx1))
            tid += 1
    logging.info(f"{len(tiles)} tiles: core {tp} native px "
                 f"({tp*ps:.0f} um), overlap {ov} px ({ov*ps:.0f} um)")

    # ---- nucleus prior ------------------------------------------------
    rings, ncx, ncy, nids = load_platform_nuclei(nb_path, meta)
    labels = np.arange(1, len(rings) + 1, dtype=np.int32)

    # ---- stream transcripts into per-tile buffers ---------------------
    ymin, ymax = y0 * ps, (y0 + sy) * ps
    xmin, xmax = x0 * ps, (x0 + sx) * ps
    pf = pq.ParquetFile(str(tx_path))
    names = set(pf.schema.names)
    cols = [c for c in ("x_location", "y_location", "z_location", "qv",
                        "feature_name") if c in names]
    buffers = {t["tile_id"]: [] for t in tiles}
    n_total = n_kept = 0
    genes_seen = set()

    for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
        df = batch.to_pandas()
        n_total += len(df)
        if "qv" in df:
            df = df[df["qv"].to_numpy() >= args.qv_min]
        if df.empty:
            continue
        xu = df["x_location"].to_numpy(dtype=np.float64)
        yu = df["y_location"].to_numpy(dtype=np.float64)
        sel = (xu >= xmin) & (xu < xmax) & (yu >= ymin) & (yu < ymax)
        if not sel.any():
            continue
        df, xu, yu = df.loc[sel], xu[sel], yu[sel]
        feat = df["feature_name"].to_numpy()
        if feat.dtype == object and len(feat) and isinstance(feat[0], (bytes, bytearray)):
            feat = np.array([f.decode("utf-8") for f in feat])
        else:
            feat = feat.astype(str)
        keep = ~np.array([f.startswith(CONTROL_PREFIXES) for f in feat], dtype=bool)
        feat, xu, yu = feat[keep], xu[keep], yu[keep]
        if len(feat) == 0:
            continue
        zu = (df["z_location"].to_numpy(dtype=np.float64)[keep]
              if args.use_z and "z_location" in df else np.zeros(len(feat)))
        genes_seen.update(np.unique(feat).tolist())

        # native-pixel coordinates local to the ROI
        px_ = xu / ps - x0
        py_ = yu / ps - y0
        for t in tiles:
            m = ((py_ >= t["read_y0"]) & (py_ < t["read_y1"]) &
                 (px_ >= t["read_x0"]) & (px_ < t["read_x1"]))
            if not m.any():
                continue
            buffers[t["tile_id"]].append(pd.DataFrame({
                "gene": feat[m],
                "x": (px_[m] - t["read_x0"]) / mb,   # mask-pixel units, local
                "y": (py_[m] - t["read_y0"]) / mb,
                "z": zu[m],
                "global_x": px_[m], "global_y": py_[m],
            }))
        n_kept += len(feat)
        del df, xu, yu, feat, px_, py_, zu

    if n_kept == 0:
        raise RuntimeError("no transcripts inside the ROI - check pixel_size / ROI")
    logging.info(f"transcripts scanned = {n_total:,}   in ROI = {n_kept:,}   "
                 f"genes = {len(genes_seen)}")

    # ---- write per-tile csv + mask ------------------------------------
    rows = []
    for t in tiles:
        parts = buffers.pop(t["tile_id"])
        n_tx = int(sum(len(p) for p in parts))
        h = (t["read_y1"] - t["read_y0"]) // mb
        w = (t["read_x1"] - t["read_x0"]) // mb
        csv_p = csv_dir / f"{t['tile_name']}.csv"
        mask_p = mask_dir / f"{t['tile_name']}.tiff"

        mask, drawn = rasterise_nuclei(
            rings, labels, t["read_y0"] / mb, t["read_x0"] / mb, h, w)
        tifffile.imwrite(str(mask_p), mask)

        if n_tx:
            df = pd.concat(parts, ignore_index=True)
            df = df[(df["x"] >= 0) & (df["x"] < w) & (df["y"] >= 0) & (df["y"] < h)]
            df.to_csv(csv_p, index=False)
        else:
            pd.DataFrame(columns=["gene", "x", "y", "z", "global_x", "global_y"]
                         ).to_csv(csv_p, index=False)
        n_nuc = int(len(np.unique(mask)) - (1 if (mask == 0).any() else 0))
        rows.append({**t, "n_transcripts": n_tx, "mask_h": h, "mask_w": w,
                     "n_nuclei_in_read": n_nuc, "csv": str(csv_p),
                     "mask": str(mask_p)})
        logging.info(f"{t['tile_name']}: {n_tx:,} spots, mask {h}x{w}, "
                     f"{n_nuc:,} nuclei ({drawn:,} polygons drawn)")
        del parts
        gc.collect()

    tdf = pd.DataFrame(rows)
    tdf.to_csv(tiles_p, sep="\t", index=False)
    logging.info(f"tile metadata -> {tiles_p}")

    gt = int(np.sum((ncx * mb >= 0) & (ncx * mb < sx) &
                    (ncy * mb >= 0) & (ncy * mb < sy)))
    logging.info(f"platform cells with centroid in ROI (GT reference) = {gt:,}")

    out = dict(meta)
    out.update(n_tiles=len(tiles), n_transcripts=int(n_kept),
               n_genes=len(genes_seen), mask_h=Hm, mask_w=Wm,
               unit_um=unit_um, gt_expected_cells=gt,
               csv_dir=str(csv_dir), mask_dir=str(mask_dir))
    meta_p.write_text(json.dumps(out, indent=2))
    (out_root / "input" / "genes.txt").write_text(
        "\n".join(sorted(genes_seen)) + "\n")
    return out, tdf


# ----------------------------------------------------------------------
# stage: run ComSeg per tile
# ----------------------------------------------------------------------
def build_config(prep, args):
    unit = prep["unit_um"]
    return {
        # dataset initialisation
        "dict_scale": {"x": unit, "y": unit, "z": 1.0},
        "mean_cell_diameter": args.mean_cell_diameter_um,
        "gene_column": "gene",
        # prior computation from the mask
        "prior_name": "in_nucleus",
        "overwrite": True,
        "compute_centroid": True,
        # co-expression
        "n_neighbors": args.n_neighbors,
        "sampling": True,
        "sampling_size": args.sampling_size,
        # knn graph
        "k_nearest_neighbors": args.k_nearest_neighbors,
        # in-situ clustering
        "size_commu_min": args.size_commu_min,
        "norm_vector": args.norm_vector,
        "n_pcs": args.n_pcs,
        "clustering_method": args.clustering_method,
        "resolution": args.resolution,
        "n_clusters_kmeans": args.n_clusters_kmeans,
        "nb_min_cluster": 1,
        "min_merge_correlation": args.min_merge_correlation,
        # rna -> centroid association
        "max_cell_radius": args.max_cell_radius_um,
        # final result
        "alpha": args.alpha,
        "min_rna_per_cell": args.min_rna_per_cell,
        "return_polygon": False,
        "allow_disconnected_polygon": True,
        "disable_tqdm": True,
    }


def extract_assignment(comsegdict, img_name, n_spots):
    """Per-RNA cell label from the ComSeg graph, defensively."""
    import networkx as nx  # noqa: F401

    obj = comsegdict[img_name]
    G = getattr(obj, "G", None)
    if G is None:
        raise RuntimeError(f"no graph on ComSeg result for {img_name}")
    keys = set()
    for _, d in list(G.nodes(data=True))[:5]:
        keys.update(d.keys())
    logging.info(f"  graph node attributes: {sorted(keys)}")
    label_key = next((k for k in ("cell_index_pred", "cell_index", "cell")
                      if k in keys), None)
    if label_key is None:
        raise RuntimeError(f"no cell label attribute on graph nodes: {sorted(keys)}")

    lab = np.zeros(n_spots, dtype=np.int64)
    n_set = 0
    for node, d in G.nodes(data=True):
        if not isinstance(node, (int, np.integer)) or not (0 <= node < n_spots):
            continue
        v = d.get(label_key, 0)
        try:
            v = int(v)
        except (TypeError, ValueError):
            v = 0
        lab[node] = max(v, 0)
        n_set += 1
    logging.info(f"  assigned attribute on {n_set:,}/{n_spots:,} nodes, "
                 f"{int((lab > 0).sum()):,} spots in a cell")
    return lab


def run_tiles(out_root, prep, tiles, args):
    from comseg import dataset as ds
    from comseg import dictionary

    work = out_root / "comseg_out"
    work.mkdir(parents=True, exist_ok=True)
    summary_p = out_root / "logs" / "tile_summary.tsv"
    config = build_config(prep, args)
    logging.info("ComSeg config: " + json.dumps(
        {k: v for k, v in config.items() if k != "dict_scale"}, indent=2))
    logging.info(f"dict_scale = {config['dict_scale']}")

    csv_dir = Path(prep["csv_dir"])
    mask_dir = Path(prep["mask_dir"])

    for _, t in tiles.iterrows():
        name = t["tile_name"]
        tdir = work / name
        tdir.mkdir(parents=True, exist_ok=True)
        done, failed = tdir / "DONE.flag", tdir / "FAILED.flag"
        assign_p = tdir / "spot_assignment.parquet"

        if done.exists() and assign_p.exists() and not args.force_run:
            logging.info(f"skip {name} (done)")
            continue
        if int(t["n_transcripts"]) < args.min_tile_transcripts:
            logging.info(f"skip {name}: only {t['n_transcripts']} spots")
            done.write_text("TOO_FEW_TRANSCRIPTS\n")
            pd.DataFrame(columns=["global_x", "global_y", "gene", "cell"]
                         ).to_parquet(assign_p)
            continue

        logging.info("=" * 90)
        logging.info(f"{name}: {int(t['n_transcripts']):,} spots, "
                     f"{int(t['n_nuclei_in_read']):,} nuclei in read region")
        t0 = time.time()
        status, error, n_cells = "UNKNOWN", "", 0
        try:
            # ComSegDataset consumes a folder, so give each tile its own
            # single-file folder; that also keeps failures isolated.
            one_csv = tdir / "spots"
            one_mask = tdir / "mask"
            one_csv.mkdir(exist_ok=True)
            one_mask.mkdir(exist_ok=True)
            src_csv = csv_dir / f"{name}.csv"
            src_mask = mask_dir / f"{name}.tiff"
            for src, dst in ((src_csv, one_csv / f"{name}.csv"),
                             (src_mask, one_mask / f"{name}.tiff")):
                if not dst.exists():
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        import shutil

                        shutil.copy2(src, dst)

            spots = pd.read_csv(src_csv)
            n_spots = len(spots)

            dataset = ds.ComSegDataset(
                path_dataset_folder=str(one_csv),
                dict_scale=config["dict_scale"],
                mean_cell_diameter=config["mean_cell_diameter"],
                gene_column=config["gene_column"],
                path_to_mask_prior=str(one_mask),
                prior_name=config["prior_name"],
                disable_tqdm=config["disable_tqdm"],
            )
            dataset.add_prior_from_mask(config=config)
            dataset.compute_edge_weight(config=config)

            cdict = dictionary.ComSegDict(
                dataset=dataset,
                mean_cell_diameter=config["mean_cell_diameter"],
                disable_tqdm=config["disable_tqdm"],
            )
            cdict.run_all(config=config)
            adata, _ = cdict.anndata_from_comseg_result(config=config)
            try:
                adata.write_h5ad(tdir / "comseg_tile.h5ad")
            except Exception as e:  # noqa: BLE001
                logging.info(f"  (tile h5ad skipped: {e})")

            lab = extract_assignment(cdict, name, n_spots)
            res = pd.DataFrame({
                "global_x": spots["global_x"].to_numpy(),
                "global_y": spots["global_y"].to_numpy(),
                "gene": spots["gene"].to_numpy(),
                "cell": lab,
            })
            res.to_parquet(assign_p, index=False)
            n_cells = int(res.loc[res["cell"] > 0, "cell"].nunique())
            status = "DONE"
            done.write_text("DONE\n")
            failed.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            status, error = "FAILED", repr(e)
            failed.write_text(traceback.format_exc())
            logging.error(traceback.format_exc())

        rec = {"tile_name": name, "status": status,
               "seconds": round(time.time() - t0, 1),
               "n_transcripts": int(t["n_transcripts"]),
               "n_nuclei_in_read": int(t["n_nuclei_in_read"]),
               "n_cells": n_cells, "error": error}
        pd.DataFrame([rec]).to_csv(summary_p, sep="\t", index=False,
                                   mode="a", header=not summary_p.exists())
        logging.info(f"{name}: {status} cells={n_cells} "
                     f"({time.time()-t0:.0f}s)")
        gc.collect()
    logging.info(f"tile summary -> {summary_p}")


# ----------------------------------------------------------------------
# stage: merge
# ----------------------------------------------------------------------
def merge_tiles(out_root, prep, tiles, args):
    import tifffile

    work = out_root / "comseg_out"
    ps, mb = prep["pixel_size"], prep["mask_bin"]
    bf = prep["bin_factor"]
    H, W = prep["size_y"] // bf, prep["size_x"] // bf

    frames, offset = [], 0
    for _, t in tiles.iterrows():
        p = work / t["tile_name"] / "spot_assignment.parquet"
        if not p.exists():
            logging.warning(f"missing {p}; tile skipped")
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        df = df[df["cell"] > 0]
        if df.empty:
            continue
        # cells belong to the tile whose CORE contains their centroid
        cen = df.groupby("cell")[["global_x", "global_y"]].mean()
        keep = cen.index[(cen["global_y"] >= t["core_y0"]) &
                         (cen["global_y"] < t["core_y1"]) &
                         (cen["global_x"] >= t["core_x0"]) &
                         (cen["global_x"] < t["core_x1"])]
        df = df[df["cell"].isin(keep)]
        if df.empty:
            continue
        codes = pd.factorize(df["cell"])[0] + 1 + offset
        offset = int(codes.max())
        frames.append(pd.DataFrame({"global_x": df["global_x"].to_numpy(),
                                    "global_y": df["global_y"].to_numpy(),
                                    "gene": df["gene"].to_numpy(),
                                    "cell": codes}))
        logging.info(f"{t['tile_name']}: kept {len(keep):,} core cells, "
                     f"{len(df):,} spots")
    if not frames:
        raise RuntimeError("no tile produced any cells")
    allsp = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    n_cells = int(allsp["cell"].max())
    gt = int(prep.get("gt_expected_cells", 0))
    logging.info(f"merged cells = {n_cells:,}" +
                 (f"   platform {gt:,}   ratio {n_cells/gt:.3f}" if gt else ""))
    logging.info(f"spots assigned = {len(allsp):,} / {prep['n_transcripts']:,} "
                 f"({len(allsp)/max(prep['n_transcripts'],1)*100:.1f}%)")

    # ---- raster on the coarse comparison grid -------------------------
    col = (allsp["global_x"].to_numpy() // bf).astype(np.int64)
    row = (allsp["global_y"].to_numpy() // bf).astype(np.int64)
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    flat = (row[ok] * W + col[ok])
    lab = allsp["cell"].to_numpy()[ok]
    order = np.lexsort((lab, flat))
    flat_s, lab_s = flat[order], lab[order]
    seg = np.zeros(H * W, dtype=np.int32)
    # majority label per pixel: longest run wins
    starts = np.flatnonzero(np.r_[True, (flat_s[1:] != flat_s[:-1]) |
                                  (lab_s[1:] != lab_s[:-1])])
    run_len = np.diff(np.r_[starts, len(flat_s)])
    run_pix, run_lab = flat_s[starts], lab_s[starts]
    best = np.zeros(H * W, dtype=np.int64)
    for pix, lb, ln in zip(run_pix, run_lab, run_len):
        if ln > best[pix]:
            best[pix] = ln
            seg[pix] = lb
    seg = seg.reshape(H, W)
    logging.info(f"raster {H} x {W}: "
                 f"{int(len(np.unique(seg)) - 1):,} cells, "
                 f"{(seg > 0).mean()*100:.2f}% occupied")
    tifffile.imwrite(str(out_root / "comseg_segmentation.tif"), seg)
    logging.info(f"raster -> {out_root / 'comseg_segmentation.tif'}")

    # ---- exact full-panel count matrix from per-RNA assignment --------
    genes, gcode = np.unique(allsp["gene"].to_numpy(), return_inverse=True)
    n_genes = len(genes)
    mat = np.bincount((allsp["cell"].to_numpy() - 1) * n_genes + gcode,
                      minlength=n_cells * n_genes).astype(np.int32)
    mat = mat.reshape(n_cells, n_genes)
    counts = mat.sum(axis=1)
    cen = allsp.groupby("cell")[["global_x", "global_y"]].mean()
    cen = cen.reindex(np.arange(1, n_cells + 1))
    np.save(out_root / "cell_by_gene.npy", mat)
    (out_root / "cell_by_gene_genes.txt").write_text("\n".join(genes) + "\n")
    logging.info(f"matrix -> {out_root / 'cell_by_gene.npy'}  shape {mat.shape}")
    logging.info(f"counts/cell: median={np.median(counts):.1f} "
                 f"mean={counts.mean():.1f}")

    try:
        import anndata as ad
        from scipy import sparse

        obs = pd.DataFrame({
            "n_counts": counts,
            "x_um": (cen["global_x"].to_numpy() + prep["x0"]) * ps,
            "y_um": (cen["global_y"].to_numpy() + prep["y0"]) * ps,
        }, index=[f"cell_{i+1}" for i in range(n_cells)])
        adata = ad.AnnData(X=sparse.csr_matrix(mat), obs=obs,
                           var=pd.DataFrame(index=genes))
        adata.uns["method"] = "comseg"
        adata.uns["prior_source"] = args.prior_source
        adata.write_h5ad(out_root / "comseg_cells.h5ad")
        logging.info(f"h5ad -> {out_root / 'comseg_cells.h5ad'}")
    except Exception as e:  # noqa: BLE001
        logging.info(f"(h5ad export skipped: {e})")


# ----------------------------------------------------------------------
# QC
# ----------------------------------------------------------------------
def spread(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return "empty"
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return f"med={q[2]:.1f}  IQR={q[1]:.1f}-{q[3]:.1f}  p5={q[0]:.1f} p95={q[4]:.1f}"


def qc_report(out_root, prep, compare_to=None):
    import tifffile

    banner("QC REPORT")
    seg_p = out_root / "comseg_segmentation.tif"
    if not seg_p.exists():
        logging.info("no comseg_segmentation.tif yet")
        return
    ps, bf = prep["pixel_size"], prep["bin_factor"]
    px_um2 = (bf * ps) ** 2
    seg = tifffile.imread(str(seg_p))
    lab = seg.reshape(-1).astype(np.int64)
    area = np.bincount(lab, minlength=int(lab.max()) + 1)[1:]
    area = area[area > 0]
    gt = int(prep.get("gt_expected_cells", 0))
    extra = f"   (platform {gt:,}, ratio {len(area)/gt:.3f})" if gt else ""
    print(f"  cells            {len(area):,}{extra}")
    print(f"  coverage         {(seg > 0).mean()*100:.2f}% of grid")
    print(f"  area (um^2)      {spread(area * px_um2)}")
    print(f"  equiv diam (um)  {spread(2*np.sqrt(area*px_um2/np.pi))}")
    cbg = out_root / "cell_by_gene.npy"
    if cbg.exists():
        c = np.load(cbg).sum(axis=1)
        print(f"  counts / cell    {spread(c)}")

    if compare_to:
        other_dir = Path(compare_to)
        other_p = next((p for p in (other_dir / "ucs_segmentation.tif",
                                    other_dir / "proseg_segmentation.tif",
                                    other_dir / "cellist_segmentation.tif")
                        if p.exists()), None)
        if other_p is None:
            logging.info(f"(nothing to compare against in {compare_to})")
            return
        b_img = tifffile.imread(str(other_p))
        if b_img.shape != seg.shape:
            logging.info(f"(grid mismatch {b_img.shape} vs {seg.shape})")
            return
        bb = np.bincount(b_img.reshape(-1).astype(np.int64))[1:]
        bb = bb[bb > 0]
        a, b = seg > 0, b_img > 0
        dice = 2 * int((a & b).sum()) / max(int(a.sum()) + int(b.sum()), 1)
        print(f"\n  --- vs {other_p.name} ---")
        print(f"  cells            comseg {len(area):,}   other {len(bb):,}")
        print(f"  median area      comseg {np.median(area)*px_um2:.1f} um^2   "
              f"other {np.median(bb)*px_um2:.1f} um^2")
        print(f"  foreground Dice  {dice:.3f}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(
        description="ComSeg on a centered Xenium ROI, tiled.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--stage", choices=["prep", "run", "merge", "all", "report"],
                   default="all")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-dir", default=None)

    p.add_argument("--roi-size", type=int, default=10000, help="ROI edge in NATIVE px")
    p.add_argument("--roi-origin", type=int, nargs=2, default=None, metavar=("Y0", "X0"))
    p.add_argument("--bin-factor", type=int, default=10,
                   help="coarse comparison grid; must match the UCS run")
    p.add_argument("--mask-bin", type=int, default=2,
                   help="native px per prior-mask pixel")
    p.add_argument("--tile-px", type=int, default=2000,
                   help="tile core size in NATIVE px")
    p.add_argument("--tile-overlap-px", type=int, default=250,
                   help="overlap margin in NATIVE px; >= max cell radius")
    p.add_argument("--min-tile-transcripts", type=int, default=500)
    p.add_argument("--pixel-size", type=float, default=None)
    p.add_argument("--morphology", default=None)
    p.add_argument("--qv-min", type=float, default=20.0)
    p.add_argument("--batch-size", type=int, default=2_000_000)
    p.add_argument("--use-z", action="store_true", help="keep real z_location")
    p.add_argument("--prior-source", choices=["platform", "cellist"],
                   default="platform")

    # ComSeg hyperparameters
    p.add_argument("--mean-cell-diameter-um", type=float, default=15.0)
    p.add_argument("--max-cell-radius-um", type=float, default=25.0)
    p.add_argument("--k-nearest-neighbors", type=int, default=10)
    p.add_argument("--n-neighbors", type=int, default=40)
    p.add_argument("--sampling-size", type=int, default=10000)
    p.add_argument("--size-commu-min", type=int, default=3)
    p.add_argument("--norm-vector", action="store_true",
                   help="sctransform normalisation; needs R sctransform/feather/arrow")
    p.add_argument("--n-pcs", type=int, default=4)
    p.add_argument("--clustering-method", default="leiden",
                   choices=["leiden", "louvain", "kmeans"])
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--n-clusters-kmeans", type=int, default=5)
    p.add_argument("--min-merge-correlation", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--min-rna-per-cell", type=int, default=5)

    p.add_argument("--compare-to", default=None)
    p.add_argument("--force-prep", action="store_true")
    p.add_argument("--force-run", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    cfg = DATASETS[args.dataset]
    root = Path(args.data_root or cfg["root"])
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if args.bin_factor % args.mask_bin:
        raise ValueError(f"--mask-bin {args.mask_bin} must divide "
                         f"--bin-factor {args.bin_factor}")
    out_root = Path(args.out_dir or OUT_TEMPLATE.format(tag=cfg["tag"],
                                                        roi=args.roi_size))
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "logs", f"comseg_{cfg['tag']}_{args.stage}")

    banner(f"COMSEG  dataset={args.dataset}  stage={args.stage}")
    logging.info(f"prior source = {args.prior_source}")
    logging.info(f"data root  = {root}")
    logging.info(f"output dir = {out_root}")

    morph = Path(args.morphology) if args.morphology else \
        resolve_first(root, MORPH_CANDIDATES, "morphology image")
    tx = resolve_first(root, TRANSCRIPT_CANDIDATES, "transcripts.parquet")
    nb = resolve_first(root, NUCBOUND_CANDIDATES, "nucleus_boundaries.parquet")
    ps = args.pixel_size or read_pixel_size(root)
    nc, ih, iw = image_level0_shape(morph)
    y0, x0, sy, sx = compute_roi(ih, iw, args.roi_size, args.roi_origin)
    logging.info(f"morphology = {morph}  (channels={nc}, {ih} x {iw})")
    logging.info(f"pixel size = {ps} um/px")
    logging.info(f"ROI = y[{y0},{y0+sy}) x[{x0},{x0+sx}) = {sy*ps:.1f} x {sx*ps:.1f} um")

    if args.tile_overlap_px * ps < args.max_cell_radius_um:
        logging.warning(
            f"--tile-overlap-px {args.tile_overlap_px} = "
            f"{args.tile_overlap_px*ps:.1f} um is below --max-cell-radius-um "
            f"{args.max_cell_radius_um}; cells near tile edges may be truncated")

    meta = dict(dataset=args.dataset, y0=y0, x0=x0, size_y=sy, size_x=sx,
                bin_factor=args.bin_factor, mask_bin=args.mask_bin,
                tile_px=args.tile_px, tile_overlap_px=args.tile_overlap_px,
                pixel_size=ps, qv_min=args.qv_min, roi_size=args.roi_size)

    prep_p = out_root / "input" / "prep_meta.json"
    tiles_p = out_root / "input" / "tile_metadata.tsv"

    if args.stage == "report":
        qc_report(out_root, json.loads(prep_p.read_text()), args.compare_to)
        return

    if args.stage == "prep" or not prep_p.exists() or args.force_prep:
        banner("STAGE prep")
        prep, tiles = prep_tiles(tx, nb, out_root, meta, args)
    else:
        prep = json.loads(prep_p.read_text())
        tiles = pd.read_csv(tiles_p, sep="\t")
        logging.info(f"reusing prep: {prep['n_tiles']} tiles, "
                     f"{prep['n_transcripts']:,} transcripts")

    if args.stage == "prep":
        banner("PREP DONE")
        logging.info(f"tiles: {prep['n_tiles']}   GT cells: "
                     f"{prep.get('gt_expected_cells')}")
        return

    if args.stage in ("run", "all"):
        banner("STAGE run")
        run_tiles(out_root, prep, tiles, args)
    if args.stage in ("merge", "all"):
        banner("STAGE merge")
        merge_tiles(out_root, prep, tiles, args)
        qc_report(out_root, prep, args.compare_to)
    banner("ALL DONE")


if __name__ == "__main__":
    main()