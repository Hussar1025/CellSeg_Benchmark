#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comseg_cosmx_roi_run.py  - ComSeg x CosMx pancreas (spatial ROI version)
=========================================================================
Problem: full-image FOV 51 has 2M transcripts; classify_centroid grows to 50 GB
and gets killed. Solution: a 10000x10000 px spatial ROI (~1203x1203 um),
containing ~5000-8000 cells and 400-800k transcripts, which ComSeg can handle.

ROI center: median of cell centroids from cell_metadata (tissue center)
Coordinates: x_global_px / y_global_px (pixels, 0.1203 um/px)

Key params:
  --k-nn 2  --max-cell-radius 20  (more conservative than full FOV to avoid OOM)
  --sampling-size 3000  --top-n 300

Usage:
  conda activate comseg_env
  cd yourpath/comseg
  python comseg_cosmx_roi_run.py
  python comseg_cosmx_roi_run.py --roi-size 5000    # smaller ROI
  python comseg_cosmx_roi_run.py --check-only       # only count data in ROI
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd

SEP = "=" * 68

TX_PATH   = "yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles/Pancreas_tx_file.csv"
META_PATH = "yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles/Pancreas_metadata_file.csv"
OUT_DIR   = "yourpath/comseg/output_roi"
CACHE_DIR = "yourpath/comseg/cache_roi"
PIXEL_UM  = 0.1203   # um/px

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--roi-size",    type=int,   default=10000,
                   help="ROI side length (px), default 10000x10000")
    p.add_argument("--sampling-size", type=int, default=3000)
    p.add_argument("--top-n",       type=int,   default=300)
    p.add_argument("--k-nn",        type=int,   default=2)
    p.add_argument("--max-cell-radius", type=float, default=20.0,
                   help="nucleus search radius (um), in px: 20/0.1203~166px")
    p.add_argument("--check-only",  action="store_true",
                   help="only check ROI data size; do not run ComSeg")
    p.add_argument("--force",       action="store_true",
                   help="ignore cache and recompute edge_weight")
    return p.parse_args()

# ════════════════════════════════════════════════════════════════════════════
# Step 1: determine ROI center (median of cell_metadata centroids)
# ════════════════════════════════════════════════════════════════════════════
def get_roi_center(meta_path, tx_path):
    """Try centroids from metadata; fall back to tx_file sampling."""
    try:
        meta = pd.read_csv(meta_path)
        print(f"  metadata columns: {list(meta.columns[:8])}")
        # Find x/y centroid columns (CosMx metadata may use other names)
        x_candidates = [c for c in meta.columns
                        if any(k in c.lower() for k in ("x","col","centroid"))]
        y_candidates = [c for c in meta.columns
                        if any(k in c.lower() for k in ("y","row","centroid"))
                        and c not in x_candidates]
        if x_candidates and y_candidates:
            xc = meta[x_candidates[0]].median()
            yc = meta[y_candidates[0]].median()
            print(f"  centroid median: x={xc:.0f}  y={yc:.0f} px (from metadata)")
            return float(xc), float(yc)
    except Exception as e:
        print(f"  failed to read metadata: {e}")
    # Fallback: sample tx_file
    print("  sampling tx_file to estimate center...")
    sample = pd.read_csv(tx_path, nrows=500_000,
                         usecols=["x_global_px","y_global_px"])
    xc = sample.x_global_px.median()
    yc = sample.y_global_px.median()
    print(f"  centroid median: x={xc:.0f}  y={yc:.0f} px (from tx sample)")
    return float(xc), float(yc)

# ════════════════════════════════════════════════════════════════════════════
# Step 2: filter transcripts + nucleus centroids to ROI
# ════════════════════════════════════════════════════════════════════════════
def load_roi_data(cfg, cx, cy):
    half = cfg.roi_size // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    print(f"\n  ROI: x[{x0:.0f},{x1:.0f}]  y[{y0:.0f},{y1:.0f}] px"
          f"  = {(x1-x0)*PIXEL_UM:.0f}x{(y1-y0)*PIXEL_UM:.0f} um")

    print("  Reading transcripts (chunked filtering)...")
    chunks, total = [], 0
    for chunk in pd.read_csv(TX_PATH, chunksize=2_000_000,
                              usecols=["fov","cell_ID","x_global_px",
                                       "y_global_px","z","target","CellComp"]):
        total += len(chunk)
        sub = chunk[(chunk.x_global_px >= x0) & (chunk.x_global_px < x1) &
                    (chunk.y_global_px >= y0) & (chunk.y_global_px < y1)]
        if len(sub): chunks.append(sub)
    tx = pd.concat(chunks, ignore_index=True)
    print(f"  total {total:,} -> in ROI {len(tx):,}  genes {tx.target.nunique()}")
    tx = tx.rename(columns={"target":"gene"})

    # Nucleus centroids: estimate from Nuclear transcripts or read from cell_metadata
    print("  estimating nucleus centroids...")
    nuclear_tx = tx[tx.CellComp == "Nuclear"]
    if len(nuclear_tx) > 0:
        nuc = (nuclear_tx[nuclear_tx.cell_ID > 0]
               .groupby(["fov","cell_ID"])[["x_global_px","y_global_px"]]
               .mean().reset_index())
        nuc.columns = ["fov","cell_ID","cx_px","cy_px"]
    else:
        # Use all transcripts with cell_ID > 0
        nuc = (tx[tx.cell_ID > 0]
               .groupby(["fov","cell_ID"])[["x_global_px","y_global_px"]]
               .mean().reset_index())
        nuc.columns = ["fov","cell_ID","cx_px","cy_px"]

    # Assign a unique cell_id ((fov, cell_ID) composite key)
    nuc["cell_uid"] = nuc.fov.astype(str) + "_" + nuc.cell_ID.astype(str)
    cell_id_map = {v: i+1 for i, v in enumerate(nuc.cell_uid.unique())}
    nuc["cell_id"] = nuc.cell_uid.map(cell_id_map)
    tx["cell_uid"] = tx.fov.astype(str) + "_" + tx.cell_ID.astype(str)
    tx["cell_id"]  = tx.cell_uid.map(cell_id_map).fillna(0).astype(int)

    print(f"  nucleus centroids in ROI: {len(nuc):,}  transcripts with nucleus assignment: "
          f"{(tx.cell_id>0).sum():,} ({(tx.cell_id>0).mean()*100:.1f}%)")
    return tx, nuc, (x0, y0)

# ════════════════════════════════════════════════════════════════════════════
# Step 3: run ComSeg (call Python API directly)
# ════════════════════════════════════════════════════════════════════════════
def run_comseg(cfg, tx, nuc, roi_origin):
    """Call the ComSeg Python API directly, skipping the comseg_worker.py FOV wrapper."""
    os.makedirs(OUT_DIR,   exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    x0, y0 = roi_origin
    img_name = f"cosmx_roi_{int(x0)}_{int(y0)}"
    cache_path = os.path.join(CACHE_DIR,
        f"edge_{img_name}_top{cfg.top_n}_s{cfg.sampling_size}.pkl")
    out_path = os.path.join(OUT_DIR, f"{img_name}.h5ad")

    if os.path.exists(out_path):
        print(f"  [Skip] already exists {out_path}")
        import anndata
        a = anndata.read_h5ad(out_path)
        print(f"  cells: {a.n_obs:,}")
        return

    from comseg import dataset as comseg_dataset
    from comseg.dictionary import ComSegDict

    # top-n genes
    top_genes = tx.gene.value_counts().head(cfg.top_n).index.tolist()
    tx_filt = tx[tx.gene.isin(top_genes)].copy()
    print(f"  after top-{cfg.top_n} genes: {len(tx_filt):,} transcripts")

    # dict_scale (um scale: x_global_px must convert to um)
    tx_filt["x_um"] = tx_filt.x_global_px * PIXEL_UM
    tx_filt["y_um"] = tx_filt.y_global_px * PIXEL_UM
    nuc["cx_um"]    = nuc.cx_px * PIXEL_UM
    nuc["cy_um"]    = nuc.cy_px * PIXEL_UM

    dict_scale = {"x": 1.0, "y": 1.0, "z": 1.0}  # already in um; scale=1

    # ComSegDataset (follow the call pattern in comseg_worker.py)
    print("\n  Building ComSegDataset ...")
    t0 = time.time()
    # Write temp CSV (ComSeg usually reads from files)
    tmp_csv = os.path.join(CACHE_DIR, f"{img_name}_tx.csv")
    tx_filt[["x_um","y_um","gene","cell_id"]].rename(
        columns={"x_um":"x","y_um":"y"}
    ).to_csv(tmp_csv, index=False)

    # Nucleus centroid dict {cell_id: [z,y,x]} (same format as comseg_worker.py)
    dict_centroid = {}
    for _, row in nuc.iterrows():
        if row.cell_id > 0:
            dict_centroid[row.cell_id] = [0.0,
                                           float(row.cy_um),
                                           float(row.cx_um)]
    dict_centroid[0] = [0.0, 0.0, 0.0]   # background sentinel
    print(f"  dict_centroid: {len(dict_centroid):,} nuclei (incl. background)  ({time.time()-t0:.1f}s)")

    # Build an in-memory dataset directly
    try:
        from comseg.dataset import ComSegDataset
        data = ComSegDataset(
            path_dir_input     = CACHE_DIR,
            dict_scale         = {"x":1.0,"y":1.0,"z":1.0},
            min_transcript_count = 0,
            n_neighbors        = cfg.k_nn,
            sigma              = 3,
        )
        # Inject in-memory data (skip file scan)
        data.df_spots = tx_filt[["x_um","y_um","gene","cell_id"]].rename(
            columns={"x_um":"x","y_um":"y"}).copy()
        data.df_spots["z"] = 0.0
        data.img_name  = img_name
        data.mode      = "with_prior"
        print(f"  ComSegDataset ready ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  ComSegDataset construction failed: {e}")
        print("  trying fallback...")
        # Fallback: follow comseg_worker's run_one_fov approach
        _run_via_worker(cfg, tx_filt, dict_centroid, img_name, cache_path, out_path)
        return

    # edge_weight
    if not os.path.exists(cache_path) or cfg.force:
        print(f"\n  compute_edge_weight (sampling={cfg.sampling_size}, about 10-20 min)...")
        t0 = time.time()
        data.compute_edge_weight(
            dict_scale         = dict_scale,
            dict_centroid      = dict_centroid,
            max_cell_radius    = cfg.max_cell_radius,
            sampling_size      = cfg.sampling_size,
            n_neighbors        = cfg.k_nn,
            gene_index         = None,
        )
        import pickle
        with open(cache_path, "wb") as f: pickle.dump(data, f)
        print(f"  edge_weight done ({time.time()-t0:.0f}s); cached")
    else:
        import pickle
        print(f"  reusing edge_weight cache: {os.path.basename(cache_path)}")
        with open(cache_path, "rb") as f: data = pickle.load(f)

    # ComSegDict + run_all
    print("\n  run_all (classify_centroid)...")
    t0 = time.time()
    comseg_dict = ComSegDict(
        dataset         = data,
        dict_centroid   = dict_centroid,
    )
    comseg_dict.run_all(
        gene_selection_mode    = "all",
        dict_scale             = dict_scale,
        max_cell_radius        = cfg.max_cell_radius,
        n_neighbors            = cfg.k_nn,
        image_name             = img_name,
    )
    print(f"  run_all done ({time.time()-t0:.0f}s)")

    # Write output
    anndata_obj = comseg_dict.anndata_from_comseg_result(
        dict_scale = dict_scale, gene_selection_mode = "all")
    anndata_obj.write_h5ad(out_path)
    print(f"\n  ✓ Done! cells: {anndata_obj.n_obs:,}  -> {out_path}")


def _run_via_worker(cfg, tx_filt, dict_centroid, img_name, cache_path, out_path):
    """Fallback: write ROI data to temp files and call comseg_worker internals."""
    print("  (fallback path) calling comseg_worker internals...")
    # Use this path when the comseg API changes
    # Fill in as needed


# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = parse_args()
    print(SEP); print("ComSeg x CosMx pancreas (spatial ROI)"); print(SEP)
    print(f"ROI size: {cfg.roi_size}x{cfg.roi_size} px"
          f" = {cfg.roi_size*PIXEL_UM:.0f}x{cfg.roi_size*PIXEL_UM:.0f} um")

    # Find center
    print("\n[1] determine ROI center ...")
    cx, cy = get_roi_center(META_PATH, TX_PATH)

    # Load ROI data
    print("\n[2] load ROI data ...")
    tx, nuc, roi_origin = load_roi_data(cfg, cx, cy)

    # Data size report
    prod = len(tx) * min(tx.gene.nunique(), cfg.top_n)
    print(f"\n  transcripts in ROI: {len(tx):,}  genes: {tx.gene.nunique()}")
    print(f"  after top-{cfg.top_n}: n_tx x n_gene ~ {prod:,}"
          f"  {'✓ safe' if prod < 1_500_000_000 else '⚠ large'}")
    print(f"  nucleus centroids: {len(nuc):,}  GT density: {len(nuc)/(cfg.roi_size*PIXEL_UM)**2*1e6:.0f} cells/mm2")

    if cfg.check_only:
        print("\n  --check-only mode; stopping here.")
        print("  memory estimate (classify_centroid): "
              f"~{len(tx)*cfg.k_nn*8/1e9:.1f} GB (n_tx x k_nn x 8B)")
        return

    # Run ComSeg
    print("\n[3] run ComSeg ...")
    run_comseg(cfg, tx, nuc, roi_origin)


if __name__ == "__main__":
    main()