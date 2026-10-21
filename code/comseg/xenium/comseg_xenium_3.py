#!/usr/bin/env python
"""
ComSeg x Xenium (breast cancer / lymph node) - full run, no ROI crop

The datasets are small (breast cancer 8,809 cells, lymph node 2,442 cells), so run
them in full and reuse the validated transcripts.parquet + cells.parquet loading
from ucs_value_xenium_3.py / cellist_value_xenium_3.py (qv>=20 filter; the GT
cell_id is the cell_id column in the transcript table, so unlike CosMx/Stereo
no centroid has to be rebuilt).

Two modes (same as the CosMx/Stereo version):
  default (with-prior): all cell_id values come from the official GT segmentation
  --no-transcript-prior: all cell_id set to 0, keeping only centroid landmarks

Usage:
  python comseg_xenium_full.py --dataset xenium_breast
  python comseg_xenium_full.py --dataset xenium_lymph --no-transcript-prior
"""
import os, sys, time, pickle, inspect, argparse, subprocess

import numpy as np
import pandas as pd

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

DATASETS = {
    "xenium_breast": dict(
        data_dir="yourpath/dataset/xenium_roi_crop",
        gt_cells=8809),
    "xenium_lymph": dict(
        data_dir="yourpath/dataset/xenium_roi_crop",
        gt_cells=2442),
}

QV_MIN = 20.0
TOP_N = 300


def get_dirs(dataset, no_prior):
    base = "yourpath/comseg_xenium"
    suffix = "_noprior" if no_prior else ""
    d = dict(
        cache=f"{base}/{dataset}/cache{suffix}",
        centroid=f"{base}/{dataset}/cache{suffix}_centroid_only",
        out=f"{base}/{dataset}/output{suffix}",
    )
    for v in d.values():
        os.makedirs(v, exist_ok=True)
    return d


def safe_kwargs(fn, want, label=""):
    try:
        params = set(inspect.signature(fn).parameters.keys())
    except (TypeError, ValueError):
        return want
    use = {k: v for k, v in want.items() if k in params}
    drop = [k for k in want if k not in params]
    if label:
        P(f"    {label} passed: {sorted(use.keys())}"
          + (f"   dropped (unsupported in this version): {drop}" if drop else ""))
    return use


def kill_dict_centroid(*objs):
    for owner in objs:
        if owner is None:
            continue
        targets = [owner]
        ds = getattr(owner, "dataset", None)
        if ds is not None:
            targets.append(ds)
        for t in targets:
            cur = getattr(t, "dict_centroid", "__NO_ATTR__")
            if cur is None or cur == "__NO_ATTR__":
                continue
            t.dict_centroid = None


def find_one(base, patterns):
    import glob
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(base, "**", pat), recursive=True))
        if hits:
            hits.sort(key=len)
            return hits[0]
    raise FileNotFoundError(f"not found: {patterns} in {base}")


def prepare_data(cfg, dirs):
    ds_cfg = DATASETS[cfg.dataset]
    data_dir = ds_cfg["data_dir"]

    suffix = "_noprior" if cfg.no_transcript_prior else ""
    tag = f"{cfg.dataset}_top{cfg.top_n}{suffix}"
    txp = f"{dirs['cache']}/{tag}_tx.csv"
    cenp = f"{dirs['cache']}/{tag}_centroid.csv"

    if os.path.exists(txp) and os.path.exists(cenp) and not cfg.force:
        tx = pd.read_csv(txp)
        nuc = pd.read_csv(cenp)
        P(f"  reusing transcript CSV ({len(tx):,} rows) and centroid table ({len(nuc):,} rows)")
        return tag, txp, cenp, nuc

    f_tx = find_one(data_dir, ["transcripts.parquet"])
    f_cells = find_one(data_dir, ["cells.parquet", "cells.csv"])
    P(f"  reading transcripts: {f_tx}")
    tx_raw = pd.read_parquet(f_tx, columns=["x_location", "y_location",
                                            "feature_name", "qv", "cell_id"])
    tx_raw = tx_raw[tx_raw["qv"] >= QV_MIN].reset_index(drop=True)
    cid = tx_raw["cell_id"].astype(str)
    tx_raw["cell_id_str"] = cid.where(cid != "UNASSIGNED", "0")
    P(f"    {len(tx_raw):,} transcripts (qv>={QV_MIN})")

    if f_cells.endswith(".parquet"):
        cells = pd.read_parquet(f_cells)
    else:
        cells = pd.read_csv(f_cells)
    cells["cell_id"] = cells["cell_id"].astype(str)
    P(f"  GT cells: {len(cells):,}")

    # local renumbering: official cell_id strings -> integers 1..N (ComSeg needs numeric ids)
    uniq = sorted(cells["cell_id"].unique().tolist())
    uid2i = {c: i + 1 for i, c in enumerate(uniq)}
    cells["cell_id_local"] = cells["cell_id"].map(uid2i)

    top_genes = tx_raw.feature_name.value_counts().head(cfg.top_n).index
    before_n = len(tx_raw)
    tx_raw = tx_raw[tx_raw.feature_name.isin(top_genes)].reset_index(drop=True)
    P(f"  top-{cfg.top_n} gene filter: {before_n:,} -> {len(tx_raw):,} transcripts")

    local_cell = tx_raw["cell_id_str"].map(uid2i).fillna(0).astype(int)

    if cfg.no_transcript_prior:
        tx = pd.DataFrame({
            "x": tx_raw.x_location.values, "y": tx_raw.y_location.values,
            "gene": tx_raw.feature_name.values,
            "cell_id": 0,
            "_landmark_for_centroid": local_cell.values,
        })
    else:
        tx = pd.DataFrame({
            "x": tx_raw.x_location.values, "y": tx_raw.y_location.values,
            "gene": tx_raw.feature_name.values,
            "cell_id": local_cell.values,
        })
    tx.to_csv(txp, index=False)

    landmark_col = "_landmark_for_centroid" if cfg.no_transcript_prior else "cell_id"
    xcol = "x_centroid" if "x_centroid" in cells.columns else "x"
    ycol = "y_centroid" if "y_centroid" in cells.columns else "y"
    nuc = pd.DataFrame({
        "cell_id": cells["cell_id_local"].values,
        "x": cells[xcol].values,
        "y": cells[ycol].values,
    })
    nuc["z"] = nuc.cell_id * 0.001
    nuc.to_csv(cenp, index=False)
    P(f"  centroid table: {len(nuc):,} landmarks (z range {nuc.z.min():.3f}~{nuc.z.max():.3f})")

    return tag, txp, cenp, nuc


def run_comseg(tag, txp, cenp, nuc, cfg, dirs):
    from comseg.dataset import ComSegDataset
    from comseg.dictionary import ComSegDict

    img_name = os.path.basename(txp).replace(".csv", "")
    mmc_tag = str(cfg.min_merge_correlation).replace(".", "p")
    ck_pkl = f"{dirs['cache']}/comsegdict_{tag}_mmc{mmc_tag}.pkl"

    P(f"\n[2] ComSegDataset")
    ds_kw = safe_kwargs(ComSegDataset.__init__, dict(
        path_dataset_folder=dirs["cache"], prior_name="cell_id",
        image_csv_files=[os.path.basename(txp)],
        centroid_csv_files=[os.path.basename(cenp)],
        path_cell_centroid=dirs["cache"],
        dict_scale={"x": 1, "y": 1, "z": 1},
        mean_cell_diameter=cfg.cell_diam, gene_column="gene",
    ), "ComSegDataset")
    t0 = time.time()
    data = ComSegDataset(**ds_kw)
    P(f"    ready ({time.time()-t0:.1f}s)")

    P(f"\n[3] compute_edge_weight")
    ew_kw = safe_kwargs(data.compute_edge_weight, dict(
        sampling=True, sampling_size=cfg.sampling_size,
        n_neighbors=cfg.n_neighbors,
    ), "compute_edge_weight")
    t0 = time.time()
    data.compute_edge_weight(**ew_kw)
    P(f"    done ({time.time()-t0:.0f}s)")

    P(f"\n[4] ComSegDict")
    dd_kw = safe_kwargs(ComSegDict.__init__, dict(
        dataset=data, mean_cell_diameter=cfg.cell_diam,
        community_detection="with_prior",
    ), "ComSegDict")
    comseg_dict = ComSegDict(**dd_kw)

    P(f"\n[5] clear dict_centroid")
    kill_dict_centroid(data, comseg_dict)

    P(f"\n[6] write the dedicated classify_centroid centroid file")
    nuc.to_csv(f"{dirs['centroid']}/{img_name}.csv", index=False)

    P(f"\n[7] run_all  min_merge_correlation={cfg.min_merge_correlation}")
    ra_kw = safe_kwargs(comseg_dict.run_all, dict(
        max_cell_radius=cfg.max_cell_radius,
        k_nearest_neighbors=cfg.k_nn, n_neighbors=cfg.n_neighbors,
        path_dataset_folder_centroid=dirs["centroid"],
        min_merge_correlation=cfg.min_merge_correlation,
        resolution=cfg.resolution,
    ), "run_all")
    t0 = time.time()
    comseg_dict.run_all(**ra_kw)
    P(f"    done ({time.time()-t0:.0f}s)")

    with open(ck_pkl, "wb") as f:
        pickle.dump(comseg_dict, f, protocol=4)
    return comseg_dict, img_name


def export(comseg_dict, tag, cfg, dirs):
    P(f"\n[8] export anndata")
    ad_kw = safe_kwargs(comseg_dict.anndata_from_comseg_result, dict(
        min_rna_per_cell=cfg.min_rna_per_cell, return_polygon=False,
    ), "anndata_from_comseg_result")
    ret = comseg_dict.anndata_from_comseg_result(**ad_kw)
    adata = ret[0] if isinstance(ret, tuple) else ret
    if adata is None or getattr(adata, "n_obs", 0) == 0:
        P("    anndata is empty"); return None
    mmc_tag = str(cfg.min_merge_correlation).replace(".", "p")
    out = f"{dirs['out']}/{tag}_mmc{mmc_tag}.h5ad"
    adata.write_h5ad(out)
    P(f"  {adata.n_obs:,} cells x {adata.n_vars} genes -> {out}")
    df_spots = adata.uns.get("df_spots")
    if df_spots:
        key = list(df_spots.keys())[0]
        P(f"  leiden_merged actual clusters: {df_spots[key].leiden_merged.nunique()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--no-transcript-prior", action="store_true")
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--cell-diam", type=float, default=10.0)
    ap.add_argument("--sampling-size", type=int, default=2000)
    ap.add_argument("--k-nn", type=int, default=2)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--max-cell-radius", type=float, default=15.0)
    ap.add_argument("--min-rna-per-cell", type=int, default=5)
    ap.add_argument("--min-merge-correlation", type=float, default=0.99)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args()

    dirs = get_dirs(cfg.dataset, cfg.no_transcript_prior)
    mode = "no transcript prior" if cfg.no_transcript_prior else "with prior"
    P(SEP); P(f"ComSeg x {cfg.dataset} - {mode} mode (full run, no ROI crop)"); P(SEP)

    tag, txp, cenp, nuc = prepare_data(cfg, dirs)
    comseg_dict, img_name = run_comseg(tag, txp, cenp, nuc, cfg, dirs)
    export(comseg_dict, tag, cfg, dirs)


if __name__ == "__main__":
    main()