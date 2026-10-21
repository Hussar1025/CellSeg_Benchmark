#!/usr/bin/env python
"""
ComSeg x Stereo-seq - center-ROI version (reuses the dual-mode design from comseg_comsx_roi.py)

★ This version fixes a key bug: previously the raw global GEM cell id (cell column)
  was used directly as landmark cell_id, without remapping to ROI-local 1..N ids
  as the CosMx version does. GEM's cell column is CellBin's global id for the whole
  tissue section; its magnitude depends on the cell's rank in the section, not on
  where the ROI is cut. E14.5_E1S3/E16.5_E2S7 happen to fall in regions with cell
  ids in the hundreds of thousands. With z = cell_id * 0.001 (to avoid qhull 2D
  degeneracy), the resulting z values (175~326) were far larger than the plausible
  x/y range, so z dominated the 3D distance; every transcript "looked" too far from
  every landmark, all assignments failed (cell_index_pred all 0), and export
  crashed because list_expression_vectors was empty.
  Fix: after cropping the ROI, remap the cell column to local contiguous 1..N ids
  (exactly like uid2id in comseg_comsx_roi.py) before any further computation,
  including z values and UMI expansion.

Usage:
  python comseg_stereo_roi.py --dataset E16.5_E2S6
  python comseg_stereo_roi.py --dataset E16.5_E2S6 --no-transcript-prior --min-merge-correlation 0.99
  python comseg_stereo_roi.py --dataset E14.5_E1S3 --no-transcript-prior --min-merge-correlation 0.99 --force
  python comseg_stereo_roi.py --dataset E16.5_E2S6 --status
"""
import os
import sys
import time
import gzip
import pickle
import inspect
import argparse
import subprocess

import numpy as np
import pandas as pd

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

DATA_DIR = "yourpath/dataset/stereoseq/stereo-seq_data_all"
BIN_UM = 0.5

DATASETS = {
    "E14.5_E1S3": dict(
        gem=f"{DATA_DIR}/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=4872),
    "E16.5_E2S6": dict(
        gem=f"{DATA_DIR}/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=6229),
    "E16.5_E2S7": dict(
        gem=f"{DATA_DIR}/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        gt_cells=7960),
}


def get_dirs(dataset, no_prior):
    base = "yourpath/comseg_stereo"
    suffix = "_noprior" if no_prior else ""
    d = dict(
        cache=f"{base}/{dataset}/cache_roi{suffix}",
        centroid=f"{base}/{dataset}/cache_roi{suffix}_centroid_only",
        out=f"{base}/{dataset}/output_roi{suffix}",
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
            cur = getattr(t, "dict_centroid", "···NO_ATTR···")
            if cur is None or cur == "···NO_ATTR···":
                continue
            n = len(cur) if hasattr(cur, "__len__") else "?"
            P(f"    Set {type(t).__name__}.dict_centroid = None (had {n} entries)")
            t.dict_centroid = None


# ══════════════════════════════════════════════════════════════════════════
#  Step 0: read GEM to estimate bbox + density and target ROI half-width (um)
# ══════════════════════════════════════════════════════════════════════════
def load_full_gem(gem_path):
    P(f"  Reading GEM {os.path.basename(gem_path)} ...")
    t0 = time.time()
    opener = gzip.open if gem_path.endswith(".gz") else open
    skip = 0
    with opener(gem_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                skip += 1
            else:
                break
    gem = pd.read_csv(gem_path, sep="\t", skiprows=skip)
    ren = {}
    for c in gem.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "gene"
        elif cl == "x":
            ren[c] = "x_bin"
        elif cl == "y":
            ren[c] = "y_bin"
        elif cl in ("midcounts", "midcount", "count", "counts"):
            ren[c] = "count"
        elif cl in ("cell", "cellid", "label"):
            ren[c] = "cell"
    gem = gem.rename(columns=ren)
    if "count" not in gem.columns:
        gem["count"] = 1
    if "cell" not in gem.columns:
        gem["cell"] = 0
    P(f"    {len(gem):,} rows  {gem.gene.nunique():,} genes  "
      f"({time.time()-t0:.1f}s)")
    return gem


def estimate_roi_half_um(gem, gt_cells, target_cells):
    x_span_um = (gem.x_bin.max() - gem.x_bin.min()) * BIN_UM
    y_span_um = (gem.y_bin.max() - gem.y_bin.min()) * BIN_UM
    area_um2 = x_span_um * y_span_um
    density = gt_cells / area_um2
    target_area = target_cells / density
    side_um = target_area ** 0.5
    half_um = side_um / 2
    P(f"    overall bbox: {x_span_um:.0f} x {y_span_um:.0f} um  "
      f"area={area_um2:,.0f} um^2")
    P(f"    overall density: {density:.6f} cells/um^2  ({gt_cells:,} cells)")
    P(f"    target {target_cells:,} cells -> estimated square ROI side {side_um:.0f} um "
      f"(half-width {half_um:.0f} um)")
    return half_um


def find_roi_center_um(gem, random_state=0):
    samp = gem.sample(min(400_000, len(gem)), random_state=random_state)
    cx_um = float(samp.x_bin.median()) * BIN_UM
    cy_um = float(samp.y_bin.median()) * BIN_UM
    return cx_um, cy_um


# ══════════════════════════════════════════════════════════════════════════
#  Step 1: ROI crop + ★ cell_id local remap + UMI expansion + centroid table
# ══════════════════════════════════════════════════════════════════════════
def prepare_data(cfg, dirs):
    ds_cfg = DATASETS[cfg.dataset]
    gem = load_full_gem(ds_cfg["gem"])

    if cfg.roi_half_um is not None:
        half_um = cfg.roi_half_um
        P(f"  using manual ROI half-width: {half_um:.0f} um")
    else:
        half_um = estimate_roi_half_um(gem, ds_cfg["gt_cells"], cfg.target_cells)

    cx_um, cy_um = find_roi_center_um(gem)
    x0_um, x1_um = cx_um - half_um, cx_um + half_um
    y0_um, y1_um = cy_um - half_um, cy_um + half_um
    P(f"  ROI center ({cx_um:.0f},{cy_um:.0f}) um  "
      f"range x[{x0_um:.0f},{x1_um:.0f}] y[{y0_um:.0f},{y1_um:.0f}] um")

    suffix = "_noprior" if cfg.no_transcript_prior else ""
    tag = f"stereo_{cfg.dataset}_roi_{int(cx_um)}_{int(cy_um)}_top{cfg.top_n}_v2{suffix}"
    txp = f"{dirs['cache']}/{tag}_tx.csv"
    cenp = f"{dirs['cache']}/{tag}_centroid.csv"

    if os.path.exists(txp) and not cfg.force:
        tx = pd.read_csv(txp)
        P(f"  ⚡ reused transcript CSV: {len(tx):,} rows  columns={list(tx.columns)}")
    else:
        gem_x_um = gem.x_bin.values * BIN_UM
        gem_y_um = gem.y_bin.values * BIN_UM
        in_roi = ((gem_x_um >= x0_um) & (gem_x_um < x1_um) &
                 (gem_y_um >= y0_um) & (gem_y_um < y1_um))
        roi = gem[in_roi].reset_index(drop=True)
        P(f"  GEM rows in ROI: {len(roi):,} / {len(gem):,}")

        # ★★★ Key fix: remap raw global GEM cell ids to local contiguous
        #     1..N ids inside the ROI; do not use the raw cell column.
        #     Raw ids are CellBin global ids for the whole section and can be
        #     hundreds of thousands regardless of where the ROI is cut. With
        #     z=cell_id*0.001 they yield absurd z values and break 3D distance
        #     (verified empirically).
        raw_cell = roi["cell"].values
        uniq_raw = sorted(set(raw_cell[raw_cell > 0].tolist()))
        raw2local = {c: i + 1 for i, c in enumerate(uniq_raw)}
        local_cell = np.array([raw2local.get(c, 0) for c in raw_cell], dtype=np.int64)
        roi = roi.assign(cell_local=local_cell)
        n_gt_in_roi = len(uniq_raw)
        P(f"  actual GT cells in ROI: {n_gt_in_roi:,} "
          f"(target {cfg.target_cells:,}; if far off, tune with --roi-half-um)")
        P(f"  ★ remapped global cell ids (range {raw_cell[raw_cell>0].min() if n_gt_in_roi else 0}"
          f"~{raw_cell[raw_cell>0].max() if n_gt_in_roi else 0}) to "
          f"local ROI ids 1~{n_gt_in_roi}")

        top_genes = roi.gene.value_counts().head(cfg.top_n).index
        before_n = len(roi)
        roi = roi[roi.gene.isin(top_genes)].reset_index(drop=True)
        P(f"  ★ top-{cfg.top_n} gene filter: {before_n:,} -> {len(roi):,} rows "
          f"({len(roi)/before_n*100:.1f}%)")

        P("  Expanding UMI (each count becomes one synthetic molecule)...")
        rep = roi.loc[roi.index.repeat(roi["count"].clip(1).astype(int))].reset_index(drop=True)
        P(f"    after expansion: {len(rep):,} synthetic transcripts")

        if cfg.no_transcript_prior:
            tx = pd.DataFrame({
                "x": rep.x_bin.values * BIN_UM,
                "y": rep.y_bin.values * BIN_UM,
                "gene": rep.gene.values,
                "cell_id": 0,
                "_landmark_for_centroid": rep.cell_local.values.astype(int),
            })
            P(f"    ✓ {os.path.basename(txp)} (all cell_id = 0, no-transcript-prior mode)")
        else:
            tx = pd.DataFrame({
                "x": rep.x_bin.values * BIN_UM,
                "y": rep.y_bin.values * BIN_UM,
                "gene": rep.gene.values,
                "cell_id": rep.cell_local.values.astype(int),
            })
            n_in = int((tx.cell_id > 0).sum())
            P(f"    ✓ {os.path.basename(txp)}   nuclear {n_in:,}/{len(tx):,} "
              f"({n_in/len(tx)*100:.1f}%)")
        tx.to_csv(txp, index=False)

    landmark_col = ("_landmark_for_centroid"
                    if cfg.no_transcript_prior else "cell_id")
    if landmark_col not in tx.columns:
        sys.exit(f"✘ cached {txp} lacks column {landmark_col}; rerun with --force")

    if os.path.exists(cenp) and not cfg.force:
        nuc = pd.read_csv(cenp)
        P(f"  ⚡ reused centroid table: {len(nuc):,} rows")
    else:
        sub = tx[tx[landmark_col] > 0]
        g = sub.groupby(landmark_col)[["x", "y"]].mean().reset_index()
        g = g.rename(columns={landmark_col: "cell_id"})
        # ★ cell_id is now local 1..N, so z = cell_id * 0.001 is at most a
        #   few units and cannot dominate x/y distance
        g["z"] = g.cell_id * 0.001
        nuc = g[["cell_id", "x", "y", "z"]].copy()
        nuc.to_csv(cenp, index=False)
        P(f"  ✓ centroid table: {len(nuc):,} landmarks  "
          f"(z range {nuc.z.min():.3f}~{nuc.z.max():.3f}; sane magnitude restored)")

    if not cfg.no_transcript_prior:
        nuc, changed = ensure_sentinel(nuc, cfg.max_cell_radius)
        if changed:
            nuc.to_csv(cenp, index=False)
            P(f"    ✓ wrote back {os.path.basename(cenp)} ({len(nuc):,} rows)")

    return tag, txp, cenp, nuc


def sentinel_position(nuc, max_cell_radius):
    x0, x1 = float(nuc.x.min()), float(nuc.x.max())
    y0, y1 = float(nuc.y.min()), float(nuc.y.max())
    dx = max(x0 - 0.0, 0.0, 0.0 - x1)
    dy = max(y0 - 0.0, 0.0, 0.0 - y1)
    dist = (dx ** 2 + dy ** 2) ** 0.5
    safe_margin = max_cell_radius * 5
    if dist >= safe_margin:
        return 0.0, 0.0, dist, True
    sx = x0 - safe_margin * 4
    sy = y0 - safe_margin * 4
    return sx, sy, dist, False


def ensure_sentinel(nuc, max_cell_radius):
    has_zero = bool((nuc.cell_id == 0).any())
    if has_zero:
        P(f"    ⚡ cell_id=0 sentinel already present; nothing to do")
        return nuc, False
    sx, sy, dist, used_origin = sentinel_position(nuc, max_cell_radius)
    where = "origin (0,0)" if used_origin else f"outside ROI boundary ({sx:.0f},{sy:.0f})"
    P(f"    ★ missing cell_id=0 sentinel; placed at {where}, {dist:.0f}um from nearest real nucleus")
    row = {"cell_id": 0, "x": sx, "y": sy, "z": 0.0}
    nuc2 = pd.concat([pd.DataFrame([row]), nuc], ignore_index=True)
    nuc2["cell_id"] = nuc2.cell_id.astype(int)
    return nuc2, True


def write_centroid_for_classify(nuc, img_name, dirs):
    out = f"{dirs['centroid']}/{img_name}.csv"
    nuc.to_csv(out, index=False)
    P(f"    ✓ classify_centroid centroid file -> {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Step 2-4: ComSeg main pipeline (same as before; no call logic changed)
# ══════════════════════════════════════════════════════════════════════════
def run_comseg(tag, txp, cenp, nuc, cfg, dirs):
    from comseg.dataset import ComSegDataset
    from comseg.dictionary import ComSegDict

    img_name = os.path.basename(txp).replace(".csv", "")
    mmc_tag = str(cfg.min_merge_correlation).replace(".", "p")
    ck_pkl = f"{dirs['cache']}/comsegdict_{tag}_mmc{mmc_tag}.pkl"

    if cfg.resume and os.path.exists(ck_pkl):
        P(f"\n[Resume] loaded ComSegDict from pkl (skip run_all)")
        with open(ck_pkl, "rb") as f:
            comseg_dict = pickle.load(f)
        return comseg_dict, img_name

    P(f"\n[2] ComSegDataset")
    ds_kw = safe_kwargs(ComSegDataset.__init__, dict(
        path_dataset_folder=dirs["cache"],
        prior_name="cell_id",
        image_csv_files=[os.path.basename(txp)],
        centroid_csv_files=[os.path.basename(cenp)],
        path_cell_centroid=dirs["cache"],
        dict_scale={"x": 1, "y": 1, "z": 1},
        mean_cell_diameter=cfg.cell_diam,
        gene_column="gene",
    ), "ComSegDataset")
    t0 = time.time()
    data = ComSegDataset(**ds_kw)
    P(f"    ready ({time.time()-t0:.1f}s)   list_index: {data.list_index}")

    ew_pkl = (f"{dirs['cache']}/edge_{img_name}"
              f"_s{cfg.sampling_size}.pkl")
    if os.path.exists(ew_pkl) and not cfg.force:
        P(f"\n[3] ⚡ reused edge_weight: {os.path.basename(ew_pkl)}")
        with open(ew_pkl, "rb") as f:
            data = pickle.load(f)
    else:
        P(f"\n[3] compute_edge_weight (sampling={cfg.sampling_size})...")
        ew_kw = safe_kwargs(data.compute_edge_weight, dict(
            sampling=True, sampling_size=cfg.sampling_size,
            n_neighbors=cfg.n_neighbors,
        ), "compute_edge_weight")
        t0 = time.time()
        data.compute_edge_weight(**ew_kw)
        P(f"    ✓ done ({time.time()-t0:.0f}s)")
        with open(ew_pkl, "wb") as f:
            pickle.dump(data, f, protocol=4)
        P(f"    ✓ cached -> {os.path.basename(ew_pkl)}")

    P(f"\n[4] ComSegDict")
    dd_kw = safe_kwargs(ComSegDict.__init__, dict(
        dataset=data, mean_cell_diameter=cfg.cell_diam,
        community_detection="with_prior",
    ), "ComSegDict")
    comseg_dict = ComSegDict(**dd_kw)

    P(f"\n[5] clear dict_centroid (required by classify_centroid assertion)")
    kill_dict_centroid(data, comseg_dict)

    P(f"\n[6] write dedicated classify_centroid file (separate dir, avoid same-name misread)")
    write_centroid_for_classify(nuc, img_name, dirs)

    P(f"\n[7] run_all")
    P(f"    min_merge_correlation={cfg.min_merge_correlation}"
      f"  resolution={cfg.resolution}")
    ra_kw = safe_kwargs(comseg_dict.run_all, dict(
        max_cell_radius=cfg.max_cell_radius,
        k_nearest_neighbors=cfg.k_nn, n_neighbors=cfg.n_neighbors,
        path_dataset_folder_centroid=dirs["centroid"],
        min_merge_correlation=cfg.min_merge_correlation,
        resolution=cfg.resolution,
    ), "run_all")
    t0 = time.time()
    comseg_dict.run_all(**ra_kw)
    el = time.time() - t0
    P(f"    ✓ run_all done ({el:.0f}s = {el/60:.1f} min)")

    try:
        with open(ck_pkl, "wb") as f:
            pickle.dump(comseg_dict, f, protocol=4)
        P(f"    ✓ ComSegDict saved -> {os.path.basename(ck_pkl)}")
    except Exception as e:
        P(f"    ⚠ save failed ({e}); continuing")

    return comseg_dict, img_name


def export(comseg_dict, tag, nuc, cfg, dirs):
    P(f"\n[8] export anndata")
    ad_kw = safe_kwargs(comseg_dict.anndata_from_comseg_result, dict(
        min_rna_per_cell=cfg.min_rna_per_cell, return_polygon=False,
    ), "anndata_from_comseg_result")
    ret = comseg_dict.anndata_from_comseg_result(**ad_kw)
    adata = ret[0] if isinstance(ret, tuple) else ret
    P(f"    return type {type(ret).__name__}"
      f"{' (tuple, using [0])' if isinstance(ret, tuple) else ''}")

    if adata is None or getattr(adata, "n_obs", 0) == 0:
        P("    ✘ anndata is empty - if still empty this time, something else "
          f"is wrong, not just z magnitude"); return None

    mmc_tag = str(cfg.min_merge_correlation).replace(".", "p")
    out = f"{dirs['out']}/{tag}_mmc{mmc_tag}.h5ad"
    adata.write_h5ad(out)

    P(f"\n{SEP}")
    P(f"  ✓ {adata.n_obs:,} cells x {adata.n_vars} genes")
    P(f"  → {out}")

    df_spots = adata.uns.get("df_spots")
    if df_spots:
        key = list(df_spots.keys())[0]
        n_clusters = df_spots[key].leiden_merged.nunique()
        P(f"\n  leiden_merged actual cluster count: {n_clusters}")
    return out


def show_status(dataset, no_prior):
    dirs = get_dirs(dataset, no_prior)
    mode = "no transcript prior" if no_prior else "with prior (original)"
    P(f"\n{SEP}\nStatus check - {dataset} {mode} mode\n{SEP}")

    p = subprocess.run(["pgrep", "-f", "comseg_stereo_roi"],
                       capture_output=True, text=True).stdout.strip()
    P(f"  running processes: {'PID ' + p.replace(chr(10), ', ') if p else '(none)'}")

    P(f"\n  -- cache {dirs['cache']} --")
    import glob
    for f in sorted(glob.glob(f"{dirs['cache']}/*")):
        sz = os.path.getsize(f) / 1e6
        P(f"    {sz:9.2f} MB   {os.path.basename(f)}")

    P(f"\n  -- output {dirs['out']} --")
    outs = sorted(glob.glob(f"{dirs['out']}/*.h5ad"))
    if not outs:
        P(f"    (no .h5ad output)")
    for f in outs:
        try:
            import anndata
            a = anndata.read_h5ad(f)
            P(f"    ✓ {a.n_obs:,} cells x {a.n_vars} genes   {os.path.basename(f)}")
        except Exception as e:
            P(f"    ⚠ {os.path.basename(f)} (read failed {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="E16.5_E2S6",
                    choices=list(DATASETS.keys()))
    ap.add_argument("--no-transcript-prior", action="store_true")
    ap.add_argument("--target-cells", type=int, default=2000)
    ap.add_argument("--roi-half-um", type=float, default=None)
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--cell-diam", type=float, default=13.0)
    ap.add_argument("--sampling-size", type=int, default=2000)
    ap.add_argument("--k-nn", type=int, default=2)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--max-cell-radius", type=float, default=15.0)
    ap.add_argument("--min-rna-per-cell", type=int, default=5)
    ap.add_argument("--min-merge-correlation", type=float, default=0.99)
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    cfg = ap.parse_args()

    dirs = get_dirs(cfg.dataset, cfg.no_transcript_prior)

    if cfg.status:
        show_status(cfg.dataset, cfg.no_transcript_prior)
        return

    mode = "no transcript prior" if cfg.no_transcript_prior else "with prior (original)"
    P(SEP); P(f"ComSeg x Stereo-seq {cfg.dataset} - {mode} mode"); P(SEP)
    P(f"  min_merge_correlation={cfg.min_merge_correlation}  "
      f"resolution={cfg.resolution}  target cells={cfg.target_cells:,}  "
      f"top-{cfg.top_n} genes")
    P(f"  cache dir: {dirs['cache']}")

    P(f"\n[1] data prep")
    tag, txp, cenp, nuc = prepare_data(cfg, dirs)

    mmc_tag = str(cfg.min_merge_correlation).replace(".", "p")
    out = f"{dirs['out']}/{tag}_mmc{mmc_tag}.h5ad"
    if os.path.exists(out) and not cfg.force and not cfg.resume:
        import anndata
        a = anndata.read_h5ad(out)
        P(f"\n  ✓ existing result: {a.n_obs:,} cells  ->  {out}")
        P(f"    use --force to recompute, or --resume to rerun only export")
        return

    if cfg.check_only:
        P(f"\n  --check-only: nuclei {int((nuc.cell_id > 0).sum()):,}")
        return

    comseg_dict, img_name = run_comseg(tag, txp, cenp, nuc, cfg, dirs)
    export(comseg_dict, tag, nuc, cfg, dirs)


if __name__ == "__main__":
    main()