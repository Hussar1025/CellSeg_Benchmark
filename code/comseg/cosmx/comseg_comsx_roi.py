#!/usr/bin/env python
"""
ComSeg x CosMx pancreas - unified version (with-prior / no-transcript-prior in one file)

===== Why one file =====
Previously split into two scripts (with-prior original / no-transcript-prior).
Twice we confused which content had been saved under which filename. Now one
switch, --no-transcript-prior, selects the mode; no need to remember filename
mappings - there is only one file, so the active mode is always clear.

===== Mode differences =====
- Default (with-prior): the cell_id column in the transcript CSV carries the
  real CosMx segmentation. Confirmed: transcripts with cell_id>0 are passed
  back unchanged at 100% and never reassigned (by design in with_prior mode,
  not a bug). Thus F1/count_pearson/vec_cosine here are not evidence of
  detection ability; they only reflect recovery of background (cell_id=0)
  transcripts.
- --no-transcript-prior: set all transcript cell_id values to 0, keep only
  the centroid landmark positions, and let all transcripts go through free
  community detection. Three runs (including two with-prior) all converged to
  exactly 3 clusters; suspect the min_merge_correlation default (0.8) is too
  permissive, so this version exposes it as a CLI parameter for tuning.

===== Cache dirs are fully separate, no cross-contamination =====
with-prior:        cache_roi / cache_roi_centroid_only / output_roi
no-transcript-prior: cache_roi_noprior / cache_roi_noprior_centroid_only /
                     output_roi_noprior

===== cell_id=0 sentinel: only needed in with-prior mode =====
Verified by two successful no-prior runs: when all transcripts share
cell_id=0, classify_centroid/associate_rna2landmark do not need a dedicated
landmark for id=0. The sentinel is specific to with-prior mode (some
transcripts have real priors, others are background; it is needed only when
the two identities coexist). no-transcript-prior no longer calls this logic.

===== Core historical fixes (kept in both modes) =====
- import pickle must be at module top
- unconditionally clear dict_centroid before run_all (required by the
  classify_centroid assertion)
- classify_centroid reads {img_name}.csv, which must live in a separate
  directory, not alongside the transcript CSV (otherwise the transcript
  table is misread as a centroid table)
- anndata_from_comseg_result returns a tuple; unpack it
- return_polygon=False (avoid qhull crash on collinear transcripts at ROI edges)

Usage:
  python comseg_comsx_roi.py                                        # with-prior, mmc default 0.8
  python comseg_comsx_roi.py --no-transcript-prior                  # no-prior, mmc default 0.8
  python comseg_comsx_roi.py --no-transcript-prior --min-merge-correlation 0.95
  python comseg_comsx_roi.py --status
  python comseg_comsx_roi.py --no-transcript-prior --status
"""
import os
import sys
import time
import glob
import pickle          # must be at module top
import inspect
import argparse
import subprocess

import numpy as np
import pandas as pd

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

TX_FILE = ("yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles/"
           "Pancreas_tx_file.csv")
PIXEL_SIZE = 0.1203
CELL_DIAM = 10.0
ROI_SIZE = 5000


def get_dirs(no_prior):
    """Keep mode directories separate to avoid cross-contamination."""
    if no_prior:
        base = "yourpath/comseg"
        d = dict(cache=f"{base}/cache_roi_noprior",
                 centroid=f"{base}/cache_roi_noprior_centroid_only",
                 out=f"{base}/output_roi_noprior")
    else:
        base = "yourpath/comseg"
        d = dict(cache=f"{base}/cache_roi",
                 centroid=f"{base}/cache_roi_centroid_only",
                 out=f"{base}/output_roi")
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
    """Unconditionally clear dict_centroid before run_all (required by classify_centroid)."""
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
#  Step 1: ROI data prep (shared center calculation; content branches by mode)
# ══════════════════════════════════════════════════════════════════════════
def find_roi_center():
    P("  Estimating ROI center (random_state=0 keeps runs consistent)...")
    xs, ys = [], []
    for ck in pd.read_csv(TX_FILE, chunksize=2_000_000,
                          usecols=["x_global_px", "y_global_px"]):
        xs.append(ck.x_global_px.sample(min(50_000, len(ck)), random_state=0))
        ys.append(ck.y_global_px.sample(min(50_000, len(ck)), random_state=0))
        if len(xs) >= 8:
            break
    cx = float(pd.concat(xs).median())
    cy = float(pd.concat(ys).median())
    h = ROI_SIZE / 2
    return cx - h, cx + h, cy - h, cy + h


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
    """
    Called only in with-prior mode. Verified by two successful no-prior runs:
    this sentinel is unnecessary when all transcripts share cell_id=0.
    """
    has_zero = bool((nuc.cell_id == 0).any())
    if has_zero:
        P(f"    ⚡ cell_id=0 sentinel already present; nothing to do")
        return nuc, False
    sx, sy, dist, used_origin = sentinel_position(nuc, max_cell_radius)
    where = "origin (0,0)" if used_origin else f"outside ROI boundary ({sx:.0f},{sy:.0f})"
    P(f"    ★ missing cell_id=0 sentinel (~9.7% of transcripts have no nucleus "
      f"assignment; only with-prior mode needs it)")
    P(f"      placed at {where}, {dist:.0f}um from nearest real nucleus")
    row = {"cell_id": 0, "x": sx, "y": sy, "z": 0.0}
    nuc2 = pd.concat([pd.DataFrame([row]), nuc], ignore_index=True)
    nuc2["cell_id"] = nuc2.cell_id.astype(int)
    return nuc2, True


def prepare_data(cfg, dirs):
    x0, x1, y0, y1 = find_roi_center()
    suffix = "_noprior" if cfg.no_transcript_prior else ""
    tag = f"cosmx_roi_{int(x0)}_{int(y0)}{suffix}"
    txp = f"{dirs['cache']}/{tag}_tx.csv"
    cenp = f"{dirs['cache']}/{tag}_centroid.csv"

    if os.path.exists(txp) and not cfg.force:
        tx = pd.read_csv(txp)
        P(f"  ⚡ reused transcript CSV: {len(tx):,} rows  columns={list(tx.columns)}")
    else:
        P(f"  ROI x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}] px")
        P("  Reading transcripts (~3-5 min)...")
        chunks, total = [], 0
        for ck in pd.read_csv(TX_FILE, chunksize=2_000_000,
                              usecols=["x_global_px", "y_global_px",
                                       "target", "cell_ID", "fov"]):
            total += len(ck)
            sub = ck[(ck.x_global_px >= x0) & (ck.x_global_px < x1) &
                     (ck.y_global_px >= y0) & (ck.y_global_px < y1)]
            if len(sub):
                chunks.append(sub)
            print(f"    Read {total:,}", end="\r", flush=True)
        roi = pd.concat(chunks, ignore_index=True)
        P(f"\n    ROI contains {len(roi):,}")

        roi["cell_uid"] = np.where(
            roi.cell_ID > 0,
            roi.fov.astype(str) + "_" + roi.cell_ID.astype(str), "")
        top = roi.target.value_counts().head(cfg.top_n).index
        roi = roi[roi.target.isin(top)]
        P(f"    after top-{cfg.top_n} genes: {len(roi):,} rows")

        uid2id = dict(zip(
            sorted(roi.loc[roi.cell_uid != "", "cell_uid"].unique()),
            range(1, roi.cell_uid.nunique() + 1)))
        landmark_id = roi.cell_uid.map(uid2id).fillna(0).astype(int)

        if cfg.no_transcript_prior:
            # Set all cell_id to 0; keep _landmark_for_centroid for centroids
            tx = pd.DataFrame({
                "x": roi.x_global_px.values * PIXEL_SIZE,
                "y": roi.y_global_px.values * PIXEL_SIZE,
                "gene": roi.target.values,
                "cell_id": 0,
                "_landmark_for_centroid": landmark_id.values,
            })
            P(f"    ✓ {os.path.basename(txp)}  (all cell_id = 0, "
              f"no-transcript-prior mode)")
        else:
            tx = pd.DataFrame({
                "x": roi.x_global_px.values * PIXEL_SIZE,
                "y": roi.y_global_px.values * PIXEL_SIZE,
                "gene": roi.target.values,
                "cell_id": landmark_id.values,
            })
            P(f"    ✓ {os.path.basename(txp)}   nuclear "
              f"{int((tx.cell_id > 0).sum()):,}/{len(tx):,} "
              f"({(tx.cell_id > 0).mean()*100:.1f}%)")
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
        g["z"] = g.cell_id * 0.001   # unique z; avoid degenerate 2D qhull
        nuc = g[["cell_id", "x", "y", "z"]].copy()
        nuc.to_csv(cenp, index=False)
        P(f"  ✓ centroid table: {len(nuc):,} landmarks")

    if not cfg.no_transcript_prior:
        nuc, changed = ensure_sentinel(nuc, cfg.max_cell_radius)
        if changed:
            nuc.to_csv(cenp, index=False)
            P(f"    ✓ wrote back {os.path.basename(cenp)} ({len(nuc):,} rows)")

    return tag, txp, cenp, nuc


def write_centroid_for_classify(nuc, img_name, dirs):
    """
    classify_centroid() actually reads {path_dataset_folder_centroid}/
    {img_name}.csv, where img_name is the transcript filename stem - unrelated
    to the centroid_csv_files argument of ComSegDataset.__init__. It must live
    in a separate directory; otherwise the filename equals the transcript file
    and the transcript table is misread (no z column -> degenerate 2-key dict,
    KeyError: 0).
    """
    out = f"{dirs['centroid']}/{img_name}.csv"
    nuc.to_csv(out, index=False)
    P(f"    ✓ classify_centroid centroid file -> {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Step 2-4: ComSeg main pipeline
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
        mean_cell_diameter=CELL_DIAM,
        gene_column="gene",
    ), "ComSegDataset")
    t0 = time.time()
    data = ComSegDataset(**ds_kw)
    P(f"    ready ({time.time()-t0:.1f}s)   list_index: {data.list_index}")

    ew_pkl = (f"{dirs['cache']}/edge_{img_name}"
              f"_top{cfg.top_n}_s{cfg.sampling_size}.pkl")
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
        dataset=data, mean_cell_diameter=CELL_DIAM,
        community_detection="with_prior",   # ★ this ComSeg version hardcodes
                                             #   community_vector() assert
                                             #   clustering_method in
                                             #   ["with_prior"]; other options
                                             #   are documented but unimplemented
    ), "ComSegDict")
    comseg_dict = ComSegDict(**dd_kw)

    P(f"\n[5] clear dict_centroid (required by classify_centroid assertion)")
    kill_dict_centroid(data, comseg_dict)

    P(f"\n[6] write dedicated classify_centroid file (separate dir, avoid same-name misread)")
    write_centroid_for_classify(nuc, img_name, dirs)

    P(f"\n[7] run_all")
    P(f"    min_merge_correlation={cfg.min_merge_correlation}"
      f"  resolution={cfg.resolution}")
    P(f"    Runtime varies widely: original ~220 min, also seen 102 min - "
      f"depends on clustering scale. Do not judge a hang from one run; "
      f"check whether TIME keeps growing in ps")
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
        P("    ✘ anndata is empty"); return None

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
        P(f"\n  ★ leiden_merged actual cluster count: {n_clusters}"
          f" (previous three runs all merged to exactly 3 regardless of "
          f"initial count; this run min_merge_correlation={cfg.min_merge_correlation}, "
          f"see whether this number changes)")
    return out


def show_status(no_prior):
    dirs = get_dirs(no_prior)
    mode = "no transcript prior" if no_prior else "with prior (original)"
    P(f"\n{SEP}\nStatus check - {mode} mode\n{SEP}")

    p = subprocess.run(["pgrep", "-f", "comseg_comsx_roi"],
                       capture_output=True, text=True).stdout.strip()
    P(f"  running processes: {'PID ' + p.replace(chr(10), ', ') if p else '(none)'}")

    P(f"\n  -- cache {dirs['cache']} --")
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
            P(f"    ✓ {a.n_obs:,} cells x {a.n_vars} genes   "
              f"{os.path.basename(f)}")
        except Exception as e:
            P(f"    ⚠ {os.path.basename(f)} (read failed {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-transcript-prior", action="store_true",
                    help="set all transcript cell_id to 0, keep only centroid landmarks")
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--sampling-size", type=int, default=2000)
    ap.add_argument("--k-nn", type=int, default=2)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--max-cell-radius", type=float, default=15.0)
    ap.add_argument("--min-rna-per-cell", type=int, default=5)
    ap.add_argument("--min-merge-correlation", type=float, default=0.8,
                    help="run_all original default. After three runs all "
                         "converged to 3 clusters, this looks too permissive; "
                         "try 0.95/0.99")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    cfg = ap.parse_args()

    dirs = get_dirs(cfg.no_transcript_prior)

    if cfg.status:
        show_status(cfg.no_transcript_prior)
        return

    mode = "no transcript prior" if cfg.no_transcript_prior else "with prior (original)"
    P(SEP); P(f"ComSeg x CosMx pancreas - {mode} mode"); P(SEP)
    P(f"  min_merge_correlation={cfg.min_merge_correlation}  "
      f"resolution={cfg.resolution}")
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