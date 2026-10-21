#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_value_cosmx_2.py - evaluate ComSeg @ cosmx_2 (CosMx lymph node, FOV 48)

Data status (must know before running):
  Only 14 of 25 tiles have a cell column; tile_0014-0024 are 6-column remnants
  from the 08-11 ComSegDataset API failure (only index/x/y/z/gene/in_nucleus).
  comseg_cosmx_3.py's completion test only checks whether spot_assignment
  exists, not its contents, so those 11 remnants also got DONE.flag and all
  later resumes skipped them.

  Rerunning them is not worthwhile: the 25 tiles are not spatially exclusive
  blocks; coordinate ranges are all x[0,532] y[0,532], and 97.5% of the pending
  tiles' coordinate points are already covered by the completed 14. Rerunning
  takes 82 h and 209 GB for only 2.5% new spatial positions. So this script
  evaluates only those 14 tiles and records n_tiles_used / tile_coverage for
  the manuscript.

  The quality of the 14 tiles is good: total 4,675 cells / GT 4,898 = 0.955,
  mean assigned 97.5%, and per-tile nucleus vs cell counts are one-to-one
  (zero merges, zero splits).

Coordinates: CosMx FOV-local, in um (0-531.5, i.e. 4256 px x 0.12 um/px).
  GT uses CenterX/Y_local_px x 0.12 for FOV 48 in S0_metadata_file.

Evaluation matches comseg_value_xenium_2.py:
  centroids recomputed from spot_assignment grouped by cell, not h5ad landmark centroids;
  KD-tree nearest neighbor + radius cutoff + one-to-one dedup, no direct ID join;
  radius = sqrt(ROI area / GT cells in ROI).

Usage:
  python3 comseg_value_cosmx_2.py
  python3 comseg_value_cosmx_2.py --match-radius 8
  python3 comseg_value_cosmx_2.py --no-dedup
  python3 comseg_value_cosmx_2.py --skip-tx        # skip expression vectors (save memory)
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = "yourpath/comseg_cosmx_lymph_node/fov261"
META = ("yourpath/dataset/cosmx_lymph_node/flat_files/"
        "S0_metadata_file.csv.gz")
TX_CACHE = ("yourpath/dataset/cosmx_lymph_node/_inspect_cache/"
            "tx_fov_48_261.parquet")
OUT = "yourpath/eval_results_qv20/cosmx_2"
METHOD, DATASET = "ComSeg", "cosmx_2"
FOV = 261
PX_UM = 0.12
FOV_PX = 4256
CELL_COLS = ("cell", "cell_index", "comseg", "cell_id", "label",
             "cell_label", "cell_index_pred")


def log(m):
    print(m, flush=True)


def load_pred(root, dedup=True):
    """Only include tiles with a cell column; 6-column ones are prep remnants."""
    fs = sorted(glob.glob(os.path.join(
        root, "comseg_out", "tile_*", "spot_assignment.parquet")))
    if not fs:
        sys.exit(f"✗ no spot_assignment.parquet: {root}/comseg_out/")
    good, bad, frames = [], [], []
    for f in fs:
        t = pd.read_parquet(f)
        name = os.path.basename(os.path.dirname(f))
        if any(c in t.columns for c in CELL_COLS):
            good.append(name)
            frames.append(t)
        else:
            bad.append(name)
    log(f"  {len(fs)} tiles: {len(good)} with a cell column, "
        f"{len(bad)} with only prep inputs")
    if bad:
        log(f"    remnant tiles (left by the 08-11 API failure): {bad}")
    if not frames:
        sys.exit("✗ no tile has a cell column")
    d = pd.concat(frames, ignore_index=True)
    log(f"  merged {len(d):,} rows   columns {len(d.columns)}: "
        f"{list(d.columns)[:8]}...")
    if dedup:
        n0 = len(d)
        d = d.drop_duplicates(subset=["x", "y", "gene"], keep="first")
        log(f"  tile overlap dedup: {n0:,} -> {len(d):,} "
            f"(removed {(n0-len(d))/n0*100:.1f}%)")
    return d, good, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--meta", default=META)
    ap.add_argument("--tx-cache", default=TX_CACHE)
    ap.add_argument("--fov", type=int, default=FOV)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--match-radius", type=float, default=0.0)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--skip-tx", action="store_true")
    a = ap.parse_args()

    from scipy.spatial import cKDTree
    from scipy.stats import pearsonr, spearmanr

    log("=" * 78)
    log(f"{METHOD} @ {DATASET}  (CosMx lymph node, FOV {a.fov})")
    log("=" * 78)

    log(f"\n[Prediction] {a.root}")
    d, good, bad = load_pred(a.root, not a.no_dedup)
    cc = next(c for c in CELL_COLS if c in d.columns)
    log(f"  cell column used: {cc}  (all seven cell columns are identical aliases)")
    v = pd.to_numeric(d[cc], errors="coerce")
    assigned = v > 0
    log(f"  assigned {int(assigned.sum()):,} ({assigned.mean()*100:.1f}%)")

    da = d[assigned].copy()
    g = da.groupby(cc).agg(x=("x", "mean"), y=("y", "mean"),
                           n=("x", "size"))
    q = g.n.quantile([.05, .25, .5, .75, .95])
    log(f"\n  predicted cells {len(g):,}")
    log(f"  tx per cell  median {g.n.median():.0f}  mean {g.n.mean():.0f}  "
        f"mean/median {g.n.mean()/max(g.n.median(),1):.2f}")
    log(f"             p5/p25/p75/p95 = {q.iloc[0]:.0f}/{q.iloc[1]:.0f}/"
        f"{q.iloc[3]:.0f}/{q.iloc[4]:.0f}   "
        f"<10: {(g.n<10).mean()*100:.2f}%")
    log(f"  centroid x[{g.x.min():.1f},{g.x.max():.1f}] "
        f"y[{g.y.min():.1f},{g.y.max():.1f}] um   "
        f"(FOV = {FOV_PX*PX_UM:.1f} um square)")

    sp = da.groupby(cc).agg(sx=("x", "std"), sy=("y", "std"))
    diam = 2 * np.sqrt(2) * np.sqrt((sp.sx ** 2 + sp.sy ** 2) / 2)
    log(f"  equivalent diameter (2*sqrt2*RMS spread) median {diam.median():.2f} um  "
        f"(metadata lymphocyte 7.86 um)")

    # Nuclear prior consistency: nuclei in in_nucleus vs predicted cells
    if "in_nucleus" in d.columns:
        nuc = pd.to_numeric(d["in_nucleus"], errors="coerce")
        nn = int(nuc[nuc > 0].nunique())
        both = (nuc > 0) & assigned
        t2 = pd.DataFrame({"nuc": nuc[both].astype(int),
                           "cel": v[both].astype(int)})
        per = t2.groupby("cel")["nuc"].nunique()
        per2 = t2.groupby("nuc")["cel"].nunique()
        log(f"\n  nucleus prior: {nn:,} (within the 14 tiles {t2.nuc.nunique():,} participate)")
        log(f"  nuclei covered per cell  mean {per.mean():.2f}  "
            f">1: {int((per>1).sum()):,} ({(per>1).mean()*100:.1f}%)")
        log(f"  nuclei split across >1 cell: {int((per2>1).sum()):,}")
        log(f"  -> one-to-one means no nucleus merges or splits")

    log(f"\n[GT] {os.path.basename(a.meta)}")
    meta = pd.read_csv(a.meta)
    fc = next(c for c in meta.columns if c.lower() == "fov")
    sub = meta[pd.to_numeric(meta[fc], errors="coerce") == a.fov].copy()
    cx = next((c for c in ("CenterX_local_px", "x_local_px")
               if c in sub.columns), None)
    cy = next((c for c in ("CenterY_local_px", "y_local_px")
               if c in sub.columns), None)
    if cx is None:
        sys.exit(f"✗ metadata has no local coordinate columns: {list(sub.columns)[:12]}")
    gx = sub[cx].to_numpy(float) * PX_UM
    gy = sub[cy].to_numpy(float) * PX_UM
    log(f"  GT in FOV {a.fov}: {len(sub):,}   "
        f"x[{gx.min():.1f},{gx.max():.1f}] y[{gy.min():.1f},{gy.max():.1f}] um")
    gtn = None
    for c in ("nCount_RNA", "totalcounts", "Total_counts", "nCount"):
        if c in sub.columns:
            gtn = sub[c].to_numpy(float)
            log(f"  GT tx per cell column {c}  median {np.nanmedian(gtn):.0f}")
            break
    ga = None
    for c in ("Area", "area", "NucArea"):
        if c in sub.columns:
            ga = sub[c].to_numpy(float)
            log(f"  GT area column {c}  median {np.nanmedian(ga):.0f} px2 "
                f"= diameter {2*np.sqrt(np.nanmedian(ga)/np.pi)*PX_UM:.2f} um")
            break

    r = a.match_radius or float(np.sqrt(
        (FOV_PX * PX_UM) ** 2 / max(len(sub), 1)))
    log(f"\n[Match] KD-tree one-to-one, radius {r:.2f} um")
    P = np.c_[g.x.to_numpy(float), g.y.to_numpy(float)]
    G = np.c_[gx, gy]
    dist, gi = cKDTree(G).query(P, k=1, distance_upper_bound=r)
    ok = np.isfinite(dist) & (dist < r)
    order = np.argsort(np.where(ok, dist, np.inf))
    used, pairs = set(), []
    for pi in order:
        if not ok[pi]:
            break
        if gi[pi] in used:
            continue
        used.add(gi[pi])
        pairs.append((int(pi), int(gi[pi]), float(dist[pi])))
    matched = len(pairs)
    prec = matched / max(len(P), 1)
    rec = matched / max(len(G), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    dd = np.array([p[2] for p in pairs]) if pairs else np.array([np.nan])
    log(f"  pred {len(P):,}  gt {len(G):,}  matched {matched:,}")
    log(f"  precision {prec:.4f}  recall {rec:.4f}  f1 {f1:.4f}")
    log(f"  loc mean {np.nanmean(dd):.3f}  median {np.nanmedian(dd):.3f}  "
        f"p95 {np.nanpercentile(dd,95):.3f} um")

    det_ok = not (np.nanmedian(dd) < 1e-6)
    if not det_ok:
        log("  ★ loc_median = 0 -> centroids inherited from prior landmarks")

    res = dict(method=METHOD, dataset=DATASET,
               gt_cells=len(G), pred_cells=len(P), matched=matched,
               cell_ratio=len(P) / max(len(G), 1),
               precision=prec, recall=rec, f1=f1,
               loc_mean_um=float(np.nanmean(dd)),
               loc_median_um=float(np.nanmedian(dd)),
               loc_p95_um=float(np.nanpercentile(dd, 95)),
               match_radius_um=r,
               assign_rate=float(assigned.mean()),
               coverage_fraction=float(assigned.mean()),
               tx_per_cell_median=float(g.n.median()),
               tx_per_cell_mean=float(g.n.mean()),
               frag_lt10=float((g.n < 10).mean()),
               median_diameter_um=float(diam.median()),
               n_tiles_used=len(good), n_tiles_total=len(good) + len(bad),
               uses_platform_prior=False,
               detection_comparable=det_ok, data_error=False)

    if pairs and gtn is not None:
        pi = np.array([p[0] for p in pairs])
        gj = np.array([p[1] for p in pairs])
        A = g.n.to_numpy(float)[pi]
        B = gtn[gj]
        k = np.isfinite(A) & np.isfinite(B)
        if k.sum() >= 5:
            res.update(count_pearson=float(pearsonr(A[k], B[k])[0]),
                       count_spearman=float(spearmanr(A[k], B[k])[0]),
                       count_mae=float(np.abs(A[k] - B[k]).mean()),
                       count_rmse=float(np.sqrt(((A[k]-B[k])**2).mean())))
            log(f"\n[Count] n={int(k.sum()):,}  "
                f"pearson {res['count_pearson']:.4f}  "
                f"spearman {res['count_spearman']:.4f}  "
                f"mae {res['count_mae']:.1f}  rmse {res['count_rmse']:.1f}")

    # Transcript-level assignment + expression vectors: map predicted transcripts to GT
    if pairs and not a.skip_tx and os.path.exists(a.tx_cache):
        log(f"\n[Assignment / expression vectors] {os.path.basename(a.tx_cache)}")
        tx = pd.read_parquet(a.tx_cache)
        tfc = "fov" if "fov" in tx.columns else "FOV"
        tx = tx[pd.to_numeric(tx[tfc], errors="coerce") == a.fov].copy()
        txc = next((c for c in ("x_local_px", "x") if c in tx.columns), None)
        tyc = next((c for c in ("y_local_px", "y") if c in tx.columns), None)
        tgc = next((c for c in ("target", "gene", "feature_name")
                    if c in tx.columns), None)
        tcc = next((c for c in ("cell_ID", "cell_id", "cell")
                    if c in tx.columns), None)
        log(f"  {len(tx):,} transcripts  columns x={txc} y={tyc} gene={tgc} "
            f"cell={tcc}")
        if txc and tyc and tcc:
            tx["_xum"] = tx[txc].astype(float) * PX_UM
            tx["_yum"] = tx[tyc].astype(float) * PX_UM
            gt_ids = sub["cell_ID"].astype(str).to_numpy() \
                if "cell_ID" in sub.columns \
                else sub.iloc[:, 0].astype(str).to_numpy()
            id2row = {s: i for i, s in enumerate(gt_ids)}
            tx["_gtrow"] = (tx[tcc].astype(str).map(id2row)
                            .fillna(-1).astype(np.int64))
            hit_gt = int((tx["_gtrow"] >= 0).sum())
            log(f"  GT transcripts matched to metadata: {hit_gt:,} "
                f"({hit_gt/max(len(tx),1)*100:.1f}%)")

            dq, iq = cKDTree(np.c_[tx._xum.to_numpy(float),
                                   tx._yum.to_numpy(float)]).query(
                np.c_[da.x.to_numpy(float), da.y.to_numpy(float)],
                k=1, distance_upper_bound=0.3)
            hit = np.isfinite(dq)
            log(f"  predicted transcripts matched to GT transcripts {int(hit.sum()):,} / "
                f"{len(da):,} ({hit.mean()*100:.1f}%)")
            if hit.mean() < .5:
                log("  ⚠ hit rate <50%; coordinate conventions may disagree")

            cell2row = {c: i for i, c in enumerate(g.index)}
            pred_row = np.array([cell2row.get(c, -1)
                                 for c in da[cc].to_numpy()])
            gt_row = np.full(len(da), -1, dtype=np.int64)
            gt_row[hit] = tx["_gtrow"].to_numpy()[iq[hit].astype(np.int64)]
            p2g = {p[0]: p[1] for p in pairs}
            valid = hit & (pred_row >= 0) & (gt_row >= 0)
            if valid.sum() >= 100:
                want = np.array([p2g.get(int(pr), -2)
                                 for pr in pred_row[valid]])
                acc = float((want == gt_row[valid]).mean())
                res["assign_accuracy"] = acc
                res["overall_correct"] = acc * float(assigned.mean())
                log(f"  assign_accuracy {acc:.4f}  "
                    f"(over {int(valid.sum()):,} decidable transcripts)")
                log(f"  overall_correct  {res['overall_correct']:.4f}")

            if tgc:
                from scipy.sparse import coo_matrix
                genes = sorted(set(da["gene"].astype(str)) |
                               set(tx[tgc].astype(str)))
                gidx = {v_: i for i, v_ in enumerate(genes)}
                vp = pred_row >= 0
                PV = coo_matrix(
                    (np.ones(int(vp.sum())),
                     (pred_row[vp],
                      da["gene"].astype(str).map(gidx).to_numpy()[vp])),
                    shape=(len(g), len(genes))).tocsr()
                vg = tx["_gtrow"].to_numpy() >= 0
                GV = coo_matrix(
                    (np.ones(int(vg.sum())),
                     (tx["_gtrow"].to_numpy()[vg],
                      tx[tgc].astype(str).map(gidx).to_numpy()[vg])),
                    shape=(len(sub), len(genes))).tocsr()
                pi = np.array([p[0] for p in pairs])
                gj = np.array([p[1] for p in pairs])
                A = np.asarray(PV[pi].todense(), float)
                B = np.asarray(GV[gj].todense(), float)
                na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
                k = (na > 0) & (nb > 0)
                if k.sum() >= 5:
                    res["vec_cosine"] = float(np.mean(
                        (A[k] * B[k]).sum(1) / (na[k] * nb[k])))
                    Ap = A[k] / A[k].sum(1, keepdims=True)
                    Bp = B[k] / B[k].sum(1, keepdims=True)
                    M = (Ap + Bp) / 2
                    with np.errstate(divide="ignore", invalid="ignore"):
                        kl1 = np.nansum(Ap * np.log2(Ap / M), axis=1)
                        kl2 = np.nansum(Bp * np.log2(Bp / M), axis=1)
                    res["vec_js_dist"] = float(
                        np.mean(np.sqrt((kl1 + kl2) / 2)))
                    nn = min(2000, int(k.sum()))
                    res["vec_pearson"] = float(np.mean(
                        [pearsonr(A[k][i], B[k][i])[0] for i in range(nn)]))
                    log(f"  vec_cosine {res['vec_cosine']:.4f}  "
                        f"vec_js_dist {res['vec_js_dist']:.4f}  "
                        f"vec_pearson {res['vec_pearson']:.4f}  "
                        f"(n={int(k.sum()):,}, {len(genes):,} genes)")

    res["note"] = (f"using only {len(good)}/{len(good)+len(bad)} tiles - "
                   f"tile_0014-0024 are 6-column prep remnants from the 08-11 "
                   f"ComSegDataset API failure; 97.5% of pending tiles' "
                   f"coordinate points are already covered, so rerunning takes "
                   f"82 h for only 2.5% new space; centroids recomputed from "
                   f"spot_assignment by cell (not landmarks); KD-tree one-to-one "
                   f"matching; tile overlap deduplicated")

    os.makedirs(a.out, exist_ok=True)
    f = os.path.join(a.out, f"comseg_{DATASET}_metrics.csv")
    pd.DataFrame([res]).to_csv(f, index=False)
    pd.DataFrame(pairs, columns=["pred_idx", "gt_idx", "dist_um"]).to_csv(
        os.path.join(a.out, f"comseg_{DATASET}_pairs.csv"), index=False)
    log("\n" + "=" * 78)
    for k, v_ in res.items():
        if k == "note":
            continue
        log(f"  {k:<24} {v_}")
    log("=" * 78)
    log(f"-> {f}")
    log(f"\n⚠ cite n_tiles_used={len(good)}/{len(good)+len(bad)} when tabulating")


if __name__ == "__main__":
    main()