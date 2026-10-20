#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComSeg evaluation - assemble cell centroids + gene vectors from tile_results/*.h5ad,
reusing the matching/metric logic of the reference script.
ComSeg output is per-tile h5ad (cell-level) with no global pixel label image, so evaluate
with centroids + per-cell gene vectors; area/pixel-transcript metrics are nan (ComSeg output
has no pixel footprint). Core metrics (count / detection P,R,F1 / centroid shift / gene vector
cosine,js,pearson / per-cell transcript count correlation) use the same protocol as other methods.
Usage:  python eval_comseg.py --dataset prostate1   (or prostate2)
Optional overrides: --root --run-dir --gt-mask --out-dir --ref --roi-size
"""
import argparse, importlib.util, sys, glob, re
from pathlib import Path
import numpy as np

REF_DEFAULT  = "yourpath/ucs/UCS/ucs_value_merfish_4.py"
DATA_BASE    = "yourpath/dataset"
RESULT_BASE  = "yourpath"
METHOD       = "ComSeg"
ROI_SIZE_DEF = 10000

def _load_ref(ref):
    ref = Path(ref).resolve()
    if not ref.exists():
        raise SystemExit(f"reference evaluation script not found: {ref}")
    spec = importlib.util.spec_from_file_location("refval", ref)
    V = importlib.util.module_from_spec(spec); sys.modules["refval"] = V
    spec.loader.exec_module(V)
    return V, ref

def _pick_gt_mask(run_dir, root, dataset):
    for c in [Path(run_dir)/"platform_labels.tif",
              Path(f"{RESULT_BASE}/proseg_merfish_{dataset}/platform_labels.tif") if dataset else None,
              Path(root)/"platform_labels.tif"]:
        if c and c.exists():
            return str(c)
    return None

def load_comseg_cells(run_dir, geo):
    """Join cells from 16 tiles -> global centroid (um) + per-cell gene vector (gene table aligned to the tile var)."""
    import anndata as ad, pandas as pd
    meta = pd.read_csv(Path(run_dir)/"tile_metadata.tsv", sep="\t")
    origin = {r.tile_id: (float(r.ry), float(r.rx)) for r in meta.itertuples()}
    PX_UM = geo.um_per_px
    fs = sorted(glob.glob(f"{run_dir}/tile_results/tile_*.h5ad"))
    if not fs:
        raise SystemExit(f"{run_dir}/tile_results has no h5ad")
    all_cx=[]; all_cy=[]; gms=[]; genes=None
    for f in fs:
        a = ad.read_h5ad(f)
        tid = re.search(r"tile_\d+", f).group()
        if tid not in origin:
            tid2 = a.obs.get("image_name")
            tid = tid2.iloc[0] if tid2 is not None else tid
        ry, rx = origin[tid]
        # centroid inside a tile is um (relative to the tile origin); global px = origin + um/PX_UM
        cx_um = a.obs["centroid_x"].to_numpy(float)
        cy_um = a.obs["centroid_y"].to_numpy(float)
        gpx = rx + cx_um / PX_UM
        gpy = ry + cy_um / PX_UM
        ux, uy = geo.local_to_um(gpx, gpy)     # global px -> global um
        all_cx.append(ux); all_cy.append(uy)
        X = a.X
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        gms.append((list(a.var_names), X))
        if genes is None:
            genes = list(a.var_names)
    # unify the gene table (stack directly when all tiles share the same var)
    cx = np.concatenate(all_cx); cy = np.concatenate(all_cy)
    if all(g == genes for g, _ in gms):
        gm = np.vstack([X for _, X in gms]).astype(np.float32)
    else:
        gidx = {g: i for i, g in enumerate(genes)}
        rows = []
        for gnames, X in gms:
            m = np.zeros((X.shape[0], len(genes)), np.float32)
            cols = [gidx[g] for g in gnames if g in gidx]
            src  = [j for j, g in enumerate(gnames) if g in gidx]
            m[:, cols] = X[:, src]
            rows.append(m)
        gm = np.vstack(rows)
    # keep ROI cells only
    inside = geo.inside(cx, cy)
    return cx[inside], cy[inside], gm[inside], genes

def build_gt_gene_matrix(gt, genes):
    """Align the reference-script gt per-cell gene vectors to the ComSeg gene table."""
    gt_genes = getattr(gt, "genes", None)
    gm = getattr(gt, "gene_matrix", None)
    if gm is None:
        return None
    if gt_genes is None or list(gt_genes) == list(genes):
        return np.asarray(gm)
    idx = {g: i for i, g in enumerate(gt_genes)}
    out = np.zeros((gm.shape[0], len(genes)), np.float32)
    cols = [j for j, g in enumerate(genes) if g in idx]
    src  = [idx[g] for g in genes if g in idx]
    out[:, cols] = np.asarray(gm)[:, src]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF_DEFAULT)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--gt-mask", default=None)
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE_DEF)
    args = ap.parse_args()

    if args.dataset:
        args.root    = args.root    or f"{DATA_BASE}/merfish_{args.dataset}"
        args.run_dir = args.run_dir or f"{RESULT_BASE}/comseg_merfish_{args.dataset}"
        args.out_dir = args.out_dir or f"{RESULT_BASE}/eval_merfish_{args.dataset}"
    if not (args.root and args.run_dir):
        raise SystemExit("need --dataset, or both --root and --run-dir")
    args.out_dir = args.out_dir or f"{RESULT_BASE}/eval_merfish_out"
    if not args.gt_mask:
        args.gt_mask = _pick_gt_mask(args.run_dir, args.root, args.dataset)

    V, ref = _load_ref(args.ref)
    root, out = Path(args.root), Path(args.out_dir)
    V.setup_log(out, f"eval_ComSeg_{args.dataset or 'x'}")
    V.rule(f"ComSeg  x  MERFISH {args.dataset or ''}   (centroid assembly, via {ref.name})")

    geo = V.load_geometry(root, args.roi_size)
    gt_mask = V.load_gt_mask(root, geo, args.gt_mask)
    tx = V.load_transcripts(root, geo, gt_mask)
    gt = V.load_ground_truth(root, geo, tx)

    cx, cy, gm, genes = load_comseg_cells(args.run_dir, geo)
    V.log.info(f"comseg assembled cells = {len(cx):,}  (genes {len(genes)})")

    # ---- matching + metrics (reuse reference-script functions, same protocol as other methods) ----
    gx, gy = gt.ux, gt.uy
    radius = V.infer_match_radius(gx, gy)
    mp, mg, md = V.match_sparse_hungarian(cx, cy, gx, gy, radius)
    n = len(mp)
    prec = n / max(len(cx), 1); rec = n / max(len(gx), 1)
    f1 = 2*prec*rec/(prec+rec) if prec+rec else np.nan
    V.log.info(f"match radius {radius:.2f}um  matched {n:,} of pred {len(cx):,} / GT {len(gx):,}")
    V.log.info(f"precision {prec:.4f} recall {rec:.4f} F1 {f1:.4f}")

    row = {
        "method": METHOD, "dataset": args.dataset or "x",
        "Pred cell count": int(len(cx)), "GT cell count": int(len(gx)),
        "Cell count ratio": len(cx)/max(len(gx),1),
        "Matched cell count": int(n),
        "Detection precision": prec, "Detection recall": rec, "Detection F1": f1,
        "Mean centroid shift": float(md.mean()) if n else np.nan,
        "Median centroid shift": float(np.median(md)) if n else np.nan,
        "p95 centroid shift": float(np.percentile(md,95)) if n else np.nan,
        "median_area_um2": np.nan, "median_diameter_um": np.nan,
        "coverage_fraction": np.nan, "uses_platform_prior": True,
        "gt_counts_source": getattr(gt, "counts_source", "platform"),
    }
    if n:
        # --------------------------------------------------------------
        # Gene-space alignment.
        #
        # ComSeg may contain additional genes/control features across tiles
        # (e.g. 550 columns), while the benchmark GT gene matrix is built
        # strictly from tx.genes (e.g. 500 targets).
        #
        # Never align by column position.  Restrict BOTH matrices to the
        # common gene names so every method is scored on exactly the same
        # benchmark transcript panel.
        # --------------------------------------------------------------
        pred_genes = [str(g) for g in genes]
        gt_genes = [str(g) for g in tx.genes]

        pred_idx = {g: i for i, g in enumerate(pred_genes)}
        gt_idx = {g: i for i, g in enumerate(gt_genes)}

        common_genes = [g for g in gt_genes if g in pred_idx]

        if not common_genes:
            raise RuntimeError(
                "no overlapping genes between ComSeg and benchmark GT"
            )

        pi = np.asarray(
            [pred_idx[g] for g in common_genes],
            dtype=np.int64
        )
        gi = np.asarray(
            [gt_idx[g] for g in common_genes],
            dtype=np.int64
        )

        gt_full = getattr(gt, "gene_matrix", None)
        if gt_full is None:
            raise RuntimeError("GT gene_matrix missing")

        gt_full = np.asarray(gt_full)

        if gt_full.shape[1] != len(gt_genes):
            raise RuntimeError(
                f"GT gene matrix columns={gt_full.shape[1]} "
                f"but tx.genes={len(gt_genes)}"
            )

        gm_eval = np.asarray(gm)[:, pi]
        gt_gm = gt_full[:, gi]

        V.log.info(
            f"gene alignment: ComSeg={len(pred_genes)} "
            f"GT={len(gt_genes)} common={len(common_genes)}"
        )

        # per-cell transcript count correlation:
        # also restrict to common benchmark genes so ComSeg's extra features
        # do not inflate the predicted total.
        pn = gm_eval[mp].sum(1).astype(float)
        gn = gt_gm[mg].sum(1).astype(float)

        row.update(
            V.numeric_pair_metrics(
                pn,
                gn,
                "matched_cell_transcript_count"
            )
        )

        row.update(
            V.gene_vector_metrics(
                gm_eval[mp],
                gt_gm[mg]
            )
        )

        V.log.info(
            f"gene cosine "
            f"{row.get('matched_cell_gene_vector_mean_cosine')}"
        )

    V.rule("RESULT")
    for k, v in row.items():
        V.log.info(f"  {k:52s} {v}")
    # write table: append to out/eval_results.csv
    import csv, os
    fcsv = out / "eval_results.csv"
    os.makedirs(out, exist_ok=True)
    write_header = not fcsv.exists()
    with open(fcsv, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    V.log.info(f"written -> {fcsv}")

if __name__ == "__main__":
    main()