#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComSeg 评估 —— 拼合 tile_results/*.h5ad 的细胞质心+基因向量，复用参考脚本的匹配/指标逻辑。
comseg 输出是分块 h5ad(细胞级)、无全局像素标签图，故用质心+每细胞基因向量评估，
面积/像素转录本类指标置 nan（comseg 输出本就没有像素足迹）。核心指标(数量/检测P,R,F1/
质心偏移/基因向量 cosine,js,pearson/每细胞转录本计数相关)与其它方法同口径。
用法:  python eval_comseg.py --dataset prostate1   (或 prostate2)
可选覆盖: --root --run-dir --gt-mask --out-dir --ref --roi-size
"""
import argparse, importlib.util, sys, glob, re
from pathlib import Path
import numpy as np

REF_DEFAULT  = "/data/qiuyijia/ucs/UCS/ucs_value_merfish_4.py"
DATA_BASE    = "/data/qiuyijia/dataset"
RESULT_BASE  = "/data/qiuyijia"
METHOD       = "ComSeg"
ROI_SIZE_DEF = 10000

def _load_ref(ref):
    ref = Path(ref).resolve()
    if not ref.exists():
        raise SystemExit(f"参考评估脚本不存在: {ref}")
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
    """拼 16 个 tile 的细胞 -> 全局质心(um) + 每细胞基因向量(基因表对齐到 tile 的 var)。"""
    import anndata as ad, pandas as pd
    meta = pd.read_csv(Path(run_dir)/"tile_metadata.tsv", sep="\t")
    origin = {r.tile_id: (float(r.ry), float(r.rx)) for r in meta.itertuples()}
    PX_UM = geo.um_per_px
    fs = sorted(glob.glob(f"{run_dir}/tile_results/tile_*.h5ad"))
    if not fs:
        raise SystemExit(f"{run_dir}/tile_results 下没有 h5ad")
    all_cx=[]; all_cy=[]; gms=[]; genes=None
    for f in fs:
        a = ad.read_h5ad(f)
        tid = re.search(r"tile_\d+", f).group()
        if tid not in origin:
            tid2 = a.obs.get("image_name")
            tid = tid2.iloc[0] if tid2 is not None else tid
        ry, rx = origin[tid]
        # tile 内质心是 um（relative to tile origin）；全局像素 = 原点 + um/PX_UM
        cx_um = a.obs["centroid_x"].to_numpy(float)
        cy_um = a.obs["centroid_y"].to_numpy(float)
        gpx = rx + cx_um / PX_UM
        gpy = ry + cy_um / PX_UM
        ux, uy = geo.local_to_um(gpx, gpy)     # 全局像素 -> 全局微米
        all_cx.append(ux); all_cy.append(uy)
        X = a.X
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        gms.append((list(a.var_names), X))
        if genes is None:
            genes = list(a.var_names)
    # 统一基因表（各 tile var 相同则直接堆叠）
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
    # 只留 ROI 内
    inside = geo.inside(cx, cy)
    return cx[inside], cy[inside], gm[inside], genes

def build_gt_gene_matrix(gt, genes):
    """把参考脚本 gt 的每细胞基因向量对齐到 comseg 的基因表。"""
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
        raise SystemExit("需要 --dataset，或同时给 --root 和 --run-dir")
    args.out_dir = args.out_dir or f"{RESULT_BASE}/eval_merfish_out"
    if not args.gt_mask:
        args.gt_mask = _pick_gt_mask(args.run_dir, args.root, args.dataset)

    V, ref = _load_ref(args.ref)
    root, out = Path(args.root), Path(args.out_dir)
    V.setup_log(out, f"eval_ComSeg_{args.dataset or 'x'}")
    V.rule(f"ComSeg  x  MERFISH {args.dataset or ''}   (质心拼合, via {ref.name})")

    geo = V.load_geometry(root, args.roi_size)
    gt_mask = V.load_gt_mask(root, geo, args.gt_mask)
    tx = V.load_transcripts(root, geo, gt_mask)
    gt = V.load_ground_truth(root, geo, tx)

    cx, cy, gm, genes = load_comseg_cells(args.run_dir, geo)
    V.log.info(f"comseg 拼合细胞 = {len(cx):,}  (基因 {len(genes)})")

    # ---- 匹配 + 指标（复用参考脚本函数，与其它方法同口径）----
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

        # 每细胞转录本计数相关：
        # 也只统计共同 benchmark genes，避免 ComSeg 额外 feature
        # 使 predicted total artificially larger.
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
    # 落表：追加到 out/eval_results.csv
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