#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate all ten Baysor/Xenium outputs with the ProSeg reference metrics.

The metrics are the same as in ``ucs_value_xenium_4&5&6.py``.  Baysor's
``segmentation_cell_stats.csv`` is used for predicted cell centroids/counts,
and ``segmentation.csv`` is streamed for molecule assignment and gene vectors.
All coordinates are Xenium global micrometres.  ROI bounds are the exact bounds
used for the supplied Baysor runs.  Legacy CSV and modern Parquet Baysor output
styles are both supported.

Examples::

    python baysor_value_xenium.py --dataset breast
    python baysor_value_xenium.py --all
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

VERSION = "2026.09.18"
METHOD = "Baysor"
PIXEL_SIZE_UM = 0.2125
USES_PRIOR = False

DATASETS = {
    # x0, x1, y0, y1 are micrometres.
    "breast": dict(root="/data/qiuyijia/dataset/xenium_breast",
                   pred="/data/qiuyijia/baysor/xenium_breast_output",
                   roi=(5205.6125, 6268.1125, 2382.125, 3444.625), label="xenium_breast"),
    "colon": dict(root="/data/qiuyijia/dataset/xenium_colon",
                  pred="/data/qiuyijia/baysor/xenium_colon_output",
                  roi=(1963.7125, 4088.7125, 2561.05, 4686.05), label="xenium_colon"),
    # Baysor was run on a 15000 x 15000 native-pixel ROI, but the benchmark
    # comparison intentionally uses its centered 10000 x 10000 sub-ROI.  The
    # exact bounds come from dataset/xenium_liver/central_roi_10000/
    # gt_roi_info.json (native x0=21974, y0=13704).
    "liver": dict(root="/data/qiuyijia/dataset/xenium_liver",
                  pred="/data/qiuyijia/baysor/xenium_liver_output",
                  roi=(4669.475, 6794.475, 2912.1, 5037.1), label="xenium_liver"),
    "mouse_brain": dict(root="/data/qiuyijia/dataset/xenium_mouse_brain",
                        pred="/data/qiuyijia/baysor/xenium_mouse_brain_output",
                        roi=(2566.3625, 4691.3625, 1478.15, 3603.15), label="xenium_mouse_brain"),
    "roi_crop": dict(root="/data/qiuyijia/dataset/xenium_roi_crop",
                     pred="/data/qiuyijia/baysor/xenium_roi_crop_output",
                     roi=(3982.5, 4317.5, 2522.5, 2857.5), label="xenium_roi_crop"),
    "xenium7": dict(root="/data/qiuyijia/dataset/xenium7/outs",
                    pred="/data/qiuyijia/baysor/xenium7_output",
                    roi=(4692.0, 6817.0, 2169.625, 4294.625), label="xenium7",
                    gt_tx="/data/qiuyijia/dataset/xenium7/outs/_roi_cache/tx_roi_10210_22080_10000_qv20.parquet"),
    "xenium8": dict(root="/data/qiuyijia/dataset/xenium_8",
                    pred="/data/qiuyijia/baysor/xenium_8_output",
                    roi=(4675.2125, 6800.2125, 1841.1, 3966.1), label="xenium8",
                    gt_tx="/data/qiuyijia/dataset/xenium_8/_roi_cache/tx_roi_8664_22001_10000_qv20.parquet"),
    "xenium9": dict(root="/data/qiuyijia/dataset/xenium9",
                    pred="/data/qiuyijia/baysor/xenium9_ncellsinit_output",
                    roi=(3479.475, 5604.475, 1093.7375, 3218.7375), label="xenium9",
                    gt_zip="/data/qiuyijia/dataset/xenium9/Xenium_Prime_Human_Skin_FFPE_outs.zip",
                    gt_tx="/data/qiuyijia/dataset/xenium9/_roi_cache/tx_roi_16374_5147_10000_qv20.parquet"),
    "xenium10": dict(root="/data/qiuyijia/dataset/xenium10",
                     pred="/data/qiuyijia/baysor/xenium10_ncellsinit_output",
                     roi=(4369.85, 6494.85, 2921.875, 5046.875), label="xenium10",
                     gt_h5ad_zip="/data/qiuyijia/dataset/xenium10/transcriptome.zip",
                     gt_tx="/data/qiuyijia/dataset/xenium10/_roi_cache/tx_roi_20564_13750_10000_qv20.parquet"),
    "pancreas": dict(root="/data/qiuyijia/dataset/xenium_pancreas_cancer",
                     pred="/data/qiuyijia/baysor/xenium_pancreas_output",
                     roi=(-np.inf, np.inf, -np.inf, np.inf), label="xenium_pancreas",
                     stats="cells.parquet", segmentation="molecules.parquet",
                     gt_boundaries="/data/qiuyijia/dataset/xenium_pancreas_cancer/cell_boundaries.parquet",
                     uses_platform_prior=True),
}

REQUESTED = [
    "method", "Pred cell count", "GT cell count", "Cell count ratio",
    "Matched cell count", "Detection precision", "Detection recall", "Detection F1",
    "Mean centroid shift", "Median centroid shift", "p95 centroid shift",
    "matched_pair_n_transcripts_vs_total_counts_pearson",
    "matched_pair_n_transcripts_vs_total_counts_spearman",
    "matched_pair_n_transcripts_vs_total_counts_mae",
    "matched_pair_n_transcripts_vs_total_counts_rmse", "Pred transcript rows",
    "GT transcript rows", "Pred transcript assignment rate", "Transcript matched cell pairs",
    "matched_cell_transcript_count_pearson", "matched_cell_transcript_count_spearman",
    "matched_cell_transcript_count_mae", "matched_cell_transcript_count_rmse",
    "matched_cell_gene_vector_mean_cosine", "matched_cell_gene_vector_mean_js_distance",
    "matched_cell_gene_vector_mean_pearson", "matched_cell_gene_vector_valid_pairs",
    "transcript_id_overlap_n", "transcript_assignment_accuracy_via_matched_cells",
    "dataset", "uses_platform_prior", "Overall correct fraction",
]
SELECTED = [
    "method", "dataset", "GT cell count", "Pred cell count", "Cell count ratio",
    "Matched cell count", "Detection precision", "Detection recall", "Detection F1",
    "Mean centroid shift", "Median centroid shift", "p95 centroid shift",
    "matched_cell_transcript_count_pearson", "matched_cell_transcript_count_spearman",
    "matched_cell_transcript_count_mae", "matched_cell_transcript_count_rmse",
    "matched_cell_gene_vector_mean_cosine", "matched_cell_gene_vector_mean_js_distance",
    "matched_cell_gene_vector_mean_pearson", "transcript_id_overlap_n",
    "transcript_assignment_accuracy_via_matched_cells", "Pred transcript rows",
    "GT transcript rows", "Pred transcript assignment rate", "Overall correct fraction",
    "median_area_um2", "median_diameter_um", "coverage_fraction", "uses_platform_prior",
]

CONTROLS = ("NegControl", "NegPrb", "Neg_", "BLANK", "Blank", "Unassigned",
            "DeprecatedCodeword", "GenomicControl", "antisense", "Intergenic", "intergenic")
UNASSIGNED = {"", "0", "-1", "UNASSIGNED", "nan", "NaN", "None", "none", "null"}
log = logging.getLogger("eval_baysor_xenium")


def setup_log(out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(out / f"{name}.log", mode="w"); fh.setFormatter(fmt); log.addHandler(fh)


def find_file(root: Path, *patterns: str) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(root.rglob(pat))
        if hits:
            return hits[0]
    return None


def _normalize_gt_frame(df: pd.DataFrame, roi: Tuple[float, float, float, float], source: str) -> pd.DataFrame:
    """Normalize an official Xenium cell table to the evaluator's GT schema."""
    low = {str(c).lower(): c for c in df.columns}

    def find(*names: str) -> str:
        for name in names:
            if name.lower() in low:
                return str(low[name.lower()])
        raise ValueError(f"{source} lacks one of {names}; columns={list(df.columns)}")

    idc = find("cell_id", "cell", "label_id")
    xc = find("x_centroid", "center_x", "x", "vertex_x")
    yc = find("y_centroid", "center_y", "y", "vertex_y")
    total = next(
        (c for c in df.columns if str(c).lower() in {"total_counts", "transcript_counts", "n_transcripts"}),
        None,
    )
    out = df.rename(columns={idc: "_id", xc: "_x", yc: "_y"}).copy()
    out["_id"] = out._id.map(norm_cell)
    out["_x"] = pd.to_numeric(out._x, errors="coerce")
    out["_y"] = pd.to_numeric(out._y, errors="coerce")
    out["_total"] = pd.to_numeric(out[total], errors="coerce") if total else np.nan
    x0, x1, y0, y1 = roi
    keep = out._x.ge(x0) & out._x.lt(x1) & out._y.ge(y0) & out._y.lt(y1) & out._id.ne("")
    return out.loc[keep].reset_index(drop=True)


def _h5ad_row_sums(group, row_count: int, chunk_rows: int = 10_000) -> np.ndarray:
    """Sum CSR rows from an AnnData HDF5 group without loading the full matrix."""
    indptr = np.asarray(group["indptr"], dtype=np.int64)
    data = group["data"]
    totals = np.zeros(row_count, dtype=np.float64)
    for start in range(0, row_count, chunk_rows):
        stop = min(start + chunk_rows, row_count)
        lo, hi = int(indptr[start]), int(indptr[stop])
        values = np.asarray(data[lo:hi], dtype=np.float64)
        local = indptr[start:stop + 1] - lo
        if len(values):
            cs = np.empty(len(values) + 1, dtype=np.float64)
            cs[0] = 0.0
            np.cumsum(values, out=cs[1:])
            totals[start:stop] = cs[local[1:]] - cs[local[:-1]]
    return totals


def load_configured_gt_cells(
    cfg: Dict[str, object],
    root: Path,
    roi: Tuple[float, float, float, float],
    cache_root: Path,
) -> pd.DataFrame:
    """Read official GT cells, including ZIP/AnnData/boundary-only exports."""
    if cfg.get("gt_zip"):
        archive = Path(str(cfg["gt_zip"]))
        with zipfile.ZipFile(archive) as zf:
            with zf.open("cells.parquet") as handle:
                frame = pd.read_parquet(io.BytesIO(handle.read()))
        return _normalize_gt_frame(frame, roi, f"{archive}!cells.parquet")

    if cfg.get("gt_h5ad_zip"):
        cache_root.mkdir(parents=True, exist_ok=True)
        cache = cache_root / "xenium10_official_cells.parquet"
        if not cache.is_file():
            archive = Path(str(cfg["gt_h5ad_zip"]))
            with tempfile.TemporaryDirectory(prefix="xenium10_gt_") as tmp:
                with zipfile.ZipFile(archive) as zf:
                    h5ad_path = Path(zf.extract("transcriptome/adata.h5ad", path=tmp))
                import h5py

                with h5py.File(h5ad_path, "r") as h5:
                    ids = h5["obs/_index"].asstr()[:]
                    xy = np.asarray(h5["obsm/spatial"], dtype=np.float64)
                    totals = _h5ad_row_sums(h5["X"], len(ids))
                pd.DataFrame({
                    "cell_id": ids,
                    "x_centroid": xy[:, 0],
                    "y_centroid": xy[:, 1],
                    "total_counts": totals,
                }).to_parquet(cache, index=False)
        return _normalize_gt_frame(pd.read_parquet(cache), roi, str(cache))

    if cfg.get("gt_boundaries"):
        path = Path(str(cfg["gt_boundaries"]))
        boundary = pd.read_parquet(path)
        low = {str(c).lower(): c for c in boundary.columns}
        idc = low.get("cell_id", low.get("cell"))
        xc = low.get("vertex_x", low.get("x"))
        yc = low.get("vertex_y", low.get("y"))
        if idc is None or xc is None or yc is None:
            raise ValueError(f"cannot derive GT centroids from {path}: {list(boundary.columns)}")
        frame = (boundary.groupby(idc, sort=False)[[xc, yc]].mean().reset_index()
                 .rename(columns={idc: "cell_id", xc: "x_centroid", yc: "y_centroid"}))
        return _normalize_gt_frame(frame, roi, str(path))

    gt_path = find_file(root, "cells.parquet", "cells.csv.gz", "cells.csv")
    if not gt_path:
        raise FileNotFoundError(f"official cells table not found under {root}")
    return load_cells(gt_path, roi, gt=True)


def norm_cell(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in UNASSIGNED:
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def infer_radius(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    if len(x) < 2:
        return 2.0
    d, _ = cKDTree(np.column_stack([x, y])).query(np.column_stack([x, y]), k=2)
    nn = d[:, 1]; nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(max(np.quantile(nn, .95) * 1.25, 2.0)) if len(nn) else 2.0


def sparse_hungarian(px: np.ndarray, py: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                     radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one sparse centroid matching from the reference evaluator."""
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if not len(px) or not len(gx):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    pairs = cKDTree(np.column_stack([px, py])).query_ball_tree(
        cKDTree(np.column_stack([gx, gy])), r=radius)
    pi, gi = [], []
    for i, js in enumerate(pairs):
        for j in js: pi.append(i); gi.append(j)
    if not pi:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, float)
    pi, gi = np.asarray(pi, int), np.asarray(gi, int)
    dd = np.hypot(px[pi] - gx[gi], py[pi] - gy[gi])
    graph = coo_matrix((np.ones(len(pi)), (pi, len(px) + gi)),
                       shape=(len(px) + len(gx), len(px) + len(gx)))
    _, labels = connected_components(graph + graph.T, directed=False)
    comp = labels[pi]; order = np.argsort(comp, kind="stable")
    pi, gi, dd, comp = pi[order], gi[order], dd[order], comp[order]
    starts = np.append(np.searchsorted(comp, np.unique(comp)), len(pi))
    out_p, out_g, out_d = [], [], []
    for lo, hi in zip(starts[:-1], starts[1:]):
        p, g, d = pi[lo:hi], gi[lo:hi], dd[lo:hi]
        up, ug = np.unique(p), np.unique(g)
        if len(up) * len(ug) > 400 * 400:
            used_p, used_g = set(), set()
            for k in np.argsort(d):
                if p[k] in used_p or g[k] in used_g: continue
                used_p.add(int(p[k])); used_g.add(int(g[k]))
                out_p.append(int(p[k])); out_g.append(int(g[k])); out_d.append(float(d[k]))
            continue
        pm = {int(v): i for i, v in enumerate(up)}; gm = {int(v): i for i, v in enumerate(ug)}
        big = float(d.max() * 10 + 1); cost = np.full((len(up), len(ug)), big)
        cost[[pm[int(v)] for v in p], [gm[int(v)] for v in g]] = d
        rr, cc = linear_sum_assignment(cost)
        for r, c in zip(rr, cc):
            if cost[r, c] < big:
                out_p.append(int(up[r])); out_g.append(int(ug[c])); out_d.append(float(cost[r, c]))
    return np.asarray(out_p, int), np.asarray(out_g, int), np.asarray(out_d, float)


def pair_metrics(a: Sequence[float], b: Sequence[float], prefix: str) -> Dict[str, float]:
    from scipy.stats import pearsonr, spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    result = {f"{prefix}_{k}": np.nan for k in ("pearson", "spearman", "mae", "rmse")}
    if ok.sum() < 3: return result
    a, b = a[ok], b[ok]
    if a.std() > 0 and b.std() > 0:
        result[f"{prefix}_pearson"] = float(pearsonr(a, b)[0])
        result[f"{prefix}_spearman"] = float(spearmanr(a, b)[0])
    result[f"{prefix}_mae"] = float(np.mean(np.abs(a - b)))
    result[f"{prefix}_rmse"] = float(np.sqrt(np.mean((a - b) ** 2)))
    return result


def gene_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    from scipy.spatial.distance import jensenshannon
    co, js, pr = [], [], []
    for a, b in zip(pred.astype(float), gt.astype(float)):
        if a.sum() <= 0 or b.sum() <= 0: continue
        co.append(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
        d = jensenshannon(a / a.sum(), b / b.sum(), base=2)
        if np.isfinite(d): js.append(float(d))
        if a.std() > 0 and b.std() > 0: pr.append(float(np.corrcoef(a, b)[0, 1]))
    return {"matched_cell_gene_vector_mean_cosine": float(np.mean(co)) if co else np.nan,
            "matched_cell_gene_vector_mean_js_distance": float(np.mean(js)) if js else np.nan,
            "matched_cell_gene_vector_mean_pearson": float(np.mean(pr)) if pr else np.nan,
            "matched_cell_gene_vector_valid_pairs": len(co)}


def load_cells(path: Path, roi: Tuple[float, float, float, float], gt: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if gt:
        def find(*names):
            low = {str(c).lower(): c for c in df.columns}
            for n in names:
                if n.lower() in low: return low[n.lower()]
            raise ValueError(f"{path} lacks one of {names}; columns={list(df.columns)}")
        idc, xc, yc = find("cell_id", "cell"), find("x_centroid", "center_x"), find("y_centroid", "center_y")
        total = next((c for c in df.columns if str(c).lower() in {"total_counts", "transcript_counts"}), None)
        df = df.rename(columns={idc: "_id", xc: "_x", yc: "_y"})
        df["_id"] = df._id.map(norm_cell); df["_x"] = pd.to_numeric(df._x, errors="coerce"); df["_y"] = pd.to_numeric(df._y, errors="coerce")
        df["_total"] = pd.to_numeric(df[total], errors="coerce") if total else np.nan
    else:
        names = {str(c).lower(): c for c in df.columns}
        for canonical, aliases in {"cell": ("cell", "cell_id"), "x": ("x", "centroid_x"), "y": ("y", "centroid_y")}.items():
            if canonical not in df:
                for a in aliases:
                    if a in names: df = df.rename(columns={names[a]: canonical}); break
        for c in ("x", "y", "n_transcripts", "area"):
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["cell"] = df.cell.map(norm_cell)
    x0, x1, y0, y1 = roi
    xcol, ycol = ("_x", "_y") if gt else ("x", "y")
    keep = df[xcol].ge(x0) & df[xcol].lt(x1) & df[ycol].ge(y0) & df[ycol].lt(y1)
    return df.loc[keep].reset_index(drop=True)


def transcript_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield canonical transcript columns from Parquet or CSV."""
    aliases = {"cell_id": ("cell_id", "cell", "platform_cell"), "gene": ("feature_name", "gene", "target"),
               "x": ("x_location", "x", "x_um", "global_x_px", "global_x"),
               "y": ("y_location", "y", "y_um", "global_y_px", "global_y"),
               "qv": ("qv",), "is_gene": ("is_gene",), "is_noise": ("is_noise",)}
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        names = set(pq.ParquetFile(path).schema.names)
        chosen = {k: next((a for a in v if a in names), None) for k, v in aliases.items()}
        chosen = {k: v for k, v in chosen.items() if v is not None}
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(chosen.values())):
            yield batch.to_pandas().rename(columns={v: k for k, v in chosen.items()})
    else:
        header = pd.read_csv(path, nrows=0)
        chosen = {k: next((a for a in v if a in header.columns), None) for k, v in aliases.items()}
        chosen = {k: v for k, v in chosen.items() if v is not None}
        for frame in pd.read_csv(path, usecols=list(chosen.values()), chunksize=batch_size):
            yield frame.rename(columns={v: k for k, v in chosen.items()})


def qc_transcripts(frame: pd.DataFrame, roi: Tuple[float, float, float, float], qv: float) -> pd.DataFrame:
    x0, x1, y0, y1 = roi
    frame["x"] = pd.to_numeric(frame.x, errors="coerce"); frame["y"] = pd.to_numeric(frame.y, errors="coerce")
    keep = frame.x.ge(x0) & frame.x.lt(x1) & frame.y.ge(y0) & frame.y.lt(y1)
    if qv > 0 and "qv" in frame: keep &= pd.to_numeric(frame.qv, errors="coerce").ge(qv)
    if "is_gene" in frame: keep &= frame.is_gene.fillna(False).astype(bool)
    elif "gene" in frame: keep &= ~frame.gene.astype(str).str.startswith(CONTROLS, na=False)
    return frame.loc[keep].reset_index(drop=True)


def coord_key(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Xenium coordinates are stored to 1/16 or 1/32 µm precision; 0.01 µm
    # rounding is robust to CSV formatting while preserving molecule identity.
    return np.rint(np.asarray(x, float) * 100).astype(np.int64) * 2_000_000 + np.rint(np.asarray(y, float) * 100).astype(np.int64)


def evaluate(name: str, args: argparse.Namespace) -> Dict[str, object]:
    cfg = DATASETS[name]; root = Path(args.data_root or cfg["root"]); pred_root = Path(args.run_dir or cfg["pred"])
    roi = tuple(args.roi) if args.roi else cfg["roi"]; out = Path(args.out_dir) / name
    setup_log(out, f"eval_baysor_{name}")
    stats_path = Path(args.stats) if args.stats else pred_root / str(cfg.get("stats", "segmentation_cell_stats.csv"))
    seg_path = Path(args.segmentation) if args.segmentation else pred_root / str(cfg.get("segmentation", "segmentation.csv"))
    tx_path = Path(str(cfg["gt_tx"])) if cfg.get("gt_tx") else find_file(
        root, "transcripts.parquet", "transcripts.csv.gz", "transcripts.csv")
    if not stats_path.is_file() or not seg_path.is_file() or not tx_path:
        raise FileNotFoundError(
            f"missing Baysor/GT input: stats={stats_path}, segmentation={seg_path}, transcripts={tx_path}")
    pred = load_cells(stats_path, roi)
    gt = load_configured_gt_cells(cfg, root, roi, Path(args.out_dir) / "_cache")
    if pred.empty or gt.empty: raise ValueError(f"empty prediction or GT in ROI {roi}")
    px, py, gx, gy = pred.x.to_numpy(float), pred.y.to_numpy(float), gt._x.to_numpy(float), gt._y.to_numpy(float)
    radius = args.match_radius if args.match_radius is not None else infer_radius(gx, gy)
    mp, mg, md = sparse_hungarian(px, py, gx, gy, radius)
    p_to_g = np.full(len(pred), -1, np.int64); p_to_g[mp] = mg
    pred_idx = {str(v): i for i, v in enumerate(pred.cell)}; gt_idx = {str(v): i for i, v in enumerate(gt._id)}
    # Stream both tables.  The coordinate index contains only molecules from
    # predicted cells matched to a GT centroid, mirroring the reference code.
    pred_total_rows = 0; pred_rows = 0
    pred_coord_parts: List[np.ndarray] = []; pred_target_parts: List[np.ndarray] = []
    pred_pairs = []; genes: Dict[str, int] = {}
    for frame in transcript_batches(seg_path, args.batch_size):
        if "cell_id" not in frame: raise ValueError(f"{seg_path} has no cell column")
        frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
        frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
        cells = frame.cell_id.map(norm_cell)
        in_roi = frame.x.ge(roi[0]) & frame.x.lt(roi[1]) & frame.y.ge(roi[2]) & frame.y.lt(roi[3])
        pred_total_rows += int(in_roi.sum())
        keep = cells.ne("") & in_roi
        if "is_noise" in frame: keep &= ~frame.is_noise.fillna(False).astype(bool)
        if not keep.any(): continue
        cells = cells.loc[keep]; pred_rows += int(len(cells)); p = cells.map(pred_idx).fillna(-1).to_numpy(np.int64)
        valid = p >= 0; x, y = frame.loc[keep, "x"].to_numpy(float), frame.loc[keep, "y"].to_numpy(float)
        matched = valid & (p_to_g[p.clip(min=0)] >= 0)
        if matched.any():
            ck = coord_key(x[matched], y[matched])
            pred_coord_parts.append(ck)
            pred_target_parts.append(ck * (len(gt) + 1) + p_to_g[p[matched]])
        if valid.any() and "gene" in frame:
            gg = frame.loc[keep, "gene"].astype(str).to_numpy(); rr = p[valid]; gi = []
            for g in gg[valid]:
                if g not in genes: genes[g] = len(genes)
                gi.append(genes[g])
            z = pd.DataFrame({"r": rr, "g": gi}).groupby(["r", "g"], sort=False).size().reset_index(name="n")
            pred_pairs.append((z.r.to_numpy(np.int32), z.g.to_numpy(np.int32), z.n.to_numpy(np.int32)))
    coord_index = np.unique(np.concatenate(pred_coord_parts)) if pred_coord_parts else np.zeros(0, np.int64)
    target_index = np.unique(np.concatenate(pred_target_parts)) if pred_target_parts else np.zeros(0, np.int64)
    coord_index.sort(); target_index.sort()
    gt_rows = gt_assigned = overlap = correct = 0; gt_pairs = []
    gt_stream_counts = np.zeros(len(gt), dtype=np.int64)
    for frame in transcript_batches(tx_path, args.batch_size):
        frame = qc_transcripts(frame, roi, args.qv_min)
        if frame.empty: continue
        gt_rows += len(frame)
        if "cell_id" not in frame: continue
        c = frame.cell_id.map(norm_cell); assigned = c.ne(""); gt_assigned += int(assigned.sum())
        gi_cell = c.map(gt_idx).fillna(-1).to_numpy(np.int64); keys = coord_key(frame.x, frame.y)
        query = keys * (len(gt) + 1) + np.maximum(gi_cell, 0)
        if len(coord_index):
            pos_any = np.searchsorted(coord_index, keys); safe_any = np.minimum(pos_any, len(coord_index) - 1)
            hit_any = (pos_any < len(coord_index)) & (coord_index[safe_any] == keys)
        else: hit_any = np.zeros(len(frame), bool)
        if len(target_index):
            pos_target = np.searchsorted(target_index, query); safe_target = np.minimum(pos_target, len(target_index) - 1)
            hit_target = (pos_target < len(target_index)) & (target_index[safe_target] == query)
        else: hit_target = np.zeros(len(frame), bool)
        # As in the reference evaluator, the denominator contains only
        # transcripts whose official cell is one of the GT cells retained by
        # the ROI (a molecule from a cell centred outside the ROI is not a
        # valid matched-cell transcript).
        overlap += int((hit_any & assigned & (gi_cell >= 0)).sum())
        correct += int((hit_target & assigned & (gi_cell >= 0)).sum())
        valid = gi_cell >= 0
        if valid.any():
            gt_stream_counts += np.bincount(gi_cell[valid], minlength=len(gt))
        if valid.any() and "gene" in frame:
            g = frame.loc[valid, "gene"].astype(str).to_numpy(); ids = []
            for name_g in g:
                if name_g not in genes: genes[name_g] = len(genes)
                ids.append(genes[name_g])
            z = pd.DataFrame({"r": gi_cell[valid], "g": ids}).groupby(["r", "g"], sort=False).size().reset_index(name="n")
            gt_pairs.append((z.r.to_numpy(np.int32), z.g.to_numpy(np.int32), z.n.to_numpy(np.int32)))

    # Boundary-only exports (currently pancreas) have no official cell count
    # column.  Fill it from the same QC-filtered GT transcript stream.
    if gt["_total"].isna().all():
        gt["_total"] = gt_stream_counts.astype(float)

    n = len(mp); prec = n / max(len(pred), 1); rec = n / max(len(gt), 1)
    row: Dict[str, object] = {"method": METHOD, "dataset": cfg["label"], "Pred cell count": len(pred), "GT cell count": len(gt),
        "Cell count ratio": len(pred) / max(len(gt), 1), "Matched cell count": n,
        "Detection precision": prec, "Detection recall": rec, "Detection F1": 2 * prec * rec / max(prec + rec, 1e-12),
        "Mean centroid shift": float(md.mean()) if n else np.nan, "Median centroid shift": float(np.median(md)) if n else np.nan,
        "p95 centroid shift": float(np.percentile(md, 95)) if n else np.nan,
        "Pred transcript rows": int(pred_total_rows), "GT transcript rows": int(gt_rows),
        "Pred transcript assignment rate": pred_rows / max(pred_total_rows, 1), "Transcript matched cell pairs": n,
        "transcript_id_overlap_n": int(overlap), "transcript_assignment_accuracy_via_matched_cells": correct / overlap if overlap else np.nan,
        "Overall correct fraction": (pred_rows / max(pred_total_rows, 1)) * (correct / overlap) if overlap else np.nan,
        "median_area_um2": float(pred.area.median()) if "area" in pred else np.nan,
        "median_diameter_um": float(2 * np.sqrt(pred.area.median() / np.pi)) if "area" in pred else np.nan,
        "coverage_fraction": (float(pred.area.sum() / ((roi[1] - roi[0]) * (roi[3] - roi[2])))
                              if "area" in pred and np.isfinite(roi).all() else np.nan),
        "uses_platform_prior": bool(cfg.get("uses_platform_prior", USES_PRIOR))}
    n_tx = pred.n_transcripts.to_numpy(float) if "n_transcripts" in pred else np.full(len(pred), np.nan)
    if n:
        row.update(pair_metrics(n_tx[mp], gt._total.to_numpy(float)[mg], "matched_pair_n_transcripts_vs_total_counts"))
        row.update(pair_metrics(n_tx[mp], gt._total.to_numpy(float)[mg], "matched_cell_transcript_count"))
    else:
        for pfx in ("matched_pair_n_transcripts_vs_total_counts", "matched_cell_transcript_count"):
            row.update({f"{pfx}_{k}": np.nan for k in ("pearson", "spearman", "mae", "rmse")})
    if n and genes:
        def matrix(parts):
            m = np.zeros((len(pred if parts is pred_pairs else gt), len(genes)), np.float32)
            for r, c, v in parts: np.add.at(m, (r, c), v)
            return m
        row.update(gene_metrics(matrix(pred_pairs)[mp], matrix(gt_pairs)[mg]))
    else:
        row.update({"matched_cell_gene_vector_mean_cosine": np.nan, "matched_cell_gene_vector_mean_js_distance": np.nan,
                    "matched_cell_gene_vector_mean_pearson": np.nan, "matched_cell_gene_vector_valid_pairs": 0})
    tag = f"baysor_{name}"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: row.get(k, np.nan) for k in REQUESTED}]).to_csv(out / f"{tag}_sparse_hungarian_qc_metrics_full.csv", index=False)
    pd.DataFrame([{k: row.get(k, np.nan) for k in SELECTED}]).to_csv(out / f"{tag}_sparse_hungarian_qc_metrics_selected.csv", index=False)
    if n:
        pd.DataFrame({"pred_id": pred.cell.iloc[mp].to_numpy(), "gt_id": gt._id.iloc[mg].to_numpy(), "centroid_shift_um": md,
                      "pred_n_transcripts": n_tx[mp], "gt_total_counts": gt._total.to_numpy(float)[mg],
                      "pred_area_um2": pred.area.iloc[mp].to_numpy() if "area" in pred else np.nan}).to_csv(out / f"{tag}_matched_pairs.csv", index=False)
    log.info("%s: pred=%d gt=%d matched=%d precision=%.4f recall=%.4f F1=%.4f", name, len(pred), len(gt), n, prec, rec, row["Detection F1"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(DATASETS), default="breast"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--run-dir"); ap.add_argument("--segmentation"); ap.add_argument("--stats"); ap.add_argument("--data-root")
    ap.add_argument("--out-dir", default="/data/qiuyijia/eval/baysor"); ap.add_argument("--qv-min", type=float, default=20.0)
    ap.add_argument("--match-radius", type=float); ap.add_argument("--batch-size", type=int, default=1_000_000)
    ap.add_argument("--roi", nargs=4, type=float, metavar=("X0", "X1", "Y0", "Y1")); ap.add_argument("--version", action="store_true")
    args = ap.parse_args()
    if args.version: print(VERSION); return
    if args.all and any(getattr(args, x) is not None for x in ("run_dir", "segmentation", "stats", "data_root", "roi")):
        ap.error("--all cannot be combined with path/ROI overrides")
    names = list(DATASETS) if args.all else [args.dataset]; rows = [evaluate(name, args) for name in names]
    if len(rows) > 1:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{k: r.get(k, np.nan) for k in REQUESTED} for r in rows]).to_csv(out / "baysor_xenium_evaluation.csv", index=False)
        pd.DataFrame([{k: r.get(k, np.nan) for k in SELECTED} for r in rows]).to_csv(out / "baysor_xenium_all_metrics_selected.csv", index=False)


if __name__ == "__main__":
    main()
