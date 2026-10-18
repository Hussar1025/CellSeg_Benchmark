#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一的、reference-assisted Bering benchmark runner。

输入使用 Xenium 风格的逐分子 Parquet schema；``cell_id`` 是 Bering 的粗分割
种子（raw_cells），可以来自 Xenium 或 CellBin。它不是 seed-free 分割。所有
数据集使用配置中明确记录的 QC、标签生成、Bering 超参数和随机种子。

配置由 ``configs/*.json`` 提供。脚本不会把官方 cell_boundaries 当作预测，
不会把独立的人工真值用于调参。Stereo-seq reference-assisted 模式会用已经
作为粗分割输入的 CellBin 种子数，在预先给定的 Leiden 候选值中逐 tile 选择。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
# Respect the launcher/user selection.  Device 6 remains the historical
# default when neither variable is provided.
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", os.environ.get("BERING_CUDA_VISIBLE_DEVICES", "6")
)
#os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("bering_xenium_benchmark")


def setup_log(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    # ``force=`` was added in Python 3.8, while the Bering environment is
    # Python 3.7.  Remove existing root handlers explicitly so this remains
    # compatible with both versions and also captures logs from legacy
    # ``bering_xenium.py`` (which uses its own named logger).
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); root.addHandler(sh)
    fh = logging.FileHandler(str(out / "run.log")); fh.setFormatter(fmt); root.addHandler(fh)


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_input(cfg: dict, name: str) -> Path:
    """Resolve an input under data_root; optionally extract one member from outs.zip."""
    raw = Path(cfg.get(name, ""))
    root = Path(cfg.get("data_root", "."))
    p = raw if raw.is_absolute() else root / raw
    if p.exists(): return p
    zip_name = cfg.get("outs_zip")
    if zip_name:
        z = root / zip_name
        if z.exists():
            extract = root / "outs"
            extract.mkdir(exist_ok=True)
            member = cfg.get(name, "")
            with zipfile.ZipFile(z) as zh:
                hit = next((n for n in zh.namelist()
                            if n == member or n.endswith("/" + member)
                            or Path(n).name == Path(member).name), None)
                if hit:
                    target = extract / Path(member).name
                    if not target.exists():
                        LOG.info("extracting %s from %s", member, z.name)
                        partial = target.with_name(target.name + f".partial.{os.getpid()}")
                        try:
                            with zh.open(hit) as src, open(partial, "wb") as dst:
                                while True:
                                    b = src.read(16 * 1024 * 1024)
                                    if not b: break
                                    dst.write(b)
                            os.replace(str(partial), str(target))
                        except BaseException:
                            try:
                                partial.unlink()
                            except FileNotFoundError:
                                pass
                            raise
                    return target
    raise FileNotFoundError(f"{name} not found: {p} (and no usable outs_zip)")


def roi_bounds(cfg: dict, tx: pd.DataFrame):
    r = cfg["roi"]
    mode = r["mode"]
    if mode == "full_dataset":
        return (float(tx.x.min()), float(tx.x.max()) + 1e-6,
                float(tx.y.min()), float(tx.y.max()) + 1e-6)
    if mode == "absolute_um":
        return tuple(float(r[k]) for k in ("x0_um", "x1_um", "y0_um", "y1_um"))
    if mode == "absolute_px":
        px = float(r["pixel_size_um"])
        return (float(r["x0_px"]) * px, float(r["x0_px"] + r["width_px"]) * px,
                float(r["y0_px"]) * px, float(r["y0_px"] + r["height_px"]) * px)
    if mode != "center_px": raise ValueError(f"unknown ROI mode {mode}")
    W, H = map(float, r["image_size_px"])
    px = float(r["pixel_size_um"])
    w, h = float(r["width_px"]), float(r["height_px"])
    x0, y0 = (W - w) / 2.0, (H - h) / 2.0
    return ((x0 * px), ((x0 + w) * px),
            (y0 * px), ((y0 + h) * px))


def load_tx(cfg: dict) -> tuple[pd.DataFrame, tuple[float, float, float, float]]:
    p = resolve_input(cfg, "transcripts")
    cols = ["transcript_id", "cell_id", "feature_name", "x_location",
            "y_location", "z_location", "qv"]
    # is_gene is absent in older Xenium exports; read only columns present.
    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(p).schema.names)
    use = [c for c in cols if c in available] + (["is_gene"] if "is_gene" in available else [])
    q = cfg["qc"]

    # Fixed benchmark ROIs can be computed without loading transcript data.
    # Push QV/gene/coordinate predicates into PyArrow before conversion to
    # pandas.  This is essential for xenium10 (~350 million raw transcripts):
    # reading the full table first would require tens of GiB of RAM.
    if cfg["roi"]["mode"] == "full_dataset":
        tx = pd.read_parquet(p, columns=use)
        b = None
    else:
        b = roi_bounds(cfg, pd.DataFrame({"x": [0.0], "y": [0.0]}))
        x0, x1, y0, y1 = b
        filters = [
            ("x_location", ">=", x0),
            ("x_location", "<", x1),
            ("y_location", ">=", y0),
            ("y_location", "<", y1),
        ]
        if "qv" in available:
            filters.append(("qv", ">=", float(q["qv_min"])))
        if q.get("is_gene_only", True) and "is_gene" in available:
            filters.append(("is_gene", "=", True))
        LOG.info("Parquet predicate pushdown: ROI/QV/is_gene before pandas conversion")
        tx = pq.read_table(str(p), columns=use, filters=filters).to_pandas()

    if "cell_id" not in tx or "feature_name" not in tx:
        raise RuntimeError(f"transcripts missing cell_id/feature_name: {list(tx.columns)}")
    # Keep these checks for full-dataset mode and for compatibility with
    # exports/readers that cannot apply one of the predicates at scan time.
    if "qv" in tx: tx = tx[pd.to_numeric(tx.qv, errors="coerce") >= float(q["qv_min"])]
    if q.get("is_gene_only", True) and "is_gene" in tx:
        tx = tx[tx.is_gene.astype(bool)]
    genes = tx.feature_name.map(
        lambda v: v.decode(errors="replace") if isinstance(v, (bytes, bytearray)) else str(v))
    prefixes = tuple(q.get("control_prefixes", []))
    if prefixes: tx = tx[~genes.str.startswith(prefixes).values]
    tx = tx.copy()
    tx["cell_id"] = tx.cell_id.map(
        lambda v: v.decode(errors="replace") if isinstance(v, (bytes, bytearray)) else str(v))
    tx["gene"] = tx.feature_name.map(
        lambda v: v.decode(errors="replace") if isinstance(v, (bytes, bytearray)) else str(v))
    tx["x"] = pd.to_numeric(tx.x_location, errors="coerce").astype(float)
    tx["y"] = pd.to_numeric(tx.y_location, errors="coerce").astype(float)
    tx["z"] = (pd.to_numeric(tx.z_location, errors="coerce").fillna(0).astype(float)
               if "z_location" in tx else 0.0)
    tx = tx[np.isfinite(tx.x) & np.isfinite(tx.y)].reset_index(drop=True)
    if b is None:
        b = roi_bounds(cfg, tx)
    x0, x1, y0, y1 = b
    tx = tx[(tx.x >= x0) & (tx.x < x1) & (tx.y >= y0) & (tx.y < y1)].reset_index(drop=True)
    LOG.info("input=%s; QC+ROI transcripts=%s; genes=%d; ROI um x[%.2f,%.2f] y[%.2f,%.2f]",
             p, f"{len(tx):,}", tx.gene.nunique(), x0, x1, y0, y1)
    return tx, b


def train_tile(tx: pd.DataFrame, cfg: dict, bounds):
    r = cfg["training"]
    size = float(r["tile_um"])
    cx, cy = (float(r["center_um"][0]), float(r["center_um"][1])) if r.get("center_um") else \
              ((bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2)
    h = size / 2
    t = tx[(tx.x >= cx-h) & (tx.x < cx+h) & (tx.y >= cy-h) & (tx.y < cy+h)].copy()
    if t.cell_id.replace({"UNASSIGNED": np.nan}).dropna().nunique() < 10:
        raise RuntimeError("training tile has fewer than 10 coarse-seed cells")
    LOG.info("training tile center=(%.1f,%.1f), size=%.1f um, transcripts=%s, cells=%d",
             cx, cy, size, f"{len(t):,}", t.cell_id.nunique())
    return t


def add_labels(seg: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Deterministic expression clustering; no ground-truth annotation is consumed."""
    from sklearn.cluster import MiniBatchKMeans
    ncl = int(cfg["training"].get("label_clusters", 8))
    cells = seg.loc[seg.raw_cells != "UNASSIGNED", "raw_cells"].drop_duplicates().astype(str).values
    genes = seg.gene.astype(str)
    top = genes.value_counts().head(int(cfg["training"].get("label_genes", 128))).index
    ci = {c: i for i, c in enumerate(cells)}; gi = {g: i for i, g in enumerate(top)}
    X = np.zeros((len(cells), len(top)), np.float32)
    for c, g in zip(seg.raw_cells.astype(str).values, genes.values):
        if c in ci and g in gi: X[ci[c], gi[g]] += 1
    k = max(2, min(ncl, len(cells)))
    labs = MiniBatchKMeans(n_clusters=k, random_state=int(cfg["seed"]),
                           n_init=3, batch_size=1024).fit_predict(X)
    mp = {c: f"expr_{int(l)}" for c, l in zip(cells, labs)}
    out = seg.copy(); out["raw_labels"] = out.raw_cells.map(mp).fillna("background")
    return out


def compress_bering_features(base: pd.DataFrame, training: pd.DataFrame, cfg: dict):
    """Bound Bering's dense NGC width without dropping any molecules.

    Bering 1.x materializes an ``n_spots x n_features`` dense float64 matrix
    for every training/prediction graph.  Whole-transcriptome Stereo-seq has
    ~20k genes, which made the previous run consume >400 GiB.  Retain the most
    abundant genes in the fixed training tile and map every other gene to one
    explicit OTHER feature.  Original genes remain in ``gene`` for export;
    only the model feature is compressed.
    """
    limit = int(cfg["training"].get("feature_genes", 512))
    if limit < 2:
        raise ValueError("training.feature_genes must be at least 2")
    counts = training["gene"].astype(str).value_counts()
    selected = counts.head(min(limit, len(counts))).index.astype(str)
    selected_set = set(selected)
    other = "__BERING_OTHER__"
    if other in selected_set or (base["gene"] == other).any():
        raise ValueError(f"Reserved Bering feature name occurs in input: {other}")
    base["bering_feature"] = base["gene"].where(base["gene"].isin(selected_set), other)
    training["bering_feature"] = training["gene"].where(
        training["gene"].isin(selected_set), other
    )
    stats = {
        "method": "top genes in fixed training tile plus one OTHER feature",
        "top_gene_limit": limit,
        "model_features_expected": int(len(selected) + 1),
        "original_features_full_roi": int(base["gene"].nunique()),
        "original_features_training_tile": int(len(counts)),
        "other_transcripts_full_roi": int((base["bering_feature"] == other).sum()),
        "other_fraction_full_roi": float((base["bering_feature"] == other).mean()),
        "top_gene_names": selected.tolist(),
        "reserved_other_feature": other,
        "all_molecules_retained": True,
    }
    LOG.info(
        "Bering feature compression: %d original genes -> %d top genes + OTHER; "
        "all %s molecules retained (OTHER %.1f%%)",
        stats["original_features_full_roi"], len(selected), f"{len(base):,}",
        100.0 * stats["other_fraction_full_roi"],
    )
    return base, training, stats


def stitch_tiles(
    pred: pd.DataFrame,
    tile_cores,
    radius_um: float = 8.0,
    boundary_band_um: float = 12.0,
    min_seed_purity: float = 0.8,
):
    """Safely reconnect one seed cell split across adjacent tile cores.

    A candidate pair must be close in space, lie near the *shared boundary* of
    two directly adjacent tiles, and have the same dominant coarse seed in
    both predicted components.  Requiring a shared seed also prevents the
    transitive cross-cell chains produced by the former global 15-um centroid
    union.  The operation is intentionally reference-assisted, like the rest
    of this benchmark, and its counts are recorded in ``run_summary.json``.
    """
    stats = {
        "method": "adjacent-boundary_same-dominant-seed",
        "radius_um": float(radius_um),
        "boundary_band_um": float(boundary_band_um),
        "min_seed_purity": float(min_seed_purity),
        "cells_before": int(pred.cell_id.nunique()) if not pred.empty else 0,
        "candidate_pairs": 0,
        "merged_pairs": 0,
        "cells_after": int(pred.cell_id.nunique()) if not pred.empty else 0,
    }
    if pred.empty or stats["cells_before"] < 2:
        return pred, stats
    required = {"cell_id", "x", "y", "tile_index", "seed_cell_id"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise RuntimeError(
            "Boundary-aware stitching requires prediction columns: " + ", ".join(missing)
        )
    if not tile_cores:
        raise RuntimeError("Boundary-aware stitching requires tile core metadata")
    if radius_um <= 0 or boundary_band_um <= 0:
        raise ValueError("stitch radius and boundary band must both be positive")
    if not 0.0 <= min_seed_purity <= 1.0:
        raise ValueError("stitch_min_seed_purity must be in [0, 1]")

    from scipy.spatial import cKDTree

    ids = pred["cell_id"].astype(str).drop_duplicates().to_numpy()
    grouped = pred.groupby("cell_id", sort=False)
    meta = grouped.agg(x=("x", "mean"), y=("y", "mean"),
                       tile_index=("tile_index", "first"), n=("x", "size")).loc[ids]

    valid_seed = ~pred["seed_cell_id"].astype(str).isin(
        ["UNASSIGNED", "0", "-1", "nan", "None", ""]
    )
    seed_counts = (pred.loc[valid_seed]
                   .groupby(["cell_id", "seed_cell_id"], sort=False)
                   .size().rename("seed_n").reset_index())
    if seed_counts.empty:
        raise RuntimeError("No assigned coarse seeds are available for safe tile stitching")
    dominant_idx = seed_counts.groupby("cell_id", sort=False)["seed_n"].idxmax()
    dominant = seed_counts.loc[dominant_idx].set_index("cell_id")
    meta["dominant_seed"] = dominant["seed_cell_id"].reindex(ids).astype(object)
    meta["seed_purity"] = dominant["seed_n"].reindex(ids).fillna(0).to_numpy() / meta["n"]

    # Determine exactly which tile pairs share a line segment.  Tiles that
    # meet only at one corner are not adjacent for stitching purposes.
    cores = {i: tuple(map(float, core)) for i, core in enumerate(tile_cores)}
    adjacency = {}
    eps = 1e-6
    for i in range(len(cores)):
        ax0, ax1, ay0, ay1 = cores[i]
        for j in range(i + 1, len(cores)):
            bx0, bx1, by0, by1 = cores[j]
            ylo, yhi = max(ay0, by0), min(ay1, by1)
            xlo, xhi = max(ax0, bx0), min(ax1, bx1)
            if yhi - ylo > eps and (abs(ax1 - bx0) <= eps or abs(bx1 - ax0) <= eps):
                boundary = bx0 if abs(ax1 - bx0) <= eps else ax0
                adjacency[(i, j)] = ("x", boundary, ylo, yhi)
            elif xhi - xlo > eps and (abs(ay1 - by0) <= eps or abs(by1 - ay0) <= eps):
                boundary = by0 if abs(ay1 - by0) <= eps else ay0
                adjacency[(i, j)] = ("y", boundary, xlo, xhi)

    cent = meta[["x", "y"]].to_numpy(float)
    tile_index = meta["tile_index"].to_numpy(dtype=int)
    dominant_seed = meta["dominant_seed"].to_numpy(dtype=object)
    seed_purity = meta["seed_purity"].to_numpy(dtype=float)
    centroid_x, centroid_y = cent[:, 0], cent[:, 1]
    tree = cKDTree(cent)
    pairs = tree.query_pairs(float(radius_um))
    stats["candidate_pairs"] = int(len(pairs))
    parent = np.arange(len(ids))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ta, tb = int(tile_index[a]), int(tile_index[b])
        if ta == tb:
            continue
        boundary = adjacency.get(tuple(sorted((ta, tb))))
        if boundary is None:
            continue
        seed_a, seed_b = dominant_seed[a], dominant_seed[b]
        if pd.isna(seed_a) or pd.isna(seed_b) or str(seed_a) != str(seed_b):
            continue
        if (seed_purity[a] < min_seed_purity or seed_purity[b] < min_seed_purity):
            continue
        axis, line, span_lo, span_hi = boundary
        if axis == "x":
            near = (abs(centroid_x[a] - line) <= boundary_band_um and
                    abs(centroid_x[b] - line) <= boundary_band_um and
                    span_lo - eps <= centroid_y[a] <= span_hi + eps and
                    span_lo - eps <= centroid_y[b] <= span_hi + eps)
        else:
            near = (abs(centroid_y[a] - line) <= boundary_band_um and
                    abs(centroid_y[b] - line) <= boundary_band_um and
                    span_lo - eps <= centroid_x[a] <= span_hi + eps and
                    span_lo - eps <= centroid_x[b] <= span_hi + eps)
        if not near:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            stats["merged_pairs"] += 1

    roots = {i: find(i) for i in range(len(ids))}
    mapping = {ids[i]: f"cell_{roots[i]:07d}" for i in range(len(ids))}
    out = pred.copy()
    out["cell_id"] = out["cell_id"].astype(str).map(mapping)
    stats["cells_after"] = int(out["cell_id"].nunique())
    LOG.info(
        "boundary-aware stitch: cells %s -> %s; candidate pairs=%s; merged pairs=%s",
        f"{stats['cells_before']:,}", f"{stats['cells_after']:,}",
        f"{stats['candidate_pairs']:,}", f"{stats['merged_pairs']:,}",
    )
    return out, stats


def run(cfg: dict):
    supported_seeds = {"official_transcripts_cell_id", "cellbin_transcript_cell_id"}
    if cfg.get("seed_source") not in supported_seeds:
        raise ValueError(
            f"Unsupported seed_source={cfg.get('seed_source')!r}; expected one of "
            f"{sorted(supported_seeds)}"
        )
    out = Path(cfg["output_dir"]); setup_log(out); seed_everything(int(cfg["seed"]))
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    tx, bounds = load_tx(cfg)
    seed = tx.cell_id.astype(str)
    bgmask = seed.isin(tuple(cfg.get("unassigned_values", ["UNASSIGNED", "0", "-1", "nan"])))
    base = tx[["transcript_id", "x", "y", "z", "gene", "cell_id"]].copy()
    base["raw_cells"] = base.cell_id.where(~bgmask, "UNASSIGNED")
    tr = train_tile(base, cfg, bounds)
    base, tr, feature_compression = compress_bering_features(base, tr, cfg)
    seg = tr[tr.raw_cells != "UNASSIGNED"][
        ["x", "y", "z", "gene", "bering_feature", "raw_cells"]
    ].copy()
    seg = add_labels(seg, cfg).drop(columns=["gene"]).rename(
        columns={"bering_feature": "features"}
    )
    unseg = tr[tr.raw_cells == "UNASSIGNED"][
        ["x", "y", "z", "bering_feature"]
    ].rename(columns={"bering_feature": "features"})
    if unseg.empty:
        LOG.warning(
            "The training source contains no unassigned/background molecules. "
            "Bering will train its edge model and foreground classes from CellBin "
            "seeds, but its background/noise class has no observed examples."
        )
    # Reuse the memory-safe implementation in bering_xenium.py, overriding
    # every parameter from the dataset config (never inherit its pancreas defaults).
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import bering_xenium as legacy
    P = cfg["bering"]
    # Cache inside this dataset-specific output directory only.  This cannot
    # reuse the pancreas/other-dataset model and makes a prediction-only rerun
    # practical after an interrupted or failed tiled prediction.
    legacy.CONFIG.OUTDIR = str(out)
    legacy.CONFIG.USE_CACHED_TRAIN = True
    legacy.CONFIG.REQUIRE_CACHED_TRAIN = (
        os.environ.get("BERING_REQUIRE_CACHED_TRAIN", "0") == "1"
    )
    legacy.CONFIG.STRICT_TILE_ERRORS = True
    legacy.CONFIG.TRAIN_TILE_UM = float(cfg["training"]["tile_um"])
    legacy.CONFIG.TRAIN_TILE_CENTER = cfg["training"].get("center_um")
    legacy.CONFIG.N_CELLS_PER_CLASS = P["n_cells_perClass"]
    legacy.CONFIG.N_WINDOWS_PER_CELL = int(P.get("n_windows_per_cell", 3))
    legacy.CONFIG.WINDOW_WIDTH = P["window_width"]
    legacy.CONFIG.WINDOW_HEIGHT = P["window_height"]
    legacy.CONFIG.N_NEIGHBORS = P["bwg_n_neighbors"]
    legacy.CONFIG.BATCH_SIZE = P["batch_size"]
    legacy.CONFIG.INTERNAL_TRAIN_RATIO = P["training_ratio"]
    legacy.CONFIG.TILE_SIZE_UM = P["tile_size_um"]
    legacy.CONFIG.TILE_MARGIN_UM = P["tile_margin_um"]
    legacy.CONFIG.MIN_TILE_TRANSCRIPTS = P["min_tile_transcripts"]
    legacy.CONFIG.NODE_CLASSIFICATION_DIRECT_MAX = int(
        P.get("node_classification_direct_max", 100_000)
    )
    legacy.CONFIG.NODE_CLASSIFICATION_TARGET_CHUNK = int(
        P.get("node_classification_target_chunk", 100_000)
    )
    legacy.CONFIG.SEG_POSITIVE_EDGE_THRESH = P["seg_positive_edge_thresh"]
    legacy.CONFIG.SEG_LEIDEN_RESOLUTION = P["seg_leiden_resolution"]
    legacy.CONFIG.SEG_NUM_EDGES_PER_SPOT = P["seg_num_edges_perSpot"]
    legacy.CONFIG.SEG_GRAPH_N_NEIGHBORS = P["seg_graph_n_neighbors"]
    legacy.CONFIG.MIN_TRANSCRIPTS = P["ann_min_transcripts"]
    legacy.CONFIG.MIN_DOMINANT_NODES = P["ann_min_dominant_nodes"]
    legacy.CONFIG.MIN_DOMINANT_RATIO = P["ann_min_dominant_ratio"]
    legacy.CONFIG.FORCE_CPU_TRAIN = bool(P.get("force_cpu_train", False))
    legacy.CONFIG.TRAIN_GPU_NEED_GIB = float(P.get("train_gpu_need_gib", 12.0))
    # Bering's legacy Training() writes performance plots to relative paths
    # such as ``figures/performance_*``.  The normal legacy entrypoint creates
    # these directories and changes into its work directory, but this
    # benchmark runner calls train_on_B() directly.  Reproduce that setup here
    # so a fresh benchmark run does not fail before the first training epoch.
    work_dir = out / "bering_work"
    for sub in ("figures", "checkpoints", "models"):
        (work_dir / sub).mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    LOG.info("Bering work directory=%s (relative figures/checkpoints/models ready)", work_dir)
    bg = legacy._try_load_cached_bg_train()
    if bg is None:
        bg = legacy.train_on_B(seg, unseg)
        # Save immediately after training so a later prediction failure does
        # not force another multi-hour training run.
        legacy._save_bg_train_cache(bg)
    # Prediction is deliberately restricted to the pre-registered ROI.
    # Keep transcript_id: predict_all_B uses it to restore molecule identity
    # after Bering internally reindexes spots.
    seg_all = base[~bgmask][
        ["transcript_id", "x", "y", "z", "gene", "bering_feature", "cell_id"]
    ].rename(columns={"gene": "original_gene", "bering_feature": "features",
                      "cell_id": "raw_cells"})
    seg_all["raw_labels"] = "background"
    unseg_all = base[bgmask][
        ["transcript_id", "x", "y", "z", "gene", "bering_feature"]
    ].rename(columns={"gene": "original_gene", "bering_feature": "features"})
    # Delegate tiling/stitching to the shared implementation after replacing its inputs.
    # This keeps Bering's tested node/edge code while recording the exact seed protocol.
    pred = legacy.predict_all_B(bg, seg_all, unseg_all)
    tile_stats = dict(pred.attrs.get("tile_stats", {}))
    feature_stats = dict(pred.attrs.get("feature_stats", {}))
    tile_cores = pred.attrs.get("tile_cores", [])
    feature_stats["compression"] = feature_compression
    pred, stitch_stats = stitch_tiles(
        pred,
        tile_cores=tile_cores,
        radius_um=float(P.get("stitch_radius_um", 8.0)),
        boundary_band_um=float(P.get("stitch_boundary_band_um", 12.0)),
        min_seed_purity=float(P.get("stitch_min_seed_purity", 0.8)),
    )
    # Auxiliary provenance is required for stitching but the public output
    # schema stays compatible with the previous benchmark files.
    pred = pred.drop(columns=["seed_cell_id", "tile_index"], errors="ignore")
    pred.to_parquet(out / "transcript_assignments.parquet", index=False)
    with open(out / "run_summary.json", "w") as f:
        json.dump({"dataset": cfg["dataset"], "roi_um": bounds,
                   "seed_source": cfg.get("seed_description", cfg["seed_source"]),
                   "label_source": "deterministic expression MiniBatchKMeans on training tile",
                   "coarse_cell_boundaries_used": bool(cfg.get("coarse_cell_boundaries_used", False)),
                   "official_cell_boundaries_not_used": not bool(cfg.get("coarse_cell_boundaries_used", False)),
                   "benchmark_mode": "reference_assisted",
                   "source_limitations": cfg.get("source_limitations", []),
                   "qc": cfg["qc"], "seed": cfg["seed"],
                   "n_input_transcripts": len(base),
                   "n_seed_cells": int(base.loc[~bgmask, "raw_cells"].nunique()),
                   "n_background_transcripts": int(bgmask.sum()),
                   "n_assigned": len(pred), "n_cells": int(pred.cell_id.nunique()),
                   "prediction_features": feature_stats,
                   "prediction_tiles": tile_stats,
                   "stitching": stitch_stats,
                   "prediction_complete": bool(tile_stats) and tile_stats.get("failed", 1) == 0,
                   "fairness_note": cfg.get(
                       "fairness_note",
                       "All datasets use identical QC, label generation, Bering parameters and random seed; ROI is fixed a priori."
                   )},
                  f, indent=2, ensure_ascii=False)


def check_config(cfg: dict):
    """Cheap preflight: verify paths/schema and print the fixed ROI."""
    p = resolve_input(cfg, "transcripts")
    import pyarrow.parquet as pq
    names = pq.ParquetFile(p).schema.names
    required = {"transcript_id", "cell_id", "feature_name", "x_location", "y_location"}
    missing = sorted(required - set(names))
    if missing: raise RuntimeError(f"{p} missing columns: {missing}")
    if cfg["roi"]["mode"] == "full_dataset":
        # Read only coordinate columns for the pancreas/full-dataset case.
        d = pd.read_parquet(p, columns=["x_location", "y_location"])
        b = roi_bounds(cfg, d.rename(columns={"x_location": "x", "y_location": "y"}))
    else:
        b = roi_bounds(cfg, pd.DataFrame({"x": [0.0], "y": [0.0]}))
    print(json.dumps({"dataset": cfg["dataset"], "transcripts": str(p),
                      "roi_um": b, "schema_ok": True,
                      "seed_source": cfg["seed_source"],
                      "seed_description": cfg.get("seed_description", cfg["seed_source"]),
                      "benchmark_mode": "reference_assisted"}, indent=2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--check", action="store_true", help="only verify input/schema/ROI")
    a = ap.parse_args(); cfg = json.loads(Path(a.config).read_text())
    check_config(cfg) if a.check else run(cfg)


if __name__ == "__main__": main()
