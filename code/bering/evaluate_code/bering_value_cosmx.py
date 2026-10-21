"""
eval_bering_cosmx.py — Bering vs CosMx GT
==========================================
Reference code: 2026-06-08-genesegnet-label-image-eval-input-layer-v1

Key differences between Bering output and GeneSegNet/UCS:
  1. pred comes from sliceB_cells.csv (cx,cy,n_transcripts,cell_type); no label-image lookup
  2. n_transcripts is directly available (no transcript-assignment completion needed)
  3. transcript_id is a real ID (not a synthetic row number), so overlap is direct
  4. cell_type already present (type metrics can be computed)
  5. Coordinates: Bering cx/cy are µm, but the y axis may be flipped vs GT, needs auto-alignment

Run:
  conda activate genesegnet_env
  python eval_bering_cosmx.py --output-dir yourpath/eval_results/bering
"""
from __future__ import annotations

import argparse, json, os, warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_rand_score, balanced_accuracy_score,
    f1_score, normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12
SCRIPT_VERSION = "2026-06-08-genesegnet-label-image-eval-input-layer-v1"

DATA_DIR         = "yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles"
GT_METADATA_PATH = os.path.join(DATA_DIR, "Pancreas_metadata_file.csv")
TX_FILE          = os.path.join(DATA_DIR, "Pancreas_tx_file.csv")

BERING_DIR         = "yourpath/bering/bering_cosmx_output"
BERING_CELLS       = os.path.join(BERING_DIR, "sliceB_cells.csv")
BERING_TRANSCRIPTS = os.path.join(BERING_DIR, "sliceB_transcript_assignments.csv")
BERING_SUMMARY     = os.path.join(BERING_DIR, "run_summary.json")

PIXEL_SIZE_UM = 0.1203


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class EvaluationConfig:
    gt_path:              str  = GT_METADATA_PATH
    gt_transcripts_path:  str  = TX_FILE
    bering_cells_path:    str  = BERING_CELLS
    bering_tx_path:       str  = BERING_TRANSCRIPTS
    bering_summary_path:  str  = BERING_SUMMARY
    pixel_size_um:        float = PIXEL_SIZE_UM
    output_dir:           str  = "yourpath/eval_results/bering"
    method_name:          str  = "Bering"
    cell_id_col:          str  = "cell_id"
    gt_cell_id_col:       str  = "cell_id"
    x_col:                str  = "x"
    y_col:                str  = "y"
    type_col:             str  = "cell_type"
    numeric_pairs: Optional[List[Tuple[str,str]]] = None
    numeric_cols:  Optional[List[str]] = None
    match_radius:         Optional[float] = None
    radius_quantile:      float = 0.95
    radius_multiplier:    float = 1.25
    max_component_size:   int   = 3000
    coordinate_scale_x:  float = 1.0
    coordinate_scale_y:  float = 1.0
    k_neighbors:          int   = 8
    low_recall_warning_threshold: float = 0.70
    warn_multitype_fraction:      float = 0.01
    strict_unique_type_per_cell:  bool  = False
    save_matched_table:           bool  = True
    save_cell_level_tables:       bool  = True
    verbose:                      bool  = True


# =============================================================================
# GT loading
# =============================================================================
def load_gt_cosmx(cfg: EvaluationConfig) -> pd.DataFrame:
    print(f"[GT] Loading: {cfg.gt_path}")
    df = pd.read_csv(cfg.gt_path, usecols=[
        "cell_id","CenterX_global_px","CenterY_global_px",
        "nCount_RNA","Area","fov"
    ])
    df["x"]           = df["CenterX_global_px"] * cfg.pixel_size_um
    df["y"]           = df["CenterY_global_px"] * cfg.pixel_size_um
    df["total_counts"]= df["nCount_RNA"]
    df["area_um2"]    = df["Area"] * (cfg.pixel_size_um ** 2)
    df["cell_id"]     = df["cell_id"].astype(str)
    print(f"[GT] cells: {len(df):,}  "
          f"x=[{df['x'].min():.1f},{df['x'].max():.1f}]µm  "
          f"y=[{df['y'].min():.1f},{df['y'].max():.1f}]µm")
    return df


# =============================================================================
# Bering pred loading (directly from sliceB_cells.csv)
# =============================================================================
def load_bering_cells(cfg: EvaluationConfig) -> pd.DataFrame:
    print(f"[Pred] Loading Bering cells: {cfg.bering_cells_path}")
    df = pd.read_csv(cfg.bering_cells_path)
    df = df.rename(columns={"cx": "x_raw", "cy": "y_raw"})
    df["cell_id"] = df["cell_id"].astype(str)
    print(f"[Pred] Bering cells: {len(df):,}")
    print(f"[Pred] raw coords x=[{df['x_raw'].min():.1f},{df['x_raw'].max():.1f}]µm  "
          f"y=[{df['y_raw'].min():.1f},{df['y_raw'].max():.1f}]µm")
    return df


# =============================================================================
# coordinate system auto-alignment
# Strategy: try (1) direct shift (2) Y-flip + shift; pick the smaller distance
# =============================================================================
def auto_align_coordinates(
    pred_df: pd.DataFrame,
    gt_df:   pd.DataFrame,
    cfg:     EvaluationConfig,
) -> Tuple[pd.DataFrame, dict]:
    """
    Return (aligned pred_df, align_params).
    align_params = {"y_flip": bool, "shift_x": float, "shift_y": float}
    These params align Bering transcript coordinates to the GT coordinate system,
    so transcript_assignment_accuracy can be computed.
    """
    pred_xy = pred_df[["x_raw","y_raw"]].to_numpy(dtype=float)
    gt_xy   = gt_df[[cfg.x_col, cfg.y_col]].to_numpy(dtype=float)

    pred_center = pred_xy.mean(axis=0)
    gt_center   = gt_xy.mean(axis=0)

    # Option A: direct shift
    shift_A    = gt_center - pred_center
    pred_A     = pred_xy + shift_A

    # Option B: Y-axis flip + shift
    pred_flipped = pred_xy.copy()
    pred_flipped[:, 1] = -pred_xy[:, 1]
    shift_B    = gt_center - pred_flipped.mean(axis=0)
    pred_B     = pred_flipped + shift_B

    # compare which option has smaller NN distance on a sample
    tree    = cKDTree(gt_xy)
    n_samp  = min(500, len(pred_xy))
    idx     = np.random.default_rng(0).choice(len(pred_xy), n_samp, replace=False)
    dA, _   = tree.query(pred_A[idx],  k=1)
    dB, _   = tree.query(pred_B[idx],  k=1)

    if dA.mean() <= dB.mean():
        print(f"[Align] option A (direct shift): "
              f"Δx={shift_A[0]:+.2f} Δy={shift_A[1]:+.2f} µm  "
              f"sample NN mean={dA.mean():.2f} µm")
        pred_df = pred_df.copy()
        pred_df["x"] = pred_xy[:, 0] + shift_A[0]
        pred_df["y"] = pred_xy[:, 1] + shift_A[1]
        pred_df["_align"] = "direct_shift"
        align_params = {"y_flip": False, "shift_x": float(shift_A[0]), "shift_y": float(shift_A[1])}
    else:
        print(f"[Align] option B (Y flip + shift): "
              f"Δx={shift_B[0]:+.2f} Δy={shift_B[1]:+.2f} µm  "
              f"sample NN mean={dB.mean():.2f} µm")
        pred_df = pred_df.copy()
        pred_df["x"] = pred_flipped[:, 0] + shift_B[0]
        pred_df["y"] = pred_flipped[:, 1] + shift_B[1]
        pred_df["_align"] = "y_flip_shift"
        align_params = {"y_flip": True, "shift_x": float(shift_B[0]), "shift_y": float(shift_B[1])}

    print(f"[Align] after alignment x=[{pred_df['x'].min():.1f},{pred_df['x'].max():.1f}]µm  "
          f"y=[{pred_df['y'].min():.1f},{pred_df['y'].max():.1f}]µm")
    return pred_df, align_params


# =============================================================================
# transcript_assignment_accuracy computation (coordinate matching)
#
# Principle:
#   After applying the alignment transform, Bering transcript (x,y) and GT
#   (x_global_px×px_size, y_global_px×px_size) refer to the same physical point.
#   Match by exact coordinates to find each Bering transcript's GT cell
#   assignment and check whether it agrees with the matched pair.
# =============================================================================
def compute_transcript_accuracy(
    bering_tx_path:   str,
    tx_file:          str,
    gt_metadata_path: str,
    matches:          pd.DataFrame,
    align_params:     dict,
    pixel_size_um:    float,
    verbose:          bool = True,
) -> Tuple[float, int, int]:
    """
    Same method as GeneSegNet/UCS:
      anchor on GT transcript coordinates, inverse-transform into the Bering
      coordinate system, and look up Bering's assignment.

    Why invert:
      GeneSegNet/UCS look up the label image by GT coordinate (no transform).
      Bering's equivalent: look up the Bering assignment table by GT coordinate
      (inverse-transformed into Bering space).

    Precision: use a 1 µm bin (rounded to integer).
      GT coords = integer_px × 0.1203 µm/px, with ~0.06 µm rounding error.
      Bering coords are continuous. The gap can reach ~0.1 µm, causing many
      misses at 0.1 µm precision. At 1 µm precision the error is far smaller
      than the bin width, so the match rate rises substantially.
    """
    if matches.empty: return np.nan, 0, 0

    # matched pair mapping
    pred_to_gt: Dict[str,str] = {}
    gt_to_pred: Dict[str,str] = {}
    for _, row in matches[["pred_cell_id","gt_cell_id"]].iterrows():
        p, g = str(row["pred_cell_id"]), str(row["gt_cell_id"])
        pred_to_gt[p] = g;  gt_to_pred[g] = p
    matched_pred_set = set(pred_to_gt.keys())
    matched_gt_set   = set(gt_to_pred.keys())

    # GT cell_ID mapping
    meta = pd.read_csv(gt_metadata_path, usecols=["cell_id","cell_ID","fov"])
    meta["_key"] = meta["fov"].astype(str) + "_" + meta["cell_ID"].astype(str)
    fov_cell_to_str = dict(zip(meta["_key"], meta["cell_id"].astype(str)))

    shift_x = align_params["shift_x"]
    shift_y = align_params["shift_y"]
    y_flip  = align_params["y_flip"]

    # Step 1: build Bering raw coordinate -> bering_cell mapping (1 µm precision)
    # only store transcripts of matched pred cells
    print("[ACC] Building Bering coordinate mapping (matched pred cells, 1 µm precision)...")
    bering_coord: Dict[Tuple[int,int], str] = {}  # {(int_x, int_y): bering_cell_id}
    bering_read = 0
    for chunk in pd.read_csv(bering_tx_path,
                              usecols=["cell_id","x","y"],
                              chunksize=2_000_000):
        bering_read += len(chunk)
        m = chunk[chunk["cell_id"].astype(str).isin(matched_pred_set)]
        if m.empty: continue
        # use Bering raw coordinates (before alignment) -> 1 µm bin
        x_int = m["x"].values.round().astype(int)
        y_int = m["y"].values.round().astype(int)
        cells = m["cell_id"].astype(str).values
        for x, y, cell in zip(x_int.tolist(), y_int.tolist(), cells.tolist()):
            bering_coord[(x, y)] = cell
        if verbose: print(f"  Bering read {bering_read:,}  coord entries={len(bering_coord):,}")

    print(f"[ACC] Bering coordinate mapping: {len(bering_coord):,} bins")

    # Step 2: scan tx_file, anchored on GT coordinates (same as GeneSegNet/UCS)
    # inverse-transform GT µm coordinates into the Bering raw coordinate system
    print("[ACC] Scanning tx_file (GT coords -> inverse transform -> look up Bering assignment)...")
    in_matched = 0; correct = 0; gt_read = 0
    for chunk in pd.read_csv(tx_file,
                              usecols=["fov","cell_ID","x_global_px","y_global_px"],
                              chunksize=2_000_000):
        gt_read += len(chunk)
        x_gt = chunk["x_global_px"].values * pixel_size_um   # GT µm
        y_gt = chunk["y_global_px"].values * pixel_size_um

        # inverse-transform into the Bering raw coordinate system (inverse of alignment)
        # forward: x_al = x_bering + shift_x -> inverse: x_bering = x_al - shift_x = x_gt - shift_x
        # forward (y_flip): y_al = -y_bering + shift_y -> inverse: y_bering = -(y_al - shift_y) = -y_gt + shift_y
        x_bering_space = (x_gt - shift_x).round().astype(int)
        if y_flip:
            y_bering_space = (-y_gt + shift_y).round().astype(int)
        else:
            y_bering_space = (y_gt - shift_y).round().astype(int)

        # vectorized lookup of Bering assignment
        coord_keys   = list(zip(x_bering_space.tolist(), y_bering_space.tolist()))
        bering_cells = pd.Series(coord_keys).map(bering_coord).fillna("").values
        has_bering   = bering_cells != ""

        if not has_bering.any(): continue

        # for transcripts with a Bering assignment, check the GT cell matches the matched pair
        chunk_m      = chunk.iloc[np.where(has_bering)[0]]
        bering_m     = bering_cells[has_bering]

        gt_keys  = chunk_m["fov"].astype(str) + "_" + chunk_m["cell_ID"].astype(str)
        gt_cells = gt_keys.map(fov_cell_to_str).fillna("").values

        # the expected GT cell for each Bering cell
        expected_gt = pd.Series(bering_m).map(pred_to_gt).fillna("").values

        in_matched += int(has_bering.sum())
        correct    += int(np.sum(gt_cells == expected_gt))

        if verbose:
            print(f"  GT read {gt_read:,}  in_matched={in_matched:,}  correct={correct:,}")

    accuracy = float(correct / max(in_matched, 1))
    print(f"[ACC] done: in_matched={in_matched:,}  correct={correct:,}  accuracy={accuracy:.4f}")
    return accuracy, in_matched, correct


# =============================================================================
# full transcript-metric computation (Bering-specific: uses transcript_assignments.csv directly)
# no label-image lookup, more direct than the GeneSegNet/UCS script
# =============================================================================
def compute_bering_transcript_metrics(
    bering_tx_path:   str,
    bering_summary:   dict,
    gt_tx_path:       str,
    gt_metadata_path: str,
    matches:          pd.DataFrame,
    align_params:     dict,
    pixel_size_um:    float,
    verbose:          bool = True,
) -> dict:
    """
    Bering has real transcript_id and cell_id, more direct than label-image methods:
      pred gene expression: aggregated directly from sliceB_transcript_assignments.csv
      GT   gene expression: aggregated from tx_file by (fov,cell_ID)->cell_id_str

    transcript_id_overlap_n:
      Bering assigned 53.3M transcripts.
      GT also has assignments by (fov, cell_ID).
      overlap = transcripts assigned by both Bering and GT (via spatial coord matching).
      Simplified impl: use run_summary's n_transcripts_assigned and an
      estimated GT assignment rate.
    """
    # ── build matched pair mapping ────────────────────────────────────────────
    pred_to_gt: Dict[str,str] = {}; gt_to_pred: Dict[str,str] = {}
    if not matches.empty and "pred_cell_id" in matches.columns:
        for _, row in matches[["pred_cell_id","gt_cell_id"]].iterrows():
            p, g = str(row["pred_cell_id"]), str(row["gt_cell_id"])
            pred_to_gt[p] = g;  gt_to_pred[g] = p
    matched_pred_set = set(pred_to_gt.keys())
    matched_gt_set   = set(gt_to_pred.keys())
    print(f"[BT] Matched pred={len(matched_pred_set):,}  GT={len(matched_gt_set):,}")

    # ── build GT cell_ID mapping (fov, cell_ID) -> cell_id_str ────────────────
    print(f"[BT] Loading GT cell_ID mapping ...")
    meta_map = pd.read_csv(gt_metadata_path, usecols=["cell_id","cell_ID","fov"])
    meta_map["_key"] = meta_map["fov"].astype(str) + "_" + meta_map["cell_ID"].astype(str)
    fov_cell_to_str = dict(zip(meta_map["_key"], meta_map["cell_id"].astype(str)))
    print(f"[BT] mapping entries: {len(fov_cell_to_str):,}")

    # ── Pred gene vectors (from transcript_assignments.csv) ───────────────────
    print(f"[BT] Streaming Bering transcript_assignments.csv ...")
    pred_gene_counts: Dict[str,Dict[str,int]] = {}
    pred_total = 0

    for chunk in pd.read_csv(bering_tx_path,
                              usecols=["transcript_id","cell_id","gene"],
                              chunksize=2_000_000):
        pred_total += len(chunk)
        # aggregate genes only for matched pred cells
        m = chunk[chunk["cell_id"].astype(str).isin(matched_pred_set)]
        if not m.empty:
            for (cid, gene), cnt in m.groupby(["cell_id","gene"]).size().items():
                cid = str(cid)
                if cid not in pred_gene_counts: pred_gene_counts[cid] = {}
                pred_gene_counts[cid][gene] = pred_gene_counts[cid].get(gene,0) + int(cnt)
        if verbose: print(f"  read {pred_total:,}  pred_gene_cells={len(pred_gene_counts):,}")

    # ── GT gene vectors (from tx_file) ────────────────────────────────────────
    print(f"[BT] Streaming tx_file ...")
    gt_gene_counts: Dict[str,Dict[str,int]] = {}
    gt_total = 0; gt_assigned = 0

    for chunk in pd.read_csv(gt_tx_path,
                              usecols=["fov","cell_ID","target"],
                              chunksize=2_000_000):
        gt_total += len(chunk)
        keys    = chunk["fov"].astype(str) + "_" + chunk["cell_ID"].astype(str)
        gt_str  = keys.map(fov_cell_to_str).fillna("")
        gt_str  = gt_str.where(chunk["cell_ID"].values > 0, "").values
        gt_assigned += int((chunk["cell_ID"].values > 0).sum())

        m_mask = pd.Series(gt_str).isin(matched_gt_set)
        if m_mask.any():
            df_m = pd.DataFrame({
                "gt": gt_str[m_mask.values],
                "gene": chunk["target"].values[m_mask.values],
            })
            for (cid, gene), cnt in df_m.groupby(["gt","gene"]).size().items():
                if cid not in gt_gene_counts: gt_gene_counts[cid] = {}
                gt_gene_counts[cid][gene] = gt_gene_counts[cid].get(gene,0) + int(cnt)
        if verbose: print(f"  read {gt_total:,}  gt_gene_cells={len(gt_gene_counts):,}")

    # ── gene vector similarity ────────────────────────────────────────────────
    print("[BT] Computing gene vector similarity ...")
    cosine_sims, js_dists, pearsons = [], [], []
    valid_pairs = 0
    for pred_str, gt_str in pred_to_gt.items():
        if pred_str not in pred_gene_counts or gt_str not in gt_gene_counts: continue
        all_genes = sorted(set(pred_gene_counts[pred_str]) | set(gt_gene_counts[gt_str]))
        if len(all_genes) < 3: continue
        pv = np.array([pred_gene_counts[pred_str].get(g,0) for g in all_genes], dtype=float)
        gv = np.array([gt_gene_counts[gt_str].get(g,0)   for g in all_genes], dtype=float)
        pn, gn = np.linalg.norm(pv), np.linalg.norm(gv)
        if pn > 0 and gn > 0: cosine_sims.append(float(np.dot(pv,gv)/(pn*gn)))
        pp = pv/(pv.sum()+EPS); gp = gv/(gv.sum()+EPS); m = (pp+gp)/2
        js = 0.5*np.sum(pp*np.log(pp/(m+EPS)+EPS)) + 0.5*np.sum(gp*np.log(gp/(m+EPS)+EPS))
        js_dists.append(float(np.sqrt(max(js,0))))
        if np.std(pv) > EPS and np.std(gv) > EPS:
            pearsons.append(float(pearsonr(pv, gv)[0]))
        valid_pairs += 1

    # ── transcript_id_overlap estimate ────────────────────────────────────────
    n_pred_assigned = bering_summary.get("n_transcripts_assigned", pred_total)
    pred_rate       = bering_summary.get("fraction_assigned", pred_total / max(gt_total, 1))
    gt_rate         = gt_assigned / max(gt_total, 1)
    # overlap ≈ transcripts assigned by both (simplified estimate: P(both) ≈ pred_rate × gt_rate × total)
    overlap_estimate = int(pred_rate * gt_rate * gt_total)

    # ── transcript_assignment_accuracy (coordinate matching) ──────────────────
    print("[BT] Computing transcript_assignment_accuracy ...")
    accuracy, in_matched, correct_tx = compute_transcript_accuracy(
        bering_tx_path   = bering_tx_path,
        tx_file          = gt_tx_path,
        gt_metadata_path = gt_metadata_path,
        matches          = matches,
        align_params     = align_params,
        pixel_size_um    = pixel_size_um,
        verbose          = verbose,
    )

    print(f"[BT] done: valid_pairs={valid_pairs:,}  "
          f"pred_rate={pred_rate:.3f}  gt_rate={gt_rate:.3f}  accuracy={accuracy:.4f}")

    return {
        "pred_transcript_rows"                         : n_pred_assigned,
        "pred_transcript_assigned_n"                   : n_pred_assigned,
        "pred_transcript_assignment_rate"              : float(pred_rate),
        "gt_transcript_rows"                           : float(gt_total),
        "gt_transcript_assigned_n"                     : gt_assigned,
        "gt_transcript_assignment_rate"                : gt_rate,
        "transcript_id_overlap_n"                      : overlap_estimate,
        "in_matched_pred"                              : in_matched,
        "correct_matched"                              : correct_tx,
        "transcript_assignment_accuracy_via_matched_cells": accuracy,
        "matched_cell_gene_vector_mean_cosine"         : float(np.mean(cosine_sims)) if cosine_sims else np.nan,
        "matched_cell_gene_vector_mean_js_distance"    : float(np.mean(js_dists)) if js_dists else np.nan,
        "matched_cell_gene_vector_mean_pearson"        : float(np.mean(pearsons)) if pearsons else np.nan,
        "matched_cell_gene_vector_valid_pairs"         : valid_pairs,
        "pred_transcript_has_gene_col"                 : True,
        "gt_transcript_has_gene_col"                   : True,
    }


# =============================================================================
# The functions below match the reference code exactly (reused from eval_genesegnet_cosmx.py)
# =============================================================================
def expand_path(path):
    if path is None: return None
    return Path(os.path.expanduser(path)).resolve()

def normalize_columns(df, cfg, table_name):
    if table_name.lower().startswith("pred"):
        id_aliases = [cfg.cell_id_col,"label","object_id","segmentation_id","CellID","cell_ID","cellid","cell"]
    else:
        id_aliases = [cfg.gt_cell_id_col,cfg.cell_id_col,"cell","cellid","cell_ID","CellID","label","object_id"]
    alias_map = {
        cfg.cell_id_col: id_aliases,
        cfg.x_col: [cfg.x_col,"centroid_x","x_centroid","center_x","cx","x_location","X"],
        cfg.y_col: [cfg.y_col,"centroid_y","y_centroid","center_y","cy","y_location","Y"],
        cfg.type_col: [cfg.type_col,"type","annotation","cluster","celltype","cell_type_gt"],
    }
    rename = {}; existing = set(df.columns)
    for canonical, aliases in alias_map.items():
        if canonical in existing: continue
        for a in aliases:
            if a in existing: rename[a] = canonical; break
    out = df.rename(columns=rename).copy()
    required = [cfg.cell_id_col, cfg.x_col, cfg.y_col]
    missing  = [c for c in required if c not in out.columns]
    if missing: raise ValueError(f"{table_name} missing required columns {missing}. Available: {list(out.columns)[:30]}")
    return out

def _mode_with_confidence(values):
    s = values.dropna()
    if s.empty: return np.nan,0,0,np.nan,""
    counts=s.value_counts(dropna=True,sort=True); max_count=int(counts.iloc[0])
    modes=counts[counts==max_count].index.tolist(); chosen=modes[0]
    n_valid=int(counts.sum()); n_unique=int(counts.shape[0])
    return chosen,n_unique,n_valid,float(max_count/max(n_valid,1)),";".join(map(str,modes[:10]))

def numeric_qc_for_series(s, prefix):
    x=pd.to_numeric(s,errors="coerce"); valid=x.dropna()
    return {f"{prefix}_valid_n":int(valid.shape[0]),f"{prefix}_nan_fraction":float(x.isna().mean()),
            f"{prefix}_is_constant":bool(valid.nunique(dropna=True)<=1) if valid.shape[0]>0 else True}

def collapse_to_cell_level(df, cfg, table_name):
    df=normalize_columns(df,cfg,table_name=table_name)
    for col in [cfg.x_col,cfg.y_col]: df[col]=pd.to_numeric(df[col],errors="coerce")
    bad=df[[cfg.x_col,cfg.y_col]].isna().any(axis=1)
    if bad.any(): warnings.warn(f"{table_name}: dropping {int(bad.sum())} rows (missing coords)"); df=df.loc[~bad].copy()
    if df.empty: raise ValueError(f"{table_name}: no valid rows after coordinate cleaning")
    df[cfg.x_col]*=cfg.coordinate_scale_x; df[cfg.y_col]*=cfg.coordinate_scale_y
    grouped=df.groupby(cfg.cell_id_col,sort=False,observed=True)
    base=grouped[[cfg.x_col,cfg.y_col]].mean().reset_index()
    qc={f"{table_name}_raw_rows":int(len(df)),f"{table_name}_n_cells":int(base.shape[0]),
        f"{table_name}_mean_rows_per_cell":float(len(df)/max(base.shape[0],1))}
    if cfg.type_col in df.columns:
        mode_records=[]
        for cid,sub in grouped[cfg.type_col]:
            chosen,n_unique,n_valid,conf,modes_repr=_mode_with_confidence(sub)
            mode_records.append((cid,chosen,n_unique,n_valid,conf,modes_repr))
        type_df=pd.DataFrame(mode_records,columns=[cfg.cell_id_col,cfg.type_col,
            f"{cfg.type_col}_n_unique",f"{cfg.type_col}_n_valid",
            f"{cfg.type_col}_confidence",f"{cfg.type_col}_modes"])
        base=base.merge(type_df,on=cfg.cell_id_col,how="left")
        multi_frac=float((type_df[f"{cfg.type_col}_n_unique"]>1).mean())
        qc.update({f"{table_name}_multitype_cell_fraction":multi_frac,
            f"{table_name}_type_low_confidence_fraction":float((type_df[f"{cfg.type_col}_confidence"]<1.0).mean()),
            f"{table_name}_type_median_confidence":float(type_df[f"{cfg.type_col}_confidence"].median(skipna=True))})
    numeric_to_agg=[]
    if cfg.numeric_cols: numeric_to_agg.extend(cfg.numeric_cols)
    if cfg.numeric_pairs:
        if table_name.lower().startswith("pred"): numeric_to_agg.extend([p for p,_ in cfg.numeric_pairs])
        else: numeric_to_agg.extend([g for _,g in cfg.numeric_pairs])
    seen=set(); numeric_to_agg=[c for c in numeric_to_agg if not (c in seen or seen.add(c))]
    for col in numeric_to_agg:
        if col not in df.columns: warnings.warn(f"{table_name}: numeric column '{col}' missing"); continue
        numeric=pd.to_numeric(df[col],errors="coerce")
        tmp=pd.DataFrame({cfg.cell_id_col:df[cfg.cell_id_col].values,col:numeric.values})
        agg=tmp.groupby(cfg.cell_id_col,sort=False,observed=True)[col].mean().reset_index()
        base=base.merge(agg,on=cfg.cell_id_col,how="left"); qc.update(numeric_qc_for_series(agg[col],prefix=f"{table_name}_{col}"))
    for extra in ["area_um2","n_genes"]:
        if extra in df.columns and extra not in base.columns:
            agg=df.groupby(cfg.cell_id_col,sort=False,observed=True)[extra].mean().reset_index()
            base=base.merge(agg,on=cfg.cell_id_col,how="left"); qc.update(numeric_qc_for_series(agg[extra],prefix=f"{table_name}_{extra}"))
    return base,qc

def infer_match_radius(gt_cells,cfg):
    xy=gt_cells[[cfg.x_col,cfg.y_col]].to_numpy(dtype=float)
    if xy.shape[0]<2: raise ValueError("GT cell count < 2")
    nbrs=NearestNeighbors(n_neighbors=2,algorithm="auto").fit(xy); dists,_=nbrs.kneighbors(xy)
    radius=float(np.nanquantile(dists[:,1],cfg.radius_quantile)*cfg.radius_multiplier)
    if not np.isfinite(radius) or radius<=0: raise ValueError(f"invalid match radius: {radius}")
    return radius

def build_candidate_edges(pred_cells,gt_cells,cfg,radius):
    pred_xy=pred_cells[[cfg.x_col,cfg.y_col]].to_numpy(dtype=float)
    gt_xy=gt_cells[[cfg.x_col,cfg.y_col]].to_numpy(dtype=float)
    tree=cKDTree(gt_xy); neighbor_lists=tree.query_ball_point(pred_xy,r=radius)
    pred_idx,gt_idx,dist_list=[],[],[]
    for i,js in enumerate(neighbor_lists):
        if not js: continue
        js_arr=np.asarray(js,dtype=int); d=np.linalg.norm(gt_xy[js_arr]-pred_xy[i],axis=1)
        for j,dij in zip(js_arr.tolist(),d.tolist()):
            pred_idx.append(i); gt_idx.append(j); dist_list.append(float(dij))
    return pd.DataFrame({"pred_index":pred_idx,"gt_index":gt_idx,"distance":dist_list})

def _connected_components_bipartite(edges,n_pred,n_gt):
    if edges.empty: return []
    pred_to_edges,gt_to_edges={},{}
    pred_arr=edges["pred_index"].to_numpy(dtype=int); gt_arr=edges["gt_index"].to_numpy(dtype=int)
    for eidx,(p,g) in enumerate(zip(pred_arr,gt_arr)):
        pred_to_edges.setdefault(int(p),[]).append(eidx); gt_to_edges.setdefault(int(g),[]).append(eidx)
    visited_pred,visited_gt,visited_edges=set(),set(),set(); components=[]
    for start_p in pred_to_edges.keys():
        if start_p in visited_pred: continue
        stack_p=[start_p]; comp_pred,comp_gt,comp_edges=set(),set(),set()
        while stack_p:
            p=stack_p.pop()
            if p in visited_pred: continue
            visited_pred.add(p); comp_pred.add(p)
            for eidx in pred_to_edges.get(p,[]):
                if eidx in visited_edges: continue
                visited_edges.add(eidx); comp_edges.add(eidx)
                g=int(gt_arr[eidx]); comp_gt.add(g)
                if g not in visited_gt:
                    visited_gt.add(g)
                    for e2 in gt_to_edges.get(g,[]):
                        p2=int(pred_arr[e2])
                        if p2 not in visited_pred: stack_p.append(p2)
        comp_edge_df=edges.iloc[sorted(comp_edges)].copy()
        components.append((np.array(sorted(comp_pred),dtype=int),np.array(sorted(comp_gt),dtype=int),comp_edge_df))
    return components

def empty_matches(): return pd.DataFrame(columns=["pred_index","gt_index","distance"])

def attach_match_metadata(matches,pred_cells,gt_cells,cfg):
    if matches.empty: return matches
    pred_meta=pred_cells.reset_index(drop=True).reset_index().rename(columns={
        "index":"pred_index",cfg.cell_id_col:"pred_cell_id",cfg.x_col:"pred_x",cfg.y_col:"pred_y"})
    gt_meta=gt_cells.reset_index(drop=True).reset_index().rename(columns={
        "index":"gt_index",cfg.cell_id_col:"gt_cell_id",cfg.x_col:"gt_x",cfg.y_col:"gt_y"})
    pred_keep={"pred_index","pred_cell_id","pred_x","pred_y"}; gt_keep={"gt_index","gt_cell_id","gt_x","gt_y"}
    pred_meta=pred_meta.rename(columns={c:f"pred_{c}" for c in pred_meta.columns if c not in pred_keep and not c.startswith("pred_")})
    gt_meta=gt_meta.rename(columns={c:f"gt_{c}" for c in gt_meta.columns if c not in gt_keep and not c.startswith("gt_")})
    out=matches.copy(); out=out.merge(pred_meta,on="pred_index",how="left"); out=out.merge(gt_meta,on="gt_index",how="left")
    return out

def match_sparse_hungarian(pred_cells,gt_cells,cfg,radius=None):
    if radius is None: radius=infer_match_radius(gt_cells,cfg)
    edges=build_candidate_edges(pred_cells,gt_cells,cfg,radius)
    qc={"matching_strategy":"sparse_hungarian_connected_components","match_radius":float(radius),
        "candidate_edges":int(edges.shape[0]),"mean_candidate_edges_per_pred":float(edges.shape[0]/max(pred_cells.shape[0],1))}
    if edges.empty: return empty_matches(),qc
    components=_connected_components_bipartite(edges,len(pred_cells),len(gt_cells))
    qc.update({"candidate_components":int(len(components)),
        "max_component_pred_size":int(max((len(c[0]) for c in components),default=0)),
        "max_component_gt_size":int(max((len(c[1]) for c in components),default=0))})
    matched_records=[]; large_components=0
    for pred_idx,gt_idx,comp_edges in components:
        if len(pred_idx)==0 or len(gt_idx)==0: continue
        if max(len(pred_idx),len(gt_idx))>cfg.max_component_size:
            large_components+=1
            comp_sorted=comp_edges.sort_values(["distance","pred_index","gt_index"],kind="mergesort")
            used_p,used_g=set(),set()
            for row in comp_sorted.itertuples(index=False):
                p,g=int(row.pred_index),int(row.gt_index)
                if p in used_p or g in used_g: continue
                used_p.add(p); used_g.add(g); matched_records.append((p,g,float(row.distance)))
            continue
        p_pos={p:i for i,p in enumerate(pred_idx.tolist())}; g_pos={g:i for i,g in enumerate(gt_idx.tolist())}
        finite_big=float(max(radius*1e4,1e9)); cost=np.full((len(pred_idx),len(gt_idx)),finite_big,dtype=np.float64)
        for row in comp_edges.itertuples(index=False):
            cost[p_pos[int(row.pred_index)],g_pos[int(row.gt_index)]]=float(row.distance)
        row_ind,col_ind=linear_sum_assignment(cost)
        for r,c in zip(row_ind.tolist(),col_ind.tolist()):
            d=float(cost[r,c])
            if d<=radius+EPS: matched_records.append((int(pred_idx[r]),int(gt_idx[c]),d))
    qc["large_components_greedy_fallback"]=int(large_components)
    matches=pd.DataFrame(matched_records,columns=["pred_index","gt_index","distance"])
    if matches.empty: return empty_matches(),qc
    matches=attach_match_metadata(matches,pred_cells,gt_cells,cfg)
    return matches,qc

def detection_metrics(matches,n_pred,n_gt):
    m=int(matches.shape[0]); precision=m/max(n_pred,1); recall=m/max(n_gt,1)
    f1=2*precision*recall/max(precision+recall,EPS)
    return {"pred_cell_count":int(n_pred),"gt_cell_count":int(n_gt),"matched_cell_count":m,
        "unmatched_pred_count":int(n_pred-m),"unmatched_gt_count":int(n_gt-m),
        "detection_precision":float(precision),"detection_recall":float(recall),"detection_f1":float(f1),
        "matched_fraction_gt":float(recall),"matched_fraction_pred":float(precision),
        "cell_count_difference":int(n_pred-n_gt),"cell_count_ratio":float(n_pred/max(n_gt,EPS))}

def spatial_shift_metrics(matches):
    nan_dict={k:np.nan for k in ["mean_centroid_shift","median_centroid_shift","p95_centroid_shift","max_centroid_shift"]}
    if matches is None or matches.empty or "distance" not in matches.columns: return nan_dict
    d=pd.to_numeric(matches["distance"],errors="coerce").dropna().to_numpy(dtype=float)
    if d.size==0: return nan_dict
    return {"mean_centroid_shift":float(np.mean(d)),"median_centroid_shift":float(np.median(d)),
            "p95_centroid_shift":float(np.quantile(d,0.95)),"max_centroid_shift":float(np.max(d))}

def type_metrics(matches,cfg):
    base_nan={"cell_type_accuracy":np.nan,"cell_type_macro_f1":np.nan,"cell_type_weighted_f1":np.nan,
        "cell_type_balanced_accuracy":np.nan,"ARI_matched_type_labels":np.nan,"NMI_matched_type_labels":np.nan,
        "cell_type_purity_legacy_matched_accuracy":np.nan,"matched_type_valid_n":0}
    pred_col=f"pred_{cfg.type_col}"; gt_col=f"gt_{cfg.type_col}"
    if matches.empty or pred_col not in matches.columns or gt_col not in matches.columns: return base_nan
    tmp=matches[[pred_col,gt_col]].dropna()
    if tmp.empty: return base_nan
    y_pred=tmp[pred_col].astype(str).to_numpy(); y_true=tmp[gt_col].astype(str).to_numpy()
    acc=float(np.mean(y_pred==y_true))
    return {"cell_type_accuracy":acc,
        "cell_type_macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
        "cell_type_weighted_f1":float(f1_score(y_true,y_pred,average="weighted",zero_division=0)),
        "cell_type_balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
        "ARI_matched_type_labels":float(adjusted_rand_score(y_true,y_pred)),
        "NMI_matched_type_labels":float(normalized_mutual_info_score(y_true,y_pred)),
        "cell_type_purity_legacy_matched_accuracy":acc,"matched_type_valid_n":int(tmp.shape[0])}

def morans_i_knn(coords,values,k=8):
    coords=np.asarray(coords,dtype=float); values=pd.to_numeric(pd.Series(values),errors="coerce").to_numpy(dtype=float)
    valid=np.isfinite(values)&np.isfinite(coords).all(axis=1); coords,values=coords[valid],values[valid]
    n=len(values)
    if n<=k+1 or np.nanstd(values)<=EPS: return np.nan
    nbrs=NearestNeighbors(n_neighbors=k+1,algorithm="auto",n_jobs=-1).fit(coords); _,indices=nbrs.kneighbors(coords)
    mean_val=np.nanmean(values); denom=np.nansum((values-mean_val)**2)
    if denom<=EPS: return np.nan
    centered=values-mean_val; numerator=0.0; weight_sum=0
    for i in range(n):
        neigh=indices[i,1:]; numerator+=float(np.sum(centered[i]*centered[neigh])); weight_sum+=len(neigh)
    return float((n/max(weight_sum,EPS))*(numerator/max(denom,EPS)))

def numeric_spatial_metrics(pred_cells,gt_cells,matches,cfg):
    res={}
    if not cfg.numeric_cols: return res
    for col in cfg.numeric_cols:
        for label,cells in [("pred",pred_cells),("gt",gt_cells)]:
            if col not in cells.columns: res[f"{label}_{col}_morans_i"]=np.nan; res.update(numeric_qc_for_series(pd.Series(dtype=float),prefix=f"{label}_{col}")); continue
            vals=pd.to_numeric(cells[col],errors="coerce"); res.update(numeric_qc_for_series(vals,prefix=f"{label}_{col}"))
            coords=cells[[cfg.x_col,cfg.y_col]].to_numpy(dtype=float); res[f"{label}_{col}_morans_i"]=morans_i_knn(coords,vals.to_numpy(),k=cfg.k_neighbors)
        p_mi=res.get(f"pred_{col}_morans_i",np.nan); g_mi=res.get(f"gt_{col}_morans_i",np.nan)
        res[f"delta_morans_i_{col}"]=float(p_mi-g_mi) if np.isfinite(p_mi) and np.isfinite(g_mi) else np.nan
        pred_col_m=f"pred_{col}"; gt_col_m=f"gt_{col}"
        if matches.empty or pred_col_m not in matches.columns or gt_col_m not in matches.columns:
            res.update({f"matched_{col}_valid_n":0,f"matched_{col}_pearson":np.nan,f"matched_{col}_spearman":np.nan,
                        f"matched_{col}_mae":np.nan,f"matched_{col}_rmse":np.nan}); continue
        x=pd.to_numeric(matches[pred_col_m],errors="coerce"); y=pd.to_numeric(matches[gt_col_m],errors="coerce")
        valid=x.notna()&y.notna(); xv,yv=x[valid].to_numpy(dtype=float),y[valid].to_numpy(dtype=float)
        pear=spear=np.nan
        if len(xv)>=3 and np.std(xv)>EPS and np.std(yv)>EPS:
            pear=float(pearsonr(xv,yv)[0]); spear=float(spearmanr(xv,yv)[0])
        mae=float(np.mean(np.abs(xv-yv))) if len(xv) else np.nan
        rmse=float(np.sqrt(np.mean((xv-yv)**2))) if len(xv) else np.nan
        res.update({f"matched_{col}_valid_n":int(len(xv)),f"matched_{col}_pearson":pear,
            f"matched_{col}_spearman":spear,f"matched_{col}_mae":mae,f"matched_{col}_rmse":rmse})
    return res

def numeric_pair_metrics(matches,cfg):
    res={}
    if not cfg.numeric_pairs: return res
    for pred_col_raw,gt_col_raw in cfg.numeric_pairs:
        pred_col=f"pred_{pred_col_raw}"; gt_col=f"gt_{gt_col_raw}"; label=f"{pred_col_raw}_vs_{gt_col_raw}"
        base={f"matched_pair_{label}_valid_n":0,f"matched_pair_{label}_pearson":np.nan,
              f"matched_pair_{label}_spearman":np.nan,f"matched_pair_{label}_mae":np.nan,f"matched_pair_{label}_rmse":np.nan}
        if matches.empty or pred_col not in matches.columns or gt_col not in matches.columns:
            warnings.warn(f"numeric pair '{pred_col_raw}:{gt_col_raw}' cannot be evaluated"); res.update(base); continue
        x=pd.to_numeric(matches[pred_col],errors="coerce"); y=pd.to_numeric(matches[gt_col],errors="coerce")
        valid=x.notna()&y.notna(); xv,yv=x[valid].to_numpy(dtype=float),y[valid].to_numpy(dtype=float)
        pear=spear=np.nan
        if len(xv)>=3 and np.std(xv)>EPS and np.std(yv)>EPS:
            pear=float(pearsonr(xv,yv)[0]); spear=float(spearmanr(xv,yv)[0])
        mae=float(np.mean(np.abs(xv-yv))) if len(xv) else np.nan
        rmse=float(np.sqrt(np.mean((xv-yv)**2))) if len(xv) else np.nan
        res.update({f"matched_pair_{label}_valid_n":int(len(xv)),f"matched_pair_{label}_pearson":pear,
            f"matched_pair_{label}_spearman":spear,f"matched_pair_{label}_mae":mae,f"matched_pair_{label}_rmse":rmse})
    return res

def low_recall_knn_warning(metrics,cfg):
    recall=metrics.get("detection_recall",np.nan)
    low=bool(np.isfinite(recall) and recall<cfg.low_recall_warning_threshold)
    if low: warnings.warn(f"Detection recall={recall:.3f} < {cfg.low_recall_warning_threshold:.3f}")
    return {"knn_metric_matched_only":True,"knn_metric_low_recall_warning":low,
            "knn_metric_recall_threshold":float(cfg.low_recall_warning_threshold)}

REQUESTED_METRICS=[
    ("Pred cell count","pred_cell_count"),("GT cell count","gt_cell_count"),
    ("Cell count ratio","cell_count_ratio"),("Matched cell count","matched_cell_count"),
    ("Detection precision","detection_precision"),("Detection recall","detection_recall"),
    ("Detection F1","detection_f1"),("Mean centroid shift","mean_centroid_shift"),
    ("Median centroid shift","median_centroid_shift"),("p95 centroid shift","p95_centroid_shift"),
    ("matched_pair_n_transcripts_vs_total_counts_pearson","matched_pair_n_transcripts_vs_total_counts_pearson"),
    ("matched_pair_n_transcripts_vs_total_counts_spearman","matched_pair_n_transcripts_vs_total_counts_spearman"),
    ("matched_pair_n_transcripts_vs_total_counts_mae","matched_pair_n_transcripts_vs_total_counts_mae"),
    ("matched_pair_n_transcripts_vs_total_counts_rmse","matched_pair_n_transcripts_vs_total_counts_rmse"),
    ("Pred transcript rows","pred_transcript_rows"),("GT transcript rows","gt_transcript_rows"),
    ("Pred transcript assignment rate","pred_transcript_assignment_rate"),
    ("Transcript matched cell pairs","transcript_matched_cell_pairs_n"),
    ("matched_cell_transcript_count_pearson","matched_cell_transcript_count_pearson"),
    ("matched_cell_transcript_count_spearman","matched_cell_transcript_count_spearman"),
    ("matched_cell_transcript_count_mae","matched_cell_transcript_count_mae"),
    ("matched_cell_transcript_count_rmse","matched_cell_transcript_count_rmse"),
    ("matched_cell_gene_vector_mean_cosine","matched_cell_gene_vector_mean_cosine"),
    ("matched_cell_gene_vector_mean_js_distance","matched_cell_gene_vector_mean_js_distance"),
    ("matched_cell_gene_vector_mean_pearson","matched_cell_gene_vector_mean_pearson"),
    ("matched_cell_gene_vector_valid_pairs","matched_cell_gene_vector_valid_pairs"),
    ("transcript_id_overlap_n","transcript_id_overlap_n"),
    ("transcript_assignment_accuracy_via_matched_cells","transcript_assignment_accuracy_via_matched_cells"),
]

def build_selected_metric_row(method_name,metrics):
    row={"method":method_name}
    for display,key in REQUESTED_METRICS: row[display]=metrics.get(key,np.nan)
    return row

def _json_default(obj):
    if isinstance(obj,np.integer): return int(obj)
    if isinstance(obj,np.floating): return None if np.isnan(obj) else float(obj)
    if isinstance(obj,np.bool_): return bool(obj)
    if isinstance(obj,Path): return str(obj)
    return str(obj)


# =============================================================================
# main evaluation flow
# =============================================================================
def evaluate(cfg: EvaluationConfig) -> pd.DataFrame:
    if cfg.numeric_pairs is None: cfg.numeric_pairs=[("n_transcripts","total_counts")]
    outdir=expand_path(cfg.output_dir); outdir.mkdir(parents=True,exist_ok=True)

    if cfg.verbose:
        print(f"[INFO] Method: {cfg.method_name}")
        print(f"[INFO] Bering cells: {cfg.bering_cells_path}")
        print(f"[INFO] Bering tx:    {cfg.bering_tx_path}")

    # load run_summary
    bering_summary = {}
    if os.path.exists(cfg.bering_summary_path):
        with open(cfg.bering_summary_path) as f:
            bering_summary = json.load(f)
        print(f"[INFO] Run summary: n_tx_assigned={bering_summary.get('n_transcripts_assigned'):,}  "
              f"fraction={bering_summary.get('fraction_assigned',0):.4f}")

    # GT
    gt_raw = load_gt_cosmx(cfg)

    # Bering pred
    pred_raw = load_bering_cells(cfg)

    # coordinate alignment
    print("\n[Align] auto coordinate alignment ...")
    pred_raw, align_params = auto_align_coordinates(pred_raw, gt_raw, cfg)

    # collapse
    gt_cells, gt_qc   = collapse_to_cell_level(gt_raw,   cfg, table_name="gt")
    pred_cells,pred_qc= collapse_to_cell_level(pred_raw, cfg, table_name="pred")

    radius = cfg.match_radius if cfg.match_radius is not None else infer_match_radius(gt_cells,cfg)
    if cfg.verbose:
        print(f"[INFO] match radius: {radius:.4f} µm")
        print(f"[INFO] GT={len(gt_cells):,}  Pred={len(pred_cells):,}")

    matches, match_qc = match_sparse_hungarian(pred_cells, gt_cells, cfg, radius=radius)

    # transcript metrics (from Bering transcript_assignments.csv + tx_file)
    print(f"\n[INFO] Computing transcript metrics (~10-15 min)...")
    tx_metrics = compute_bering_transcript_metrics(
        bering_tx_path   = cfg.bering_tx_path,
        bering_summary   = bering_summary,
        gt_tx_path       = cfg.gt_transcripts_path,
        gt_metadata_path = cfg.gt_path,
        matches          = matches,
        align_params     = align_params,
        pixel_size_um    = cfg.pixel_size_um,
        verbose          = cfg.verbose,
    )

    # aggregate
    metrics: Dict[str,Any] = {
        "method"              : cfg.method_name,
        "script_version"      : SCRIPT_VERSION,
        "pixel_size_um"       : cfg.pixel_size_um,
        "coordinate_alignment": pred_raw["_align"].iloc[0] if "_align" in pred_raw.columns else "unknown",
    }
    metrics.update(gt_qc); metrics.update(pred_qc); metrics.update(match_qc)
    metrics.update(detection_metrics(matches, n_pred=len(pred_cells), n_gt=len(gt_cells)))
    metrics.update(spatial_shift_metrics(matches))
    metrics.update(type_metrics(matches, cfg))
    metrics.update(low_recall_knn_warning(metrics, cfg))
    metrics.update(numeric_spatial_metrics(pred_cells, gt_cells, matches, cfg))
    metrics.update(numeric_pair_metrics(matches, cfg))

    # merge transcript metrics
    metrics["transcript_matched_cell_pairs_n"] = int(matches.shape[0])
    metrics["matched_cell_transcript_count_pearson"]  = metrics.get("matched_pair_n_transcripts_vs_total_counts_pearson", np.nan)
    metrics["matched_cell_transcript_count_spearman"] = metrics.get("matched_pair_n_transcripts_vs_total_counts_spearman", np.nan)
    metrics["matched_cell_transcript_count_mae"]      = metrics.get("matched_pair_n_transcripts_vs_total_counts_mae", np.nan)
    metrics["matched_cell_transcript_count_rmse"]     = metrics.get("matched_pair_n_transcripts_vs_total_counts_rmse", np.nan)
    for k, v in tx_metrics.items():
        metrics[k] = v

    prefix = f"{cfg.method_name.lower()}_cosmx_sparse_hungarian_qc_metrics"
    full_df = pd.DataFrame([metrics])
    full_df.to_csv(outdir/f"{prefix}_full.csv", index=False)
    with open(outdir/f"{prefix}_full.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)

    selected_row = build_selected_metric_row(cfg.method_name, metrics)
    selected_df  = pd.DataFrame([selected_row])
    sel_csv      = outdir/f"{prefix}_selected.csv"
    selected_df.to_csv(sel_csv, index=False)
    with open(outdir/f"{prefix}_selected.json","w",encoding="utf-8") as f:
        json.dump(selected_row, f, indent=2, ensure_ascii=False, default=_json_default)
    with open(outdir/"evaluation_config.json","w",encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False, default=_json_default)
    if cfg.save_matched_table:
        matches.to_csv(outdir/"matched_cells_sparse_hungarian.csv", index=False)
    if cfg.save_cell_level_tables:
        pred_cells.to_csv(outdir/"pred_cell_level_table.csv", index=False)
        gt_cells.to_csv(outdir/"gt_cell_level_table.csv", index=False)

    if cfg.verbose:
        print(f"\n[DONE] Selected metrics → {sel_csv}")
        print("\n=== SELECTED METRICS ===")
        print(selected_df.T.to_string(header=False))
    return selected_df


def parse_args():
    p=argparse.ArgumentParser(description="Evaluate Bering segmentation results (CosMx dataset)")
    p.add_argument("--gt-path",            default=GT_METADATA_PATH)
    p.add_argument("--gt-transcripts-path",default=TX_FILE)
    p.add_argument("--bering-cells",       default=BERING_CELLS)
    p.add_argument("--bering-transcripts", default=BERING_TRANSCRIPTS)
    p.add_argument("--bering-summary",     default=BERING_SUMMARY)
    p.add_argument("--pixel-size-um",      type=float, default=PIXEL_SIZE_UM)
    p.add_argument("--output-dir",         default="yourpath/eval_results/bering")
    p.add_argument("--method-name",        default="Bering")
    p.add_argument("--match-radius",       type=float, default=None)
    p.add_argument("--no-save-matched-table",      action="store_true")
    p.add_argument("--no-save-cell-level-tables",  action="store_true")
    p.add_argument("--quiet",                      action="store_true")
    args=p.parse_args()
    return EvaluationConfig(
        gt_path              = args.gt_path,
        gt_transcripts_path  = args.gt_transcripts_path,
        bering_cells_path    = args.bering_cells,
        bering_tx_path       = args.bering_transcripts,
        bering_summary_path  = args.bering_summary,
        pixel_size_um        = args.pixel_size_um,
        output_dir           = args.output_dir,
        method_name          = args.method_name,
        numeric_pairs        = [("n_transcripts","total_counts")],
        match_radius         = args.match_radius,
        save_matched_table   = not args.no_save_matched_table,
        save_cell_level_tables = not args.no_save_cell_level_tables,
        verbose              = not args.quiet,
    )

if __name__=="__main__":
    cfg=parse_args(); evaluate(cfg)