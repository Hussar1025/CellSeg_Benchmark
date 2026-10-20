"""
eval_cellist_cosmx.py - Cellist vs CosMx GT
===========================================
Reference: 2026-06-08-genesegnet-label-image-eval-input-layer-v1

Cellist coordinate system (critical):
  In cell_coord / segmentation.txt:
    x column = ROW direction (vertical = image row = GT y_global_px direction)
    y column = COL direction (horizontal = image column = GT x_global_px direction)

  Therefore Cellist x <-> GT y, Cellist y <-> GT x (swapped).

  Global coordinate conversion:
    GT x_um = (Cellist_y_local + fov_ox) * pixel_size  (horizontal)
    GT y_um = (Cellist_x_local + fov_oy) * pixel_size  (vertical)

  Transcript coordinate query:
    cellist_x_key = (y_global_px - fov_oy) // 4 * 4    (GT y -> Cellist x = ROW)
    cellist_y_key = (x_global_px - fov_ox) // 4 * 4    (GT x -> Cellist y = COL)

Run:
  conda activate genesegnet_env
  python eval_cellist_cosmx.py --output-dir yourpath
"""
from __future__ import annotations

import argparse, glob, json, os, warnings
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

DATA_DIR         = "yourpath"
GT_METADATA_PATH = os.path.join(DATA_DIR, "Pancreas_metadata_file.csv")
TX_FILE          = os.path.join(DATA_DIR, "Pancreas_tx_file.csv")

CELLIST_DIR   = "yourpath"
PIXEL_SIZE_UM = 0.1203
CELLIST_BIN   = 4      # Cellist output bin resolution (pixels)
FOV_IDS       = list(range(51, 69))


@dataclass
class EvaluationConfig:
    gt_path:              str   = GT_METADATA_PATH
    gt_transcripts_path:  str   = TX_FILE
    cellist_dir:          str   = CELLIST_DIR
    pixel_size_um:        float = PIXEL_SIZE_UM
    output_dir:           str   = "yourpath"
    method_name:          str   = "Cellist"
    cell_id_col:          str   = "cell_id"
    gt_cell_id_col:       str   = "cell_id"
    x_col:                str   = "x"
    y_col:                str   = "y"
    type_col:             str   = "cell_type"
    numeric_pairs: Optional[List[Tuple[str,str]]] = None
    numeric_cols:  Optional[List[str]] = None
    match_radius:         Optional[float] = None
    radius_quantile:      float = 0.95
    radius_multiplier:    float = 1.25
    max_component_size:   int   = 3000
    coordinate_scale_x:  float = 1.0
    coordinate_scale_y:  float = 1.0
    low_recall_warning_threshold: float = 0.70
    save_matched_table:           bool  = True
    save_cell_level_tables:       bool  = True
    verbose:                      bool  = True


# Utility: recursively find Cellist files
def find_cellist_file(cellist_dir: str, fov_id: int, suffix: str) -> Optional[str]:
    filename = f"F{fov_id:03d}_{suffix}"
    for p in [
        os.path.join(cellist_dir, filename),
        os.path.join(cellist_dir, f"F{fov_id:03d}", filename),
    ]:
        if os.path.exists(p): return p
    hits = glob.glob(os.path.join(cellist_dir, "**", filename), recursive=True)
    return hits[0] if hits else None


# GT loading
def load_gt(cfg: EvaluationConfig) -> pd.DataFrame:
    print(f"[GT] Loading: {cfg.gt_path}")
    df = pd.read_csv(cfg.gt_path, usecols=[
        "cell_id","CenterX_global_px","CenterY_global_px","nCount_RNA","Area","fov"
    ])
    df["x"]           = df["CenterX_global_px"] * cfg.pixel_size_um
    df["y"]           = df["CenterY_global_px"] * cfg.pixel_size_um
    df["total_counts"]= df["nCount_RNA"]
    df["cell_id"]     = df["cell_id"].astype(str)
    print(f"[GT] Cell count: {len(df):,}  "
          f"x=[{df['x'].min():.1f},{df['x'].max():.1f}]  "
          f"y=[{df['y'].min():.1f},{df['y'].max():.1f}]")
    return df


# FOV offset computation
def compute_fov_offsets(tx_file: str, verbose: bool = True) -> Dict[int, Tuple[int,int]]:
    print("[Offset] Computing FOV offsets...")
    meta     = pd.read_csv(GT_METADATA_PATH, usecols=["fov"])
    all_fovs = set(int(f) for f in meta["fov"].unique())
    found: Dict[int, Tuple[int,int]] = {}
    for chunk in pd.read_csv(tx_file,
                              usecols=["fov","x_local_px","y_local_px",
                                       "x_global_px","y_global_px"],
                              chunksize=5_000_000):
        for fov_id, grp in chunk.groupby("fov"):
            fid = int(fov_id)
            if fid not in found:
                s  = grp.head(50)
                ox = (s["x_global_px"] - s["x_local_px"]).median()
                oy = (s["y_global_px"] - s["y_local_px"]).median()
                found[fid] = (int(round(ox)), int(round(oy)))
        if set(found.keys()) >= all_fovs: break
    print(f"[Offset] Done: {len(found)} FOVs")
    return found


# Cellist cell table loading
def load_cellist_cells(cfg: EvaluationConfig,
                       fov_offsets: Dict[int, Tuple[int,int]]) -> pd.DataFrame:
    """
    Critical coordinate correction:
      Cellist x_local = ROW (vertical) = GT y direction
      Cellist y_local = COL (horizontal) = GT x direction

    Global um coordinates:
      x_um (horizontal) = (Cellist_y_local + ox) * pixel_size
      y_um (vertical) = (Cellist_x_local + oy) * pixel_size
    """
    print("[Pred] Loading Cellist cell centroids...")
    records = []
    for fov_id in FOV_IDS:
        fpath = find_cellist_file(cfg.cellist_dir, fov_id,
                                  "Cellist_segmentation_cell_coord.txt")
        if fpath is None:
            if cfg.verbose: print(f"  WARNING FOV {fov_id}: cell_coord file not found")
            continue

        df = pd.read_csv(fpath, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        if not all(c in df.columns for c in ["Cellist","x","y","nSpot"]):
            print(f"  WARNING FOV {fov_id}: column names mismatch {list(df.columns)}")
            continue

        if fov_id not in fov_offsets: continue
        ox, oy = fov_offsets[fov_id]

        for _, row in df.iterrows():
            cell_local = row["Cellist"]
            cx_row     = float(row["x"])   # Cellist x = ROW = vertical = GT y direction
            cy_col     = float(row["y"])   # Cellist y = COL = horizontal = GT x direction
            n_spot     = int(row["nSpot"])

            cell_id_global = f"F{fov_id:03d}_{int(cell_local)}"
            # horizontal (GT x) = Cellist y_col + ox
            # vertical (GT y) = Cellist x_row + oy
            x_um = (cy_col + ox) * cfg.pixel_size_um
            y_um = (cx_row + oy) * cfg.pixel_size_um

            records.append({
                "cell_id"      : cell_id_global,
                "fov"          : fov_id,
                "cell_local_id": int(cell_local),
                "x"            : x_um,
                "y"            : y_um,
                "n_transcripts": n_spot,
            })

        if cfg.verbose: print(f"  FOV {fov_id}: {len(df):,} cells")

    pred_df = pd.DataFrame(records)
    print(f"[Pred] Total cells: {len(pred_df):,}  "
          f"x=[{pred_df['x'].min():.1f},{pred_df['x'].max():.1f}]  "
          f"y=[{pred_df['y'].min():.1f},{pred_df['y'].max():.1f}]")
    return pred_df


# All-transcript metrics
def compute_all_transcript_metrics(
    cfg:         EvaluationConfig,
    fov_offsets: Dict[int, Tuple[int,int]],
    matches:     pd.DataFrame,
    verbose:     bool = True,
) -> dict:
    """
    Coordinate query correction (Cellist x=ROW=GT y, Cellist y=COL=GT x):
      cellist_x_key = floor((y_global_px - oy) / BIN) * BIN   # GT y -> Cellist x
      cellist_y_key = floor((x_global_px - ox) / BIN) * BIN   # GT x -> Cellist y
    """
    if matches.empty: return {}

    pred_to_gt: Dict[str,str] = {}
    gt_to_pred: Dict[str,str] = {}
    for _, row in matches[["pred_cell_id","gt_cell_id"]].iterrows():
        p, g = str(row["pred_cell_id"]), str(row["gt_cell_id"])
        pred_to_gt[p] = g;  gt_to_pred[g] = p
    matched_pred_set = set(pred_to_gt.keys())
    matched_gt_set   = set(gt_to_pred.keys())
    print(f"[TX] Matched pred={len(matched_pred_set):,}  GT={len(matched_gt_set):,}")

    meta = pd.read_csv(cfg.gt_path, usecols=["cell_id","cell_ID","fov"])
    meta["_key"] = meta["fov"].astype(str) + "_" + meta["cell_ID"].astype(str)
    fov_cell_to_str = dict(zip(meta["_key"], meta["cell_id"].astype(str)))

    # Step 1: build coordinate-to-cell map from segmentation.txt
    print("[TX] Step 1: Reading Cellist segmentation.txt (coordinate-corrected)...")
    # key = (fov, cellist_x=ROW, cellist_y=COL)
    cellist_coord: Dict[Tuple[int,int,int], str] = {}
    pred_gene_counts: Dict[str, Dict[str,int]] = {}
    total_pred_assigned = 0

    for fov_id in FOV_IDS:
        seg_path = find_cellist_file(cfg.cellist_dir, fov_id, "Cellist_segmentation.txt")
        if seg_path is None: continue

        fov_assigned = 0
        for chunk in pd.read_csv(seg_path, sep="\t", chunksize=500_000,
                                  usecols=["x","y","Cellist"]):
            chunk = chunk.dropna(subset=["Cellist"])
            chunk = chunk[chunk["Cellist"].astype(str).str.strip() != ""]
            if len(chunk) == 0: continue

            # x=ROW, y=COL (Cellist internal coordinates)
            x_int = chunk["x"].astype(int).values   # ROW
            y_int = chunk["y"].astype(int).values   # COL
            cells = chunk["Cellist"].values

            for x_row, y_col, cell in zip(x_int.tolist(), y_int.tolist(), cells.tolist()):
                cell_global = f"F{fov_id:03d}_{int(float(cell))}"
                key = (fov_id, x_row, y_col)
                if key not in cellist_coord:
                    cellist_coord[key] = cell_global
                fov_assigned += 1
                total_pred_assigned += 1

        if verbose: print(f"  FOV {fov_id}: {fov_assigned:,} assigned positions")

    print(f"[TX] Cellist coordinate map: {len(cellist_coord):,} entries")

    # Step 2: scan GT tx_file using x_local_px/y_local_px directly
    # Cellist x (ROW=vertical) <-> tx_file y_local_px; Cellist y (COL=horizontal) <-> tx_file x_local_px
    print("[TX] Step 2: Scanning GT tx_file (using x_local_px/y_local_px directly)...")
    gt_gene_counts: Dict[str, Dict[str,int]] = {}
    total_gt = 0; gt_assigned = 0; both = 0
    in_matched = 0; correct = 0; gt_read = 0

    for chunk in pd.read_csv(
        cfg.gt_transcripts_path,
        usecols=["fov","cell_ID","x_local_px","y_local_px","target"],
        chunksize=2_000_000,
    ):
        gt_read += len(chunk)
        total_gt += len(chunk)
        fov_ids_chunk = chunk["fov"].astype(int).values
        x_local_arr   = chunk["x_local_px"].values   # horizontal (COL) = Cellist y direction
        y_local_arr   = chunk["y_local_px"].values   # vertical (ROW) = Cellist x direction
        gt_id_v       = chunk["cell_ID"].values.astype(np.int64)
        genes         = chunk["target"].values

        gt_keys = chunk["fov"].astype(str) + "_" + chunk["cell_ID"].astype(str)
        gt_str  = gt_keys.map(fov_cell_to_str).fillna("").where(
            gt_id_v > 0, "").to_numpy(dtype=object)
        gt_assigned += int((gt_id_v > 0).sum())

        cellist_cells = []
        for i in range(len(chunk)):
            fov_id = int(fov_ids_chunk[i])
            x_local = int(x_local_arr[i])   # horizontal -> Cellist y (COL)
            y_local = int(y_local_arr[i])   # vertical -> Cellist x (ROW)
            cellist_x_key = int(y_local // CELLIST_BIN) * CELLIST_BIN
            cellist_y_key = int(x_local // CELLIST_BIN) * CELLIST_BIN
            key = (fov_id, cellist_x_key, cellist_y_key)
            cellist_cells.append(cellist_coord.get(key, ""))

        cellist_arr = np.array(cellist_cells, dtype=object)
        has_cellist = cellist_arr != ""
        both       += int(((gt_id_v > 0) & has_cellist).sum())

        df = pd.DataFrame({
            "pred"  : cellist_arr,
            "gt"    : gt_str,
            "gene"  : genes,
            "gt_id" : gt_id_v,
        })

        pred_m = df[has_cellist & pd.Series(cellist_arr).isin(matched_pred_set).values]
        if not pred_m.empty:
            in_matched += len(pred_m)
            correct    += int((pred_m["gt"] == pred_m["pred"].map(pred_to_gt)).sum())
            for (cell, gene), cnt in pred_m.groupby(["pred","gene"]).size().items():
                if cell not in pred_gene_counts: pred_gene_counts[cell] = {}
                pred_gene_counts[cell][gene] = pred_gene_counts[cell].get(gene,0) + int(cnt)

        gt_m = df[(df["gt_id"] > 0) & df["gt"].isin(matched_gt_set)]
        if not gt_m.empty:
            for (cell, gene), cnt in gt_m.groupby(["gt","gene"]).size().items():
                if cell not in gt_gene_counts: gt_gene_counts[cell] = {}
                gt_gene_counts[cell][gene] = gt_gene_counts[cell].get(gene,0) + int(cnt)

        if verbose and gt_read % 10_000_000 == 0:
            print(f"  GT read {gt_read:,}  in_matched={in_matched:,}  "
                  f"correct={correct:,}  both={both:,}")

    # Step 3: gene vector similarity
    print("[TX] Step 3: Computing gene vector similarity...")
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
        if np.std(pv) > EPS and np.std(gv) > EPS: pearsons.append(float(pearsonr(pv,gv)[0]))
        valid_pairs += 1

    accuracy  = float(correct / max(in_matched, 1))
    pred_rate = float(total_pred_assigned / max(total_gt, 1))
    print(f"[TX] Done: valid_pairs={valid_pairs:,}  accuracy={accuracy:.4f}  "
          f"pred_rate={pred_rate:.4f}  both={both:,}")

    return {
        "pred_transcript_rows"                          : total_pred_assigned,
        "pred_transcript_assignment_rate"               : pred_rate,
        "gt_transcript_rows"                            : float(total_gt),
        "gt_transcript_assignment_rate"                 : float(gt_assigned / max(total_gt,1)),
        "transcript_id_overlap_n"                       : both,
        "transcript_assignment_accuracy_via_matched_cells": accuracy,
        "transcript_matched_cell_pairs_n"               : int(matches.shape[0]),
        "matched_cell_gene_vector_mean_cosine"          : float(np.mean(cosine_sims)) if cosine_sims else np.nan,
        "matched_cell_gene_vector_mean_js_distance"     : float(np.mean(js_dists)) if js_dists else np.nan,
        "matched_cell_gene_vector_mean_pearson"         : float(np.mean(pearsons)) if pearsons else np.nan,
        "matched_cell_gene_vector_valid_pairs"          : valid_pairs,
    }


def expand_path(path):
    return Path(os.path.expanduser(path)).resolve()

def normalize_columns(df, cfg, table_name):
    alias_map = {
        cfg.cell_id_col: [cfg.cell_id_col,"cell","label"],
        cfg.x_col: [cfg.x_col,"centroid_x","cx","X"],
        cfg.y_col: [cfg.y_col,"centroid_y","cy","Y"],
        cfg.type_col: [cfg.type_col,"type","cluster","celltype"],
    }
    rename = {}; existing = set(df.columns)
    for canonical, aliases in alias_map.items():
        if canonical in existing: continue
        for a in aliases:
            if a in existing: rename[a] = canonical; break
    out = df.rename(columns=rename).copy()
    missing = [c for c in [cfg.cell_id_col, cfg.x_col, cfg.y_col] if c not in out.columns]
    if missing: raise ValueError(f"{table_name} missing columns {missing}")
    return out

def collapse_to_cell_level(df, cfg, table_name):
    df = normalize_columns(df, cfg, table_name)
    for col in [cfg.x_col, cfg.y_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[[cfg.x_col, cfg.y_col]].isna().any(axis=1)
    if bad.any(): df = df.loc[~bad].copy()
    if df.empty: raise ValueError(f"{table_name}: no valid rows")
    grouped = df.groupby(cfg.cell_id_col, sort=False, observed=True)
    base = grouped[[cfg.x_col, cfg.y_col]].mean().reset_index()
    qc = {f"{table_name}_n_cells": int(base.shape[0])}
    numeric_to_agg = []
    if cfg.numeric_pairs:
        if table_name.lower().startswith("pred"): numeric_to_agg.extend([p for p,_ in cfg.numeric_pairs])
        else: numeric_to_agg.extend([g for _,g in cfg.numeric_pairs])
    for col in numeric_to_agg:
        if col not in df.columns: continue
        tmp = pd.DataFrame({cfg.cell_id_col: df[cfg.cell_id_col].values,
                            col: pd.to_numeric(df[col], errors="coerce").values})
        agg = tmp.groupby(cfg.cell_id_col, sort=False)[col].mean().reset_index()
        base = base.merge(agg, on=cfg.cell_id_col, how="left")
    return base, qc

def infer_match_radius(gt_cells, cfg):
    xy = gt_cells[[cfg.x_col, cfg.y_col]].to_numpy(float)
    nbrs = NearestNeighbors(n_neighbors=2).fit(xy); dists, _ = nbrs.kneighbors(xy)
    return float(np.nanquantile(dists[:,1], cfg.radius_quantile) * cfg.radius_multiplier)

def build_candidate_edges(pred_cells, gt_cells, cfg, radius):
    pred_xy = pred_cells[[cfg.x_col, cfg.y_col]].to_numpy(float)
    gt_xy   = gt_cells[[cfg.x_col, cfg.y_col]].to_numpy(float)
    tree = cKDTree(gt_xy); nb = tree.query_ball_point(pred_xy, r=radius)
    pi, gi, di = [], [], []
    for i, js in enumerate(nb):
        if not js: continue
        js_arr = np.asarray(js, int); d = np.linalg.norm(gt_xy[js_arr]-pred_xy[i], axis=1)
        for j, dij in zip(js_arr.tolist(), d.tolist()):
            pi.append(i); gi.append(j); di.append(float(dij))
    return pd.DataFrame({"pred_index":pi,"gt_index":gi,"distance":di})

def _connected_components_bipartite(edges):
    if edges.empty: return []
    p2e, g2e = {}, {}
    pa = edges["pred_index"].to_numpy(int); ga = edges["gt_index"].to_numpy(int)
    for eidx,(p,g) in enumerate(zip(pa,ga)):
        p2e.setdefault(int(p),[]).append(eidx); g2e.setdefault(int(g),[]).append(eidx)
    vp, vg, ve = set(), set(), set(); components = []
    for start in p2e:
        if start in vp: continue
        stack=[start]; cp,cg,ce=set(),set(),set()
        while stack:
            p=stack.pop()
            if p in vp: continue
            vp.add(p); cp.add(p)
            for eidx in p2e.get(p,[]):
                if eidx in ve: continue
                ve.add(eidx); ce.add(eidx)
                g=int(ga[eidx]); cg.add(g)
                if g not in vg:
                    vg.add(g)
                    for e2 in g2e.get(g,[]):
                        p2=int(pa[e2])
                        if p2 not in vp: stack.append(p2)
        components.append((np.array(sorted(cp),int), np.array(sorted(cg),int),
                           edges.iloc[sorted(ce)].copy()))
    return components

def match_sparse_hungarian(pred_cells, gt_cells, cfg, radius=None):
    if radius is None: radius = infer_match_radius(gt_cells, cfg)
    edges = build_candidate_edges(pred_cells, gt_cells, cfg, radius)
    qc = {"match_radius": float(radius), "candidate_edges": int(edges.shape[0])}
    if edges.empty: return pd.DataFrame(columns=["pred_index","gt_index","distance"]), qc
    components = _connected_components_bipartite(edges)
    records = []
    for pi, gi, comp in components:
        if not len(pi) or not len(gi): continue
        if max(len(pi),len(gi)) > cfg.max_component_size:
            used_p, used_g = set(), set()
            for row in comp.sort_values("distance").itertuples(index=False):
                p,g=int(row.pred_index),int(row.gt_index)
                if p in used_p or g in used_g: continue
                used_p.add(p); used_g.add(g); records.append((p,g,float(row.distance)))
            continue
        p_pos={p:i for i,p in enumerate(pi.tolist())}; g_pos={g:i for i,g in enumerate(gi.tolist())}
        big=float(max(radius*1e4,1e9)); cost=np.full((len(pi),len(gi)),big)
        for row in comp.itertuples(index=False):
            cost[p_pos[int(row.pred_index)],g_pos[int(row.gt_index)]]=float(row.distance)
        ri,ci=linear_sum_assignment(cost)
        for r,c in zip(ri.tolist(),ci.tolist()):
            d=float(cost[r,c])
            if d<=radius+EPS: records.append((int(pi[r]),int(gi[c]),d))
    matches=pd.DataFrame(records,columns=["pred_index","gt_index","distance"])
    if matches.empty: return matches, qc
    # Attach metadata
    pm=pred_cells.reset_index(drop=True).reset_index().rename(columns={
        "index":"pred_index",cfg.cell_id_col:"pred_cell_id",
        cfg.x_col:"pred_x",cfg.y_col:"pred_y"})
    gm=gt_cells.reset_index(drop=True).reset_index().rename(columns={
        "index":"gt_index",cfg.cell_id_col:"gt_cell_id",
        cfg.x_col:"gt_x",cfg.y_col:"gt_y"})
    matches=matches.merge(pm[["pred_index","pred_cell_id","pred_x","pred_y"]],on="pred_index",how="left")
    matches=matches.merge(gm[["gt_index","gt_cell_id","gt_x","gt_y"]],on="gt_index",how="left")
    return matches, qc

def detection_metrics(matches, n_pred, n_gt):
    m=int(matches.shape[0]); p=m/max(n_pred,1); r=m/max(n_gt,1)
    return {"pred_cell_count":int(n_pred),"gt_cell_count":int(n_gt),"matched_cell_count":m,
            "detection_precision":float(p),"detection_recall":float(r),
            "detection_f1":float(2*p*r/max(p+r,EPS)),"cell_count_ratio":float(n_pred/max(n_gt,EPS))}

def spatial_shift_metrics(matches):
    nan_d={k:np.nan for k in ["mean_centroid_shift","median_centroid_shift",
                               "p95_centroid_shift","max_centroid_shift"]}
    if matches.empty or "distance" not in matches.columns: return nan_d
    d=pd.to_numeric(matches["distance"],errors="coerce").dropna().to_numpy(float)
    if d.size==0: return nan_d
    return {"mean_centroid_shift":float(np.mean(d)),"median_centroid_shift":float(np.median(d)),
            "p95_centroid_shift":float(np.quantile(d,.95)),"max_centroid_shift":float(np.max(d))}

def numeric_pair_metrics(matches, cfg):
    res = {}
    if not cfg.numeric_pairs: return res
    for pc_raw, gc_raw in cfg.numeric_pairs:
        pc=f"pred_{pc_raw}"; gc=f"gt_{gc_raw}"; label=f"{pc_raw}_vs_{gc_raw}"
        base={f"matched_pair_{label}_pearson":np.nan, f"matched_pair_{label}_spearman":np.nan,
              f"matched_pair_{label}_mae":np.nan,     f"matched_pair_{label}_rmse":np.nan}
        if matches.empty or pc not in matches.columns or gc not in matches.columns:
            res.update(base); continue
        x=pd.to_numeric(matches[pc],errors="coerce"); y=pd.to_numeric(matches[gc],errors="coerce")
        valid=x.notna()&y.notna(); xv,yv=x[valid].to_numpy(float),y[valid].to_numpy(float)
        pear=spear=np.nan
        if len(xv)>=3 and np.std(xv)>EPS and np.std(yv)>EPS:
            pear=float(pearsonr(xv,yv)[0]); spear=float(spearmanr(xv,yv)[0])
        mae=float(np.mean(np.abs(xv-yv))) if len(xv) else np.nan
        rmse=float(np.sqrt(np.mean((xv-yv)**2))) if len(xv) else np.nan
        res.update({f"matched_pair_{label}_pearson":pear, f"matched_pair_{label}_spearman":spear,
                    f"matched_pair_{label}_mae":mae,      f"matched_pair_{label}_rmse":rmse})
    return res

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

def build_selected_metric_row(method_name, metrics):
    row = {"method": method_name}
    for display, key in REQUESTED_METRICS: row[display] = metrics.get(key, np.nan)
    return row

def _json_default(obj):
    if isinstance(obj,np.integer): return int(obj)
    if isinstance(obj,np.floating): return None if np.isnan(obj) else float(obj)
    if isinstance(obj,Path): return str(obj)
    return str(obj)


# Main evaluation flow
def evaluate(cfg: EvaluationConfig) -> pd.DataFrame:
    if cfg.numeric_pairs is None: cfg.numeric_pairs=[("n_transcripts","total_counts")]
    outdir = expand_path(cfg.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    gt_raw = load_gt(cfg)
    print("\n[INFO] Computing FOV offsets...")
    fov_offsets = compute_fov_offsets(cfg.gt_transcripts_path, verbose=cfg.verbose)
    pred_raw = load_cellist_cells(cfg, fov_offsets)

    gt_cells,  gt_qc   = collapse_to_cell_level(gt_raw,   cfg, "gt")
    pred_cells,pred_qc = collapse_to_cell_level(pred_raw, cfg, "pred")

    radius = cfg.match_radius or infer_match_radius(gt_cells, cfg)
    print(f"[INFO] Match radius: {radius:.4f} um  GT={len(gt_cells):,}  Pred={len(pred_cells):,}")

    matches, match_qc = match_sparse_hungarian(pred_cells, gt_cells, cfg, radius=radius)
    print(f"[INFO] Matched cells: {len(matches):,}")

    # Merge n_transcripts
    if not matches.empty:
        nt = pred_cells[[cfg.cell_id_col,"n_transcripts"]].rename(
            columns={cfg.cell_id_col:"pred_cell_id","n_transcripts":"pred_n_transcripts"})
        if "pred_n_transcripts" not in matches.columns:
            matches = matches.merge(nt, on="pred_cell_id", how="left")
        gt_nt = gt_cells[[cfg.cell_id_col,"total_counts"]].rename(
            columns={cfg.cell_id_col:"gt_cell_id","total_counts":"gt_total_counts"})
        if "gt_total_counts" not in matches.columns:
            matches = matches.merge(gt_nt, on="gt_cell_id", how="left")

    print("\n[INFO] Computing transcript metrics (about 20-30 min)...")
    tx_result = compute_all_transcript_metrics(
        cfg=cfg, fov_offsets=fov_offsets,
        matches=matches, verbose=cfg.verbose,
    )

    metrics: Dict[str,Any] = {"method":cfg.method_name,"script_version":SCRIPT_VERSION,
                               "pixel_size_um":cfg.pixel_size_um}
    metrics.update(gt_qc); metrics.update(pred_qc); metrics.update(match_qc)
    metrics.update(detection_metrics(matches, n_pred=len(pred_cells), n_gt=len(gt_cells)))
    metrics.update(spatial_shift_metrics(matches))
    metrics.update(numeric_pair_metrics(matches, cfg))
    metrics["transcript_matched_cell_pairs_n"] = int(matches.shape[0])
    for s in ["pearson","spearman","mae","rmse"]:
        metrics[f"matched_cell_transcript_count_{s}"] = \
            metrics.get(f"matched_pair_n_transcripts_vs_total_counts_{s}", np.nan)
    for k, v in tx_result.items():
        metrics[k] = v

    prefix = f"{cfg.method_name.lower()}_cosmx_sparse_hungarian_qc_metrics"
    pd.DataFrame([metrics]).to_csv(outdir/f"{prefix}_full.csv", index=False)
    with open(outdir/f"{prefix}_full.json","w",encoding="utf-8") as f:
        json.dump(metrics,f,indent=2,ensure_ascii=False,default=_json_default)
    selected = build_selected_metric_row(cfg.method_name, metrics)
    sel_df = pd.DataFrame([selected])
    sel_csv = outdir/f"{prefix}_selected.csv"
    sel_df.to_csv(sel_csv, index=False)
    with open(outdir/f"{prefix}_selected.json","w",encoding="utf-8") as f:
        json.dump(selected,f,indent=2,ensure_ascii=False,default=_json_default)
    if cfg.save_matched_table: matches.to_csv(outdir/"matched_cells.csv", index=False)
    if cfg.save_cell_level_tables:
        pred_cells.to_csv(outdir/"pred_cell_level_table.csv", index=False)
        gt_cells.to_csv(outdir/"gt_cell_level_table.csv", index=False)

    if cfg.verbose:
        print(f"\n[DONE] -> {sel_csv}")
        print("\n=== SELECTED METRICS ===")
        print(sel_df.T.to_string(header=False))
    return sel_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-path",            default=GT_METADATA_PATH)
    p.add_argument("--gt-transcripts-path",default=TX_FILE)
    p.add_argument("--cellist-dir",        default=CELLIST_DIR)
    p.add_argument("--output-dir",         default="yourpath")
    p.add_argument("--method-name",        default="Cellist")
    p.add_argument("--match-radius",       type=float, default=None)
    p.add_argument("--no-save-matched-table",     action="store_true")
    p.add_argument("--no-save-cell-level-tables", action="store_true")
    p.add_argument("--quiet",                     action="store_true")
    args = p.parse_args()
    return EvaluationConfig(
        gt_path=args.gt_path, gt_transcripts_path=args.gt_transcripts_path,
        cellist_dir=args.cellist_dir, output_dir=args.output_dir,
        method_name=args.method_name, numeric_pairs=[("n_transcripts","total_counts")],
        match_radius=args.match_radius,
        save_matched_table=not args.no_save_matched_table,
        save_cell_level_tables=not args.no_save_cell_level_tables,
        verbose=not args.quiet,
    )

if __name__=="__main__":
    cfg = parse_args(); evaluate(cfg)