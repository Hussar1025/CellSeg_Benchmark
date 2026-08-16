"""
eval_ucs_xenium.py — UCS vs Xenium Human Breast Cancer GT
==========================================================
参考代码: 2026-06-08-genesegnet-label-image-eval-input-layer-v1

数据特点：
  坐标单位：µm（直接用 round(µm) 作为 mask 像素索引）
  ROI：5000×5000 px，以图像中心为中心
    x: [5205, 6269] µm，y: [2382, 3445] µm
  mask[row, col] → x_um = col + ROI_IX0,  y_um = row + ROI_IY0
  GT cell_id：字符串（'aaaafije-1'）
  UCS pred_id：整数（1~8809）
  pred 转录本数：从 transcripts.parquet 逐个查 mask 得到真实计数

运行：
  conda activate genesegnet_env
  python eval_ucs_xenium.py --output-dir /data/qiuyijia/eval_results/ucs_xenium
"""
from __future__ import annotations

import argparse, json, os, warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12
SCRIPT_VERSION = "2026-06-08-genesegnet-label-image-eval-input-layer-v1"

DATA_DIR   = "/data/qiuyijia/dataset/xenium_breast"
MASK_PATH  = "/data/qiuyijia/ucs_xenium_breast/ucs_log/pred/segmentation_mask.tif"
TX_PARQUET = os.path.join(DATA_DIR, "transcripts.parquet")
CELLS_PQ   = os.path.join(DATA_DIR, "cells.parquet")

# ROI 坐标参数
PIXEL_SIZE = 0.2125
CX_UM  = 53994 / 2 * PIXEL_SIZE
CY_UM  = 27420 / 2 * PIXEL_SIZE
HALF   = 5000 * PIXEL_SIZE / 2

ROI_X0 = CX_UM - HALF;  ROI_X1 = CX_UM + HALF
ROI_Y0 = CY_UM - HALF;  ROI_Y1 = CY_UM + HALF
ROI_IX0 = int(ROI_X0)   # 5205
ROI_IY0 = int(ROI_Y0)   # 2382


@dataclass
class EvaluationConfig:
    mask_path:            str   = MASK_PATH
    gt_cells_path:        str   = CELLS_PQ
    gt_transcripts_path:  str   = TX_PARQUET
    pixel_size_um:        float = PIXEL_SIZE
    output_dir:           str   = "/data/qiuyijia/eval_results/ucs_xenium"
    method_name:          str   = "UCS"
    cell_id_col:          str   = "cell_id"
    x_col:                str   = "x"
    y_col:                str   = "y"
    type_col:             str   = "cell_type"
    numeric_pairs: Optional[List[Tuple[str,str]]] = None
    match_radius:         Optional[float] = None
    radius_quantile:      float = 0.95
    radius_multiplier:    float = 1.25
    max_component_size:   int   = 3000
    low_recall_warning_threshold: float = 0.70
    save_matched_table:           bool  = True
    save_cell_level_tables:       bool  = True
    verbose:                      bool  = True


def load_gt(cfg: EvaluationConfig) -> pd.DataFrame:
    print(f"[GT] 加载: {cfg.gt_cells_path}")
    cells = pd.read_parquet(cfg.gt_cells_path,
                            columns=["cell_id","x_centroid","y_centroid",
                                     "total_counts","cell_area"])
    mask = ((cells.x_centroid >= ROI_X0) & (cells.x_centroid <= ROI_X1) &
            (cells.y_centroid >= ROI_Y0) & (cells.y_centroid <= ROI_Y1))
    cells = cells[mask].copy()
    cells.rename(columns={"x_centroid":"x","y_centroid":"y"}, inplace=True)
    cells["cell_id"] = cells["cell_id"].astype(str)
    print(f"[GT] ROI 内细胞数: {len(cells):,}")
    return cells


def load_pred(cfg: EvaluationConfig):
    print(f"[Pred] 加载 UCS mask: {cfg.mask_path}")
    mask = tifffile.imread(cfg.mask_path)
    ids  = np.unique(mask); ids = ids[ids > 0]
    print(f"[Pred] 细胞数: {len(ids):,}")

    print("[Pred] 计算质心...")
    centroids = ndimage.center_of_mass(mask > 0, mask, ids.tolist())
    records   = []
    for cell_id, (row, col) in zip(ids.tolist(), centroids):
        records.append({
            "cell_id": str(cell_id),
            "x"      : col + ROI_IX0,
            "y"      : row + ROI_IY0,
        })
    pred_df = pd.DataFrame(records)
    print(f"[Pred] 质心完成: {len(pred_df):,} 个细胞")
    return pred_df, mask


def compute_all_transcript_metrics(
    cfg:     EvaluationConfig,
    mask:    np.ndarray,
    matches: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    if matches.empty: return {}

    pred_to_gt: Dict[str,str] = {}
    gt_to_pred: Dict[str,str] = {}
    for _, row in matches[["pred_cell_id","gt_cell_id"]].iterrows():
        p, g = str(row["pred_cell_id"]), str(row["gt_cell_id"])
        pred_to_gt[p] = g;  gt_to_pred[g] = p
    matched_pred_set = set(pred_to_gt.keys())
    matched_gt_set   = set(gt_to_pred.keys())
    MAP_H, MAP_W = mask.shape
    print(f"[TX] Matched pred={len(matched_pred_set):,}  GT={len(matched_gt_set):,}")

    print(f"[TX] 加载 ROI 转录本...")
    pq = pd.read_parquet(cfg.gt_transcripts_path,
                         columns=["cell_id","x_location","y_location",
                                  "feature_name","is_gene","qv"])
    roi_m = ((pq["is_gene"] == True) &
             (pq["x_location"] >= ROI_X0) & (pq["x_location"] <= ROI_X1) &
             (pq["y_location"] >= ROI_Y0) & (pq["y_location"] <= ROI_Y1))
    tx = pq[roi_m].copy(); del pq
    total_gt = len(tx)
    print(f"[TX] ROI 内转录本: {total_gt:,}")

    # 坐标转换 → mask 查询
    tx["col"] = (tx["x_location"].round().astype(int) - ROI_IX0).clip(0, MAP_W-1)
    tx["row"] = (tx["y_location"].round().astype(int) - ROI_IY0).clip(0, MAP_H-1)
    pred_ids  = mask[tx["row"].values, tx["col"].values]
    tx["pred_id"] = np.where(pred_ids > 0, pred_ids.astype(str), "")

    gt_assigned_mask = tx["cell_id"] != "UNASSIGNED"
    gt_assigned      = int(gt_assigned_mask.sum())
    total_pred_assigned = int((tx["pred_id"] != "").sum())
    both = int((gt_assigned_mask & (tx["pred_id"] != "")).sum())

    in_matched_mask = (tx["pred_id"] != "") & tx["pred_id"].isin(matched_pred_set)
    in_matched      = int(in_matched_mask.sum())
    correct         = 0
    if in_matched > 0:
        expected_gt = tx.loc[in_matched_mask, "pred_id"].map(pred_to_gt)
        correct     = int((tx.loc[in_matched_mask, "cell_id"] == expected_gt).sum())

    print(f"[TX] gt_assigned={gt_assigned:,}  pred_assigned={total_pred_assigned:,}  "
          f"both={both:,}  in_matched={in_matched:,}  correct={correct:,}")

    # ── per-cell 真实转录本数（用于 Pearson/Spearman）────────────────────────
    pred_tx_count: Dict[str,int] = {}
    for pid in tx["pred_id"].values:
        if pid == "": continue
        pred_tx_count[pid] = pred_tx_count.get(pid, 0) + 1

    # ── 基因向量 ──────────────────────────────────────────────────────────────
    print("[TX] 计算基因向量...")
    pred_gene_counts: Dict[str, Dict[str,int]] = {}
    gt_gene_counts:   Dict[str, Dict[str,int]] = {}

    tx_mp = tx[in_matched_mask]
    for pid, gene in zip(tx_mp["pred_id"].values, tx_mp["feature_name"].values):
        if pid not in pred_gene_counts: pred_gene_counts[pid] = {}
        pred_gene_counts[pid][gene] = pred_gene_counts[pid].get(gene,0) + 1

    tx_gt = tx[gt_assigned_mask & tx["cell_id"].isin(matched_gt_set)]
    for gid, gene in zip(tx_gt["cell_id"].values, tx_gt["feature_name"].values):
        if gid not in gt_gene_counts: gt_gene_counts[gid] = {}
        gt_gene_counts[gid][gene] = gt_gene_counts[gid].get(gene,0) + 1

    cosine_sims, js_dists, pearsons = [], [], []
    valid_pairs = 0
    for pred_str, gt_str in pred_to_gt.items():
        if pred_str not in pred_gene_counts or gt_str not in gt_gene_counts: continue
        all_genes = sorted(set(pred_gene_counts[pred_str]) | set(gt_gene_counts[gt_str]))
        if len(all_genes) < 3: continue
        pv = np.array([pred_gene_counts[pred_str].get(g,0) for g in all_genes], float)
        gv = np.array([gt_gene_counts[gt_str].get(g,0)   for g in all_genes], float)
        pn, gn = np.linalg.norm(pv), np.linalg.norm(gv)
        if pn>0 and gn>0: cosine_sims.append(float(np.dot(pv,gv)/(pn*gn)))
        pp=pv/(pv.sum()+EPS); gp=gv/(gv.sum()+EPS); m=(pp+gp)/2
        js=0.5*np.sum(pp*np.log(pp/(m+EPS)+EPS))+0.5*np.sum(gp*np.log(gp/(m+EPS)+EPS))
        js_dists.append(float(np.sqrt(max(js,0))))
        if np.std(pv)>EPS and np.std(gv)>EPS: pearsons.append(float(pearsonr(pv,gv)[0]))
        valid_pairs += 1

    accuracy  = float(correct / max(in_matched,1))
    pred_rate = float(total_pred_assigned / max(total_gt,1))
    print(f"[TX] 完成: valid_pairs={valid_pairs:,}  accuracy={accuracy:.4f}  pred_rate={pred_rate:.4f}")

    return {
        "pred_transcript_rows"                            : total_pred_assigned,
        "pred_transcript_assignment_rate"                 : pred_rate,
        "gt_transcript_rows"                              : float(total_gt),
        "gt_transcript_assignment_rate"                   : float(gt_assigned/max(total_gt,1)),
        "transcript_id_overlap_n"                         : both,
        "transcript_assignment_accuracy_via_matched_cells": accuracy,
        "transcript_matched_cell_pairs_n"                 : int(matches.shape[0]),
        "matched_cell_gene_vector_mean_cosine"            : float(np.mean(cosine_sims)) if cosine_sims else np.nan,
        "matched_cell_gene_vector_mean_js_distance"       : float(np.mean(js_dists)) if js_dists else np.nan,
        "matched_cell_gene_vector_mean_pearson"           : float(np.mean(pearsons)) if pearsons else np.nan,
        "matched_cell_gene_vector_valid_pairs"            : valid_pairs,
        "_pred_tx_count"                                  : pred_tx_count,
    }


# ── 参考代码函数（不变）──────────────────────────────────────────────────────
def expand_path(path): return Path(os.path.expanduser(path)).resolve()

def collapse_to_cell_level(df, cfg, table_name):
    for col in [cfg.x_col, cfg.y_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[[cfg.x_col, cfg.y_col]].isna().any(axis=1)
    if bad.any(): df = df.loc[~bad].copy()
    grouped = df.groupby(cfg.cell_id_col, sort=False, observed=True)
    base    = grouped[[cfg.x_col, cfg.y_col]].mean().reset_index()
    qc      = {f"{table_name}_n_cells": int(base.shape[0])}
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
    xy   = gt_cells[[cfg.x_col, cfg.y_col]].to_numpy(float)
    nbrs = NearestNeighbors(n_neighbors=2).fit(xy)
    dists, _ = nbrs.kneighbors(xy)
    return float(np.nanquantile(dists[:,1], cfg.radius_quantile)*cfg.radius_multiplier)

def build_candidate_edges(pred_cells, gt_cells, cfg, radius):
    pred_xy = pred_cells[[cfg.x_col, cfg.y_col]].to_numpy(float)
    gt_xy   = gt_cells[[cfg.x_col,  cfg.y_col]].to_numpy(float)
    tree    = cKDTree(gt_xy)
    nb      = tree.query_ball_point(pred_xy, r=radius)
    pi, gi, di = [], [], []
    for i, js in enumerate(nb):
        if not js: continue
        ja = np.asarray(js,int); d = np.linalg.norm(gt_xy[ja]-pred_xy[i], axis=1)
        for j, dij in zip(ja.tolist(), d.tolist()):
            pi.append(i); gi.append(j); di.append(float(dij))
    return pd.DataFrame({"pred_index":pi,"gt_index":gi,"distance":di})

def _cc_bipartite(edges):
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
    qc    = {"match_radius":float(radius),"candidate_edges":int(edges.shape[0])}
    if edges.empty: return pd.DataFrame(columns=["pred_index","gt_index","distance"]),qc
    components = _cc_bipartite(edges)
    records    = []
    for pi,gi,comp in components:
        if not len(pi) or not len(gi): continue
        if max(len(pi),len(gi)) > cfg.max_component_size:
            used_p,used_g=set(),set()
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
    if matches.empty: return matches,qc
    pm=pred_cells.reset_index(drop=True).reset_index().rename(columns={
        "index":"pred_index",cfg.cell_id_col:"pred_cell_id",cfg.x_col:"pred_x",cfg.y_col:"pred_y"})
    gm=gt_cells.reset_index(drop=True).reset_index().rename(columns={
        "index":"gt_index",cfg.cell_id_col:"gt_cell_id",cfg.x_col:"gt_x",cfg.y_col:"gt_y"})
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
    for pc_raw,gc_raw in cfg.numeric_pairs:
        pc=f"pred_{pc_raw}"; gc=f"gt_{gc_raw}"; label=f"{pc_raw}_vs_{gc_raw}"
        base={f"matched_pair_{label}_pearson":np.nan,f"matched_pair_{label}_spearman":np.nan,
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
        res.update({f"matched_pair_{label}_pearson":pear,f"matched_pair_{label}_spearman":spear,
                    f"matched_pair_{label}_mae":mae,f"matched_pair_{label}_rmse":rmse})
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


def evaluate(cfg: EvaluationConfig) -> pd.DataFrame:
    if cfg.numeric_pairs is None:
        cfg.numeric_pairs = [("n_transcripts","total_counts")]
    outdir = expand_path(cfg.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    gt_raw = load_gt(cfg)
    pred_raw, mask = load_pred(cfg)

    # n_transcripts 在 tx_result 后填入，先用空列占位
    pred_raw["n_transcripts"] = 0

    gt_cells,  gt_qc   = collapse_to_cell_level(gt_raw,   cfg, "gt")
    pred_cells,pred_qc = collapse_to_cell_level(pred_raw, cfg, "pred")

    radius = cfg.match_radius or infer_match_radius(gt_cells, cfg)
    print(f"[INFO] 匹配半径: {radius:.4f} µm  GT={len(gt_cells):,}  Pred={len(pred_cells):,}")

    matches, match_qc = match_sparse_hungarian(pred_cells, gt_cells, cfg, radius=radius)
    print(f"[INFO] 匹配细胞数: {len(matches):,}")

    # GT total_counts 加入 matches
    if not matches.empty:
        tc_df = gt_cells[[cfg.cell_id_col,"total_counts"]].rename(
            columns={cfg.cell_id_col:"gt_cell_id","total_counts":"gt_total_counts"})
        matches = matches.merge(tc_df, on="gt_cell_id", how="left")

    print(f"\n[INFO] 计算转录本指标（约 3-5 分钟）...")
    tx_result = compute_all_transcript_metrics(
        cfg=cfg, mask=mask, matches=matches, verbose=cfg.verbose,
    )

    # 将真实 per-cell 转录本数加入 pred_cells 和 matches
    pred_tx_count = tx_result.pop("_pred_tx_count", {})
    pred_cells["n_transcripts"] = pred_cells[cfg.cell_id_col].map(pred_tx_count).fillna(0).astype(int)

    if not matches.empty:
        nt_df = pred_cells[[cfg.cell_id_col,"n_transcripts"]].rename(
            columns={cfg.cell_id_col:"pred_cell_id","n_transcripts":"pred_n_transcripts"})
        matches = matches.merge(nt_df, on="pred_cell_id", how="left")

    # 汇总指标
    metrics: Dict[str,Any] = {"method":cfg.method_name,"script_version":SCRIPT_VERSION,
                               "pixel_size_um":cfg.pixel_size_um}
    metrics.update(gt_qc); metrics.update(pred_qc); metrics.update(match_qc)
    metrics.update(detection_metrics(matches, n_pred=len(pred_cells), n_gt=len(gt_cells)))
    metrics.update(spatial_shift_metrics(matches))
    metrics.update(numeric_pair_metrics(matches, cfg))
    metrics["transcript_matched_cell_pairs_n"] = int(matches.shape[0])

    # 统一 key 名（n_transcripts_vs_total_counts → 标准名）
    for s in ["pearson","spearman","mae","rmse"]:
        val = metrics.get(f"matched_pair_n_transcripts_vs_total_counts_{s}", np.nan)
        metrics[f"matched_cell_transcript_count_{s}"] = val
        metrics[f"matched_pair_n_transcripts_vs_total_counts_{s}"] = val

    for k,v in tx_result.items(): metrics[k] = v

    # 保存
    prefix  = f"{cfg.method_name.lower()}_xenium_sparse_hungarian_qc_metrics"
    pd.DataFrame([metrics]).to_csv(outdir/f"{prefix}_full.csv", index=False)
    with open(outdir/f"{prefix}_full.json","w",encoding="utf-8") as f:
        json.dump(metrics,f,indent=2,ensure_ascii=False,default=_json_default)
    selected = build_selected_metric_row(cfg.method_name, metrics)
    sel_df   = pd.DataFrame([selected])
    sel_csv  = outdir/f"{prefix}_selected.csv"
    sel_df.to_csv(sel_csv, index=False)
    with open(outdir/f"{prefix}_selected.json","w",encoding="utf-8") as f:
        json.dump(selected,f,indent=2,ensure_ascii=False,default=_json_default)
    if cfg.save_matched_table: matches.to_csv(outdir/"matched_cells.csv",index=False)
    if cfg.save_cell_level_tables:
        pred_cells.to_csv(outdir/"pred_cell_level_table.csv",index=False)
        gt_cells.to_csv(outdir/"gt_cell_level_table.csv",index=False)

    if cfg.verbose:
        print(f"\n[DONE] → {sel_csv}")
        print("\n=== SELECTED METRICS ===")
        print(sel_df.T.to_string(header=False))
    return sel_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask-path",           default=MASK_PATH)
    p.add_argument("--gt-cells-path",       default=CELLS_PQ)
    p.add_argument("--gt-transcripts-path", default=TX_PARQUET)
    p.add_argument("--output-dir",          default="/data/qiuyijia/eval_results/ucs_xenium")
    p.add_argument("--method-name",         default="UCS")
    p.add_argument("--match-radius",        type=float, default=None)
    p.add_argument("--quiet",               action="store_true")
    args = p.parse_args()
    return EvaluationConfig(
        mask_path=args.mask_path,
        gt_cells_path=args.gt_cells_path,
        gt_transcripts_path=args.gt_transcripts_path,
        output_dir=args.output_dir,
        method_name=args.method_name,
        numeric_pairs=[("n_transcripts","total_counts")],
        match_radius=args.match_radius,
        verbose=not args.quiet,
    )

if __name__ == "__main__":
    cfg = parse_args(); evaluate(cfg)