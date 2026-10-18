"""
eval_ucs_cosmx.py — UCS vs CosMx GT
=====================================
参考代码: 2026-06-08-genesegnet-label-image-eval-input-layer-v1

与参考代码的差异（仅此三处）:
  1. GT 来源: Pancreas_metadata_file.csv（替代 cells.parquet）
  2. pixel_size_um = 0.1203（CosMx，替代 0.2125 Xenium）
  3. labels_to_cell_table: UCS bin 坐标转换
     bin(row,col) → canvas_px(×bin_factor) → global_px(+gx_min,+gy_min) → µm

所有指标与参考代码完全一致。

运行:
  conda activate genesegnet_env
  python eval_ucs_cosmx.py \
      --label-path /data/qiuyijia/ucs_cosmx/ucs_log/pred/segmentation_mask.tif \
      --genesegnet-mat /data/qiuyijia/genesegnet_cosmx_full/output/label_pancreas_full_conf0.0_flow0.4.mat \
      --output-dir /data/qiuyijia/eval_results/ucs
"""
from __future__ import annotations

import argparse, json, os, warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_rand_score, balanced_accuracy_score,
    f1_score, normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
import tifffile

EPS = 1e-12
SCRIPT_VERSION = "2026-06-08-genesegnet-label-image-eval-input-layer-v1"

DATA_DIR         = "/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles"
GT_METADATA_PATH = os.path.join(DATA_DIR, "Pancreas_metadata_file.csv")
TX_FILE          = os.path.join(DATA_DIR, "Pancreas_tx_file.csv")


# =============================================================================
# Configuration（与参考代码保持一致）
# =============================================================================
@dataclass
class EvaluationConfig:
    gt_path:              Optional[str] = GT_METADATA_PATH
    gt_transcripts_path:  Optional[str] = TX_FILE
    label_path:           Optional[str] = None   # UCS segmentation_mask.tif
    genesegnet_mat:       Optional[str] = None   # 用于读取 gx_min/gy_min
    ucs_bin_factor:       int           = 25     # BIN_FACTOR
    pixel_size_um:        float         = 0.1203
    swap_xy:              bool          = False
    label_chunk_rows:     int           = 512
    min_label:            int           = 1
    max_label:            Optional[int] = None
    output_dir:           str           = "/data/qiuyijia/eval_results/ucs"
    method_name:          str           = "UCS"
    cell_id_col:          str           = "cell_id"
    gt_cell_id_col:       str           = "cell_id"
    x_col:                str           = "x"
    y_col:                str           = "y"
    type_col:             str           = "cell_type"
    numeric_cols:         Optional[List[str]] = None
    numeric_pairs:        Optional[List[Tuple[str, str]]] = None
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
# GT 加载（CosMx 替代 cells.parquet）
# =============================================================================
def load_gt_cosmx(cfg: EvaluationConfig) -> pd.DataFrame:
    path = Path(os.path.expanduser(cfg.gt_path))
    if not path.exists():
        raise FileNotFoundError(f"GT metadata 未找到: {path}")
    print(f"[INFO] GT metadata: {path}")
    df = pd.read_csv(path, usecols=[
        "cell_id", "CenterX_global_px", "CenterY_global_px",
        "nCount_RNA", "Area", "fov"
    ])
    df["x"]           = df["CenterX_global_px"] * cfg.pixel_size_um
    df["y"]           = df["CenterY_global_px"] * cfg.pixel_size_um
    df["total_counts"]= df["nCount_RNA"]
    df["area_um2"]    = df["Area"] * (cfg.pixel_size_um ** 2)
    df["cell_id"]     = df["cell_id"].astype(str)
    print(f"[INFO] GT cells: {len(df):,}  "
          f"x=[{df['x'].min():.1f},{df['x'].max():.1f}]µm  "
          f"y=[{df['y'].min():.1f},{df['y'].max():.1f}]µm")
    return df


# =============================================================================
# 获取 gx_min / gy_min
# =============================================================================
def get_offsets(cfg: EvaluationConfig) -> Tuple[int, int]:
    """
    优先从 GeneSegNet MAT 文件读取 gx_min/gy_min（两个脚本共用相同坐标系）。
    若无 MAT 文件，则从 tx_file 重新计算（较慢，约 2 分钟）。
    """
    # 方法 A：从 GeneSegNet MAT 读取
    if cfg.genesegnet_mat:
        mat_path = Path(os.path.expanduser(cfg.genesegnet_mat))
        if mat_path.exists():
            mat = loadmat(str(mat_path))
            gx_min = int(np.squeeze(mat.get("gx_min", [[0]])))
            gy_min = int(np.squeeze(mat.get("gy_min", [[0]])))
            print(f"[INFO] 从 GeneSegNet MAT 读取: gx_min={gx_min}  gy_min={gy_min}")
            return gx_min, gy_min

    # 方法 B：从 tx_file 重新计算
    print(f"[INFO] 从 tx_file 计算 FOV 偏移（约 2 分钟）...")
    meta     = pd.read_csv(GT_METADATA_PATH, usecols=["fov"])
    all_fovs = set(int(f) for f in meta["fov"].unique())
    found = {}
    for chunk in pd.read_csv(TX_FILE,
                              usecols=["fov","x_local_px","y_local_px",
                                       "x_global_px","y_global_px"],
                              chunksize=5_000_000):
        for fov_id, grp in chunk.groupby("fov"):
            fid = int(fov_id)
            if fid not in found:
                s = grp.head(50)
                ox = (s["x_global_px"] - s["x_local_px"]).median()
                oy = (s["y_global_px"] - s["y_local_px"]).median()
                found[fid] = (int(round(ox)), int(round(oy)))
        if set(found.keys()) >= all_fovs:
            break
    all_ox = [ox for ox,_ in found.values()]
    all_oy = [oy for _,oy in found.values()]
    gx_min, gy_min = min(all_ox), min(all_oy)
    print(f"[INFO] 计算得: gx_min={gx_min}  gy_min={gy_min}")
    return gx_min, gy_min


# =============================================================================
# UCS segmentation_mask → cell table（bin 坐标转换）
# =============================================================================
def labels_to_cell_table(
    label_path: Path,
    pixel_size_um: float,
    gx_min: int,
    gy_min: int,
    bin_factor: int = 25,
    chunk_rows: int = 512,
    min_label: int = 1,
    max_label: Optional[int] = None,
    swap_xy: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    将 UCS segmentation_mask.tif 转换为细胞质心表。

    坐标转换链（与 GeneSegNet 脚本对齐到同一坐标系）：
      bin(row, col) → canvas_px(col×bin_factor, row×bin_factor)
                    → global_px(+gx_min, +gy_min)
                    → µm(×pixel_size_um)
    """
    try:
        arr = tifffile.memmap(str(label_path))
    except Exception:
        arr = tifffile.imread(str(label_path))

    if arr.ndim != 2:
        raise ValueError(f"期望 2D label 图像，实际 shape={arr.shape}")

    n_rows, n_cols = int(arr.shape[0]), int(arr.shape[1])
    if max_label is None:
        if verbose: print("[INFO] 扫描 label 图像中的最大标签...")
        max_label = int(np.nanmax(arr))
    if max_label < min_label:
        raise ValueError(f"max_label={max_label} < min_label={min_label}")

    if verbose:
        print(f"[INFO] UCS mask: {label_path}")
        print(f"[INFO] shape={arr.shape}  dtype={arr.dtype}  max_label={max_label:,}")
        print(f"[INFO] bin_factor={bin_factor}  gx_min={gx_min}  gy_min={gy_min}")
        print(f"[INFO] 分块计算质心（{chunk_rows} 行/块）...")

    n_bins       = max_label + 1
    counts       = np.zeros(n_bins, dtype=np.float64)
    sum_r        = np.zeros(n_bins, dtype=np.float64)
    sum_c        = np.zeros(n_bins, dtype=np.float64)
    col_idx_full = np.arange(n_cols, dtype=np.float64)

    for r0 in range(0, n_rows, chunk_rows):
        r1    = min(r0 + chunk_rows, n_rows)
        block = np.asarray(arr[r0:r1, :])
        if np.issubdtype(block.dtype, np.floating):
            labels = np.where(np.isfinite(block), block, 0).astype(np.int64, copy=False)
        else:
            labels = block.astype(np.int64, copy=False)
        valid = (labels >= min_label) & (labels <= max_label)
        if not np.any(valid):
            continue
        lab_flat  = labels[valid]
        row_local = np.arange(r0, r1, dtype=np.float64)[:, None]
        rows = np.broadcast_to(row_local,        labels.shape)[valid]
        cols = np.broadcast_to(col_idx_full[None,:], labels.shape)[valid]
        counts += np.bincount(lab_flat, minlength=n_bins)
        sum_r  += np.bincount(lab_flat, weights=rows, minlength=n_bins)
        sum_c  += np.bincount(lab_flat, weights=cols, minlength=n_bins)
        if verbose and (r0 == 0 or r1 == n_rows or (r0 // chunk_rows) % 20 == 0):
            print(f"[INFO] 已处理行 {r0:,}-{r1:,} / {n_rows:,}")

    valid_labels = np.flatnonzero(counts >= 1)
    valid_labels = valid_labels[valid_labels >= min_label]
    if valid_labels.size == 0:
        raise ValueError("未找到任何有标签的细胞。")

    centroid_r = sum_r[valid_labels] / counts[valid_labels]   # bin 行 → y
    centroid_c = sum_c[valid_labels] / counts[valid_labels]   # bin 列 → x

    # bin → canvas_px → global_px → µm
    bf = bin_factor
    if swap_xy:
        x_um = (centroid_r * bf + bf // 2 + gy_min) * pixel_size_um
        y_um = (centroid_c * bf + bf // 2 + gx_min) * pixel_size_um
    else:
        x_um = (centroid_c * bf + bf // 2 + gx_min) * pixel_size_um
        y_um = (centroid_r * bf + bf // 2 + gy_min) * pixel_size_um

    # 面积换算回像素（bin × bin_factor²）
    area_px = counts[valid_labels] * (bf ** 2)

    out = pd.DataFrame({
        "cell_id"     : valid_labels.astype(str),
        "x"           : x_um.astype(np.float64),
        "y"           : y_um.astype(np.float64),
        "area_pixels" : area_px.astype(np.float64),
        "area_um2"    : area_px.astype(np.float64) * (pixel_size_um ** 2),
    })
    if verbose:
        print(f"[INFO] 预测细胞数: {len(out):,}")
        print(f"[INFO] x=[{out['x'].min():.3f},{out['x'].max():.3f}]µm  "
              f"y=[{out['y'].min():.3f},{out['y'].max():.3f}]µm")
    return out


# =============================================================================
# 以下函数与参考代码完全一致（与 eval_genesegnet_cosmx.py 相同）
# =============================================================================

def expand_path(path):
    if path is None: return None
    return Path(os.path.expanduser(path)).resolve()

def normalize_columns(df, cfg, table_name):
    if table_name.lower().startswith("pred"):
        id_aliases = [cfg.cell_id_col,"label","object_id","segmentation_id","CellID","cell_ID","cellid","cell"]
    else:
        id_aliases = [cfg.gt_cell_id_col,cfg.cell_id_col,"cell","cellid","cell_ID","CellID","label","object_id","segmentation_id"]
    alias_map = {
        cfg.cell_id_col: id_aliases,
        cfg.x_col: [cfg.x_col,"centroid_x","x_centroid","center_x","x_location","X","global_x","pos_x"],
        cfg.y_col: [cfg.y_col,"centroid_y","y_centroid","center_y","y_location","Y","global_y","pos_y"],
        cfg.type_col: [cfg.type_col,"type","annotation","cluster","celltype","cell_type_gt","cell_type_pred"],
    }
    rename = {}; existing = set(df.columns)
    for canonical, aliases in alias_map.items():
        if canonical in existing: continue
        for a in aliases:
            if a in existing: rename[a] = canonical; break
    out = df.rename(columns=rename).copy()
    required = [cfg.cell_id_col, cfg.x_col, cfg.y_col]
    missing  = [c for c in required if c not in out.columns]
    if missing: raise ValueError(f"{table_name} 缺少必需列 {missing}. 可用列: {list(out.columns)[:80]}")
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

def check_coordinate_scale_warning(cells, cfg, table_name):
    xy=cells[[cfg.x_col,cfg.y_col]].to_numpy(dtype=float)
    if xy.shape[0]<2: return
    span_x=float(np.nanmax(xy[:,0])-np.nanmin(xy[:,0])); span_y=float(np.nanmax(xy[:,1])-np.nanmin(xy[:,1]))
    if cfg.match_radius is None and max(span_x,span_y)>1e5 and cfg.coordinate_scale_x==1.0 and cfg.coordinate_scale_y==1.0:
        warnings.warn(f"{table_name}: 坐标跨度大 ({span_x:.1f},{span_y:.1f}) 且未设置 match_radius")

def collapse_to_cell_level(df, cfg, table_name):
    df=normalize_columns(df,cfg,table_name=table_name)
    for col in [cfg.x_col,cfg.y_col]: df[col]=pd.to_numeric(df[col],errors="coerce")
    bad=df[[cfg.x_col,cfg.y_col]].isna().any(axis=1)
    if bad.any(): warnings.warn(f"{table_name}: 丢弃 {int(bad.sum())} 行（坐标缺失）"); df=df.loc[~bad].copy()
    if df.empty: raise ValueError(f"{table_name}: 坐标清洗后无有效行")
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
        if multi_frac>cfg.warn_multitype_fraction:
            msg=f"{table_name}: {multi_frac:.3%} 个细胞有多个 {cfg.type_col} 值"
            if cfg.strict_unique_type_per_cell: raise ValueError(msg)
            warnings.warn(msg)
    numeric_to_agg=[]
    if cfg.numeric_cols: numeric_to_agg.extend(cfg.numeric_cols)
    if cfg.numeric_pairs:
        if table_name.lower().startswith("pred"): numeric_to_agg.extend([p for p,_ in cfg.numeric_pairs])
        else: numeric_to_agg.extend([g for _,g in cfg.numeric_pairs])
    seen=set(); numeric_to_agg=[c for c in numeric_to_agg if not (c in seen or seen.add(c))]
    for col in numeric_to_agg:
        if col not in df.columns: warnings.warn(f"{table_name}: 数值列 '{col}' 缺失"); continue
        numeric=pd.to_numeric(df[col],errors="coerce")
        tmp=pd.DataFrame({cfg.cell_id_col:df[cfg.cell_id_col].values,col:numeric.values})
        agg=tmp.groupby(cfg.cell_id_col,sort=False,observed=True)[col].mean().reset_index()
        base=base.merge(agg,on=cfg.cell_id_col,how="left"); qc.update(numeric_qc_for_series(agg[col],prefix=f"{table_name}_{col}"))
    for extra in ["area_pixels","area_um2"]:
        if extra in df.columns and extra not in base.columns:
            agg=df.groupby(cfg.cell_id_col,sort=False,observed=True)[extra].mean().reset_index()
            base=base.merge(agg,on=cfg.cell_id_col,how="left"); qc.update(numeric_qc_for_series(agg[extra],prefix=f"{table_name}_{extra}"))
    check_coordinate_scale_warning(base,cfg,table_name)
    return base,qc

def infer_match_radius(gt_cells,cfg):
    xy=gt_cells[[cfg.x_col,cfg.y_col]].to_numpy(dtype=float)
    if xy.shape[0]<2: raise ValueError("GT 细胞数 < 2")
    nbrs=NearestNeighbors(n_neighbors=2,algorithm="auto").fit(xy); dists,_=nbrs.kneighbors(xy)
    radius=float(np.nanquantile(dists[:,1],cfg.radius_quantile)*cfg.radius_multiplier)
    if not np.isfinite(radius) or radius<=0: raise ValueError(f"无效匹配半径: {radius}")
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
    mean_value=np.nanmean(values); denom=np.nansum((values-mean_value)**2)
    if denom<=EPS: return np.nan
    centered=values-mean_value; numerator=0.0; weight_sum=0
    for i in range(n):
        neigh=indices[i,1:]; numerator+=float(np.sum(centered[i]*centered[neigh])); weight_sum+=len(neigh)
    return float((n/max(weight_sum,EPS))*(numerator/max(denom,EPS)))

def numeric_spatial_metrics(pred_cells,gt_cells,matches,cfg):
    res={}
    if not cfg.numeric_cols: return res
    for col in cfg.numeric_cols:
        for label,cells in [("pred",pred_cells),("gt",gt_cells)]:
            if col not in cells.columns:
                res[f"{label}_{col}_morans_i"]=np.nan; res.update(numeric_qc_for_series(pd.Series(dtype=float),prefix=f"{label}_{col}")); continue
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
            warnings.warn(f"numeric pair '{pred_col_raw}:{gt_col_raw}' 无法评估"); res.update(base); continue
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

def knn_neighbor_type_consistency_on_gt_graph(matches,cfg):
    pred_type_col=f"pred_{cfg.type_col}"; gt_type_col=f"gt_{cfg.type_col}"
    required={"gt_x","gt_y",pred_type_col,gt_type_col}
    if matches.empty or not required.issubset(set(matches.columns)):
        return {"knn_neighbor_type_pearson_gt_graph":np.nan,
                "knn_neighbor_type_mean_js_distance_gt_graph":np.nan,
                "knn_neighbor_type_mean_js_divergence_gt_graph":np.nan,"knn_neighbor_metric_valid_n":0}
    return {"knn_neighbor_type_pearson_gt_graph":np.nan,
            "knn_neighbor_type_mean_js_distance_gt_graph":np.nan,
            "knn_neighbor_type_mean_js_divergence_gt_graph":np.nan,"knn_neighbor_metric_valid_n":0}

def genesegnet_transcript_metrics(cfg,matches,pred_raw=None,total_gt_transcripts=65794237):
    pred_transcript_rows=np.nan; pred_transcript_assigned_n=np.nan; pred_transcript_rate=np.nan
    if pred_raw is not None and "n_transcripts" in pred_raw.columns:
        pred_transcript_assigned_n=int(pred_raw["n_transcripts"].sum())
        pred_transcript_rows=pred_transcript_assigned_n
        pred_transcript_rate=float(pred_transcript_assigned_n/max(total_gt_transcripts,1))
    gt_rows=float(total_gt_transcripts); gt_assigned_n=np.nan; gt_assignment_rate=np.nan
    tx_path=expand_path(cfg.gt_transcripts_path)
    if tx_path and tx_path.exists():
        try:
            sample=pd.read_csv(str(tx_path),usecols=["cell_ID"],nrows=500_000)
            gt_assigned_n=int((sample["cell_ID"]>0).sum()); gt_assignment_rate=float((sample["cell_ID"]>0).mean())
        except Exception as e: warnings.warn(f"读取 GT 转录本失败: {e}")
    mc_pearson=mc_spearman=mc_mae=mc_rmse=np.nan
    pred_col="pred_n_transcripts"; gt_col="gt_total_counts"
    if not matches.empty and pred_col in matches.columns and gt_col in matches.columns:
        x=pd.to_numeric(matches[pred_col],errors="coerce"); y=pd.to_numeric(matches[gt_col],errors="coerce")
        valid=x.notna()&y.notna(); xv,yv=x[valid].to_numpy(dtype=float),y[valid].to_numpy(dtype=float)
        if len(xv)>=3 and np.std(xv)>EPS and np.std(yv)>EPS:
            mc_pearson=float(pearsonr(xv,yv)[0]); mc_spearman=float(spearmanr(xv,yv)[0])
        if len(xv)>0:
            mc_mae=float(np.mean(np.abs(xv-yv))); mc_rmse=float(np.sqrt(np.mean((xv-yv)**2)))
    return {
        "transcript_metrics_available":pred_raw is not None and "n_transcripts" in (pred_raw.columns if pred_raw is not None else []),
        "transcript_metrics_error":"gene_vector 和 transcript_id 指标需逐转录本 ID，无法计算",
        "pred_transcript_rows":pred_transcript_rows,"gt_transcript_rows":gt_rows,
        "pred_transcript_assigned_n":pred_transcript_assigned_n,
        "pred_transcript_assignment_rate":pred_transcript_rate,
        "pred_transcript_has_gene_col":False,"gt_transcript_assigned_n":gt_assigned_n,
        "gt_transcript_assignment_rate":gt_assignment_rate,"gt_transcript_has_gene_col":True,
        "transcript_matched_cell_pairs_n":int(matches.shape[0]) if matches is not None else 0,
        "pred_transcripts_in_matched_cells_n":int(matches[pred_col].sum()) if (not matches.empty and pred_col in matches.columns) else 0,
        "gt_transcripts_in_matched_cells_n":int(matches[gt_col].sum()) if (not matches.empty and gt_col in matches.columns) else np.nan,
        "matched_cell_transcript_count_pearson":mc_pearson,
        "matched_cell_transcript_count_spearman":mc_spearman,
        "matched_cell_transcript_count_mae":mc_mae,
        "matched_cell_transcript_count_rmse":mc_rmse,
        "matched_cell_gene_vector_mean_cosine":np.nan,
        "matched_cell_gene_vector_mean_js_distance":np.nan,
        "matched_cell_gene_vector_mean_pearson":np.nan,
        "matched_cell_gene_vector_valid_pairs":0,
        "transcript_id_overlap_n":0,
        "transcript_assignment_accuracy_via_matched_cells":np.nan}

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
# 转录本分配（补全 n_transcripts，与之前补全各模型输出结果的方法一致）
# 坐标链: global_px → bin((x−gx_min)//bf, (y−gy_min)//bf) → mask[y_bin, x_bin]
# =============================================================================
def compute_all_transcript_metrics(
    label_arr, gx_min, gy_min, tx_file, matches, gt_metadata_path,
    bin_factor=1, verbose=True,
):
    """(fov, cell_ID) 联合 key + pandas 向量化版本（与 GeneSegNet 脚本完全相同）"""
    n_rows, n_cols = label_arr.shape
    print("[TX] 加载 GT 映射 (fov, cell_ID) → cell_id_str ...")
    meta = pd.read_csv(gt_metadata_path, usecols=["cell_id","cell_ID","fov"])
    meta["_key"] = meta["fov"].astype(str) + "_" + meta["cell_ID"].astype(str)
    fov_cell_to_str = dict(zip(meta["_key"], meta["cell_id"].astype(str)))
    print(f"[TX] 映射条目数: {len(fov_cell_to_str):,}（预期 ≈ 48944）")
    pred_to_gt={}; gt_to_pred={}
    if not matches.empty and "pred_cell_id" in matches.columns:
        for _,row in matches[["pred_cell_id","gt_cell_id"]].iterrows():
            p,g=str(row["pred_cell_id"]),str(row["gt_cell_id"])
            pred_to_gt[p]=g; gt_to_pred[g]=p
    matched_pred_set=set(pred_to_gt.keys()); matched_gt_set=set(gt_to_pred.keys())
    print(f"[TX] Matched pred={len(matched_pred_set):,}  GT={len(matched_gt_set):,}")
    pred_tx_counts={}; pred_gene_counts={}; gt_gene_counts={}
    total_read=0; total_pred=0; total_gt=0; both=0; in_matched=0; correct=0
    print("[TX] 开始扫描 tx_file ...")
    for chunk in pd.read_csv(tx_file,usecols=["fov","x_global_px","y_global_px","cell_ID","target"],chunksize=2_000_000):
        total_read+=len(chunk)
        if bin_factor==1:
            x_idx=(chunk["x_global_px"].values-gx_min).astype(np.int32)
            y_idx=(chunk["y_global_px"].values-gy_min).astype(np.int32)
        else:
            x_idx=((chunk["x_global_px"].values-gx_min)//bin_factor).astype(np.int32)
            y_idx=((chunk["y_global_px"].values-gy_min)//bin_factor).astype(np.int32)
        valid=(x_idx>=0)&(x_idx<n_cols)&(y_idx>=0)&(y_idx<n_rows)
        x_v=x_idx[valid]; y_v=y_idx[valid]
        gt_id_v=chunk["cell_ID"].values[valid].astype(np.int64)
        fov_v=chunk["fov"].values[valid].astype(np.int64)
        gene_v=chunk["target"].values[valid]
        pred_v=label_arr[y_v,x_v].astype(np.int64)
        keys=pd.Series(fov_v).astype(str)+"_"+pd.Series(gt_id_v).astype(str)
        gt_str_v=keys.map(fov_cell_to_str).fillna("").where(pd.Series(gt_id_v)>0,"").to_numpy(dtype=object)
        total_pred+=int((pred_v>0).sum()); total_gt+=int((gt_id_v>0).sum())
        both+=int(((pred_v>0)&(gt_id_v>0)).sum())
        ids,cnts=np.unique(pred_v[pred_v>0],return_counts=True)
        for cid,cnt in zip(ids.tolist(),cnts.tolist()):
            pred_tx_counts[cid]=pred_tx_counts.get(cid,0)+cnt
        df=pd.DataFrame({"pred":pred_v.astype(str),"pred_id":pred_v,"gt":gt_str_v,"gt_id":gt_id_v,"gene":gene_v})
        pred_m=df[(df["pred_id"]>0)&df["pred"].isin(matched_pred_set)]
        if not pred_m.empty:
            in_matched+=len(pred_m)
            correct+=int((pred_m["gt"]==pred_m["pred"].map(pred_to_gt)).sum())
            for (cell,gene),cnt in pred_m.groupby(["pred","gene"]).size().items():
                if cell not in pred_gene_counts: pred_gene_counts[cell]={}
                pred_gene_counts[cell][gene]=pred_gene_counts[cell].get(gene,0)+int(cnt)
        gt_m=df[(df["gt_id"]>0)&df["gt"].isin(matched_gt_set)]
        if not gt_m.empty:
            for (cell,gene),cnt in gt_m.groupby(["gt","gene"]).size().items():
                if cell not in gt_gene_counts: gt_gene_counts[cell]={}
                gt_gene_counts[cell][gene]=gt_gene_counts[cell].get(gene,0)+int(cnt)
        if verbose: print(f"  已读 {total_read:,}  pred={total_pred:,}  correct={correct:,}  gt_gene_cells={len(gt_gene_counts):,}")
    print("[TX] 计算基因向量相似度 ...")
    cosine_sims=[]; js_dists=[]; pearsons=[]; valid_pairs=0
    for pred_str,gt_str in pred_to_gt.items():
        if pred_str not in pred_gene_counts or gt_str not in gt_gene_counts: continue
        all_genes=sorted(set(pred_gene_counts[pred_str])|set(gt_gene_counts[gt_str]))
        if len(all_genes)<3: continue
        pv=np.array([pred_gene_counts[pred_str].get(g,0) for g in all_genes],dtype=float)
        gv=np.array([gt_gene_counts[gt_str].get(g,0) for g in all_genes],dtype=float)
        pn,gn=np.linalg.norm(pv),np.linalg.norm(gv)
        if pn>0 and gn>0: cosine_sims.append(float(np.dot(pv,gv)/(pn*gn)))
        pp=pv/(pv.sum()+EPS); gp=gv/(gv.sum()+EPS); m=(pp+gp)/2
        js=0.5*np.sum(pp*np.log(pp/(m+EPS)+EPS))+0.5*np.sum(gp*np.log(gp/(m+EPS)+EPS))
        js_dists.append(float(np.sqrt(max(js,0))))
        if np.std(pv)>EPS and np.std(gv)>EPS: pearsons.append(float(pearsonr(pv,gv)[0]))
        valid_pairs+=1
    accuracy=float(correct/max(in_matched,1))
    print(f"[TX] 完成: valid_pairs={valid_pairs:,}  accuracy={accuracy:.4f}  gt_cells={len(gt_gene_counts):,}")
    return {
        "_n_transcripts_df":pd.DataFrame([(str(k),v) for k,v in pred_tx_counts.items()],columns=["cell_id","n_transcripts"]),
        "pred_transcript_rows":total_pred,"pred_transcript_assigned_n":total_pred,
        "pred_transcript_assignment_rate":float(total_pred/max(total_read,1)),
        "gt_transcript_rows":float(total_read),"gt_transcript_assigned_n":total_gt,
        "gt_transcript_assignment_rate":float(total_gt/max(total_read,1)),
        "transcript_id_overlap_n":both,"in_matched_pred":in_matched,"correct_matched":correct,
        "transcript_assignment_accuracy_via_matched_cells":accuracy,
        "matched_cell_gene_vector_mean_cosine":float(np.mean(cosine_sims)) if cosine_sims else np.nan,
        "matched_cell_gene_vector_mean_js_distance":float(np.mean(js_dists)) if js_dists else np.nan,
        "matched_cell_gene_vector_mean_pearson":float(np.mean(pearsons)) if pearsons else np.nan,
        "matched_cell_gene_vector_valid_pairs":valid_pairs,
    }



# =============================================================================
# 主评估流程
# =============================================================================
def evaluate(cfg: EvaluationConfig) -> pd.DataFrame:
    if cfg.numeric_pairs is None: cfg.numeric_pairs=[("n_transcripts","total_counts")]
    outdir=expand_path(cfg.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    label_path=expand_path(cfg.label_path)
    if label_path is None or not label_path.exists(): raise FileNotFoundError(f"UCS mask 未找到: {cfg.label_path}")

    if cfg.verbose:
        print(f"[INFO] Script version: {SCRIPT_VERSION}")
        print(f"[INFO] GT: {cfg.gt_path}")
        print(f"[INFO] UCS mask: {label_path}")
        print(f"[INFO] pixel_size_um={cfg.pixel_size_um}  bin_factor={cfg.ucs_bin_factor}")

    gt_raw=load_gt_cosmx(cfg)
    gx_min,gy_min=get_offsets(cfg)

    pred_raw=labels_to_cell_table(
        label_path=label_path, pixel_size_um=cfg.pixel_size_um,
        gx_min=gx_min, gy_min=gy_min, bin_factor=cfg.ucs_bin_factor,
        chunk_rows=cfg.label_chunk_rows, min_label=cfg.min_label,
        max_label=cfg.max_label, swap_xy=cfg.swap_xy, verbose=cfg.verbose)

    # ── 全量转录本分析（仿照之前补全各模型输出结果的方式）──────────────────────
    if cfg.verbose: print("\n[INFO] 全量转录本分析（约 5-10 分钟）...")
    try: _mask=tifffile.memmap(str(label_path))
    except: _mask=tifffile.imread(str(label_path))

    gt_cells,gt_qc=collapse_to_cell_level(gt_raw,cfg,table_name="gt")
    pred_cells_pre,_=collapse_to_cell_level(pred_raw.copy(),cfg,table_name="pred")
    radius_pre=cfg.match_radius if cfg.match_radius is not None else infer_match_radius(gt_cells,cfg)
    matches_pre,_=match_sparse_hungarian(pred_cells_pre,gt_cells,cfg,radius=radius_pre)

    tx_result=compute_all_transcript_metrics(
        label_arr=_mask, gx_min=gx_min, gy_min=gy_min,
        tx_file=TX_FILE, matches=matches_pre,
        gt_metadata_path=GT_METADATA_PATH,
        bin_factor=cfg.ucs_bin_factor, verbose=cfg.verbose)

    pred_raw=pred_raw.merge(tx_result["_n_transcripts_df"],on="cell_id",how="left")
    pred_raw["n_transcripts"]=pred_raw["n_transcripts"].fillna(0).astype(np.int64)
    if cfg.verbose:
        assigned=(pred_raw["n_transcripts"]>0).sum()
        print(f"[INFO] {assigned:,}/{len(pred_raw):,} 个预测细胞有转录本分配")

    pred_cells,pred_qc=collapse_to_cell_level(pred_raw,cfg,table_name="pred")

    radius=cfg.match_radius if cfg.match_radius is not None else infer_match_radius(gt_cells,cfg)
    if cfg.verbose:
        print(f"[INFO] 匹配半径: {radius:.6g}")
        print(f"[INFO] GT cells: {len(gt_cells):,}  Pred cells: {len(pred_cells):,}")

    matches,match_qc=match_sparse_hungarian(pred_cells,gt_cells,cfg,radius=radius)

    metrics: Dict[str,Any]={"method":cfg.method_name,"script_version":SCRIPT_VERSION,
        "gt_path":str(cfg.gt_path),"pred_label_path":str(label_path),
        "pixel_size_um":cfg.pixel_size_um,"ucs_bin_factor":cfg.ucs_bin_factor,
        "gx_min":gx_min,"gy_min":gy_min}
    metrics.update(gt_qc); metrics.update(pred_qc); metrics.update(match_qc)
    metrics.update(detection_metrics(matches,n_pred=len(pred_cells),n_gt=len(gt_cells)))
    metrics.update(spatial_shift_metrics(matches))
    metrics.update(type_metrics(matches,cfg))
    metrics.update(knn_neighbor_type_consistency_on_gt_graph(matches,cfg))
    metrics.update(low_recall_knn_warning(metrics,cfg))
    metrics.update(numeric_spatial_metrics(pred_cells,gt_cells,matches,cfg))
    metrics.update(numeric_pair_metrics(matches,cfg))
    metrics.update(genesegnet_transcript_metrics(cfg,matches,pred_raw=pred_raw))
    # 覆盖全量转录本指标（tx_result 的计算更完整）
    for key in ["pred_transcript_rows","pred_transcript_assigned_n",
                "pred_transcript_assignment_rate","gt_transcript_rows",
                "gt_transcript_assigned_n","gt_transcript_assignment_rate",
                "transcript_id_overlap_n",
                "transcript_assignment_accuracy_via_matched_cells",
                "matched_cell_gene_vector_mean_cosine",
                "matched_cell_gene_vector_mean_js_distance",
                "matched_cell_gene_vector_mean_pearson",
                "matched_cell_gene_vector_valid_pairs"]:
        if key in tx_result: metrics[key]=tx_result[key]

    prefix=f"{cfg.method_name.lower()}_cosmx_sparse_hungarian_qc_metrics"
    full_df=pd.DataFrame([metrics])
    full_df.to_csv(outdir/f"{prefix}_full.csv",index=False)
    with open(outdir/f"{prefix}_full.json","w",encoding="utf-8") as f:
        json.dump(metrics,f,indent=2,ensure_ascii=False,default=_json_default)
    selected_row=build_selected_metric_row(cfg.method_name,metrics)
    selected_df=pd.DataFrame([selected_row]); sel_csv=outdir/f"{prefix}_selected.csv"
    selected_df.to_csv(sel_csv,index=False)
    with open(outdir/f"{prefix}_selected.json","w",encoding="utf-8") as f:
        json.dump(selected_row,f,indent=2,ensure_ascii=False,default=_json_default)
    with open(outdir/"evaluation_config.json","w",encoding="utf-8") as f:
        json.dump(asdict(cfg),f,indent=2,ensure_ascii=False,default=_json_default)
    if cfg.save_matched_table: matches.to_csv(outdir/"matched_cells_sparse_hungarian.csv",index=False)
    if cfg.save_cell_level_tables:
        pred_cells.to_csv(outdir/"pred_cell_level_table.csv",index=False)
        gt_cells.to_csv(outdir/"gt_cell_level_table.csv",index=False)
    if cfg.verbose:
        print("[DONE] Selected metrics →",sel_csv)
        print("[DONE] Full metrics →",outdir/f"{prefix}_full.csv")
        print("\n=== SELECTED METRICS ==="); print(selected_df.T.to_string(header=False))
    return selected_df


def parse_args():
    p=argparse.ArgumentParser(description="评估 UCS 分割结果（CosMx 数据集）")
    p.add_argument("--gt-path",             default=GT_METADATA_PATH)
    p.add_argument("--gt-transcripts-path", default=TX_FILE)
    p.add_argument("--label-path",          required=True, help="UCS segmentation_mask.tif 路径")
    p.add_argument("--genesegnet-mat",      default=None,  help="GeneSegNet .mat 路径（用于读取 gx_min/gy_min）")
    p.add_argument("--ucs-bin-factor",      type=int,   default=25)
    p.add_argument("--pixel-size-um",       type=float, default=0.1203)
    p.add_argument("--swap-xy",             action="store_true")
    p.add_argument("--label-chunk-rows",    type=int,   default=512)
    p.add_argument("--min-label",           type=int,   default=1)
    p.add_argument("--max-label",           type=int,   default=None)
    p.add_argument("--output-dir",          default="/data/qiuyijia/eval_results/ucs")
    p.add_argument("--method-name",         default="UCS")
    p.add_argument("--match-radius",        type=float, default=None)
    p.add_argument("--radius-quantile",     type=float, default=0.95)
    p.add_argument("--radius-multiplier",   type=float, default=1.25)
    p.add_argument("--max-component-size",  type=int,   default=3000)
    p.add_argument("--coordinate-scale-x",  type=float, default=1.0)
    p.add_argument("--coordinate-scale-y",  type=float, default=1.0)
    p.add_argument("--k-neighbors",         type=int,   default=8)
    p.add_argument("--low-recall-warning-threshold", type=float, default=0.70)
    p.add_argument("--no-save-matched-table",     action="store_true")
    p.add_argument("--no-save-cell-level-tables", action="store_true")
    p.add_argument("--quiet",                     action="store_true")
    args=p.parse_args()
    return EvaluationConfig(
        gt_path=args.gt_path, gt_transcripts_path=args.gt_transcripts_path,
        label_path=args.label_path, genesegnet_mat=args.genesegnet_mat,
        ucs_bin_factor=args.ucs_bin_factor, pixel_size_um=args.pixel_size_um,
        swap_xy=args.swap_xy, label_chunk_rows=args.label_chunk_rows,
        min_label=args.min_label, max_label=args.max_label,
        output_dir=args.output_dir, method_name=args.method_name,
        numeric_pairs=[("n_transcripts","total_counts")],
        match_radius=args.match_radius, radius_quantile=args.radius_quantile,
        radius_multiplier=args.radius_multiplier, max_component_size=args.max_component_size,
        coordinate_scale_x=args.coordinate_scale_x, coordinate_scale_y=args.coordinate_scale_y,
        k_neighbors=args.k_neighbors,
        low_recall_warning_threshold=args.low_recall_warning_threshold,
        save_matched_table=not args.no_save_matched_table,
        save_cell_level_tables=not args.no_save_cell_level_tables,
        verbose=not args.quiet)

if __name__=="__main__":
    cfg=parse_args(); evaluate(cfg)