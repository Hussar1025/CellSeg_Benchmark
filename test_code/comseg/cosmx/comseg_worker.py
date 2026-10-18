#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
comseg_cosmx_roi_run.py  —— ComSeg × CosMx 胰腺（空间 ROI 版）
=========================================================================
问题：全图 FOV 51 有 200 万转录本，classify_centroid 会把内存涨到 50 GB 被杀。
解法：取 10000×10000 px 的空间 ROI（≈ 1203×1203 µm），
      约含 5000-8000 细胞和 40-80 万转录本，ComSeg 可以承受。

ROI 中心：从 cell_metadata 的细胞质心中位数自动计算（组织中心）
坐标：x_global_px / y_global_px（像素，0.1203 µm/px）

关键参数：
  --k-nn 2  --max-cell-radius 20  （比 full FOV 更保守，防止内存爆）
  --sampling-size 3000  --top-n 300

用法：
  conda activate comseg_env
  cd /data/qiuyijia/comseg
  python comseg_cosmx_roi_run.py
  python comseg_cosmx_roi_run.py --roi-size 5000    # 更小的 ROI
  python comseg_cosmx_roi_run.py --check-only       # 只统计 ROI 内的数据量
"""

import os, sys, time, argparse
import numpy as np
import pandas as pd

SEP = "=" * 68

TX_PATH   = "/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles/Pancreas_tx_file.csv"
META_PATH = "/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles/Pancreas_metadata_file.csv"
OUT_DIR   = "/data/qiuyijia/comseg/output_roi"
CACHE_DIR = "/data/qiuyijia/comseg/cache_roi"
PIXEL_UM  = 0.1203   # µm/px

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--roi-size",    type=int,   default=10000,
                   help="ROI 边长（px），默认 10000×10000")
    p.add_argument("--sampling-size", type=int, default=3000)
    p.add_argument("--top-n",       type=int,   default=300)
    p.add_argument("--k-nn",        type=int,   default=2)
    p.add_argument("--max-cell-radius", type=float, default=20.0,
                   help="核搜索半径（µm），对应 px: 20/0.1203≈166px")
    p.add_argument("--check-only",  action="store_true",
                   help="只检查 ROI 数据量，不跑 ComSeg")
    p.add_argument("--force",       action="store_true",
                   help="忽略缓存，重新计算 edge_weight")
    return p.parse_args()

# ════════════════════════════════════════════════════════════════════════════
# Step 1: 确定 ROI 中心（从 cell_metadata 质心中位数）
# ════════════════════════════════════════════════════════════════════════════
def get_roi_center(meta_path, tx_path):
    """尝试从 metadata 读质心，失败则从 tx_file 抽样"""
    try:
        meta = pd.read_csv(meta_path)
        print(f"  metadata 列名: {list(meta.columns[:8])}")
        # 找 x/y 质心列（CosMx metadata 可能用不同列名）
        x_candidates = [c for c in meta.columns
                        if any(k in c.lower() for k in ("x","col","centroid"))]
        y_candidates = [c for c in meta.columns
                        if any(k in c.lower() for k in ("y","row","centroid"))
                        and c not in x_candidates]
        if x_candidates and y_candidates:
            xc = meta[x_candidates[0]].median()
            yc = meta[y_candidates[0]].median()
            print(f"  质心中位数: x={xc:.0f}  y={yc:.0f} px（来自 metadata）")
            return float(xc), float(yc)
    except Exception as e:
        print(f"  读 metadata 失败: {e}")
    # 备用：从 tx_file 抽样
    print("  从 tx_file 抽样估算中心...")
    sample = pd.read_csv(tx_path, nrows=500_000,
                         usecols=["x_global_px","y_global_px"])
    xc = sample.x_global_px.median()
    yc = sample.y_global_px.median()
    print(f"  质心中位数: x={xc:.0f}  y={yc:.0f} px（来自 tx 抽样）")
    return float(xc), float(yc)

# ════════════════════════════════════════════════════════════════════════════
# Step 2: 过滤转录本 + 核质心到 ROI
# ════════════════════════════════════════════════════════════════════════════
def load_roi_data(cfg, cx, cy):
    half = cfg.roi_size // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    print(f"\n  ROI: x[{x0:.0f},{x1:.0f}]  y[{y0:.0f},{y1:.0f}] px"
          f"  = {(x1-x0)*PIXEL_UM:.0f}×{(y1-y0)*PIXEL_UM:.0f} µm")

    print("  读取转录本（按块过滤）...")
    chunks, total = [], 0
    for chunk in pd.read_csv(TX_PATH, chunksize=2_000_000,
                              usecols=["fov","cell_ID","x_global_px",
                                       "y_global_px","z","target","CellComp"]):
        total += len(chunk)
        sub = chunk[(chunk.x_global_px >= x0) & (chunk.x_global_px < x1) &
                    (chunk.y_global_px >= y0) & (chunk.y_global_px < y1)]
        if len(sub): chunks.append(sub)
    tx = pd.concat(chunks, ignore_index=True)
    print(f"  全量 {total:,} → ROI 内 {len(tx):,}  基因 {tx.target.nunique()}")
    tx = tx.rename(columns={"target":"gene"})

    # 核质心：用 Nuclear 转录本的质心估算，或从 cell_metadata 读
    print("  估算核质心...")
    nuclear_tx = tx[tx.CellComp == "Nuclear"]
    if len(nuclear_tx) > 0:
        nuc = (nuclear_tx[nuclear_tx.cell_ID > 0]
               .groupby(["fov","cell_ID"])[["x_global_px","y_global_px"]]
               .mean().reset_index())
        nuc.columns = ["fov","cell_ID","cx_px","cy_px"]
    else:
        # 用所有 cell_ID > 0 的转录本
        nuc = (tx[tx.cell_ID > 0]
               .groupby(["fov","cell_ID"])[["x_global_px","y_global_px"]]
               .mean().reset_index())
        nuc.columns = ["fov","cell_ID","cx_px","cy_px"]

    # 赋予唯一 cell_id（(fov, cell_ID) 联合键）
    nuc["cell_uid"] = nuc.fov.astype(str) + "_" + nuc.cell_ID.astype(str)
    cell_id_map = {v: i+1 for i, v in enumerate(nuc.cell_uid.unique())}
    nuc["cell_id"] = nuc.cell_uid.map(cell_id_map)
    tx["cell_uid"] = tx.fov.astype(str) + "_" + tx.cell_ID.astype(str)
    tx["cell_id"]  = tx.cell_uid.map(cell_id_map).fillna(0).astype(int)

    print(f"  ROI 内核质心: {len(nuc):,}  转录本有核分配: "
          f"{(tx.cell_id>0).sum():,} ({(tx.cell_id>0).mean()*100:.1f}%)")
    return tx, nuc, (x0, y0)

# ════════════════════════════════════════════════════════════════════════════
# Step 3: 运行 ComSeg（直接调 Python API）
# ════════════════════════════════════════════════════════════════════════════
def run_comseg(cfg, tx, nuc, roi_origin):
    """直接调 ComSeg Python API，跳过 comseg_worker.py 的 FOV 封装"""
    os.makedirs(OUT_DIR,   exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    x0, y0 = roi_origin
    img_name = f"cosmx_roi_{int(x0)}_{int(y0)}"
    cache_path = os.path.join(CACHE_DIR,
        f"edge_{img_name}_top{cfg.top_n}_s{cfg.sampling_size}.pkl")
    out_path = os.path.join(OUT_DIR, f"{img_name}.h5ad")

    if os.path.exists(out_path):
        print(f"  [跳过] 已存在 {out_path}")
        import anndata
        a = anndata.read_h5ad(out_path)
        print(f"  细胞数: {a.n_obs:,}")
        return

    from comseg import dataset as comseg_dataset
    from comseg.dictionary import ComSegDict

    # top-n 基因
    top_genes = tx.gene.value_counts().head(cfg.top_n).index.tolist()
    tx_filt = tx[tx.gene.isin(top_genes)].copy()
    print(f"  top-{cfg.top_n} 基因后: {len(tx_filt):,} 转录本")

    # dict_scale（µm 尺度：x_global_px 需要转 µm）
    tx_filt["x_um"] = tx_filt.x_global_px * PIXEL_UM
    tx_filt["y_um"] = tx_filt.y_global_px * PIXEL_UM
    nuc["cx_um"]    = nuc.cx_px * PIXEL_UM
    nuc["cy_um"]    = nuc.cy_px * PIXEL_UM

    dict_scale = {"x": 1.0, "y": 1.0, "z": 1.0}  # 已转 µm，scale=1

    # ComSegDataset（参考 comseg_worker.py 的调用方式）
    print("\n  构建 ComSegDataset ...")
    t0 = time.time()
    # 写临时 CSV（ComSeg 通常从文件读）
    tmp_csv = os.path.join(CACHE_DIR, f"{img_name}_tx.csv")
    tx_filt[["x_um","y_um","gene","cell_id"]].rename(
        columns={"x_um":"x","y_um":"y"}
    ).to_csv(tmp_csv, index=False)

    # 核质心字典 {cell_id: [z,y,x]}（格式与 comseg_worker.py 一致）
    dict_centroid = {}
    for _, row in nuc.iterrows():
        if row.cell_id > 0:
            dict_centroid[row.cell_id] = [0.0,
                                           float(row.cy_um),
                                           float(row.cx_um)]
    dict_centroid[0] = [0.0, 0.0, 0.0]   # 背景哨兵
    print(f"  dict_centroid: {len(dict_centroid):,} 核（含背景）  ({time.time()-t0:.1f}s）")

    # 直接构造 In-memory dataset
    try:
        from comseg.dataset import ComSegDataset
        data = ComSegDataset(
            path_dir_input     = CACHE_DIR,
            dict_scale         = {"x":1.0,"y":1.0,"z":1.0},
            min_transcript_count = 0,
            n_neighbors        = cfg.k_nn,
            sigma              = 3,
        )
        # 注入内存数据（跳过文件扫描）
        data.df_spots = tx_filt[["x_um","y_um","gene","cell_id"]].rename(
            columns={"x_um":"x","y_um":"y"}).copy()
        data.df_spots["z"] = 0.0
        data.img_name  = img_name
        data.mode      = "with_prior"
        print(f"  ComSegDataset 就绪（{time.time()-t0:.1f}s）")
    except Exception as e:
        print(f"  ComSegDataset 构建失败: {e}")
        print("  尝试备用方式...")
        # 备用：直接用 comseg_worker 的 run_one_fov 思路
        _run_via_worker(cfg, tx_filt, dict_centroid, img_name, cache_path, out_path)
        return

    # edge_weight
    if not os.path.exists(cache_path) or cfg.force:
        print(f"\n  compute_edge_weight（sampling={cfg.sampling_size}，约需 10-20 分钟）...")
        t0 = time.time()
        data.compute_edge_weight(
            dict_scale         = dict_scale,
            dict_centroid      = dict_centroid,
            max_cell_radius    = cfg.max_cell_radius,
            sampling_size      = cfg.sampling_size,
            n_neighbors        = cfg.k_nn,
            gene_index         = None,
        )
        import pickle
        with open(cache_path, "wb") as f: pickle.dump(data, f)
        print(f"  edge_weight 完成（{time.time()-t0:.0f}s），已缓存")
    else:
        import pickle
        print(f"  复用 edge_weight 缓存: {os.path.basename(cache_path)}")
        with open(cache_path, "rb") as f: data = pickle.load(f)

    # ComSegDict + run_all
    print("\n  run_all（classify_centroid）...")
    t0 = time.time()
    comseg_dict = ComSegDict(
        dataset         = data,
        dict_centroid   = dict_centroid,
    )
    comseg_dict.run_all(
        gene_selection_mode    = "all",
        dict_scale             = dict_scale,
        max_cell_radius        = cfg.max_cell_radius,
        n_neighbors            = cfg.k_nn,
        image_name             = img_name,
    )
    print(f"  run_all 完成（{time.time()-t0:.0f}s）")

    # 写出
    anndata_obj = comseg_dict.anndata_from_comseg_result(
        dict_scale = dict_scale, gene_selection_mode = "all")
    anndata_obj.write_h5ad(out_path)
    print(f"\n  ✓ 完成！细胞数: {anndata_obj.n_obs:,}  → {out_path}")


def _run_via_worker(cfg, tx_filt, dict_centroid, img_name, cache_path, out_path):
    """备用：把 ROI 数据写成临时文件，借用 comseg_worker 的内部函数"""
    print("  (备用路径) 调用 comseg_worker 内部逻辑...")
    # 这个路径在 comseg 的 API 接口变化时使用
    # 根据实际情况补充


# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = parse_args()
    print(SEP); print("ComSeg × CosMx 胰腺（空间 ROI）"); print(SEP)
    print(f"ROI 大小: {cfg.roi_size}×{cfg.roi_size} px"
          f" = {cfg.roi_size*PIXEL_UM:.0f}×{cfg.roi_size*PIXEL_UM:.0f} µm")

    # 找中心
    print("\n[1] 确定 ROI 中心 ...")
    cx, cy = get_roi_center(META_PATH, TX_PATH)

    # 加载 ROI 数据
    print("\n[2] 加载 ROI 数据 ...")
    tx, nuc, roi_origin = load_roi_data(cfg, cx, cy)

    # 数据量报告
    prod = len(tx) * min(tx.gene.nunique(), cfg.top_n)
    print(f"\n  ROI 内转录本: {len(tx):,}  基因: {tx.gene.nunique()}")
    print(f"  top-{cfg.top_n} 后 n_tx×n_gene ≈ {prod:,}"
          f"  {'✓ 安全' if prod < 1_500_000_000 else '⚠ 较大'}")
    print(f"  核质心: {len(nuc):,}  GT 密度: {len(nuc)/(cfg.roi_size*PIXEL_UM)**2*1e6:.0f} 细胞/mm²")

    if cfg.check_only:
        print("\n  --check-only 模式，到此为止。")
        print("  内存估算（classify_centroid）: "
              f"~{len(tx)*cfg.k_nn*8/1e9:.1f} GB (n_tx × k_nn × 8B)")
        return

    # 运行 ComSeg
    print("\n[3] 运行 ComSeg ...")
    run_comseg(cfg, tx, nuc, roi_origin)


if __name__ == "__main__":
    main()