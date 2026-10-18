#!/usr/bin/env python
"""
ComSeg × CosMx 胰腺 —— 统一版（with-prior / no-transcript-prior 合并一个文件）

===== 为什么合并成一个文件 =====
之前分成两份独立脚本（带先验的原版 / 无转录本先验版），先后两次
出现"往哪个文件名里存了哪份内容"搞混的情况。这次合并成一个文件，
用 --no-transcript-prior 这一个开关控制模式，不再需要记住文件名
对应关系——只有一份文件，永远清楚现在跑的是哪种模式。

===== 两种模式的区别 =====
· 默认（with-prior）：转录本 CSV 的 cell_id 列带 CosMx 平台真实分割
  结果。已确认：cell_id>0 的转录本会被 100% 原样传回，不参与任何
  重新分配（with_prior 模式的设计行为，不是bug）。这种模式的
  F1/count_pearson/vec_cosine 不能当作检测能力的证据，只反映它
  挽回背景（cell_id=0）转录本的能力。
· --no-transcript-prior：转录本 CSV 的 cell_id 全部置 0，只保留
  质心地标位置，全部转录本走自由社区检测。三次运行（含两次
  with-prior）都收敛到恰好3个簇，怀疑 min_merge_correlation
  默认值(0.8)太宽松，这版把它暴露成 CLI 参数可以直接调。

===== 两种模式的缓存目录完全分开，互不污染 =====
with-prior:        cache_roi / cache_roi_centroid_only / output_roi
no-transcript-prior: cache_roi_noprior / cache_roi_noprior_centroid_only /
                     output_roi_noprior

===== cell_id=0 哨兵：只在 with-prior 模式需要 =====
已用两次成功的 no-prior 运行验证：全部转录本统一是 cell_id=0 时，
classify_centroid/associate_rna2landmark 不需要专门给 id=0 准备
一个地标——这个哨兵是 with-prior 模式特有的需求（该模式下部分转
录本有真实先验、部分是背景，两种身份混合存在时才用得到）。
no-transcript-prior 模式不再调用这个逻辑。

===== 核心历史修复（两种模式都保留）=====
· import pickle 必须在文件顶层
· run_all 前无条件清空 dict_centroid（classify_centroid 的前置断言要求）
· classify_centroid 真正读取的文件名规则是 {img_name}.csv，必须放在
  独立目录，不能和转录本 CSV 同目录（否则会把转录本表误读成质心表）
· anndata_from_comseg_result 返回二元组，需解包
· return_polygon=False（避免ROI边缘共线转录本让qhull崩溃）

用法：
  python comseg_comsx_roi.py                                        # with-prior,mmc默认0.8
  python comseg_comsx_roi.py --no-transcript-prior                  # no-prior,mmc默认0.8
  python comseg_comsx_roi.py --no-transcript-prior --min-merge-correlation 0.95
  python comseg_comsx_roi.py --status
  python comseg_comsx_roi.py --no-transcript-prior --status
"""
import os
import sys
import time
import glob
import pickle          # ★ 必须在文件顶层
import inspect
import argparse
import subprocess

import numpy as np
import pandas as pd

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

TX_FILE = ("/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles/"
           "Pancreas_tx_file.csv")
PIXEL_SIZE = 0.1203
CELL_DIAM = 10.0
ROI_SIZE = 5000


def get_dirs(no_prior):
    """★ 两种模式的目录完全分开，避免互相污染"""
    if no_prior:
        base = "/data/qiuyijia/comseg"
        d = dict(cache=f"{base}/cache_roi_noprior",
                 centroid=f"{base}/cache_roi_noprior_centroid_only",
                 out=f"{base}/output_roi_noprior")
    else:
        base = "/data/qiuyijia/comseg"
        d = dict(cache=f"{base}/cache_roi",
                 centroid=f"{base}/cache_roi_centroid_only",
                 out=f"{base}/output_roi")
    for v in d.values():
        os.makedirs(v, exist_ok=True)
    return d


def safe_kwargs(fn, want, label=""):
    try:
        params = set(inspect.signature(fn).parameters.keys())
    except (TypeError, ValueError):
        return want
    use = {k: v for k, v in want.items() if k in params}
    drop = [k for k in want if k not in params]
    if label:
        P(f"    {label} 传入: {sorted(use.keys())}"
          + (f"   丢弃(此版本不支持): {drop}" if drop else ""))
    return use


def kill_dict_centroid(*objs):
    """run_all 前无条件清空 dict_centroid（classify_centroid 的前置断言要求）"""
    for owner in objs:
        if owner is None:
            continue
        targets = [owner]
        ds = getattr(owner, "dataset", None)
        if ds is not None:
            targets.append(ds)
        for t in targets:
            cur = getattr(t, "dict_centroid", "···NO_ATTR···")
            if cur is None or cur == "···NO_ATTR···":
                continue
            n = len(cur) if hasattr(cur, "__len__") else "?"
            P(f"    置 {type(t).__name__}.dict_centroid = None（原有 {n} 项）")
            t.dict_centroid = None


# ══════════════════════════════════════════════════════════════════════════
#  Step 1: ROI 数据准备（两种模式共用中心计算，内容按模式分叉）
# ══════════════════════════════════════════════════════════════════════════
def find_roi_center():
    P("  估 ROI 中心（random_state=0 保证跨次运行一致）...")
    xs, ys = [], []
    for ck in pd.read_csv(TX_FILE, chunksize=2_000_000,
                          usecols=["x_global_px", "y_global_px"]):
        xs.append(ck.x_global_px.sample(min(50_000, len(ck)), random_state=0))
        ys.append(ck.y_global_px.sample(min(50_000, len(ck)), random_state=0))
        if len(xs) >= 8:
            break
    cx = float(pd.concat(xs).median())
    cy = float(pd.concat(ys).median())
    h = ROI_SIZE / 2
    return cx - h, cx + h, cy - h, cy + h


def sentinel_position(nuc, max_cell_radius):
    x0, x1 = float(nuc.x.min()), float(nuc.x.max())
    y0, y1 = float(nuc.y.min()), float(nuc.y.max())
    dx = max(x0 - 0.0, 0.0, 0.0 - x1)
    dy = max(y0 - 0.0, 0.0, 0.0 - y1)
    dist = (dx ** 2 + dy ** 2) ** 0.5
    safe_margin = max_cell_radius * 5
    if dist >= safe_margin:
        return 0.0, 0.0, dist, True
    sx = x0 - safe_margin * 4
    sy = y0 - safe_margin * 4
    return sx, sy, dist, False


def ensure_sentinel(nuc, max_cell_radius):
    """
    只在 with-prior 模式调用。已用两次成功的 no-prior 运行验证：
    全部转录本统一 cell_id=0 时不需要这个哨兵。
    """
    has_zero = bool((nuc.cell_id == 0).any())
    if has_zero:
        P(f"    ⚡ cell_id=0 哨兵已存在，无需处理")
        return nuc, False
    sx, sy, dist, used_origin = sentinel_position(nuc, max_cell_radius)
    where = "原点 (0,0)" if used_origin else f"ROI 边界外 ({sx:.0f},{sy:.0f})"
    P(f"    ★ 缺 cell_id=0 哨兵（约 9.7% 的转录本无核归属，仅 with-prior "
      f"模式需要）")
    P(f"      放置于 {where}，距最近真实核 {dist:.0f}µm")
    row = {"cell_id": 0, "x": sx, "y": sy, "z": 0.0}
    nuc2 = pd.concat([pd.DataFrame([row]), nuc], ignore_index=True)
    nuc2["cell_id"] = nuc2.cell_id.astype(int)
    return nuc2, True


def prepare_data(cfg, dirs):
    x0, x1, y0, y1 = find_roi_center()
    suffix = "_noprior" if cfg.no_transcript_prior else ""
    tag = f"cosmx_roi_{int(x0)}_{int(y0)}{suffix}"
    txp = f"{dirs['cache']}/{tag}_tx.csv"
    cenp = f"{dirs['cache']}/{tag}_centroid.csv"

    if os.path.exists(txp) and not cfg.force:
        tx = pd.read_csv(txp)
        P(f"  ⚡ 复用转录本 CSV: {len(tx):,} 条  列={list(tx.columns)}")
    else:
        P(f"  ROI x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}] px")
        P("  读转录本（约 3-5 分钟）...")
        chunks, total = [], 0
        for ck in pd.read_csv(TX_FILE, chunksize=2_000_000,
                              usecols=["x_global_px", "y_global_px",
                                       "target", "cell_ID", "fov"]):
            total += len(ck)
            sub = ck[(ck.x_global_px >= x0) & (ck.x_global_px < x1) &
                     (ck.y_global_px >= y0) & (ck.y_global_px < y1)]
            if len(sub):
                chunks.append(sub)
            print(f"    已读 {total:,}", end="\r", flush=True)
        roi = pd.concat(chunks, ignore_index=True)
        P(f"\n    ROI 内 {len(roi):,}")

        roi["cell_uid"] = np.where(
            roi.cell_ID > 0,
            roi.fov.astype(str) + "_" + roi.cell_ID.astype(str), "")
        top = roi.target.value_counts().head(cfg.top_n).index
        roi = roi[roi.target.isin(top)]
        P(f"    top-{cfg.top_n} 基因后: {len(roi):,} 条")

        uid2id = dict(zip(
            sorted(roi.loc[roi.cell_uid != "", "cell_uid"].unique()),
            range(1, roi.cell_uid.nunique() + 1)))
        landmark_id = roi.cell_uid.map(uid2id).fillna(0).astype(int)

        if cfg.no_transcript_prior:
            # ★ cell_id 全部置 0，_landmark_for_centroid 单独保留用于建质心
            tx = pd.DataFrame({
                "x": roi.x_global_px.values * PIXEL_SIZE,
                "y": roi.y_global_px.values * PIXEL_SIZE,
                "gene": roi.target.values,
                "cell_id": 0,
                "_landmark_for_centroid": landmark_id.values,
            })
            P(f"    ✓ {os.path.basename(txp)}  （cell_id 全部为 0，"
              f"无转录本先验模式）")
        else:
            tx = pd.DataFrame({
                "x": roi.x_global_px.values * PIXEL_SIZE,
                "y": roi.y_global_px.values * PIXEL_SIZE,
                "gene": roi.target.values,
                "cell_id": landmark_id.values,
            })
            P(f"    ✓ {os.path.basename(txp)}   核内 "
              f"{int((tx.cell_id > 0).sum()):,}/{len(tx):,} "
              f"({(tx.cell_id > 0).mean()*100:.1f}%)")
        tx.to_csv(txp, index=False)

    landmark_col = ("_landmark_for_centroid"
                    if cfg.no_transcript_prior else "cell_id")
    if landmark_col not in tx.columns:
        sys.exit(f"✘ 缓存的 {txp} 缺 {landmark_col} 列，请加 --force 重建")

    if os.path.exists(cenp) and not cfg.force:
        nuc = pd.read_csv(cenp)
        P(f"  ⚡ 复用质心表: {len(nuc):,} 行")
    else:
        sub = tx[tx[landmark_col] > 0]
        g = sub.groupby(landmark_col)[["x", "y"]].mean().reset_index()
        g = g.rename(columns={landmark_col: "cell_id"})
        g["z"] = g.cell_id * 0.001   # 各不相同，避免 qhull 2D 退化
        nuc = g[["cell_id", "x", "y", "z"]].copy()
        nuc.to_csv(cenp, index=False)
        P(f"  ✓ 质心表: {len(nuc):,} 个地标")

    if not cfg.no_transcript_prior:
        nuc, changed = ensure_sentinel(nuc, cfg.max_cell_radius)
        if changed:
            nuc.to_csv(cenp, index=False)
            P(f"    ✓ 已写回 {os.path.basename(cenp)}（{len(nuc):,} 行）")

    return tag, txp, cenp, nuc


def write_centroid_for_classify(nuc, img_name, dirs):
    """
    classify_centroid() 真正读取的路径规则是 {path_dataset_folder_centroid}/
    {img_name}.csv，img_name 是转录本文件去掉.csv后的主干名——和
    ComSegDataset.__init__ 的 centroid_csv_files 参数完全无关。
    必须放独立目录，否则文件名恰好等于转录本文件本身，会被误读
    （转录本表没有z列，构造出退化的2键字典，KeyError:0）。
    """
    out = f"{dirs['centroid']}/{img_name}.csv"
    nuc.to_csv(out, index=False)
    P(f"    ✓ classify_centroid 专用质心文件 → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Step 2-4: ComSeg 主流程
# ══════════════════════════════════════════════════════════════════════════
def run_comseg(tag, txp, cenp, nuc, cfg, dirs):
    from comseg.dataset import ComSegDataset
    from comseg.dictionary import ComSegDict

    img_name = os.path.basename(txp).replace(".csv", "")
    mmc_tag = str(cfg.min_merge_correlation).replace(".", "p")
    ck_pkl = f"{dirs['cache']}/comsegdict_{tag}_mmc{mmc_tag}.pkl"

    if cfg.resume and os.path.exists(ck_pkl):
        P(f"\n[恢复] 从 pkl 加载 ComSegDict（跳过 run_all）")
        with open(ck_pkl, "rb") as f:
            comseg_dict = pickle.load(f)
        return comseg_dict, img_name

    P(f"\n[2] ComSegDataset")
    ds_kw = safe_kwargs(ComSegDataset.__init__, dict(
        path_dataset_folder=dirs["cache"],
        prior_name="cell_id",
        image_csv_files=[os.path.basename(txp)],
        centroid_csv_files=[os.path.basename(cenp)],
        path_cell_centroid=dirs["cache"],
        dict_scale={"x": 1, "y": 1, "z": 1},
        mean_cell_diameter=CELL_DIAM,
        gene_column="gene",
    ), "ComSegDataset")
    t0 = time.time()
    data = ComSegDataset(**ds_kw)
    P(f"    就绪 ({time.time()-t0:.1f}s)   list_index: {data.list_index}")

    ew_pkl = (f"{dirs['cache']}/edge_{img_name}"
              f"_top{cfg.top_n}_s{cfg.sampling_size}.pkl")
    if os.path.exists(ew_pkl) and not cfg.force:
        P(f"\n[3] ⚡ 复用 edge_weight: {os.path.basename(ew_pkl)}")
        with open(ew_pkl, "rb") as f:
            data = pickle.load(f)
    else:
        P(f"\n[3] compute_edge_weight（sampling={cfg.sampling_size}）...")
        ew_kw = safe_kwargs(data.compute_edge_weight, dict(
            sampling=True, sampling_size=cfg.sampling_size,
            n_neighbors=cfg.n_neighbors,
        ), "compute_edge_weight")
        t0 = time.time()
        data.compute_edge_weight(**ew_kw)
        P(f"    ✓ 完成 ({time.time()-t0:.0f}s)")
        with open(ew_pkl, "wb") as f:
            pickle.dump(data, f, protocol=4)
        P(f"    ✓ 缓存 → {os.path.basename(ew_pkl)}")

    P(f"\n[4] ComSegDict")
    dd_kw = safe_kwargs(ComSegDict.__init__, dict(
        dataset=data, mean_cell_diameter=CELL_DIAM,
        community_detection="with_prior",   # ★ 已确认这版 ComSeg 底层
                                             #   community_vector() 硬编码
                                             #   assert clustering_method
                                             #   in ["with_prior"]，其它
                                             #   选项在文档里存在但没实现
    ), "ComSegDict")
    comseg_dict = ComSegDict(**dd_kw)

    P(f"\n[5] 清除 dict_centroid（classify_centroid 的前置断言要求）")
    kill_dict_centroid(data, comseg_dict)

    P(f"\n[6] 写入 classify_centroid 专用质心文件（独立目录，避免同名误读）")
    write_centroid_for_classify(nuc, img_name, dirs)

    P(f"\n[7] run_all")
    P(f"    min_merge_correlation={cfg.min_merge_correlation}"
      f"  resolution={cfg.resolution}")
    P(f"    实测耗时跨度很大：原版≈220分钟，也见过102分钟——"
      f"取决于聚类规模，不要拿单次经验去判断是否卡死，"
      f"用 ps 看 TIME 字段是否还在增长才是准的")
    ra_kw = safe_kwargs(comseg_dict.run_all, dict(
        max_cell_radius=cfg.max_cell_radius,
        k_nearest_neighbors=cfg.k_nn, n_neighbors=cfg.n_neighbors,
        path_dataset_folder_centroid=dirs["centroid"],
        min_merge_correlation=cfg.min_merge_correlation,
        resolution=cfg.resolution,
    ), "run_all")
    t0 = time.time()
    comseg_dict.run_all(**ra_kw)
    el = time.time() - t0
    P(f"    ✓ run_all 完成 ({el:.0f}s = {el/60:.1f} 分钟)")

    try:
        with open(ck_pkl, "wb") as f:
            pickle.dump(comseg_dict, f, protocol=4)
        P(f"    ✓ ComSegDict 落盘 → {os.path.basename(ck_pkl)}")
    except Exception as e:
        P(f"    ⚠ 落盘失败({e})，继续")

    return comseg_dict, img_name


def export(comseg_dict, tag, nuc, cfg, dirs):
    P(f"\n[8] 导出 anndata")
    ad_kw = safe_kwargs(comseg_dict.anndata_from_comseg_result, dict(
        min_rna_per_cell=cfg.min_rna_per_cell, return_polygon=False,
    ), "anndata_from_comseg_result")
    ret = comseg_dict.anndata_from_comseg_result(**ad_kw)
    adata = ret[0] if isinstance(ret, tuple) else ret
    P(f"    返回类型 {type(ret).__name__}"
      f"{'（二元组，取[0]）' if isinstance(ret, tuple) else ''}")

    if adata is None or getattr(adata, "n_obs", 0) == 0:
        P("    ✘ anndata 为空"); return None

    mmc_tag = str(cfg.min_merge_correlation).replace(".", "p")
    out = f"{dirs['out']}/{tag}_mmc{mmc_tag}.h5ad"
    adata.write_h5ad(out)

    P(f"\n{SEP}")
    P(f"  ✓ {adata.n_obs:,} 细胞 × {adata.n_vars} 基因")
    P(f"  → {out}")

    df_spots = adata.uns.get("df_spots")
    if df_spots:
        key = list(df_spots.keys())[0]
        n_clusters = df_spots[key].leiden_merged.nunique()
        P(f"\n  ★ leiden_merged 实际簇数: {n_clusters}"
          f"（此前三次运行不管初始簇数多少，合并后都恰好是3——"
          f"这次 min_merge_correlation={cfg.min_merge_correlation}，"
          f"看这个数字有没有变化）")
    return out


def show_status(no_prior):
    dirs = get_dirs(no_prior)
    mode = "无转录本先验" if no_prior else "带先验(原版)"
    P(f"\n{SEP}\n状态检查 —— {mode}模式\n{SEP}")

    p = subprocess.run(["pgrep", "-f", "comseg_comsx_roi"],
                       capture_output=True, text=True).stdout.strip()
    P(f"  运行中的进程: {'PID ' + p.replace(chr(10), ', ') if p else '（无）'}")

    P(f"\n  ── 缓存 {dirs['cache']} ──")
    for f in sorted(glob.glob(f"{dirs['cache']}/*")):
        sz = os.path.getsize(f) / 1e6
        P(f"    {sz:9.2f} MB   {os.path.basename(f)}")

    P(f"\n  ── 输出 {dirs['out']} ──")
    outs = sorted(glob.glob(f"{dirs['out']}/*.h5ad"))
    if not outs:
        P(f"    （无 .h5ad 输出）")
    for f in outs:
        try:
            import anndata
            a = anndata.read_h5ad(f)
            P(f"    ✓ {a.n_obs:,} 细胞 × {a.n_vars} 基因   "
              f"{os.path.basename(f)}")
        except Exception as e:
            P(f"    ⚠ {os.path.basename(f)}（读取失败 {e}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-transcript-prior", action="store_true",
                    help="转录本cell_id全部置0，只保留质心地标位置")
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--sampling-size", type=int, default=2000)
    ap.add_argument("--k-nn", type=int, default=2)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--max-cell-radius", type=float, default=15.0)
    ap.add_argument("--min-rna-per-cell", type=int, default=5)
    ap.add_argument("--min-merge-correlation", type=float, default=0.8,
                    help="run_all原始默认值。三次运行都收敛到3个簇后"
                         "怀疑这个值太宽松，可以尝试0.95/0.99")
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    cfg = ap.parse_args()

    dirs = get_dirs(cfg.no_transcript_prior)

    if cfg.status:
        show_status(cfg.no_transcript_prior)
        return

    mode = "无转录本先验" if cfg.no_transcript_prior else "带先验(原版)"
    P(SEP); P(f"ComSeg × CosMx 胰腺 —— {mode}模式"); P(SEP)
    P(f"  min_merge_correlation={cfg.min_merge_correlation}  "
      f"resolution={cfg.resolution}")
    P(f"  缓存目录: {dirs['cache']}")

    P(f"\n[1] 数据准备")
    tag, txp, cenp, nuc = prepare_data(cfg, dirs)

    mmc_tag = str(cfg.min_merge_correlation).replace(".", "p")
    out = f"{dirs['out']}/{tag}_mmc{mmc_tag}.h5ad"
    if os.path.exists(out) and not cfg.force and not cfg.resume:
        import anndata
        a = anndata.read_h5ad(out)
        P(f"\n  ✓ 已有结果: {a.n_obs:,} 细胞  →  {out}")
        P(f"    重算请加 --force，或 --resume 只重跑导出步骤")
        return

    if cfg.check_only:
        P(f"\n  --check-only：核 {int((nuc.cell_id > 0).sum()):,}")
        return

    comseg_dict, img_name = run_comseg(tag, txp, cenp, nuc, cfg, dirs)
    export(comseg_dict, tag, nuc, cfg, dirs)


if __name__ == "__main__":
    main()