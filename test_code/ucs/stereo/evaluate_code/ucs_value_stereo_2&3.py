#!/usr/bin/env python
"""
UCS × Stereo-seq 评估

===== 本版修复：预测细胞数不能用 mask.max() =====
之前一版用 int(mask.max()) 当预测细胞数，E14.5 上算出 181,844——
和直接验证时用 len(np.unique(m[m>0])) 算出的 4,870 差了近40倍。
这是标签值稀疏、不连续导致的（UCS 内部按 patch 处理，很可能给每个
patch 里的细胞分配了全局唯一但不连续的原始ID，没有最后重新编号成
1..N）。这个错误在这次会话里已经在 GeneSegNet 自检脚本上出现过一次，
这次踩的是同一个坑：数一个标签图里有多少个物体，必须用去重计数或者
len(regionprops(mask))，永远不能用 max()。

★ 受影响范围：precision/F1/cell_ratio 之前是错的（分母被夸大40倍，
  precision 被压到接近0）。recall/count_pearson/vec_cosine/
  assign_accuracy 不受影响，因为那几项全部通过 regionprops 枚举出的
  真实标签值（pred_labels）索引，不经过 mask.max()。

===== 循环性说明（关键，务必标注）=====
UCS 的 nuclei_mask 直接由 GEM 的 cell 列栅格化而来——先验本身就是 GT。
本评估的 precision/recall/F1 会因此虚高，不能和 Cellist（先验独立于
CellBin）放在同一张榜单里比较。这一点和 MERFISH/CosMx 上的 UCS
结果是同一种性质的问题。

用法：
  python eval_ucs_stereo.py --dataset E14.5_E1S3
  python eval_ucs_stereo.py --dataset E14.5_E1S3 --mask-path /path/to/segmentation_mask.tif
"""
import os
import argparse
import warnings

import numpy as np
import pandas as pd
import tifffile
import skimage.measure
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")
SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

DATA_DIR = "/data/qiuyijia/dataset/stereoseq/stereo-seq_data_all"
OUT_CSV = "/data/qiuyijia/eval_results/ucs/stereo.csv"

DATASETS = {
    "E16.5_E2S6": dict(
        gem=f"{DATA_DIR}/E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        mask="/data/qiuyijia/ucs_stereoseq/ucs_log/pred/segmentation_mask.tif",
        gt_cells=6229),
    "E14.5_E1S3": dict(
        gem=f"{DATA_DIR}/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        mask="/data/qiuyijia/ucs_stereoseq_E14.5/ucs_log/pred/segmentation_mask.tif",
        gt_cells=4872),
    "E16.5_E2S7": dict(
        gem=f"{DATA_DIR}/E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        mask="/data/qiuyijia/ucs_stereoseq_E2S7/ucs_log/pred/segmentation_mask.tif",
        gt_cells=7960),
}


def load_gem(dataset):
    gem_path = DATASETS[dataset]["gem"]
    P(f"  读 GEM {os.path.basename(gem_path)} ...")
    gem = pd.read_csv(gem_path, sep="\t", compression="gzip")
    ren = {}
    for c in gem.columns:
        cl = c.lower()
        if cl in ("geneid", "gene"):
            ren[c] = "geneID"
        elif cl == "x":
            ren[c] = "x"
        elif cl == "y":
            ren[c] = "y"
        elif cl in ("midcounts", "midcount", "count"):
            ren[c] = "MIDCounts"
        elif cl in ("cell", "cellid", "label"):
            ren[c] = "cell"
    gem = gem.rename(columns=ren)
    gem["geneID"] = gem["geneID"].astype("category")
    x0, y0 = int(gem.x.min()), int(gem.y.min())
    P(f"    {len(gem):,} 行  {gem.geneID.nunique():,} 基因  origin=({x0},{y0})")
    inc = gem.cell > 0
    P(f"    CellBin 细胞 {gem.loc[inc,'cell'].nunique():,}")
    return gem, x0, y0


def gt_centroids(gem, resolution):
    g = gem[gem.cell > 0]
    cen = g.groupby("cell", observed=True).agg(
        cx=("x", "mean"), cy=("y", "mean"),
        n_row=("x", "size"), n_umi=("MIDCounts", "sum")).reset_index()
    P(f"\n  GT {len(cen):,} 细胞  每细胞spot中位 {cen.n_row.median():.0f}")
    return cen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="E14.5_E1S3", choices=list(DATASETS))
    ap.add_argument("--mask-path", default=None)
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--match-radius-um", type=float, default=13.0)
    ap.add_argument("--vec-sample", type=int, default=2000)
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()

    P(SEP); P(f"UCS × Stereo-seq 评估 —— {a.dataset}"); P(SEP)
    P(f"  ★★ 循环性警告：nuclei_mask 直接由 GEM 的 cell 列栅格化而来，"
      f"先验本身就是GT。以下 precision/recall/F1 会虚高，不能和")
    P(f"     Cellist（先验独立于CellBin）放在同一张榜单里比较")

    mask_path = a.mask_path or DATASETS[a.dataset]["mask"]
    if not os.path.exists(mask_path):
        P(f"\n  ✘ 找不到 mask: {mask_path}")
        return
    mask = tifffile.imread(mask_path)
    P(f"\n  mask: {mask_path}")
    P(f"    shape={mask.shape}")

    # ── ★ 修复：细胞数用 regionprops 枚举真实标签，不用 mask.max() ──────
    props = skimage.measure.regionprops(mask)
    n_pred = len(props)
    n_pred_maxval = int(mask.max())
    P(f"    真实标签数(regionprops) = {n_pred:,}")
    if n_pred_maxval != n_pred:
        P(f"    ⚠ mask.max() = {n_pred_maxval:,} —— 与真实标签数不一致，"
          f"确认标签值本身是稀疏的（不是1..N连续编号），"
          f"这也是之前一版脚本算出预测数181,844(E14.5)那个错误的来源，"
          f"这版已改用真实标签数")

    gem, x0, y0 = load_gem(a.dataset)
    cen = gt_centroids(gem, a.resolution)
    n_gt = len(cen)

    pred_bin_xy = np.array([[x0 + p.centroid[1], y0 + p.centroid[0]]
                            for p in props])
    pred_labels = np.array([p.label for p in props])

    R_bin = a.match_radius_um / a.resolution
    gxy = cen[["cx", "cy"]].values
    tree = cKDTree(gxy)
    dd, gi = tree.query(pred_bin_xy, k=1, distance_upper_bound=R_bin)
    best = {}
    for pi in range(len(pred_bin_xy)):
        if not np.isfinite(dd[pi]) or dd[pi] >= R_bin:
            continue
        g = int(gi[pi])
        if g not in best or dd[pi] < best[g][1]:
            best[g] = (pi, dd[pi])
    pairs = [(pi, g, x) for g, (pi, x) in best.items()]
    n_match = len(pairs)
    prec = n_match / n_pred if n_pred else 0.0
    rec = n_match / n_gt if n_gt else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    sh_um = (np.array([p[2] for p in pairs]) * a.resolution
            if pairs else np.array([np.nan]))
    P(f"\n  匹配 {n_match:,} / GT {n_gt:,} / 预测 {n_pred:,}")
    P(f"  P={prec:.4f} R={rec:.4f} F1={f1:.4f}  "
      f"(预期都很高，因为先验=GT——这次分母是对的，数字才有意义)")

    H, W = mask.shape
    gx = (gem.x.values - x0).astype(np.int64)
    gy = (gem.y.values - y0).astype(np.int64)
    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    pred_label_per_row = np.zeros(len(gem), np.int64)
    pred_label_per_row[ok] = mask[gy[ok], gx[ok]]

    umi = gem.MIDCounts.values
    dfp = pd.DataFrame({"lab": pred_label_per_row, "u": umi})
    pcnt = dfp[dfp.lab > 0].groupby("lab")["u"].sum()
    gcnt = gem[gem.cell > 0].groupby("cell", observed=True)["MIDCounts"].sum()
    pc = np.array([pcnt.get(pred_labels[p[0]], 0) for p in pairs], float)
    gc = np.array([gcnt.get(cen.cell.iloc[p[1]], 0) for p in pairs], float)
    m = (pc > 0) & (gc > 0)
    cp = float(pearsonr(pc[m], gc[m])[0]) if m.sum() > 10 else np.nan
    cs = float(spearmanr(pc[m], gc[m])[0]) if m.sum() > 10 else np.nan
    cm_ = float(np.mean(np.abs(pc[m]-gc[m]))) if m.sum() > 10 else np.nan
    cr = float(np.sqrt(np.mean((pc[m]-gc[m])**2))) if m.sum() > 10 else np.nan
    P(f"  计数相关 Pearson={cp}  (预期也很高，同样是先验泄漏，不受"
      f"上面那个bug影响，因为这里一直用的是pred_labels不是mask.max())")

    rng = np.random.default_rng(0)
    samp = pairs if len(pairs) <= a.vec_sample else \
        [pairs[i] for i in rng.choice(len(pairs), a.vec_sample, replace=False)]
    sp_labels = {pred_labels[p[0]] for p in samp}
    sg_ids = {cen.cell.iloc[p[1]] for p in samp}
    pv_tbl = (pd.DataFrame({"lab": pred_label_per_row, "gene": gem.geneID.values,
                           "u": umi})
             [lambda d: d.lab.isin(sp_labels)]
             .groupby(["lab", "gene"], observed=True)["u"].sum().unstack(fill_value=0))
    gv_tbl = (gem[gem.cell.isin(sg_ids)]
             .groupby(["cell", "geneID"], observed=True)["MIDCounts"].sum()
             .unstack(fill_value=0))
    common = sorted(set(pv_tbl.columns) & set(gv_tbl.columns))
    vc = vj = vp = np.nan
    n_valid = 0
    if len(common) >= 20:
        cos_l, js_l, pr_l = [], [], []
        for pi, gj, _ in samp:
            lb, cid = pred_labels[pi], cen.cell.iloc[gj]
            if lb not in pv_tbl.index or cid not in gv_tbl.index:
                continue
            pvv = pv_tbl.loc[lb, common].values.astype(float)
            gvv = gv_tbl.loc[cid, common].values.astype(float)
            if pvv.sum() < 5 or gvv.sum() < 5:
                continue
            cos_l.append(float(pvv @ gvv / (np.linalg.norm(pvv)*np.linalg.norm(gvv))))
            js_l.append(float(jensenshannon(pvv/pvv.sum(), gvv/gvv.sum())))
            if pvv.std() > 0 and gvv.std() > 0:
                pr_l.append(float(pearsonr(pvv, gvv)[0]))
        n_valid = len(cos_l)
        if n_valid > 10:
            vc, vj = float(np.mean(cos_l)), float(np.nanmean(js_l))
            vp = float(np.mean(pr_l)) if pr_l else np.nan
    P(f"  基因向量 cosine={vc}  有效对 {n_valid:,}")

    row2gt = {pred_labels[p[0]]: cen.cell.iloc[p[1]] for p in pairs}
    exp = pd.Series(pred_label_per_row).map(row2gt)
    has_gt = gem.cell.values > 0
    both_m = has_gt & exp.notna().values & (pred_label_per_row > 0)
    w_both = float(umi[both_m].sum())
    if w_both > 0:
        agree = exp[both_m].values == gem.cell.values[both_m]
        overlap = w_both / float(umi[pred_label_per_row > 0].sum())
        acc = float(umi[both_m][agree].sum() / w_both)
    else:
        overlap = acc = np.nan
    P(f"  分配一致性 accuracy={acc}  (预期≈1.0，因为mask本身就是cell列栅格化的)")

    rnd = lambda v, n=4: round(float(v), n) if v == v else np.nan
    res = dict(
        method="UCS", dataset=a.dataset,
        gt_cells=n_gt, pred_cells=n_pred,
        cell_ratio=rnd(n_pred/n_gt), matched=n_match,
        precision=rnd(prec), recall=rnd(rec), f1=rnd(f1),
        loc_mean_um=rnd(np.nanmean(sh_um)),
        loc_median_um=rnd(np.nanmedian(sh_um)),
        loc_p95_um=rnd(np.nanpercentile(sh_um, 95)),
        count_pearson=rnd(cp), count_spearman=rnd(cs),
        count_mae=rnd(cm_, 2), count_rmse=rnd(cr, 2),
        vec_cosine=rnd(vc), vec_js_dist=rnd(vj), vec_pearson=rnd(vp),
        vec_valid_pairs=n_valid,
        assign_overlap=rnd(overlap), assign_accuracy=rnd(acc),
    )

    P(f"\n{SEP}\n结果 (UCS × {a.dataset})  ⚠ 循环，先验=GT\n{SEP}")
    for k, v in res.items():
        if k not in ("method", "dataset"):
            P(f"  {k:22} {v}")

    out_ds = a.out.replace(".csv", f"_{a.dataset}.csv")
    os.makedirs(os.path.dirname(out_ds), exist_ok=True)
    pd.DataFrame([res]).to_csv(out_ds, index=False)
    P(f"\n  已写 → {out_ds}")


if __name__ == "__main__":
    main()