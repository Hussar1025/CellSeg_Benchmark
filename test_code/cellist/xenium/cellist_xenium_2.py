#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cellist × Xenium 乳腺（xenium_2）—— 以官方 nucleus_boundaries 为核先验
======================================================================
ROI：全片中心 5000×5000 px = 1062.5×1062.5 µm，起点像素 (24497, 11210)

历史问题与修复：
  1. 轴序：skimage 的 centroid-0 是 row(y)，Cellist 却当成 x。不交换会让核与
     spot 落进不同 patch，3×3 网格只有对角线附近能对上，实测同 patch 仅 36.8%，
     产出 5 个 Data_HVG patch、2,497 个细胞，空间上呈"左上+右下有数据"的形态。
     build_inputs 里已交换。
  2. 种子：早期版本现场跑 Cellpose（diameter=30px=6.4µm，偏小）只得 3,392 核，
     导致最终仅 955 个细胞。本版改用 nucleus_boundaries 栅格化的 8,390 个核，
     与 GT 8,809 高度吻合。注意输出文件名用 nucprior_* 前缀，与旧版的
     nucleus_* 区分开，避免下游脚本读错。
  3. 输入路径：早期版本曾把 None 传给 --nucleus-prop 才报错，本版在调用前逐个校验。
  4. 完整性：跑完硬校验 patch 数、tmp/Sub 数、以及 10×10 网格是否存在空洞，
     任一不过就不返回结果，防止残缺产物流入下游。

用法：
  python cellist_xenium_2.py --check-only               # 只做 nSpot 预估
  python cellist_xenium_2.py --cell-radius 30 --force   # 正式跑
  python cellist_xenium_2.py --scan-radii 20,30,40 --force
"""

import os, glob, argparse, subprocess
import numpy as np
import pandas as pd
import cv2
import h5py
import tifffile
import skimage.measure
from scipy.sparse import csr_matrix

SEP = "=" * 68
P = lambda *a: print(*a, flush=True)

DATA_DIR = "/data/qiuyijia/dataset/xenium_breast"
OUT_DIR  = "/data/qiuyijia/cellist/xenium_breast"
PREFIX   = "xenium_breast"

PIXEL_SIZE = 0.2125
CX_UM = 53994 / 2 * PIXEL_SIZE
CY_UM = 27420 / 2 * PIXEL_SIZE
HALF  = 5000 * PIXEL_SIZE / 2
ROI_X0, ROI_X1 = CX_UM - HALF, CX_UM + HALF
ROI_Y0, ROI_Y1 = CY_UM - HALF, CY_UM + HALF
IMG_W = IMG_H = 5000
GT_CELLS  = 8809
NSPOT_MIN = 20        # Segmentation.py:287 硬编码
EXPECT_PATCHES = 9    # 3×3 网格

os.makedirs(OUT_DIR, exist_ok=True)


def write_10x_h5(path, mat, genes, barcodes):
    """CSC，shape=(n_genes,n_cells)，indptr 按列/细胞切分"""
    m = mat.tocsc()
    assert len(m.indptr) == len(barcodes)+1, \
        f"indptr {len(m.indptr)} != n_cells+1 {len(barcodes)+1}"
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data",    data=m.data.astype(np.int32))
        g.create_dataset("indices", data=m.indices.astype(np.int64))
        g.create_dataset("indptr",  data=m.indptr.astype(np.int64))
        g.create_dataset("shape",
                         data=np.array([len(genes), len(barcodes)], np.int64))
        g.create_dataset("barcodes",
                         data=np.array([str(b).encode() for b in barcodes]))
        fg = g.create_group("features")
        gb = np.array([str(x).encode() for x in genes])
        fg.create_dataset("name", data=gb)
        fg.create_dataset("id",   data=gb)
        fg.create_dataset("feature_type",
                          data=np.array([b"Gene Expression"]*len(genes)))
        fg.create_dataset("genome", data=np.array([b"unknown"]*len(genes)))


# ══════════════════════════════════════════════════════════════════════════
#  ① nucleus_boundaries → mask
# ══════════════════════════════════════════════════════════════════════════
def build_mask(force=False):
    mp = f"{OUT_DIR}/{PREFIX}_nucprior_mask.tif"
    if os.path.exists(mp) and not force:
        m = tifffile.imread(mp)
        P(f"[1] ⚡ 复用 mask: {int(m.max()):,} 核")
        return m
    P("[1] nucleus_boundaries → mask")
    nb = pd.read_parquet(f"{DATA_DIR}/nucleus_boundaries.parquet")
    P(f"    全片 {len(nb):,} 顶点")
    cen = nb.groupby("cell_id")[["vertex_x","vertex_y"]].mean()
    keep = cen.index[(cen.vertex_x >= ROI_X0) & (cen.vertex_x < ROI_X1) &
                     (cen.vertex_y >= ROI_Y0) & (cen.vertex_y < ROI_Y1)]
    nb = nb[nb.cell_id.isin(set(keep))]
    P(f"    ROI 内核 {len(keep):,}  (GT {GT_CELLS:,}, "
      f"覆盖 {len(keep)/GT_CELLS*100:.0f}%)")
    # 注意：keep 按核质心筛，但 fillPoly 画完整多边形。质心在 ROI 内而部分顶点
    # 在外的核会被 clip 压到边界上，边缘一圈的核形状会略微失真，数量不多。
    mask = np.zeros((IMG_H, IMG_W), np.int32)
    for i, (_, grp) in enumerate(nb.groupby("cell_id", sort=False), start=1):
        px = ((grp.vertex_x.values-ROI_X0)/PIXEL_SIZE).astype(np.int32)
        py = ((grp.vertex_y.values-ROI_Y0)/PIXEL_SIZE).astype(np.int32)
        pts = np.stack([np.clip(px,0,IMG_W-1),
                        np.clip(py,0,IMG_H-1)], 1).reshape(-1,1,2)
        cv2.fillPoly(mask, [pts], int(i))
    ids, cnt = np.unique(mask[mask>0], return_counts=True)
    P(f"    栅格化 {len(ids):,} 核  中位面积 {np.median(cnt):.0f}px "
      f"(直径 {2*np.sqrt(np.median(cnt)/np.pi)*PIXEL_SIZE:.1f}µm)")
    tifffile.imwrite(mp, mask)
    return mask


def load_tx():
    P("[2] 转录本（ROI 过滤）")
    tx = pd.read_parquet(f"{DATA_DIR}/transcripts.parquet",
                         columns=["feature_name","x_location",
                                  "y_location","qv"])
    tx = tx[(tx.qv >= 20) &
            (tx.x_location >= ROI_X0) & (tx.x_location < ROI_X1) &
            (tx.y_location >= ROI_Y0) & (tx.y_location < ROI_Y1)]
    tx = tx[~tx.feature_name.str.startswith(
        ("NegControl","Unassigned","BLANK","antisense"), na=False)].copy()
    tx["px"] = ((tx.x_location-ROI_X0)/PIXEL_SIZE).astype(int).clip(0,IMG_W-1)
    tx["py"] = ((tx.y_location-ROI_Y0)/PIXEL_SIZE).astype(int).clip(0,IMG_H-1)
    P(f"    {len(tx):,} 条  {tx.feature_name.nunique()} 基因")
    return tx.reset_index(drop=True)


def check_alive(mask, tx):
    nid = mask[tx.py.values, tx.px.values]
    spot = pd.DataFrame({"nid": nid,
                         "k": tx.py.values.astype(np.int64)*IMG_W + tx.px.values})
    spot = spot[spot.nid > 0].drop_duplicates("k")
    ns = spot.groupby("nid").size()
    n_tot = int(mask.max())
    P(f"\n[3] nSpot 预估（Segmentation.py:287 硬编码 ≥{NSPOT_MIN}）")
    P(f"    核总数          {n_tot:,}")
    P(f"    含 ≥1 spot 的核 {len(ns):,} ({len(ns)/max(n_tot,1)*100:.1f}%)")
    if len(ns):
        P(f"    nSpot 中位 {ns.median():.0f}  p25 {ns.quantile(.25):.0f}")
        for th in (5, 10, 20):
            P(f"      nSpot≥{th:>2}: {int((ns>=th).sum()):>6,} 存活"
              f"{'  ← Cellist 用这个' if th==NSPOT_MIN else ''}")
    alive = int((ns >= NSPOT_MIN).sum()) if len(ns) else 0
    P(f"    → 全局口径存活 ≈ {alive:,} (GT {GT_CELLS:,}, "
      f"{alive/GT_CELLS*100:.0f}%)")
    P(f"    ★ Cellist 在每个 patch 内部重算 nSpot，轴序修复前实际只出 2,497")
    return alive


# ══════════════════════════════════════════════════════════════════════════
#  ② Cellist 输入
# ══════════════════════════════════════════════════════════════════════════
def build_inputs(mask, tx, force=False):
    paths = dict(
        prop  = f"{OUT_DIR}/{PREFIX}_nucprior_prop.txt",
        coord = f"{OUT_DIR}/{PREFIX}_nucprior_coord.txt",
        gem   = f"{OUT_DIR}/{PREFIX}_nucprior_gem.txt",
        spot  = f"{OUT_DIR}/{PREFIX}_nucprior_spot_count.h5",
        count = f"{OUT_DIR}/{PREFIX}_nucprior_count.h5",
    )
    P("\n[4] 生成 Cellist 输入")
    # ── prop ────────────────────────────────────────────────────────────
    props = skimage.measure.regionprops_table(
        mask, properties=["label","area","centroid",
                          "equivalent_diameter_area"])
    prop = pd.DataFrame(props)
    # ★★ 轴序修复：Cellist 认为 centroid-0=x，skimage 给的是 row(y)
    #    不交换则核与 spot 落进不同 patch（实测同 patch 仅 36.8%）
    _c0 = prop["centroid-0"].copy()
    prop["centroid-0"] = prop["centroid-1"]   # → x / col
    prop["centroid-1"] = _c0                  # → y / row
    P(f"    ★ centroid 轴序已交换")
    P(f"      centroid-0(x) [{prop['centroid-0'].min():.0f},"
      f"{prop['centroid-0'].max():.0f}]  "
      f"centroid-1(y) [{prop['centroid-1'].min():.0f},"
      f"{prop['centroid-1'].max():.0f}]")
    prop.to_csv(paths["prop"], sep="\t", index=False)
    P(f"    ✓ prop.txt   {len(prop):,} 核（Tab 分隔，含 label）")
    # ── coord（x/y 保持原样）─────────────────────────────────────────────
    uniq = tx.drop_duplicates(subset=["px","py"])
    sx, sy = uniq.px.values, uniq.py.values
    nid = mask[sy, sx].astype(float)
    pd.DataFrame({
        "x": sx, "y": sy,
        "x_y": [f"{a}_{b}" for a,b in zip(sx,sy)],
        "Nucleus":  (nid > 0).astype(int),
        "Cellpose": np.where(nid > 0, nid, np.nan),
    }).to_csv(paths["coord"], sep="\t", index=False)
    P(f"    ✓ coord.txt  {len(uniq):,} spots  "
      f"核内 {int((nid>0).sum()):,} ({(nid>0).mean()*100:.1f}%)")
    # ── gem ─────────────────────────────────────────────────────────────
    gem = (tx.groupby(["feature_name","px","py"]).size()
             .reset_index(name="MIDCount")
             .rename(columns={"feature_name":"geneID","px":"x","py":"y"}))
    gem[["geneID","x","y","MIDCount"]].to_csv(
        paths["gem"], sep="\t", index=False)
    P(f"    ✓ gem.txt    {len(gem):,} 行（imaging 模式也必需）")
    # ── 两个 h5 ─────────────────────────────────────────────────────────
    genes = sorted(tx.feature_name.unique())
    gi = tx.feature_name.map({g:i for i,g in enumerate(genes)}).values
    skey = tx.py.values.astype(np.int64)*IMG_W + tx.px.values
    su = np.unique(skey)
    rows = pd.Series(skey).map({s:i for i,s in enumerate(su)}).values
    write_10x_h5(paths["spot"],
                 csr_matrix((np.ones(len(tx), np.int32), (gi, rows)),
                            shape=(len(genes), len(su))),
                 genes, [f"{s%IMG_W}_{s//IMG_W}" for s in su])
    P(f"    ✓ spot_count.h5     {len(genes)}×{len(su):,}  (CSC)")
    n_nuc = int(mask.max())
    tnid = mask[tx.py.values, tx.px.values]
    inn = tnid > 0
    full = csr_matrix((np.ones(int(inn.sum()), np.int32),
                       (gi[inn], tnid[inn]-1)),
                      shape=(len(genes), n_nuc))
    gk = np.where(np.asarray(full.sum(axis=1)).ravel() > 0)[0]
    write_10x_h5(paths["count"], full[gk,:], [genes[i] for i in gk],
                 [str(i) for i in range(1, n_nuc+1)])
    P(f"    ✓ nucleus_count.h5  {len(gk)}×{n_nuc:,}  (CSC)")
    P(f"      基因过滤 {len(genes)} → {len(gk)}（剔除核内零 count）")
    return paths


# ══════════════════════════════════════════════════════════════════════════
#  ③ 分割 + 完整性校验
# ══════════════════════════════════════════════════════════════════════════
def verify_output(outdir, expect=EXPECT_PATCHES):
    """校验 patch 完整性与空间覆盖，返回 True 才算可用。"""
    props = sorted(glob.glob(f"{outdir}/Data_HVG/Patch_*_property.txt"))
    subs = sorted(glob.glob(f"{outdir}/**/tmp/Sub_*_soft_clustering.txt",
                            recursive=True))
    P(f"      Data_HVG patch {len(props)} 个，tmp/Sub 输出 {len(subs)} 个"
      f"（3×3 网格应各为 {expect}）")
    if len(subs) < expect or len(props) != len(subs):
        want = {os.path.basename(p).replace("_property.txt", "")
                .replace("Patch_", "") for p in props}
        got = {os.path.basename(s).replace("_soft_clustering.txt", "")
               .replace("Sub_", "") for s in subs}
        P(f"    ✘ patch 不完整，缺失: {sorted(want - got) or '(输入侧也缺)'}")
        P(f"      这是轴序或分块问题的典型征兆，结果会出现整块空洞")
        return False

    seg = glob.glob(f"{outdir}/**/*Cellist_segmentation.txt", recursive=True)
    if not seg:
        P("    ✘ 没有 segmentation.txt")
        return False
    s = pd.read_csv(seg[0], sep="\t", usecols=["x", "y", "Cellist"])
    ok = (pd.to_numeric(s["Cellist"], errors="coerce") > 0).values
    bx = np.clip((s.x.values / IMG_W * 10).astype(int), 0, 9)
    by = np.clip((s.y.values / IMG_H * 10).astype(int), 0, 9)
    rate = pd.DataFrame({"bx": bx, "by": by, "ok": ok}) \
        .groupby(["by", "bx"])["ok"].mean().unstack()
    dead = int((rate.fillna(0) < 0.05).sum().sum())
    P(f"      整体归属率 {ok.mean():.1%}；10×10 网格中位 "
      f"{np.nanmedian(rate.values):.1%}，低于 5% 的格子 {dead}/100")
    if dead > 5:
        P(f"    ✘ 存在 {dead} 个空洞格，结果不可用")
        return False
    return True


def run_seg(paths, radius, force=False):
    tag = f"nucprior_r{radius}_fix"
    outdir = f"{OUT_DIR}/seg_{tag}"
    done = glob.glob(f"{outdir}/**/*Cellist_segmentation_cell_coord.txt",
                     recursive=True)
    if done and not force:
        n = len(pd.read_csv(done[0], sep="\t"))
        P(f"\n  ⚡ 已有结果: {n:,} 细胞  重跑加 --force")
        return done[0]
    os.makedirs(outdir, exist_ok=True)

    # 调 CLI 前逐个校验输入，早期版本曾把 None 传给 --nucleus-prop 才报错
    P(f"\n[5] 检查 Cellist 输入")
    for k, v in paths.items():
        if not (v and os.path.exists(v)):
            P(f"    ✘ 输入缺失: {k} -> {v}")
            return None
        P(f"      {k:<6} {os.path.getsize(v)/1e6:>8.1f} MB  {os.path.basename(v)}")

    cmd = ["cellist", "seg", "--platform", "imaging",
           "--resolution", str(PIXEL_SIZE),
           "--gem", paths["gem"], "--spot-count-h5", paths["spot"],
           "--nucleus-seg-method", "Cellpose",
           "--nucleus-seg", paths["coord"], "--nucleus-prop", paths["prop"],
           "--nucleus-count-h5", paths["count"],
           "--cell-radius", str(int(radius)),
           "--gene-use", "HVG",
           "--outdir", outdir, "--outprefix", PREFIX]
    P(f"\n  cellist seg → seg_{tag}/  （约 5-15 分钟，实时输出）")
    P("  " + " ".join(cmd))
    P("  " + "-" * 60)
    rc = subprocess.run(cmd).returncode
    P("  " + "-" * 60)
    if rc != 0:
        P(f"    ✘ 退出码 {rc}")
        return None

    f = glob.glob(f"{outdir}/**/*Cellist_segmentation_cell_coord.txt",
                  recursive=True)
    if not f:
        P("    ✘ 无结果文件")
        return None
    n = len(pd.read_csv(f[0], sep="\t"))
    P(f"    ✓ {n:,} 细胞  (GT {GT_CELLS:,}, ratio {n/GT_CELLS:.3f})")
    P(f"      轴序修复前是 2,497（ratio 0.283，F1=0.4407）；"
      f"旧版 Cellpose 种子只出 955")
    if not verify_output(outdir):
        P(f"    ✘ 完整性校验未通过，不返回该结果")
        return None
    P(f"      {f[0]}")
    return f[0]


def main():
    ap = argparse.ArgumentParser()
    # cell-radius 单位是像素。掩膜核等效直径中位数 5.66 µm（核半径约 2.8 µm），
    # 半径 10 px 只有 2.1 µm，扩出来的细胞半径不到 5 µm，明显小于乳腺细胞。
    # 30 px = 6.4 µm，加上核半径约 9 µm，更接近实际尺寸。
    ap.add_argument("--cell-radius", type=int, default=30)
    ap.add_argument("--scan-radii", default=None,
                    help="逗号分隔，依次试多个半径，如 20,30,40")
    ap.add_argument("--check-only", action="store_true",
                    help="只到 nSpot 预估")
    ap.add_argument("--force", action="store_true")
    cfg = ap.parse_args()

    P(SEP); P("Cellist × Xenium 乳腺（nucleus_boundaries 先验）"); P(SEP)
    P(f"  ROI x[{ROI_X0:.1f},{ROI_X1:.1f}] y[{ROI_Y0:.1f},{ROI_Y1:.1f}] µm")
    P(f"  ★ property centroid 轴序已修（Segmentation.py:283 的上游 bug）")
    P(f"  ★ 输出用 nucprior_* 前缀，与旧版 Cellpose 种子的 nucleus_* 区分")

    mask = build_mask(cfg.force)
    tx = load_tx()
    check_alive(mask, tx)
    if cfg.check_only:
        P(f"\n  --check-only，到此为止")
        return

    paths = build_inputs(mask, tx, cfg.force)
    radii = ([int(r) for r in cfg.scan_radii.split(",")]
             if cfg.scan_radii else [cfg.cell_radius])
    results = []
    for r in radii:
        out = run_seg(paths, r, cfg.force)
        if out:
            n = len(pd.read_csv(out, sep="\t"))
            results.append((r, n, n / GT_CELLS, out))
    if len(results) > 1:
        P(f"\n[6] 各半径对比（GT {GT_CELLS:,}）")
        for r, n, ratio, _ in results:
            P(f"    r={r:<3} {n:>7,} 细胞  ratio {ratio:.3f}")
    if not results:
        P(f"\n  ✘ 所有半径都没能产出可用结果")
    P(f"\n  评估: python eval_cellist_breast.py")


if __name__ == "__main__":
    main()