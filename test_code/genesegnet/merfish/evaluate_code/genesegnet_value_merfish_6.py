#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GeneSegNet 独立评估（自包含，不 import 任何现有脚本）。
读 genesegnet mask.tif + 数据集的 transcripts/cell_metadata + micron_to_mosaic 变换，
GT = cell_metadata 质心落在中心 ROI 内的 platform 核；
匹配 = kd-tree 候选边(半径=GT最近邻p95×1.25) + 连通分量内匈牙利/大分量贪心；
输出 19 指标，与 merfish benchmark 口径一致。

用法:
  python eval_genesegnet_standalone.py \
    --root /data/qiuyijia/dataset/merfish_breast_cancer \
    --mask /data/qiuyijia/genesegnet_merfish_breast/genesegnet_merfish1_roi20000_mask.tif \
    --dataset breast --roi-size 10000
"""
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile


def find_one(root, pats):
    for p in pats:
        c = sorted(glob.glob(f"{root}/**/{p}", recursive=True))
        if c:
            return c[0]
    return None


def load_geometry(root, roi_size):
    tf = find_one(root, ["*micron_to_mosaic*transform*.csv"])
    if not tf:
        raise SystemExit(f"no transform under {root}")
    M = np.loadtxt(tf)
    dapi = None
    for p in sorted(glob.glob(f"{root}/**/*.tif", recursive=True)):
        if "dapi" in Path(p).name.lower():
            dapi = p; break
    if not dapi:
        raise SystemExit("no *dapi*.tif under root")
    with tifffile.TiffFile(dapi) as t:
        H, W = t.series[0].shape[-2:]
    y0 = (H - roi_size) // 2
    x0 = (W - roi_size) // 2
    # px = M @ [um,1]; um = Minv @ [px,1]
    Minv = np.linalg.inv(M)
    return dict(M=M, Minv=Minv, H=H, W=W, y0=y0, x0=x0, size=roi_size)


def px_to_um(geo, px, py):
    v = np.c_[np.asarray(px, float), np.asarray(py, float), np.ones(len(px))] @ geo["Minv"].T
    return v[:, 0], v[:, 1]


def um_bounds(geo):
    x0, y0, s = geo["x0"], geo["y0"], geo["size"]
    corners = np.array([[x0, y0, 1], [x0+s, y0, 1], [x0, y0+s, 1], [x0+s, y0+s, 1]], float)
    um = corners @ geo["Minv"].T
    return um[:, 0].min(), um[:, 0].max(), um[:, 1].min(), um[:, 1].max()


def load_gt(root, geo):
    md = find_one(root, ["cell_metadata.csv*"])
    if not md:
        raise SystemExit("no cell_metadata.csv")
    d = pd.read_csv(md)
    xc = next(c for c in d.columns if c.lower() in ("center_x", "x_centroid", "centroid_x"))
    yc = next(c for c in d.columns if c.lower() in ("center_y", "y_centroid", "centroid_y"))
    X0, X1, Y0, Y1 = um_bounds(geo)
    m = (d[xc] >= X0) & (d[xc] < X1) & (d[yc] >= Y0) & (d[yc] < Y1)
    return d[xc].to_numpy(float)[m], d[yc].to_numpy(float)[m], (X0, X1, Y0, Y1)


def load_tx(root):
    txp = find_one(root, ["detected_transcripts.csv*"])
    if not txp:
        raise SystemExit("no detected_transcripts.csv")
    d = pd.read_csv(txp, usecols=lambda c: c in ("global_x", "global_y", "gene"))
    g = d["gene"].astype(str).to_numpy().astype("U").astype("U")
    g = np.asarray(g, dtype="U"); g = np.asarray(g, dtype="U"); keep = ~np.char.startswith(np.char.lower(g), "blank")
    genes = sorted(set(g[keep]))
    gi = {v: i for i, v in enumerate(genes)}
    gidx = np.array([gi.get(v, -1) for v in g], np.int64)
    ok = gidx >= 0
    return (d["global_x"].to_numpy(float)[ok], d["global_y"].to_numpy(float)[ok],
            gidx[ok], genes)


# ---- 匹配 ----
def infer_radius(gx, gy):
    from scipy.spatial import cKDTree
    if len(gx) < 2: return 2.0
    d, _ = cKDTree(np.c_[gx, gy]).query(np.c_[gx, gy], k=2)
    nn = d[:, 1]; nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(max(np.quantile(nn, 0.95) * 1.25, 2.0))


def match(px, py, gx, gy, radius, greedy_above=400):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.optimize import linear_sum_assignment
    tp = cKDTree(np.c_[px, py]); tg = cKDTree(np.c_[gx, gy])
    pairs = tp.query_ball_tree(tg, r=radius)
    pi, gi = [], []
    for i, js in enumerate(pairs):
        for j in js: pi.append(i); gi.append(j)
    pi = np.array(pi, np.int64); gi = np.array(gi, np.int64)
    if len(pi) == 0: return np.array([],int), np.array([],int), np.array([])
    d = np.hypot(px[pi]-gx[gi], py[pi]-gy[gi])
    n = len(px)+len(gx)
    a = coo_matrix((np.ones(len(pi)), (pi, len(px)+gi)), shape=(n, n)); a = a+a.T
    _, lab = connected_components(a, directed=False)
    comp = lab[pi]; o = np.argsort(comp, kind="stable")
    pi, gi, d, comp = pi[o], gi[o], d[o], comp[o]
    st = np.append(np.searchsorted(comp, np.unique(comp)), len(pi))
    ep, eg, ed = [], [], []
    for k in range(len(st)-1):
        a_, b_ = st[k], st[k+1]; sp, sg, sd = pi[a_:b_], gi[a_:b_], d[a_:b_]
        up, ug = np.unique(sp), np.unique(sg)
        if len(up)*len(ug) > greedy_above**2:
            oo = np.argsort(sd); usp, usg = set(), set()
            for t in oo:
                if sp[t] in usp or sg[t] in usg: continue
                usp.add(sp[t]); usg.add(sg[t]); ep.append(sp[t]); eg.append(sg[t]); ed.append(sd[t])
            continue
        pm = {v:i for i,v in enumerate(up)}; gm = {v:i for i,v in enumerate(ug)}
        big = sd.max()*10+1; cost = np.full((len(up),len(ug)), big)
        cost[[pm[v] for v in sp],[gm[v] for v in sg]] = sd
        r, c = linear_sum_assignment(cost)
        for rr, cc in zip(r, c):
            if cost[rr,cc] < big: ep.append(up[rr]); eg.append(ug[cc]); ed.append(cost[rr,cc])
    return np.array(ep,int), np.array(eg,int), np.array(ed,float)


def gene_metrics(pm, gm):
    from scipy.spatial.distance import jensenshannon
    cos, js, pr = [], [], []
    for i in range(pm.shape[0]):
        a = pm[i].astype(float); b = gm[i].astype(float)
        if a.sum() <= 0 or b.sum() <= 0: continue
        cos.append(float(np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b))))
        dd = jensenshannon(a/a.sum(), b/b.sum(), base=2)
        if np.isfinite(dd): js.append(float(dd))
        if a.std()>0 and b.std()>0: pr.append(float(np.corrcoef(a,b)[0,1]))
    return (np.mean(cos) if cos else np.nan, np.mean(js) if js else np.nan,
            np.mean(pr) if pr else np.nan, len(cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--dataset", default="x")
    ap.add_argument("--roi-size", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    geo = load_geometry(args.root, args.roi_size)
    gx, gy, roi_um = load_gt(args.root, geo)
    tx_x, tx_y, tx_g, genes = load_tx(args.root)
    print(f"GT cells in ROI = {len(gx)}   genes = {len(genes)}   ROI_um = {tuple(round(v,1) for v in roi_um)}")

    lab = np.squeeze(tifffile.imread(args.mask)).astype(np.int64); lab[lab < 0] = 0
    s = args.roi_size / lab.shape[0]
    ids = np.unique(lab); ids = ids[ids > 0]
    rows, cols = np.nonzero(lab); lv = lab[rows, cols]
    o = np.argsort(lv, kind="stable"); lv, rows, cols = lv[o], rows[o], cols[o]
    b = np.append(np.searchsorted(lv, ids), len(lv))
    cy = np.array([rows[b[i]:b[i+1]].mean() for i in range(len(ids))])
    cx = np.array([cols[b[i]:b[i+1]].mean() for i in range(len(ids))])
    area_px = (b[1:]-b[:-1]).astype(float)
    ps_um = 1.0/geo["M"][0,0]
    area_um2 = area_px * s * s * ps_um * ps_um
    # 质心 label像素 -> 全局像素 -> um
    gpx = (cx+0.5)*s + geo["x0"]; gpy = (cy+0.5)*s + geo["y0"]
    pux, puy = px_to_um(geo, gpx, gpy)

    # 转录本 -> 细胞
    lx = ((tx_x - roi_um[0]) / ps_um) / s  # 近似：um->ROI局部px->label px
    # 更稳: 用全局像素反推。global um -> global px:
    gp = np.c_[tx_x, tx_y, np.ones(len(tx_x))] @ geo["M"].T
    rr = np.clip(((gp[:,1]-geo["y0"])/s).astype(int), 0, lab.shape[0]-1)
    cc = np.clip(((gp[:,0]-geo["x0"])/s).astype(int), 0, lab.shape[1]-1)
    inroi = (gp[:,0]>=geo["x0"]) & (gp[:,0]<geo["x0"]+args.roi_size) & (gp[:,1]>=geo["y0"]) & (gp[:,1]<geo["y0"]+args.roi_size)
    tl = np.zeros(len(tx_x), np.int64); tl[inroi] = lab[rr[inroi], cc[inroi]]
    assign_rate = float((tl[inroi] > 0).mean())
    pos = {int(v): i for i, v in enumerate(ids)}
    gm_pred = np.zeros((len(ids), len(genes)), np.float32)
    sel = tl > 0
    ri = np.array([pos.get(int(v), -1) for v in tl[sel]]); ok = ri >= 0
    np.add.at(gm_pred, (ri[ok], tx_g[sel][ok]), 1.0)

    # GT 基因向量：转录本按最近 GT 质心归属
    from scipy.spatial import cKDTree
    tgt = cKDTree(np.c_[gx, gy])
    dd, jj = tgt.query(np.c_[tx_x, tx_y], k=1)
    rad = max(np.median(dd)*2, 3.0); okg = dd <= rad
    gm_gt = np.zeros((len(gx), len(genes)), np.float32)
    np.add.at(gm_gt, (jj[okg], tx_g[okg]), 1.0)

    radius = infer_radius(gx, gy)
    ep, eg, ed = match(pux, puy, gx, gy, radius)
    n = len(ep)
    prec = n/max(len(ids),1); rec = n/max(len(gx),1)
    f1 = 2*prec*rec/(prec+rec) if prec+rec else np.nan
    vc, vjs, vpr, vpair = gene_metrics(gm_pred[ep], gm_gt[eg]) if n else (np.nan,)*4
    # 计数相关
    from scipy.stats import pearsonr, spearmanr
    def corr(a,b):
        ok=np.isfinite(a)&np.isfinite(b)
        if ok.sum()<3 or a[ok].std()==0 or b[ok].std()==0: return np.nan,np.nan
        return float(pearsonr(a[ok],b[ok])[0]), float(spearmanr(a[ok],b[ok])[0])
    cp, csp = corr(gm_pred[ep].sum(1), gm_gt[eg].sum(1)) if n else (np.nan,np.nan)

    R = {
        "method":"GeneSegNet","dataset":args.dataset,
        "gt_cells":len(gx),"pred_cells":len(ids),
        "cell_ratio":round(len(ids)/max(len(gx),1),4),"matched":n,
        "precision":round(prec,4),"recall":round(rec,4),"f1":round(f1,4),
        "loc_mean_um":round(float(ed.mean()),4) if n else np.nan,
        "loc_median_um":round(float(np.median(ed)),4) if n else np.nan,
        "loc_p95_um":round(float(np.percentile(ed,95)),4) if n else np.nan,
        "count_pearson":round(cp,4) if cp==cp else np.nan,
        "count_spearman":round(csp,4) if csp==csp else np.nan,
        "vec_cosine":round(vc,4) if vc==vc else np.nan,
        "vec_js_dist":round(vjs,4) if vjs==vjs else np.nan,
        "vec_pearson":round(vpr,4) if vpr==vpr else np.nan,
        "pred_assign_rate":round(assign_rate,4),
        "median_area_um2":round(float(np.median(area_um2)),4),
        "median_diameter_um":round(float(2*np.sqrt(np.median(area_um2)/np.pi)),4),
        "coverage_fraction":round(float((lab>0).mean()),4),
        "uses_platform_prior":False,
    }
    print("="*60)
    for k,v in R.items(): print(f"  {k:24s} {v}")
    print("="*60)
    out = args.out or f"/data/qiuyijia/eval_genesegnet_{args.dataset}.csv"
    pd.DataFrame([R]).to_csv(out, index=False)
    print("written ->", out)


if __name__ == "__main__":
    main()