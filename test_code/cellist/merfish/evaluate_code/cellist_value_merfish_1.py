#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cellist 评估 —— 自包含。读 *_Cellist_segmentation_cell_coord.txt (质心, 工作网格px)
+ *_cell_count.h5 (10x CSC 基因矩阵)，换算到微米后与 platform GT 匹配，输出 19 指标。

坐标换算: cellist coord 的 x/y 是【工作网格像素】(网格 = ROI微米 / resolution)。
  µm = 坐标 * resolution + ROI微米原点
  ⚠ cellist 存在轴序交换(编排器注释: centroid-0 当 x)，用 --swap-xy 开关处理。

用法(merfish mouse_brain, ROI x[3589,5761] y[2327,4499], res 0.5):
  python cellist_value_merfish_1.py --dataset merfish_1 \
    --root /data/qiuyijia/dataset/merfish_mouse_brain \
    --coord ".../merfish_s1r1_Cellist_segmentation_cell_coord.txt" \
    --count-h5 ".../merfish_s1r1_Cellist_segmentation_cell_count.h5" \
    --roi-um 3589 5761 2327 4499 --resolution 0.5 [--swap-xy]
"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd


def find_one(root, pats):
    for p in pats:
        c = sorted(glob.glob(f"{root}/**/{p}", recursive=True))
        if c: return c[0]
    return None


def load_coord(path, roi, res, swap):
    X0, X1, Y0, Y1 = roi
    d = pd.read_csv(path, sep="\t")
    xcol = [c for c in d.columns if c.lower() == "x"][0]
    ycol = [c for c in d.columns if c.lower() == "y"][0]
    gx = d[xcol].to_numpy(float)
    gy = d[ycol].to_numpy(float)
    if swap:
        gx, gy = gy, gx
    ux = gx * res + X0
    uy = gy * res + Y0
    ids = d[d.columns[0]].to_numpy()
    return ux, uy, ids


def load_h5(path):
    import h5py
    from scipy.sparse import csc_matrix
    h = h5py.File(path, "r")
    g = h["matrix"]
    data = g["data"][:]; indices = g["indices"][:]; indptr = g["indptr"][:]
    shape = tuple(g["shape"][:])            # (n_features, n_cells) 10x 约定
    names = [x.decode() for x in g["features"]["name"][:]]
    barcodes = [x.decode() for x in g["barcodes"][:]]
    M = csc_matrix((data, indices, indptr), shape=shape)  # feature x cell
    return M.T.toarray().astype(np.float32), names, barcodes  # cell x gene + cell_id顺序


def load_gt(root, roi):
    md = find_one(root, ["cell_metadata.csv*", "*cell_metadata*.csv*"])
    if not md: raise SystemExit("no cell_metadata.csv")
    d = pd.read_csv(md)
    xc = next(c for c in d.columns if c.lower() in ("center_x","x_centroid","centroid_x"))
    yc = next(c for c in d.columns if c.lower() in ("center_y","y_centroid","centroid_y"))
    X0,X1,Y0,Y1 = roi
    m = (d[xc]>=X0)&(d[xc]<X1)&(d[yc]>=Y0)&(d[yc]<Y1)
    return d[xc].to_numpy(float)[m], d[yc].to_numpy(float)[m]


def load_tx(root):
    txp = find_one(root, ["detected_transcripts.csv*","*detected_transcripts*.csv*"])
    if not txp: raise SystemExit("no detected_transcripts.csv")
    d = pd.read_csv(txp, usecols=lambda c: c in ("global_x","global_y","gene"))
    g = np.asarray(d["gene"].astype(str).to_numpy(), dtype="U")
    keep = ~np.char.startswith(np.char.lower(g), "blank")
    genes = sorted(set(g[keep])); gi = {v:i for i,v in enumerate(genes)}
    gidx = np.array([gi.get(v,-1) for v in g], np.int64); ok = gidx>=0
    return d["global_x"].to_numpy(float)[ok], d["global_y"].to_numpy(float)[ok], gidx[ok], genes


def infer_radius(gx,gy):
    from scipy.spatial import cKDTree
    if len(gx)<2: return 2.0
    dd,_=cKDTree(np.c_[gx,gy]).query(np.c_[gx,gy],k=2); nn=dd[:,1]; nn=nn[np.isfinite(nn)&(nn>0)]
    return float(max(np.quantile(nn,0.95)*1.25,2.0))


def match(px,py,gx,gy,r,greedy=400):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.optimize import linear_sum_assignment
    tp=cKDTree(np.c_[px,py]); tg=cKDTree(np.c_[gx,gy]); pairs=tp.query_ball_tree(tg,r=r)
    pi=[];gi=[]
    for i,js in enumerate(pairs):
        for j in js: pi.append(i);gi.append(j)
    pi=np.array(pi,int);gi=np.array(gi,int)
    if not len(pi): return np.array([],int),np.array([],int),np.array([])
    d=np.hypot(px[pi]-gx[gi],py[pi]-gy[gi]); n=len(px)+len(gx)
    a=coo_matrix((np.ones(len(pi)),(pi,len(px)+gi)),shape=(n,n)); a=a+a.T
    _,lab=connected_components(a,directed=False); comp=lab[pi]
    o=np.argsort(comp,kind="stable"); pi,gi,d,comp=pi[o],gi[o],d[o],comp[o]
    st=np.append(np.searchsorted(comp,np.unique(comp)),len(pi)); ep=[];eg=[];ed=[]
    for k in range(len(st)-1):
        a_,b_=st[k],st[k+1]; sp,sg,sd=pi[a_:b_],gi[a_:b_],d[a_:b_]
        up,ug=np.unique(sp),np.unique(sg)
        if len(up)*len(ug)>greedy**2:
            usp,usg=set(),set()
            for t in np.argsort(sd):
                if sp[t] in usp or sg[t] in usg: continue
                usp.add(sp[t]);usg.add(sg[t]);ep.append(sp[t]);eg.append(sg[t]);ed.append(sd[t])
            continue
        pm={v:i for i,v in enumerate(up)};gm={v:i for i,v in enumerate(ug)}
        big=sd.max()*10+1; cost=np.full((len(up),len(ug)),big)
        cost[[pm[v] for v in sp],[gm[v] for v in sg]]=sd
        rr,cc=linear_sum_assignment(cost)
        for a2,b2 in zip(rr,cc):
            if cost[a2,b2]<big: ep.append(up[a2]);eg.append(ug[b2]);ed.append(cost[a2,b2])
    return np.array(ep,int),np.array(eg,int),np.array(ed,float)


def gene_metrics(pm,gm):
    from scipy.spatial.distance import jensenshannon
    cos=[];js=[];pr=[]
    for i in range(pm.shape[0]):
        a=pm[i].astype(float);b=gm[i].astype(float)
        if a.sum()<=0 or b.sum()<=0: continue
        cos.append(float(np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b))))
        dd=jensenshannon(a/a.sum(),b/b.sum(),base=2)
        if np.isfinite(dd): js.append(float(dd))
        if a.std()>0 and b.std()>0: pr.append(float(np.corrcoef(a,b)[0,1]))
    return (np.mean(cos) if cos else np.nan, np.mean(js) if js else np.nan,
            np.mean(pr) if pr else np.nan, len(cos))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--coord",required=True)
    ap.add_argument("--count-h5",required=True)
    ap.add_argument("--roi-um",nargs=4,type=float,required=True,metavar=("X0","X1","Y0","Y1"))
    ap.add_argument("--resolution",type=float,required=True)
    ap.add_argument("--swap-xy",action="store_true")
    ap.add_argument("--dataset",default="x")
    ap.add_argument("--out",default=None)
    a=ap.parse_args()
    roi=tuple(a.roi_um)

    px,py,pids = load_coord(a.coord, roi, a.resolution, a.swap_xy)
    gm_p, pgenes, pbarcodes = load_h5(a.count_h5)
    # 按 coord 的 cell_id(pids) 对齐 h5 的 barcodes 行顺序(barcodes乱序,不能按位置)
    bc2row = {str(b): i for i, b in enumerate(pbarcodes)}
    order = [bc2row.get(str(int(float(c))), bc2row.get(str(c), -1)) for c in pids]
    order = np.array(order)
    valid_cell = order >= 0
    gm_full = np.zeros((len(pids), gm_p.shape[1]), np.float32)
    gm_full[valid_cell] = gm_p[order[valid_cell]]
    gm_p = gm_full
    # coord 与 h5 行数应一致
    # px/py/gm_p 均已按 pids(coord)对齐,无需再截断
    gx,gy = load_gt(a.root, roi)
    tx_x,tx_y,tx_g,genes = load_tx(a.root)
    print(f"pred={len(px)}  GT={len(gx)}  roi_um={roi}  res={a.resolution}  swap={a.swap_xy}")

    # GT 基因向量
    from scipy.spatial import cKDTree
    dd,jj=cKDTree(np.c_[gx,gy]).query(np.c_[tx_x,tx_y],k=1)
    rad=max(np.median(dd)*2,3.0); ok=dd<=rad
    gm_gt=np.zeros((len(gx),len(genes)),np.float32); np.add.at(gm_gt,(jj[ok],tx_g[ok]),1.0)
    # pred 基因向量对齐基因表
    gi={x:i for i,x in enumerate(genes)}; col=np.array([gi.get(x,-1) for x in pgenes])
    gm_pred=np.zeros((len(px),len(genes)),np.float32); v=col>=0
    gm_pred[:,col[v]]=gm_p[:,np.where(v)[0]]

    r=infer_radius(gx,gy); ep,eg,ed=match(px,py,gx,gy,r); nn=len(ep)
    prec=nn/max(len(px),1); rec=nn/max(len(gx),1); f1=2*prec*rec/(prec+rec) if prec+rec else np.nan
    vc,vjs,vpr,_=gene_metrics(gm_pred[ep],gm_gt[eg]) if nn else (np.nan,)*4
    from scipy.stats import pearsonr,spearmanr
    def corr(x,y):
        m=np.isfinite(x)&np.isfinite(y)
        if m.sum()<3 or x[m].std()==0 or y[m].std()==0: return np.nan,np.nan
        return float(pearsonr(x[m],y[m])[0]),float(spearmanr(x[m],y[m])[0])
    cp,csp=corr(gm_pred[ep].sum(1),gm_gt[eg].sum(1)) if nn else (np.nan,np.nan)

    R={"method":"Cellist","dataset":a.dataset,"gt_cells":len(gx),"pred_cells":len(px),
       "cell_ratio":round(len(px)/max(len(gx),1),4),"matched":nn,
       "precision":round(prec,4),"recall":round(rec,4),"f1":round(f1,4),
       "loc_mean_um":round(float(ed.mean()),4) if nn else np.nan,
       "loc_median_um":round(float(np.median(ed)),4) if nn else np.nan,
       "loc_p95_um":round(float(np.percentile(ed,95)),4) if nn else np.nan,
       "count_pearson":round(cp,4) if cp==cp else np.nan,
       "count_spearman":round(csp,4) if csp==csp else np.nan,
       "vec_cosine":round(vc,4) if vc==vc else np.nan,
       "vec_js_dist":round(vjs,4) if vjs==vjs else np.nan,
       "vec_pearson":round(vpr,4) if vpr==vpr else np.nan}
    print("="*56)
    for k,v in R.items(): print(f"  {k:18s} {v}")
    print("="*56)
    out=a.out or f"/data/qiuyijia/cellist_value_{a.dataset}.csv"
    pd.DataFrame([R]).to_csv(out,index=False); print("written ->",out)


if __name__=="__main__":
    main()