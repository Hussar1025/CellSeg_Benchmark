#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComSeg (merfish) evaluation - self-contained, imports no other script.
One of two inputs:
  --run-dir  contains tile_results/*.h5ad + tile_metadata.tsv  (assemble all tiles)
  --h5ad     a single ComSeg output h5ad (obs has centroid_x/y, image_name)
Uniform protocol: regardless of ROI size, after assembling, only predicted cells inside
the central [10000x10000 native px] participate in evaluation; GT uses the platform
nuclei in the same central 10000 ROI. Metrics match the merfish benchmark (19 items).

Usage:
  # multi-tile (e.g. 64 tiles for merfish1, source ROI 20000, evaluate center 10000 only)
  python comseg_value_merfish_1.py --dataset mouse_brain \
    --root yourpath/dataset/merfish_mouse_brain \
    --run-dir yourpath/comseg_merfish_1_dapi_prior_roi20000 --src-roi 20000

  # single h5ad (e.g. liver1)
  python comseg_value_merfish_1.py --dataset liver1 \
    --root yourpath/dataset/merfish_liver1 \
    --h5ad yourpath/comseg/merfish_liver1/output/merfish_liver1_roi_top300_mmc0p99.h5ad --src-roi 10000
"""
import argparse, glob, re
from pathlib import Path
import numpy as np, pandas as pd, tifffile

EVAL_ROI = 10000  # uniform evaluation window: central 10000x10000 native px


def find_one(root, pats):
    for p in pats:
        c = sorted(glob.glob(f"{root}/**/{p}", recursive=True))
        if c: return c[0]
    return None


def geometry(root, src_roi, roi_origin=None, src_hw=None):
    tf = find_one(root, ["*micron_to_mosaic*transform*.csv"])
    if not tf: raise SystemExit(f"no transform under {root}")
    M = np.loadtxt(tf); Minv = np.linalg.inv(M)
    dapi = next((p for p in sorted(glob.glob(f"{root}/**/*.tif", recursive=True))
                 if "dapi" in Path(p).name.lower()), None)
    if not dapi: raise SystemExit("no *dapi*.tif")
    with tifffile.TiffFile(dapi) as t: H, W = t.series[0].shape[-2:]
    if src_hw is not None: H, W = int(src_hw[0]), int(src_hw[1])
    # source ROI origin (assembly coordinate base) and central evaluation ROI origin
    if roi_origin is not None:
        src_y0, src_x0 = int(roi_origin[0]), int(roi_origin[1])
    else:
        src_y0, src_x0 = (H-src_roi)//2, (W-src_roi)//2
    ev_y0,  ev_x0  = (H-EVAL_ROI)//2, (W-EVAL_ROI)//2
    return dict(M=M, Minv=Minv, H=H, W=W, ps=1.0/M[0,0],
                src_y0=src_y0, src_x0=src_x0, ev_y0=ev_y0, ev_x0=ev_x0)


def px_to_um(g, px, py):
    v = np.c_[np.asarray(px,float), np.asarray(py,float), np.ones(len(px))] @ g["Minv"].T
    return v[:,0], v[:,1]


def eval_um_bounds(g):
    s=EVAL_ROI; x0,y0=g["ev_x0"],g["ev_y0"]
    c=np.array([[x0,y0,1],[x0+s,y0,1],[x0,y0+s,1],[x0+s,y0+s,1]],float)@g["Minv"].T
    return c[:,0].min(),c[:,0].max(),c[:,1].min(),c[:,1].max()


def load_pred_tiles(run_dir, g, src_roi):
    """Assemble all tiles: in-tile um centroid + ry/rx origin -> global um."""
    import anndata as ad
    meta = pd.read_csv(Path(run_dir)/"tile_metadata.tsv", sep="\t")
    origin = {r.tile_id:(float(r.ry),float(r.rx)) for r in meta.itertuples()}
    cx=[]; cy=[]; gms=[]; genes=None
    for f in sorted(glob.glob(f"{run_dir}/tile_results/tile_*.h5ad")):
        a=ad.read_h5ad(f); tid=re.search(r"tile_\d+",f).group()
        if tid not in origin: continue
        ry,rx=origin[tid]
        # tile origin (native px) -> global um; then add the in-tile centroid (already um)
        ox_um, oy_um = px_to_um(g, np.array([g["src_x0"]+rx]), np.array([g["src_y0"]+ry]))
        ux = a.obs["centroid_x"].to_numpy(float) + ox_um[0]
        uy = a.obs["centroid_y"].to_numpy(float) + oy_um[0]
        cx.append(ux); cy.append(uy)
        X=a.X; X=np.asarray(X.todense()) if hasattr(X,"todense") else np.asarray(X)
        gms.append((list(a.var_names),X)); genes=genes or list(a.var_names)
    cx=np.concatenate(cx); cy=np.concatenate(cy)
    if all(gn==genes for gn,_ in gms): gm=np.vstack([X for _,X in gms]).astype(np.float32)
    else:
        gi={x:i for i,x in enumerate(genes)}; rows=[]
        for gn,X in gms:
            m=np.zeros((X.shape[0],len(genes)),np.float32)
            col=[gi[x] for x in gn if x in gi]; src=[j for j,x in enumerate(gn) if x in gi]
            m[:,col]=X[:,src]; rows.append(m)
        gm=np.vstack(rows)
    return cx,cy,gm,genes


def load_pred_h5ad(h5, g, src_roi):
    import anndata as ad
    a=ad.read_h5ad(h5)
    # centroid is already in global um (measured on liver: range matches the CFG ROI exactly), use directly
    ux = a.obs["centroid_x"].to_numpy(float)
    uy = a.obs["centroid_y"].to_numpy(float)
    X=a.X; X=np.asarray(X.todense()) if hasattr(X,"todense") else np.asarray(X)
    return ux,uy,X.astype(np.float32),list(a.var_names)


def load_gt(root, g):
    md=find_one(root,["cell_metadata.csv*","*cell_metadata*.csv*"])
    if not md: raise SystemExit("no cell_metadata.csv")
    d=pd.read_csv(md)
    xc=next(c for c in d.columns if c.lower() in ("center_x","x_centroid","centroid_x"))
    yc=next(c for c in d.columns if c.lower() in ("center_y","y_centroid","centroid_y"))
    X0,X1,Y0,Y1=eval_um_bounds(g)
    m=(d[xc]>=X0)&(d[xc]<X1)&(d[yc]>=Y0)&(d[yc]<Y1)
    return d[xc].to_numpy(float)[m], d[yc].to_numpy(float)[m], (X0,X1,Y0,Y1)


def load_tx(root):
    txp=find_one(root,["detected_transcripts.csv*","*detected_transcripts*.csv*"])
    if not txp: raise SystemExit("no detected_transcripts.csv")
    d=pd.read_csv(txp, usecols=lambda c: c in ("global_x","global_y","gene"))
    gcol=next(c for c in d.columns if c=="gene")
    g=np.asarray(d[gcol].astype(str).to_numpy(),dtype="U")
    keep=~np.char.startswith(np.char.lower(g),"blank")
    genes=sorted(set(g[keep])); gi={v:i for i,v in enumerate(genes)}
    gidx=np.array([gi.get(v,-1) for v in g],np.int64); ok=gidx>=0
    return d["global_x"].to_numpy(float)[ok], d["global_y"].to_numpy(float)[ok], gidx[ok], genes


def infer_radius(gx,gy):
    from scipy.spatial import cKDTree
    if len(gx)<2: return 2.0
    d,_=cKDTree(np.c_[gx,gy]).query(np.c_[gx,gy],k=2); nn=d[:,1]; nn=nn[np.isfinite(nn)&(nn>0)]
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
            for t in np.argsort(sd):
                pass
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
    ap.add_argument("--run-dir",default=None)
    ap.add_argument("--h5ad",default=None)
    ap.add_argument("--dataset",default="x")
    ap.add_argument("--src-roi",type=int,default=10000,help="source ROI size of the assembled result (px)")
    ap.add_argument("--roi-origin",nargs=2,type=int,default=None,metavar=("Y0","X0"),help="true source ROI origin (native px), overrides the centered assumption")
    ap.add_argument("--src-hw",nargs=2,type=int,default=None,metavar=("H","W"),help="true image H W, overrides reading the image")
    ap.add_argument("--out",default=None)
    a=ap.parse_args()
    g=geometry(a.root,a.src_roi,a.roi_origin,a.src_hw)

    if a.run_dir:
        px,py,gm_p,genes_p=load_pred_tiles(a.run_dir,g,a.src_roi)
    elif a.h5ad:
        px,py,gm_p,genes_p=load_pred_h5ad(a.h5ad,g,a.src_roi)
    else:
        raise SystemExit("provide --run-dir or --h5ad")

    # keep predicted cells inside the central evaluation ROI
    X0,X1,Y0,Y1=eval_um_bounds(g)
    keep=(px>=X0)&(px<X1)&(py>=Y0)&(py<Y1)
    px,py,gm_p=px[keep],py[keep],gm_p[keep]

    gx,gy,roi_um=load_gt(a.root,g)
    tx_x,tx_y,tx_g,genes=load_tx(a.root)
    print(f"eval ROI(um)={tuple(round(v,1) for v in roi_um)}  pred_in_ROI={len(px)}  GT={len(gx)}")

    # GT gene vectors: assign transcripts to the nearest GT centroid
    from scipy.spatial import cKDTree
    dd,jj=cKDTree(np.c_[gx,gy]).query(np.c_[tx_x,tx_y],k=1)
    rad=max(np.median(dd)*2,3.0); ok=dd<=rad
    gm_gt=np.zeros((len(gx),len(genes)),np.float32); np.add.at(gm_gt,(jj[ok],tx_g[ok]),1.0)
    # align pred gene vectors to the gene table
    gi={x:i for i,x in enumerate(genes)}
    remap=np.array([gi.get(x,-1) for x in genes_p])
    gm_pred=np.zeros((len(px),len(genes)),np.float32)
    valid=remap>=0
    gm_pred[:,remap[valid]]=gm_p[:,np.where(valid)[0]]

    r=infer_radius(gx,gy); ep,eg,ed=match(px,py,gx,gy,r); n=len(ep)
    prec=n/max(len(px),1); rec=n/max(len(gx),1); f1=2*prec*rec/(prec+rec) if prec+rec else np.nan
    vc,vjs,vpr,vp=gene_metrics(gm_pred[ep],gm_gt[eg]) if n else (np.nan,)*4
    from scipy.stats import pearsonr,spearmanr
    def corr(x,y):
        m=np.isfinite(x)&np.isfinite(y)
        if m.sum()<3 or x[m].std()==0 or y[m].std()==0: return np.nan,np.nan
        return float(pearsonr(x[m],y[m])[0]),float(spearmanr(x[m],y[m])[0])
    cp,csp=corr(gm_pred[ep].sum(1),gm_gt[eg].sum(1)) if n else (np.nan,np.nan)

    R={"method":"ComSeg","dataset":a.dataset,"gt_cells":len(gx),"pred_cells":len(px),
       "cell_ratio":round(len(px)/max(len(gx),1),4),"matched":n,
       "precision":round(prec,4),"recall":round(rec,4),"f1":round(f1,4),
       "loc_mean_um":round(float(ed.mean()),4) if n else np.nan,
       "loc_median_um":round(float(np.median(ed)),4) if n else np.nan,
       "loc_p95_um":round(float(np.percentile(ed,95)),4) if n else np.nan,
       "count_pearson":round(cp,4) if cp==cp else np.nan,
       "count_spearman":round(csp,4) if csp==csp else np.nan,
       "vec_cosine":round(vc,4) if vc==vc else np.nan,
       "vec_js_dist":round(vjs,4) if vjs==vjs else np.nan,
       "vec_pearson":round(vpr,4) if vpr==vpr else np.nan,
       "uses_platform_prior":False}
    print("="*56)
    for k,v in R.items(): print(f"  {k:20s} {v}")
    print("="*56)
    out=a.out or f"yourpath/comseg_value_{a.dataset}.csv"
    pd.DataFrame([R]).to_csv(out,index=False); print("written ->",out)


if __name__=="__main__":
    main()