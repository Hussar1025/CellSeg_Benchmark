#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComSeg × MERFISH liver1/liver2 — 19-metric evaluation.

All 19 metrics are calculated. In with-prior ComSeg, detection/localization are
explicitly flagged as prior-assisted because the supplied landmark centroids are fixed;
assignment/count/gene-vector metrics are the informative part.

The script uses the pre-segmentation ROI cache as GT transcript data, not another
method's prediction.
"""
import os, argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

CFG={
"liver1":{
 "h5ad":"yourpath/comseg/merfish_liver1/output/merfish_liver1_roi_top300_mmc0p99.h5ad",
 "cache":"yourpath/comseg/merfish_liver1/cache/merfish_liver1_roi.csv"},
"liver2":{
 "h5ad":"yourpath/comseg/merfish_liver2/output/merfish_liver2_roi_top300_mmc0p99.h5ad",
 "cache":"yourpath/comseg/merfish_liver2/cache/merfish_liver2_roi.csv"},
}
METRICS=["gt_cells","pred_cells","cell_ratio","matched","precision","recall","f1",
"loc_mean_um","loc_median_um","loc_p95_um","count_pearson","count_spearman",
"count_mae","count_rmse","vec_cosine","vec_js_dist","vec_pearson",
"assign_overlap","assign_accuracy"]

def pick(cols,cands):
    d={str(c).lower():c for c in cols}
    for x in cands:
        if x.lower() in d:return d[x.lower()]
    raise RuntimeError(f"Missing one of {cands}; columns={list(cols)}")

def corr(fn,a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return np.nan
    try:return float(fn(a,b)[0])
    except:return np.nan

def extract_spots(A):
    u=A.uns.get("df_spots",None); cand=[]
    if isinstance(u,pd.DataFrame):cand=[u]
    elif isinstance(u,dict):
        for v in u.values():
            if isinstance(v,pd.DataFrame):cand.append(v)
            elif isinstance(v,dict):
                try:cand.append(pd.DataFrame(v))
                except:pass
    if not cand:
        raise RuntimeError("No readable per-transcript DataFrame in adata.uns['df_spots']; "
                           "cannot honestly compute assignment metrics.")
    return max(cand,key=len).copy()

def greedy(gt,pr,r):
    tree=cKDTree(pr[["x_um","y_um"]].to_numpy(float))
    d,idx=tree.query(gt[["x_um","y_um"]].to_numpy(float),k=1,distance_upper_bound=r)
    used=set(); rows=[]; pids=pr.pred_id.astype(str).to_numpy()
    for i,gid in enumerate(gt.gt_id.astype(str)):
        if not np.isfinite(d[i]) or d[i]>=r:continue
        pid=pids[int(idx[i])]
        if pid in used:continue
        used.add(pid); rows.append((gid,pid,float(d[i])))
    return pd.DataFrame(rows,columns=["gt_id","pred_id","distance_um"])

def expr(T,M,nmax):
    gc=T[T.gt_cell!="0"].groupby("gt_cell").size()
    pc=T[T.pred_cell!="0"].groupby("pred_cell").size()
    g=np.array([gc.get(str(x),0) for x in M.gt_id],float)
    p=np.array([pc.get(str(x),0) for x in M.pred_id],float)
    out=dict(count_pearson=corr(pearsonr,g,p),count_spearman=corr(spearmanr,g,p),
             count_mae=float(np.mean(abs(g-p))),count_rmse=float(np.sqrt(np.mean((g-p)**2))))
    MM=M if len(M)<=nmax else M.sample(nmax,random_state=0)
    genes=sorted(T.gene.unique()); g2j={x:i for i,x in enumerate(genes)}
    gi={str(x):i for i,x in enumerate(MM.gt_id)}; pi={str(x):i for i,x in enumerate(MM.pred_id)}
    G=np.zeros((len(MM),len(genes)),np.float32); P=np.zeros_like(G)
    for (c,x),n in T[T.gt_cell.isin(gi)].groupby(["gt_cell","gene"]).size().items():
        if str(c) in gi:G[gi[str(c)],g2j[x]]=n
    for (c,x),n in T[T.pred_cell.isin(pi)].groupby(["pred_cell","gene"]).size().items():
        if str(c) in pi:P[pi[str(c)],g2j[x]]=n
    cs=[]; js=[]; ps=[]
    for a,b in zip(G,P):
        if a.sum()==0 or b.sum()==0:continue
        cs.append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
        js.append(float(jensenshannon(a/a.sum(),b/b.sum(),base=2)))
        if np.std(a)>0 and np.std(b)>0:ps.append(float(pearsonr(a,b)[0]))
    out.update(vec_cosine=np.mean(cs) if cs else np.nan,
               vec_js_dist=np.mean(js) if js else np.nan,
               vec_pearson=np.mean(ps) if ps else np.nan)
    return out

def clean(s):
    s=s.astype(str); bad=s.str.lower().isin(["0","-1","nan","none","unassigned","background"])
    return s.mask(bad,"0")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True,choices=CFG)
    ap.add_argument("--h5ad"); ap.add_argument("--cache")
    ap.add_argument("--match-radius-um",type=float,default=12.0)
    ap.add_argument("--vec-sample",type=int,default=2000)
    ap.add_argument("--no-prior",action="store_true")
    ap.add_argument("--out")
    a=ap.parse_args(); c=CFG[a.dataset]
    hp=a.h5ad or c["h5ad"]; cp=a.cache or c["cache"]
    out=a.out or f"yourpath/eval_results/comseg/merfish_{a.dataset}.csv"

    A=ad.read_h5ad(hp); G=pd.read_csv(cp); P=extract_spots(A)
    gx=pick(G.columns,["x","global_x","x_um","x_location"]); gy=pick(G.columns,["y","global_y","y_um","y_location"])
    gg=pick(G.columns,["gene","gene_name","target"]); gc=pick(G.columns,["gt_cell_id","cell_id_gt","original_cell_id","cell_id","cell"])
    px=pick(P.columns,["x","global_x","x_um","x_location"]); py=pick(P.columns,["y","global_y","y_um","y_location"])
    pg=pick(P.columns,["gene","gene_name","target"]); pc=pick(P.columns,["leiden_merged","pred_cell","cell_id_pred","cell_id","CellID","cell"])
    print("GT cols:",gx,gy,gg,gc); print("Pred cols:",px,py,pg,pc)

    # Validate same-order fast path; otherwise duplicate-safe coordinate/gene merge.
    same=False
    if len(G)==len(P):
        n=min(1000,len(G))
        same=(np.allclose(pd.to_numeric(G[gx].iloc[:n]),pd.to_numeric(P[px].iloc[:n]),atol=1e-6)
              and np.allclose(pd.to_numeric(G[gy].iloc[:n]),pd.to_numeric(P[py].iloc[:n]),atol=1e-6)
              and np.all(G[gg].iloc[:n].astype(str).to_numpy()==P[pg].iloc[:n].astype(str).to_numpy()))
    if same:
        T=pd.DataFrame({"x_um":pd.to_numeric(G[gx]),"y_um":pd.to_numeric(G[gy]),
                        "gene":G[gg].astype(str),"gt_cell":clean(G[gc]),
                        "pred_cell":clean(P[pc])})
    else:
        L=pd.DataFrame({"x":pd.to_numeric(G[gx]),"y":pd.to_numeric(G[gy]),"gene":G[gg].astype(str),"gt_cell":clean(G[gc])})
        R=pd.DataFrame({"x":pd.to_numeric(P[px]),"y":pd.to_numeric(P[py]),"gene":P[pg].astype(str),"pred_cell":clean(P[pc])})
        for D in (L,R):
            D["kx"]=D.x.round(6); D["ky"]=D.y.round(6); D["occ"]=D.groupby(["kx","ky","gene"]).cumcount()
        T=L.merge(R,on=["kx","ky","gene","occ"],how="inner")
        if len(T)<0.95*min(len(L),len(R)):
            raise RuntimeError(f"Transcript alignment only {len(T):,}; stop.")
        T["x_um"]=T.kx; T["y_um"]=T.ky
        T=T[["x_um","y_um","gene","gt_cell","pred_cell"]]

    gt=T[T.gt_cell!="0"].groupby("gt_cell")[["x_um","y_um"]].mean().reset_index().rename(columns={"gt_cell":"gt_id"})
    pr=T[T.pred_cell!="0"].groupby("pred_cell")[["x_um","y_um"]].mean().reset_index().rename(columns={"pred_cell":"pred_id"})
    M=greedy(gt,pr,a.match_radius_um)
    gt_n=len(gt); pred_n=len(pr)
    os.makedirs(os.path.dirname(out),exist_ok=True); M.to_csv(os.path.splitext(out)[0]+"_matched_cells.csv",index=False)
    E=expr(T,M,a.vec_sample)

    ga=T.gt_cell!="0"; pa=T.pred_cell!="0"; both=ga&pa
    E["assign_overlap"]=float(both.sum()/max(int(ga.sum()),1))
    p2g={str(r.pred_id):str(r.gt_id) for r in M.itertuples()}
    mapped=T.loc[both,"pred_cell"].map(p2g)
    E["assign_accuracy"]=float((mapped.to_numpy(object)==T.loc[both,"gt_cell"].to_numpy(object)).mean()) if both.any() else np.nan

    precision=len(M)/max(pred_n,1); recall=len(M)/max(gt_n,1); f1=2*precision*recall/max(precision+recall,1e-12)
    d=M.distance_um.to_numpy(float)
    row=dict(method="ComSeg",dataset=f"merfish_{a.dataset}",gt_cells=gt_n,pred_cells=pred_n,
             cell_ratio=pred_n/max(gt_n,1),matched=len(M),precision=precision,recall=recall,f1=f1,
             loc_mean_um=np.mean(d) if len(d) else np.nan,loc_median_um=np.median(d) if len(d) else np.nan,
             loc_p95_um=np.percentile(d,95) if len(d) else np.nan,**E,
             note=("ComSeg no-prior mode." if a.no_prior else
                   "WITH-PRIOR ComSeg: all 19 metrics calculated, but detection/localization are "
                   "prior-assisted by fixed landmark centroids; assignment/count/vector metrics are informative."))
    pd.DataFrame([row]).reindex(columns=["method","dataset"]+METRICS+["note"]).to_csv(out,index=False)
    for k in METRICS:print(f"{k:22s} {row[k]}")
    print("saved ->",out)

if __name__=="__main__":main()
