#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCS × STARmap BY1 CLEAN — 19-metric evaluation.

Preserves the previous benchmark logic:
GT-order nearest-neighbour matching with cKDTree, a finite match radius,
and each predicted cell can be used only once.

Metrics:
gt_cells, pred_cells, cell_ratio, matched, precision, recall, f1,
loc_mean_um, loc_median_um, loc_p95_um,
count_pearson, count_spearman, count_mae, count_rmse,
vec_cosine, vec_js_dist, vec_pearson,
assign_overlap, assign_accuracy.

IMPORTANT:
STARmap coordinates here are native image pixels. --um-per-pixel is required;
the script will not silently call pixels microns.
"""
import os, argparse
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

ROOT="/data/qiuyijia/dataset/starmap_BY1"
SPOTS=os.path.join(ROOT,"spots_all.csv")
DEFAULT_MASK="/data/qiuyijia/ucs_starmap_BY1_clean/ucs_log/pred/segmentation_mask.tif"
DEFAULT_OUT="/data/qiuyijia/eval_results/ucs/starmap_BY1_clean.csv"

METRICS=[
 "gt_cells","pred_cells","cell_ratio","matched","precision","recall","f1",
 "loc_mean_um","loc_median_um","loc_p95_um",
 "count_pearson","count_spearman","count_mae","count_rmse",
 "vec_cosine","vec_js_dist","vec_pearson","assign_overlap","assign_accuracy"
]

def corr(fn,a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ok=np.isfinite(a)&np.isfinite(b); a=a[ok]; b=b[ok]
    if len(a)<3 or np.std(a)==0 or np.std(b)==0: return np.nan
    try: return float(fn(a,b)[0])
    except Exception: return np.nan

def match_cells(gt,pred,radius):
    if len(gt)==0 or len(pred)==0:
        return pd.DataFrame(columns=["gt_id","pred_id","distance_um"])
    tree=cKDTree(pred[["x_um","y_um"]].to_numpy(float))
    d,idx=tree.query(gt[["x_um","y_um"]].to_numpy(float),
                     k=1,distance_upper_bound=radius)
    used=set(); rows=[]
    pids=pred.pred_id.astype(str).to_numpy()
    gids=gt.gt_id.astype(str).to_numpy()
    for gi in range(len(gt)):
        if not np.isfinite(d[gi]) or d[gi]>=radius: continue
        pi=int(idx[gi]); pid=pids[pi]
        if pid in used: continue
        used.add(pid); rows.append((gids[gi],pid,float(d[gi])))
    return pd.DataFrame(rows,columns=["gt_id","pred_id","distance_um"])

def expression_metrics(tx,match,vec_sample):
    if match.empty:
        return dict(count_pearson=np.nan,count_spearman=np.nan,count_mae=np.nan,
                    count_rmse=np.nan,vec_cosine=np.nan,vec_js_dist=np.nan,
                    vec_pearson=np.nan)
    gt_counts=tx[tx.gt_cell!="0"].groupby("gt_cell").size()
    pr_counts=tx[tx.pred_cell!="0"].groupby("pred_cell").size()
    g=[]; p=[]
    for r in match.itertuples():
        g.append(gt_counts.get(str(r.gt_id),0))
        p.append(pr_counts.get(str(r.pred_id),0))
    g=np.asarray(g,float); p=np.asarray(p,float)
    out=dict(count_pearson=corr(pearsonr,g,p),
             count_spearman=corr(spearmanr,g,p),
             count_mae=float(np.mean(np.abs(g-p))),
             count_rmse=float(np.sqrt(np.mean((g-p)**2))))
    mm=match if len(match)<=vec_sample else match.sample(vec_sample,random_state=0)
    gt2i={str(x):i for i,x in enumerate(mm.gt_id)}
    pr2i={str(x):i for i,x in enumerate(mm.pred_id)}
    genes=sorted(tx.gene.astype(str).unique()); g2j={x:i for i,x in enumerate(genes)}
    G=np.zeros((len(mm),len(genes)),np.float32); P=np.zeros_like(G)
    for (c,gn),n in tx[tx.gt_cell.isin(gt2i)].groupby(["gt_cell","gene"]).size().items():
        if str(c) in gt2i: G[gt2i[str(c)],g2j[str(gn)]]=n
    for (c,gn),n in tx[tx.pred_cell.isin(pr2i)].groupby(["pred_cell","gene"]).size().items():
        if str(c) in pr2i: P[pr2i[str(c)],g2j[str(gn)]]=n
    cos=[]; js=[]; prs=[]
    for a,b in zip(G,P):
        na=np.linalg.norm(a); nb=np.linalg.norm(b)
        if na==0 or nb==0: continue
        cos.append(float(np.dot(a,b)/(na*nb)))
        js.append(float(jensenshannon(a/a.sum(),b/b.sum(),base=2)))
        if np.std(a)>0 and np.std(b)>0:
            prs.append(float(pearsonr(a,b)[0]))
    out.update(vec_cosine=float(np.mean(cos)) if cos else np.nan,
               vec_js_dist=float(np.mean(js)) if js else np.nan,
               vec_pearson=float(np.mean(prs)) if prs else np.nan)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mask",default=DEFAULT_MASK)
    ap.add_argument("--bin-factor",type=float,default=10.0)
    ap.add_argument("--um-per-pixel",type=float,required=True)
    ap.add_argument("--match-radius-um",type=float,default=12.0)
    ap.add_argument("--vec-sample",type=int,default=2000)
    ap.add_argument("--out",default=DEFAULT_OUT)
    a=ap.parse_args()

    print("="*78); print("UCS × STARmap BY1 CLEAN evaluation"); print("="*78)
    s=pd.read_csv(SPOTS,usecols=["gene_name","spot_location_1","spot_location_2","clustermap"])
    gt_num=pd.to_numeric(s.clustermap,errors="coerce").fillna(-1)
    s["gt_cell"]=gt_num.astype(np.int64).astype(str)
    s.loc[gt_num<0,"gt_cell"]="0"
    s["gene"]=s.gene_name.astype(str)
    s["x_native"]=s.spot_location_1.astype(float)
    s["y_native"]=s.spot_location_2.astype(float)

    mask=np.squeeze(tifffile.imread(a.mask))
    if mask.ndim!=2: raise RuntimeError(f"mask must be 2D, got {mask.shape}")
    H,W=mask.shape
    col=np.floor(s.x_native.to_numpy()/a.bin_factor).astype(np.int64)
    row=np.floor(s.y_native.to_numpy()/a.bin_factor).astype(np.int64)
    inb=(row>=0)&(row<H)&(col>=0)&(col<W)
    pred=np.zeros(len(s),np.int64); pred[inb]=mask[row[inb],col[inb]]
    s["pred_cell"]=pred.astype(str); s.loc[pred<=0,"pred_cell"]="0"

    gt=s[s.gt_cell!="0"].groupby("gt_cell")[["x_native","y_native"]].mean().reset_index()
    gt=gt.rename(columns={"gt_cell":"gt_id","x_native":"x_um","y_native":"y_um"})
    pr=s[s.pred_cell!="0"].groupby("pred_cell")[["x_native","y_native"]].mean().reset_index()
    pr=pr.rename(columns={"pred_cell":"pred_id","x_native":"x_um","y_native":"y_um"})
    gt[["x_um","y_um"]]*=a.um_per_pixel; pr[["x_um","y_um"]]*=a.um_per_pixel

    gt_n=len(gt); pred_n=int(np.sum(np.unique(mask)>0))
    m=match_cells(gt,pr,a.match_radius_um)
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    m.to_csv(os.path.splitext(a.out)[0]+"_matched_cells.csv",index=False)

    em=expression_metrics(s[["gene","gt_cell","pred_cell"]],m,a.vec_sample)
    gt_ass=s.gt_cell!="0"; pr_ass=s.pred_cell!="0"; both=gt_ass&pr_ass
    em["assign_overlap"]=float(both.sum()/max(int(gt_ass.sum()),1))
    p2g={str(r.pred_id):str(r.gt_id) for r in m.itertuples()}
    mapped=s.loc[both,"pred_cell"].map(p2g)
    em["assign_accuracy"]=float((mapped.to_numpy(object)==s.loc[both,"gt_cell"].to_numpy(object)).mean()) if both.any() else np.nan

    precision=len(m)/max(pred_n,1); recall=len(m)/max(gt_n,1)
    f1=2*precision*recall/max(precision+recall,1e-12)
    d=m.distance_um.to_numpy(float)
    rowout=dict(method="UCS",dataset="starmap_BY1",gt_cells=gt_n,pred_cells=pred_n,
      cell_ratio=pred_n/max(gt_n,1),matched=len(m),precision=precision,recall=recall,f1=f1,
      loc_mean_um=np.mean(d) if len(d) else np.nan,
      loc_median_um=np.median(d) if len(d) else np.nan,
      loc_p95_um=np.percentile(d,95) if len(d) else np.nan,**em,
      note=("CLEAN UCS. ClusterMap segmentation was not supplied to inference. "
            "DAPI-prior parameters were selected using GT cell-count proximity "
            "(count-tuned, non-circular inference)."))
    cols=["method","dataset"]+METRICS+["note"]
    pd.DataFrame([rowout]).reindex(columns=cols).to_csv(a.out,index=False)
    for k in METRICS: print(f"{k:22s} {rowout[k]}")
    print("saved ->",a.out)

if __name__=="__main__": main()
