#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boms_value_xenium_5.py
Comprehensive evaluator for the FINAL BOMS xenium5 mouse-brain run.

Default prediction:
  /data/qiuyijia/boms_xenium_5_hs9_g3000/boms_xenium_5_hs9_g3000.npz

Important:
  Uses NPZ key `cell_loc` (the completed run contains it).
  Automatically determines whether BOMS coordinates are:
    local xy / local yx / global xy / global yx
  by comparing nearest-neighbour distance to Xenium GT centroids.

Metrics:
  detection/localization
  cell count ratio
  tx/cell distribution
  count correlation on matched cells
  expression-vector cosine / JS distance / Pearson on matched cells
  median predicted area/diameter (Voronoi-free transcript-support estimate is
  not used; if no raster mask exists, area metrics remain NaN)
  assignment rate = fraction of BOMS transcript rows assigned to a valid cell

GT:
  Xenium cells.parquet + transcripts.parquet, QV>=20, ROI centroid filtered.

This evaluator does NOT use the rescued/stitched postprocessing unless you
explicitly pass that NPZ with --pred.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

PIX = 0.2125
ROI_Y0 = 6956
ROI_X0 = 12077
ROI_SIZE = 10000

PRED_DEFAULT = "/data/qiuyijia/boms_xenium_5_hs9_g3000/boms_xenium_5_hs9_g3000.npz"
ROOT_DEFAULT = "/data/qiuyijia/dataset/xenium_mouse_brain"
OUT_DEFAULT = "/data/qiuyijia/eval_results_qv20/mouse_brain/boms_xenium_5_metrics.csv"


def safe_corr(a,b,kind):
    a=np.asarray(a,float); b=np.asarray(b,float)
    m=np.isfinite(a)&np.isfinite(b)
    a=a[m]; b=b[m]
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:
        return np.nan
    r = pearsonr(a,b) if kind=="pearson" else spearmanr(a,b)
    return float(r[0])


def choose_col(cols,cands,required=True):
    low={str(c).lower():c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise KeyError(f"cannot find any of {cands} in {list(cols)}")
    return None


def load_gt_cells(root):
    p=Path(root)/"outs/cells.parquet"
    if not p.exists():
        p=Path(root)/"cells.parquet"
    d=pd.read_parquet(p)
    xc=choose_col(d.columns,("x_centroid","center_x","x"))
    yc=choose_col(d.columns,("y_centroid","center_y","y"))
    idc=choose_col(d.columns,("cell_id","id"))
    x0=ROI_X0*PIX; x1=(ROI_X0+ROI_SIZE)*PIX
    y0=ROI_Y0*PIX; y1=(ROI_Y0+ROI_SIZE)*PIX
    m=(pd.to_numeric(d[xc],errors="coerce").ge(x0)&
       pd.to_numeric(d[xc],errors="coerce").lt(x1)&
       pd.to_numeric(d[yc],errors="coerce").ge(y0)&
       pd.to_numeric(d[yc],errors="coerce").lt(y1))
    d=d.loc[m,[idc,xc,yc]].copy().reset_index(drop=True)
    d.columns=["cell_id","x","y"]
    return d


def coordinate_candidates(loc):
    ox=ROI_X0*PIX; oy=ROI_Y0*PIX
    a=np.asarray(loc,float)[:,:2]
    return {
        "global_xy": np.c_[a[:,0],a[:,1]],
        "global_yx": np.c_[a[:,1],a[:,0]],
        "local_xy":  np.c_[a[:,0]+ox,a[:,1]+oy],
        "local_yx":  np.c_[a[:,1]+ox,a[:,0]+oy],
    }


def pick_coordinate_mode(loc,gt_xy):
    tree=cKDTree(gt_xy)
    rows=[]
    for name,xy in coordinate_candidates(loc).items():
        dist,_=tree.query(xy,k=1)
        rows.append((float(np.median(dist)),float(np.percentile(dist,90)),name,xy))
    rows.sort(key=lambda x:x[0])
    print("coordinate mode audit:")
    for med,p90,name,_ in rows:
        print(f"  {name:10s} medianNN={med:.3f} um  p90={p90:.3f} um")
    return rows[0][2],rows[0][3]


def component_hungarian(pred,gt,radius):
    """
    Exact Hungarian within connected components of the sparse radius graph.
    Avoids constructing a 10k x 12k dense matrix.
    """
    tp=cKDTree(pred); tg=cKDTree(gt)
    neigh=tp.query_ball_tree(tg,r=radius)
    edges=[(i,j) for i,js in enumerate(neigh) for j in js]
    if not edges:
        return np.empty(0,int),np.empty(0,int),np.empty(0,float)

    p2g={}
    g2p={}
    for i,j in edges:
        p2g.setdefault(i,[]).append(j)
        g2p.setdefault(j,[]).append(i)

    seenp=set(); seeng=set()
    MP=[]; MG=[]; MD=[]
    for start in p2g:
        if start in seenp: continue
        ps=set([start]); gs=set(); frontier_p=[start]
        while frontier_p:
            i=frontier_p.pop()
            if i in seenp: continue
            seenp.add(i)
            for j in p2g.get(i,[]):
                if j not in gs:
                    gs.add(j)
                    if j not in seeng:
                        seeng.add(j)
                        for ii in g2p.get(j,[]):
                            if ii not in ps:
                                ps.add(ii); frontier_p.append(ii)
        pl=sorted(ps); gl=sorted(gs)
        cost=np.full((len(pl),len(gl)),radius*1000.0,dtype=np.float64)
        gpos={g:k for k,g in enumerate(gl)}
        for ii,pidx in enumerate(pl):
            for gidx in p2g.get(pidx,[]):
                if gidx in gpos:
                    cost[ii,gpos[gidx]]=np.linalg.norm(pred[pidx]-gt[gidx])
        rr,cc=linear_sum_assignment(cost)
        for r,c in zip(rr,cc):
            if cost[r,c] <= radius:
                MP.append(pl[r]); MG.append(gl[c]); MD.append(cost[r,c])
    return np.asarray(MP,int),np.asarray(MG,int),np.asarray(MD,float)


def load_gt_expression(root,gt_cells,genes,qv_min):
    """
    Aggregate ROI transcript counts for the GT cells and selected BOMS genes.
    """
    import pyarrow.parquet as pq
    p=Path(root)/"outs/transcripts.parquet"
    if not p.exists(): p=Path(root)/"transcripts.parquet"
    schema=pq.read_schema(p)
    cols=schema.names
    idc=choose_col(cols,("cell_id",))
    gc=choose_col(cols,("feature_name","gene","gene_name"))
    qc=choose_col(cols,("qv","quality_value"),False)

    wanted={str(x):i for i,x in enumerate(genes)}
    cell_map={str(x):i for i,x in enumerate(gt_cells["cell_id"].astype(str))}
    M=np.zeros((len(gt_cells),len(genes)),dtype=np.int32)

    pf=pq.ParquetFile(p)
    for bi in range(pf.num_row_groups):
        use=[idc,gc]+([qc] if qc else [])
        q=pf.read_row_group(bi,columns=use).to_pandas()
        if qc:
            q=q[pd.to_numeric(q[qc],errors="coerce")>=qv_min]
        if q.empty: continue
        cs=q[idc].astype(str)
        gs=q[gc].astype(str)
        cm=cs.map(cell_map)
        gm=gs.map(wanted)
        keep=cm.notna()&gm.notna()
        if not keep.any(): continue
        ci=cm[keep].astype(np.int64).to_numpy()
        gi=gm[keep].astype(np.int64).to_numpy()
        np.add.at(M,(ci,gi),1)
    return M


def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pred",default=PRED_DEFAULT)
    ap.add_argument("--root",default=ROOT_DEFAULT)
    ap.add_argument("--out",default=OUT_DEFAULT)
    ap.add_argument("--qv-min",type=float,default=20)
    ap.add_argument("--match-radius",type=float,default=10.0)
    ap.add_argument("--vec-limit",type=int,default=3000,
                    help="matched pairs used for expression-vector metrics")
    args=ap.parse_args()

    z=np.load(args.pred,allow_pickle=True)
    print("BOMS keys:",z.files)
    if "cell_loc" not in z.files:
        raise RuntimeError(f"cell_loc missing; keys={z.files}")
    loc=np.asarray(z["cell_loc"])
    C=np.asarray(z["count_mat"])
    if len(loc)!=C.shape[0]:
        raise RuntimeError(f"cell_loc rows {len(loc)} != count_mat rows {C.shape[0]}")

    genes=np.asarray(z["gene_names"]).astype(str) if "gene_names" in z.files else \
          np.asarray(z["cluster_gene_names"]).astype(str)
    if C.shape[1]!=len(genes):
        raise RuntimeError(f"count_mat genes {C.shape[1]} != gene_names {len(genes)}")

    gt=load_gt_cells(args.root)
    gt_xy=gt[["x","y"]].to_numpy(float)
    mode,pred_xy=pick_coordinate_mode(loc,gt_xy)

    mp,mg,dist=component_hungarian(pred_xy,gt_xy,args.match_radius)
    pred_n=len(pred_xy); gt_n=len(gt)
    precision=len(mp)/pred_n
    recall=len(mp)/gt_n
    f1=2*precision*recall/max(precision+recall,1e-12)

    pred_counts=np.asarray(C.sum(axis=1)).ravel().astype(float)

    print(f"GT cells={gt_n:,} pred={pred_n:,} matched={len(mp):,}")
    print("building matched GT expression matrix...")
    gtM=load_gt_expression(args.root,gt,genes,args.qv_min)
    gt_counts=gtM.sum(axis=1).astype(float)

    pc=pred_counts[mp]; gc=gt_counts[mg]
    count_pearson=safe_corr(pc,gc,"pearson")
    count_spearman=safe_corr(pc,gc,"spearman")
    count_mae=float(np.mean(np.abs(pc-gc))) if len(pc) else np.nan
    count_rmse=float(np.sqrt(np.mean((pc-gc)**2))) if len(pc) else np.nan

    # Vector metrics on a deterministic subset to bound memory/time.
    take=np.arange(min(len(mp),args.vec_limit))
    cos=[]; js=[]; vp=[]
    for k in take:
        a=C[mp[k]].astype(float)
        b=gtM[mg[k]].astype(float)
        den=np.linalg.norm(a)*np.linalg.norm(b)
        cos.append(float(a@b/den) if den else np.nan)
        sa=a.sum(); sb=b.sum()
        if sa>0 and sb>0:
            js.append(float(jensenshannon(a/sa,b/sb,base=2)))
            vp.append(safe_corr(a,b,"pearson"))
    vec_cos=float(np.nanmean(cos)) if cos else np.nan
    vec_js=float(np.nanmean(js)) if js else np.nan
    vec_pear=float(np.nanmean(vp)) if vp else np.nan

    # BOMS assigns every transcript row to seg; compute actual valid fraction.
    seg=np.asarray(z["seg"])
    if seg.min()>=1:
        valid=(seg>=1)&(seg<=pred_n)
    else:
        valid=(seg>=0)&(seg<pred_n)
    assign_rate=float(valid.mean())

    result=dict(
        method="BOMS",dataset="xenium_5",tissue="mouse_brain",
        gt_cells=gt_n,pred_cells=pred_n,matched=int(len(mp)),
        cell_ratio=pred_n/gt_n,precision=precision,recall=recall,f1=f1,
        loc_mean_um=float(np.mean(dist)),loc_median_um=float(np.median(dist)),
        loc_p95_um=float(np.percentile(dist,95)),match_radius_um=args.match_radius,
        coordinate_mode=mode,
        assign_rate=assign_rate,
        tx_per_cell_median=float(np.median(pred_counts)),
        tx_per_cell_mean=float(np.mean(pred_counts)),
        tx_per_cell_p5=float(np.percentile(pred_counts,5)),
        tx_per_cell_p95=float(np.percentile(pred_counts,95)),
        frag_lt10=float((pred_counts<10).mean()),
        count_pearson=count_pearson,count_spearman=count_spearman,
        count_mae=count_mae,count_rmse=count_rmse,
        vec_cosine=vec_cos,vec_js_dist=vec_js,vec_pearson=vec_pear,
        vec_n=int(len(take)),vec_genes=int(len(genes)),
        uses_platform_prior=False,detection_comparable=True,
        assignment_independent=True,vector_independent=True,
        pred_path=args.pred,qv_min=args.qv_min,
    )
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([result]).to_csv(out,index=False)
    pd.DataFrame({
        "pred_index":mp,"gt_index":mg,"distance_um":dist,
        "pred_tx":pc,"gt_tx":gc,
        "gt_cell_id":gt.iloc[mg]["cell_id"].astype(str).to_numpy()
    }).to_csv(out.with_name(out.stem.replace("_metrics","")+"_pairs.csv"),index=False)
    out.with_suffix(".json").write_text(json.dumps(result,indent=2,ensure_ascii=False))

    print("="*88)
    for k,v in result.items():
        print(f"{k:28s} {v}")
    print("="*88)
    print("metrics ->",out)


if __name__=="__main__":
    main()
