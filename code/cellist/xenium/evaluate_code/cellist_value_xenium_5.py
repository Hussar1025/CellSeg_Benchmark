#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellist_value_xenium_5_mouse_brain.py
Evaluator for the COMPLETED Cellist Xenium-5 mouse brain run.

Known completed result:
  /data/qiuyijia/cellist_xenium_liver_tune/r7_a0.05_b2_two1_local0
  QC: 2,942 cells, platform reference ~11,963.

The evaluator auto-locates:
  cellist_segmentation.tif / segmentation*.tif / label*.tif
and uses the raster itself for predicted centroids and area.

Cellist raster grid:
  gem-bin=2 native Xenium pixels
  native pixel=0.2125 um
  raster pixel=0.425 um

ROI:
  y0=13704 x0=21974 size=10000 native px

Metrics:
  detection/localization
  count ratio
  median area / equivalent diameter
  transcript assignment rate
  matched-cell transcript count correlation

It intentionally evaluates the completed result even though QC is poor.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.stats import pearsonr,spearmanr

NATIVE_PIX=0.2125
# Final Cellist raster is not necessarily on the GEM input grid.
# Infer raster pixel size from ROI native-pixel extent / final mask shape.
ROI_Y0=6956
ROI_X0=12077
ROI_SIZE=10000

ROOT_DEFAULT="/data/qiuyijia/dataset/xenium_mouse_brain"
RUN_DEFAULT="/data/qiuyijia/cellist_xenium_mouse_brain_roi10000"
OUT_DEFAULT="/data/qiuyijia/eval_results_qv20/mouse_brain/cellist_xenium_5_metrics.csv"


def choose_col(cols,cands,required=True):
    low={str(c).lower():c for c in cols}
    for x in cands:
        if x.lower() in low:return low[x.lower()]
    if required:raise KeyError(f"missing {cands} in {list(cols)}")
    return None


def safe_corr(a,b,kind):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return np.nan
    return float((pearsonr(a,b) if kind=="pearson" else spearmanr(a,b))[0])


def find_mask(run):
    run=Path(run)
    exact=[
        run/"cellist_segmentation.tif",
        run/"segmentation/cellist_segmentation.tif",
        run/"output/cellist_segmentation.tif",
        run/"cellist_segmentation_bin.tif",
        run/"segmentation_mask.tif",
    ]
    for p in exact:
        if p.exists() and "bin" not in p.name:
            return p
    fs=sorted(run.rglob("*.tif"))
    fs=[p for p in fs if any(k in p.name.lower() for k in ("segmentation","label","mask"))
        and "bin" not in p.name.lower()]
    if not fs:raise FileNotFoundError(f"no segmentation tif under {run}")
    return fs[0]


def mask_props(mask):
    lab,cnt=np.unique(mask,return_counts=True)
    keep=lab>0
    labels=lab[keep].astype(np.int64); area=cnt[keep].astype(np.int64)
    lut={int(l):i for i,l in enumerate(labels)}
    yy,xx=np.nonzero(mask>0)
    vv=mask[yy,xx]
    idx=np.fromiter((lut[int(v)] for v in vv),dtype=np.int64,count=len(vv))
    n=len(labels)
    nn=np.bincount(idx,minlength=n)
    cx=np.bincount(idx,weights=xx,minlength=n)/np.maximum(nn,1)
    cy=np.bincount(idx,weights=yy,minlength=n)/np.maximum(nn,1)
    return labels,area,np.c_[cx,cy]


def load_gt(root):
    p=Path(root)/"outs/cells.parquet"
    if not p.exists():p=Path(root)/"cells.parquet"
    d=pd.read_parquet(p)
    idc=choose_col(d.columns,("cell_id","id"))
    xc=choose_col(d.columns,("x_centroid","center_x","x"))
    yc=choose_col(d.columns,("y_centroid","center_y","y"))
    x0=ROI_X0*NATIVE_PIX;x1=(ROI_X0+ROI_SIZE)*NATIVE_PIX
    y0=ROI_Y0*NATIVE_PIX;y1=(ROI_Y0+ROI_SIZE)*NATIVE_PIX
    m=(d[xc].ge(x0)&d[xc].lt(x1)&d[yc].ge(y0)&d[yc].lt(y1))
    q=d.loc[m,[idc,xc,yc]].copy().reset_index(drop=True)
    q.columns=["cell_id","x","y"]
    return q


def sparse_match(pred,gt,radius):
    tp=cKDTree(pred);tg=cKDTree(gt)
    neigh=tp.query_ball_tree(tg,radius)
    edges=[(i,j) for i,x in enumerate(neigh) for j in x]
    if not edges:return np.array([],int),np.array([],int),np.array([],float)
    p2g={};g2p={}
    for i,j in edges:p2g.setdefault(i,[]).append(j);g2p.setdefault(j,[]).append(i)
    seen=set();MP=[];MG=[];MD=[]
    for s in p2g:
        if s in seen:continue
        ps={s};gs=set();stack=[s]
        while stack:
            i=stack.pop()
            if i in seen:continue
            seen.add(i)
            for j in p2g.get(i,[]):
                if j not in gs:
                    gs.add(j)
                    for ii in g2p.get(j,[]):
                        if ii not in ps:ps.add(ii);stack.append(ii)
        pl=sorted(ps);gl=sorted(gs);gp={g:k for k,g in enumerate(gl)}
        C=np.full((len(pl),len(gl)),radius*1000.0)
        for r,i in enumerate(pl):
            for j in p2g.get(i,[]):
                C[r,gp[j]]=np.linalg.norm(pred[i]-gt[j])
        rr,cc=linear_sum_assignment(C)
        for r,c in zip(rr,cc):
            if C[r,c]<=radius:MP.append(pl[r]);MG.append(gl[c]);MD.append(C[r,c])
    return np.asarray(MP),np.asarray(MG),np.asarray(MD)


def gt_counts(root,gt,qv):
    import pyarrow.parquet as pq
    p=Path(root)/"outs/transcripts.parquet"
    if not p.exists():p=Path(root)/"transcripts.parquet"
    cols=pq.read_schema(p).names
    idc=choose_col(cols,("cell_id",))
    qc=choose_col(cols,("qv","quality_value"),False)
    cmap={str(x):i for i,x in enumerate(gt.cell_id.astype(str))}
    out=np.zeros(len(gt),dtype=np.int64)
    pf=pq.ParquetFile(p)
    for b in range(pf.num_row_groups):
        use=[idc]+([qc] if qc else [])
        q=pf.read_row_group(b,columns=use).to_pandas()
        if qc:q=q[pd.to_numeric(q[qc],errors="coerce")>=qv]
        ci=q[idc].astype(str).map(cmap)
        ci=ci[ci.notna()].astype(np.int64).to_numpy()
        if len(ci):np.add.at(out,ci,1)
    return out


def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run",default=RUN_DEFAULT)
    ap.add_argument("--root",default=ROOT_DEFAULT)
    ap.add_argument("--out",default=OUT_DEFAULT)
    ap.add_argument("--match-radius",type=float,default=10)
    ap.add_argument("--qv-min",type=float,default=20)
    args=ap.parse_args()

    maskp=find_mask(args.run)
    mask=tifffile.imread(maskp)
    labels,area_px,loc_px=mask_props(mask)

    # Infer final raster scale from ROI size and mask dimensions.
    # Example mouse-brain run: 10000 native px ROI -> 1000x1000 final mask,
    # therefore 10 native pixels per mask pixel = 2.125 um/px.
    mh,mw=mask.shape[-2],mask.shape[-1]
    scale_x_um=(ROI_SIZE/mw)*NATIVE_PIX
    scale_y_um=(ROI_SIZE/mh)*NATIVE_PIX
    print(f"mask shape={mask.shape}; inferred raster scale "
          f"x={scale_x_um:.6f} um/px y={scale_y_um:.6f} um/px")

    pred_xy=np.c_[
        ROI_X0*NATIVE_PIX + loc_px[:,0]*scale_x_um,
        ROI_Y0*NATIVE_PIX + loc_px[:,1]*scale_y_um
    ]
    area_um2=area_px*(scale_x_um*scale_y_um)
    diam=2*np.sqrt(area_um2/np.pi)

    gt=load_gt(args.root);gtxy=gt[["x","y"]].to_numpy(float)
    mp,mg,dist=sparse_match(pred_xy,gtxy,args.match_radius)

    # Pred transcript counts: prefer cell_by_gene.npy.
    cbg=Path(args.run)/"cell_by_gene.npy"
    if cbg.exists():
        X=np.load(cbg,mmap_mode="r")
        pred_counts=np.asarray(X.sum(axis=1)).ravel().astype(float)
        # If raster contains a few labels lost after quantification, align conservatively.
        if len(pred_counts)!=len(labels):
            print(f"WARNING cell_by_gene rows={len(pred_counts)} raster labels={len(labels)}")
            n=min(len(pred_counts),len(labels))
            pred_counts=pred_counts[:n]
            keep=mp<n
            mp=mp[keep];mg=mg[keep];dist=dist[keep]
    else:
        pred_counts=np.full(len(labels),np.nan)

    gc=gt_counts(args.root,gt,args.qv_min)
    pc=pred_counts[mp] if len(pred_counts) else np.array([])
    mc=gc[mg] if len(mg) else np.array([])

    n=len(labels);ng=len(gt);nm=len(mp)
    prec=nm/n;rec=nm/ng;f1=2*prec*rec/max(prec+rec,1e-12)

    result=dict(
        method="Cellist",dataset="xenium_5",tissue="mouse_brain",
        gt_cells=ng,pred_cells=n,matched=nm,cell_ratio=n/ng,
        precision=prec,recall=rec,f1=f1,
        loc_mean_um=float(np.mean(dist)),loc_median_um=float(np.median(dist)),
        loc_p95_um=float(np.percentile(dist,95)),match_radius_um=args.match_radius,
        median_area_um2=float(np.median(area_um2)),
        median_diameter_um=float(np.median(diam)),
        tx_per_cell_median=float(np.nanmedian(pred_counts)),
        tx_per_cell_mean=float(np.nanmean(pred_counts)),
        tx_per_cell_p5=float(np.nanpercentile(pred_counts,5)),
        tx_per_cell_p95=float(np.nanpercentile(pred_counts,95)),
        frag_lt10=float(np.nanmean(pred_counts<10)),
        count_pearson=safe_corr(pc,mc,"pearson"),
        count_spearman=safe_corr(pc,mc,"spearman"),
        count_mae=float(np.nanmean(np.abs(pc-mc))),
        count_rmse=float(np.sqrt(np.nanmean((pc-mc)**2))),
        uses_platform_prior=False,detection_comparable=True,
        mask=str(maskp),qv_min=args.qv_min,
        raster_scale_x_um=float(scale_x_um),
        raster_scale_y_um=float(scale_y_um),
    )
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([result]).to_csv(out,index=False)
    pd.DataFrame({
        "pred_index":mp,"gt_index":mg,"distance_um":dist,
        "pred_tx":pc,"gt_tx":mc,
        "gt_cell_id":gt.iloc[mg].cell_id.astype(str).to_numpy()
    }).to_csv(out.with_name("cellist_xenium_5_pairs.csv"),index=False)
    out.with_suffix(".json").write_text(json.dumps(result,indent=2))
    print("="*88)
    for k,v in result.items():print(f"{k:28s} {v}")
    print("="*88)
    print("metrics ->",out)

if __name__=="__main__":
    main()
