#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr

NATIVE_PIX = 0.2125
ROI_Y0 = 14052
ROI_X0 = 11241
ROI_SIZE = 6000

ROOT_DEFAULT = "/data/qiuyijia/dataset/xenium_colon"
RUN_DEFAULT = "/data/qiuyijia/cellist_xenium_colon_downstream/ref_r5_imp2p5_fill1p5"
OUT_DEFAULT = "/data/qiuyijia/eval_results_qv20/colon/cellist_xenium_6_metrics.csv"


def choose_col(cols, candidates, required=True):
    low = {str(c).lower(): c for c in cols}
    for x in candidates:
        if x.lower() in low:
            return low[x.lower()]
    if required:
        raise KeyError(f"cannot find any of {candidates} in {list(cols)}")
    return None


def safe_corr(a, b, kind):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]; b = b[m]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r = pearsonr(a, b) if kind == "pearson" else spearmanr(a, b)
    return float(r[0])


def find_mask(run):
    run = Path(run)
    exact = [
        run / "cellist_segmentation.tif",
        run / "segmentation/cellist_segmentation.tif",
        run / "output/cellist_segmentation.tif",
        run / "segmentation_mask.tif",
    ]
    for p in exact:
        if p.exists():
            return p
    fs = sorted(run.rglob("*.tif"))
    fs = [p for p in fs if any(k in p.name.lower() for k in ("segmentation","label","mask"))
          and "bin" not in p.name.lower() and "nuc" not in p.name.lower()]
    if not fs:
        raise FileNotFoundError(f"no final segmentation tif under {run}")
    return fs[0]


def mask_props(mask):
    lab, cnt = np.unique(mask, return_counts=True)
    keep = lab > 0
    labels = lab[keep].astype(np.int64)
    area = cnt[keep].astype(np.int64)

    lut = {int(l): i for i, l in enumerate(labels)}
    yy, xx = np.nonzero(mask > 0)
    vv = mask[yy, xx]
    idx = np.fromiter((lut[int(v)] for v in vv), dtype=np.int64, count=len(vv))

    n = len(labels)
    nn = np.bincount(idx, minlength=n)
    cx = np.bincount(idx, weights=xx, minlength=n) / np.maximum(nn, 1)
    cy = np.bincount(idx, weights=yy, minlength=n) / np.maximum(nn, 1)
    return labels, area, np.c_[cx, cy]


def load_gt_cells(root):
    p = Path(root) / "outs/cells.parquet"
    if not p.exists():
        p = Path(root) / "cells.parquet"
    d = pd.read_parquet(p)

    idc = choose_col(d.columns, ("cell_id","id"))
    xc = choose_col(d.columns, ("x_centroid","center_x","x"))
    yc = choose_col(d.columns, ("y_centroid","center_y","y"))

    x0 = ROI_X0 * NATIVE_PIX
    x1 = (ROI_X0 + ROI_SIZE) * NATIVE_PIX
    y0 = ROI_Y0 * NATIVE_PIX
    y1 = (ROI_Y0 + ROI_SIZE) * NATIVE_PIX

    m = (pd.to_numeric(d[xc],errors="coerce").ge(x0) &
         pd.to_numeric(d[xc],errors="coerce").lt(x1) &
         pd.to_numeric(d[yc],errors="coerce").ge(y0) &
         pd.to_numeric(d[yc],errors="coerce").lt(y1))
    q = d.loc[m,[idc,xc,yc]].copy().reset_index(drop=True)
    q.columns = ["cell_id","x","y"]
    return q


def sparse_hungarian(pred, gt, radius):
    tp = cKDTree(pred); tg = cKDTree(gt)
    neigh = tp.query_ball_tree(tg, r=radius)
    edges = [(i,j) for i, js in enumerate(neigh) for j in js]
    if not edges:
        return np.empty(0,int), np.empty(0,int), np.empty(0,float)

    p2g, g2p = {}, {}
    for i,j in edges:
        p2g.setdefault(i,[]).append(j)
        g2p.setdefault(j,[]).append(i)

    seen = set()
    mp=[]; mg=[]; md=[]
    for s in p2g:
        if s in seen: continue
        ps={s}; gs=set(); stack=[s]
        while stack:
            i=stack.pop()
            if i in seen: continue
            seen.add(i)
            for j in p2g.get(i,[]):
                if j not in gs:
                    gs.add(j)
                    for ii in g2p.get(j,[]):
                        if ii not in ps:
                            ps.add(ii); stack.append(ii)
        pl=sorted(ps); gl=sorted(gs); gp={g:k for k,g in enumerate(gl)}
        C=np.full((len(pl),len(gl)), radius*1000.0)
        for r,i in enumerate(pl):
            for j in p2g.get(i,[]):
                C[r,gp[j]] = np.linalg.norm(pred[i]-gt[j])
        rr,cc = linear_sum_assignment(C)
        for r,c in zip(rr,cc):
            if C[r,c] <= radius:
                mp.append(pl[r]); mg.append(gl[c]); md.append(C[r,c])
    return np.asarray(mp,int), np.asarray(mg,int), np.asarray(md,float)


def locate_cbg(run):
    run = Path(run)
    for p in [run/"cell_by_gene.npy", run/"cell_by_gene.npz"]:
        if p.exists(): return p
    fs = list(run.rglob("cell_by_gene.npy"))
    return fs[0] if fs else None


def locate_genes(run):
    run = Path(run)
    for p in [run/"cell_by_gene_genes.txt", run/"genes.txt", run/"input/genes.txt"]:
        if p.exists():
            genes=[x.strip() for x in p.read_text().splitlines() if x.strip()]
            if genes: return np.asarray(genes,dtype=str)
    return None


def load_gt_expression(root, gt_cells, genes, qv_min):
    import pyarrow.parquet as pq
    p = Path(root)/"outs/transcripts.parquet"
    if not p.exists(): p = Path(root)/"transcripts.parquet"

    cols = pq.read_schema(p).names
    idc = choose_col(cols,("cell_id",))
    gc = choose_col(cols,("feature_name","gene","gene_name"))
    qc = choose_col(cols,("qv","quality_value"),False)

    cmap={str(x):i for i,x in enumerate(gt_cells["cell_id"].astype(str))}
    if genes is None:
        out=np.zeros((len(gt_cells),1),dtype=np.int32)
        gmap=None
    else:
        out=np.zeros((len(gt_cells),len(genes)),dtype=np.int32)
        gmap={str(g):i for i,g in enumerate(genes)}

    pf=pq.ParquetFile(p)
    for bi in range(pf.num_row_groups):
        use=[idc]+([gc] if genes is not None else [])+([qc] if qc else [])
        q=pf.read_row_group(bi,columns=use).to_pandas()
        if qc:
            q=q[pd.to_numeric(q[qc],errors="coerce")>=qv_min]
        if q.empty: continue
        ci=q[idc].astype(str).map(cmap)

        if genes is None:
            keep=ci.notna()
            arr=ci[keep].astype(np.int64).to_numpy()
            if len(arr): np.add.at(out[:,0],arr,1)
        else:
            gi=q[gc].astype(str).map(gmap)
            keep=ci.notna() & gi.notna()
            if not keep.any(): continue
            ca=ci[keep].astype(np.int64).to_numpy()
            ga=gi[keep].astype(np.int64).to_numpy()
            np.add.at(out,(ca,ga),1)
    return out


def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run",default=RUN_DEFAULT)
    ap.add_argument("--root",default=ROOT_DEFAULT)
    ap.add_argument("--out",default=OUT_DEFAULT)
    ap.add_argument("--match-radius",type=float,default=10.0)
    ap.add_argument("--qv-min",type=float,default=20.0)
    ap.add_argument("--vec-limit",type=int,default=3000)
    args=ap.parse_args()

    maskp=find_mask(args.run)
    mask=tifffile.imread(maskp)
    if mask.ndim!=2:
        raise RuntimeError(f"expected 2D mask, got {mask.shape}")

    labels,area_px,loc_px=mask_props(mask)

    mh,mw=mask.shape
    sx=(ROI_SIZE/mw)*NATIVE_PIX
    sy=(ROI_SIZE/mh)*NATIVE_PIX
    print(f"mask={maskp}")
    print(f"mask shape={mask.shape}; raster scale x={sx:.6f} um/px y={sy:.6f} um/px")

    pred_xy=np.c_[ROI_X0*NATIVE_PIX+loc_px[:,0]*sx,
                  ROI_Y0*NATIVE_PIX+loc_px[:,1]*sy]
    area_um2=area_px*sx*sy
    diam_um=2*np.sqrt(area_um2/np.pi)

    gt=load_gt_cells(args.root)
    gt_xy=gt[["x","y"]].to_numpy(float)
    mp,mg,dist=sparse_hungarian(pred_xy,gt_xy,args.match_radius)

    pred_n=len(labels); gt_n=len(gt)
    cbg=locate_cbg(args.run)
    genes=locate_genes(args.run)

    pred_counts=np.full(pred_n,np.nan)
    X=None
    if cbg:
        X=np.load(cbg,mmap_mode="r") if cbg.suffix==".npy" else np.load(cbg)[np.load(cbg).files[0]]
        print(f"cell_by_gene={cbg} shape={X.shape}")
        n=min(X.shape[0],pred_n)
        pred_counts[:n]=np.asarray(X[:n].sum(axis=1)).ravel()
        keep=mp<n
        mp=mp[keep]; mg=mg[keep]; dist=dist[keep]

    matched=len(mp)
    precision=matched/pred_n if pred_n else np.nan
    recall=matched/gt_n if gt_n else np.nan
    f1=2*precision*recall/(precision+recall) if precision+recall>0 else np.nan

    vector_ok = X is not None and genes is not None and X.shape[1]==len(genes)
    gtM=load_gt_expression(args.root,gt,genes if vector_ok else None,args.qv_min)
    gt_counts=(gtM.sum(axis=1) if vector_ok else gtM[:,0]).astype(float)

    pc=pred_counts[mp]
    gc=gt_counts[mg]

    count_pearson=safe_corr(pc,gc,"pearson")
    count_spearman=safe_corr(pc,gc,"spearman")
    count_mae=float(np.nanmean(np.abs(pc-gc))) if len(pc) else np.nan
    count_rmse=float(np.sqrt(np.nanmean((pc-gc)**2))) if len(pc) else np.nan

    vec_cos=vec_js=vec_pear=np.nan
    vec_n=vec_genes=0
    if vector_ok:
        take=np.arange(min(len(mp),args.vec_limit))
        cos=[]; js=[]; vp=[]
        for k in take:
            a=np.asarray(X[mp[k]],float).ravel()
            b=gtM[mg[k]].astype(float)
            den=np.linalg.norm(a)*np.linalg.norm(b)
            cos.append(float(a@b/den) if den else np.nan)
            if a.sum()>0 and b.sum()>0:
                js.append(float(jensenshannon(a/a.sum(),b/b.sum(),base=2)))
                vp.append(safe_corr(a,b,"pearson"))
        vec_cos=float(np.nanmean(cos)) if cos else np.nan
        vec_js=float(np.nanmean(js)) if js else np.nan
        vec_pear=float(np.nanmean(vp)) if vp else np.nan
        vec_n=len(take); vec_genes=len(genes)

    assign_rate=np.nan
    for lp in list(Path(args.run).glob("*.log"))+list(Path(args.run).parent.glob(f"{Path(args.run).name}.log")):
        try: txt=lp.read_text(errors="ignore")
        except Exception: continue
        m=re.search(r"assigned\s*=\s*([0-9,]+)\s*\(([0-9.]+)%\)",txt)
        if m:
            assign_rate=float(m.group(2))/100.0
            break

    result=dict(
        method="Cellist",dataset="xenium_6",tissue="colon",
        gt_cells=gt_n,pred_cells=pred_n,matched=matched,
        cell_ratio=pred_n/gt_n,
        precision=precision,recall=recall,f1=f1,
        loc_mean_um=float(np.mean(dist)) if len(dist) else np.nan,
        loc_median_um=float(np.median(dist)) if len(dist) else np.nan,
        loc_p95_um=float(np.percentile(dist,95)) if len(dist) else np.nan,
        match_radius_um=args.match_radius,
        median_area_um2=float(np.median(area_um2)),
        median_diameter_um=float(np.median(diam_um)),
        tx_per_cell_median=float(np.nanmedian(pred_counts)),
        tx_per_cell_mean=float(np.nanmean(pred_counts)),
        tx_per_cell_p5=float(np.nanpercentile(pred_counts,5)),
        tx_per_cell_p95=float(np.nanpercentile(pred_counts,95)),
        frag_lt10=float(np.nanmean(pred_counts<10)),
        count_pearson=count_pearson,count_spearman=count_spearman,
        count_mae=count_mae,count_rmse=count_rmse,
        assign_rate=assign_rate,
        vec_cosine=vec_cos,vec_js_dist=vec_js,vec_pearson=vec_pear,
        vec_n=vec_n,vec_genes=vec_genes,
        uses_platform_prior=False,detection_comparable=True,
        assignment_independent=True,vector_independent=True,
        mask=str(maskp),qv_min=args.qv_min,
        raster_scale_x_um=sx,raster_scale_y_um=sy,
        run_dir=str(args.run),
    )

    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([result]).to_csv(out,index=False)

    pairs=out.with_name("cellist_xenium_6_pairs.csv")
    pd.DataFrame({
        "pred_index":mp,"gt_index":mg,"distance_um":dist,
        "pred_tx":pc,"gt_tx":gc,
        "gt_cell_id":gt.iloc[mg]["cell_id"].astype(str).to_numpy()
    }).to_csv(pairs,index=False)

    out.with_suffix(".json").write_text(json.dumps(result,indent=2,ensure_ascii=False))

    print("="*88)
    for k,v in result.items():
        print(f"{k:28s} {v}")
    print("="*88)
    print("metrics ->",out)
    print("pairs   ->",pairs)


if __name__=="__main__":
    main()
