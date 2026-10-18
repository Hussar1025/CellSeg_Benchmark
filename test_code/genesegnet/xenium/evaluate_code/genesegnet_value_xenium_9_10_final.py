#!/usr/bin/env python3
import argparse, gzip
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import pyarrow.dataset as pds

from scipy.spatial import cKDTree
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from skimage.draw import polygon
from skimage.measure import regionprops_table

PIX = 0.2125
ROI = 10000

CFG = {
    "xenium9": dict(
        root=Path("/data/qiuyijia/dataset/xenium9"),
        run=Path("/data/qiuyijia/genesegnet_xenium9"),
        x0=16374, y0=5147,
        expected_gt=17689,
        label="xenium_9",
    ),
    "xenium10": dict(
        root=Path("/data/qiuyijia/dataset/xenium10"),
        run=Path("/data/qiuyijia/genesegnet_xenium10"),
        x0=20564, y0=13750,
        expected_gt=36604,
        label="xenium_10",
    ),
}

CTRL = ("NegControl", "BLANK", "Unassigned",
        "antisense", "DeprecatedCodeword", "Intergenic")

def log(*x):
    print(*x, flush=True)

def polygon_centroid(x, y):
    if len(x) > 1 and x[0] == x[-1] and y[0] == y[-1]:
        x, y = x[:-1], y[:-1]
    if len(x) < 3:
        return float(np.mean(x)), float(np.mean(y))
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cr = x*y2 - x2*y
    a2 = cr.sum()
    if abs(a2) < 1e-12:
        return float(np.mean(x)), float(np.mean(y))
    return (
        float(((x+x2)*cr).sum()/(3*a2)),
        float(((y+y2)*cr).sum()/(3*a2))
    )

def build_gt_mask(cfg):
    bp = cfg["root"]/"benchmark_roi_10000"/"cell_boundaries.parquet"
    b = pd.read_parquet(bp)

    idc = next(c for c in b.columns
               if c.lower() in ("label_id","cell_id","cell","id"))
    xc = next(c for c in b.columns
              if c.lower() in ("vertex_x_local_um","vertex_x"))
    yc = next(c for c in b.columns
              if c.lower() in ("vertex_y_local_um","vertex_y"))

    groups = list(b.groupby(idc, sort=False))
    if len(groups) != cfg["expected_gt"]:
        raise RuntimeError(
            f"canonical GT mismatch: {len(groups)} != {cfg['expected_gt']}"
        )

    gtmask = np.zeros((ROI, ROI), np.int32)
    ids, gx, gy = [], [], []

    for row, (cid, g) in enumerate(groups, start=1):
        x = g[xc].to_numpy(float)
        y = g[yc].to_numpy(float)

        cx, cy = polygon_centroid(x, y)
        ids.append(str(cid))
        gx.append(cx)
        gy.append(cy)

        px = x / PIX
        py = y / PIX

        rr, cc = polygon(py, px, shape=gtmask.shape)
        gtmask[rr, cc] = row

    return gtmask, np.asarray(ids), np.asarray(gx), np.asarray(gy)

def pred_geometry(mask):
    rp = pd.DataFrame(regionprops_table(
        mask.astype(np.int32, copy=False),
        properties=("label","area","centroid")
    ))
    rp = rp.rename(columns={
        "centroid-0":"cy",
        "centroid-1":"cx"
    })
    rp["x_um"] = rp.cx * PIX
    rp["y_um"] = rp.cy * PIX
    rp["area_um2"] = rp.area * PIX * PIX
    rp["diameter_um"] = 2*np.sqrt(rp.area_um2/np.pi)
    return rp

def greedy_pairs(pred_xy, gt_xy, radius):
    pt = cKDTree(pred_xy)
    gt = cKDTree(gt_xy)

    cand = []
    neighbours = pt.query_ball_tree(gt, r=radius)
    for pi, vv in enumerate(neighbours):
        if not vv:
            continue
        d = np.sqrt(((gt_xy[vv]-pred_xy[pi])**2).sum(axis=1))
        for gi, dd in zip(vv, d):
            cand.append((float(dd), pi, int(gi)))

    cand.sort()
    up=set(); ug=set(); pairs=[]; dists=[]
    for d,pi,gi in cand:
        if pi in up or gi in ug:
            continue
        up.add(pi); ug.add(gi)
        pairs.append((pi,gi)); dists.append(d)

    return np.asarray(pairs, np.int64), np.asarray(dists, float)

def pair_numeric(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ok=np.isfinite(a)&np.isfinite(b)
    a,b=a[ok],b[ok]
    if len(a)<2:
        return np.nan,np.nan,np.nan,np.nan
    return (
        float(pearsonr(a,b)[0]),
        float(spearmanr(a,b)[0]),
        float(np.mean(np.abs(a-b))),
        float(np.sqrt(np.mean((a-b)**2)))
    )

def vector_metrics(P,G,batch=256):
    n=P.shape[0]
    cos=[]; js=[]; pr=[]

    for a in range(0,n,batch):
        b=min(a+batch,n)
        A=P[a:b].toarray().astype(np.float64)
        B=G[a:b].toarray().astype(np.float64)

        for x,y in zip(A,B):
            nx=np.linalg.norm(x); ny=np.linalg.norm(y)
            if nx>0 and ny>0:
                cos.append(float(np.dot(x,y)/(nx*ny)))

                sx=x.sum(); sy=y.sum()
                if sx>0 and sy>0:
                    js.append(float(jensenshannon(x/sx,y/sy)))

                if np.std(x)>0 and np.std(y)>0:
                    pr.append(float(np.corrcoef(x,y)[0,1]))

    return (
        float(np.mean(cos)) if cos else np.nan,
        float(np.mean(js)) if js else np.nan,
        float(np.mean(pr)) if pr else np.nan,
        len(cos)
    )

def evaluate(name,qv,outdir):
    cfg=CFG[name]
    root=cfg["root"]
    run=cfg["run"]
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)

    mpath=run/f"genesegnet_{cfg['label']}_mask.tif"
    if not mpath.exists():
        # compatibility with existing names
        c=list(run.glob("genesegnet_*_mask.tif"))
        if not c:
            raise RuntimeError(f"mask missing under {run}")
        mpath=c[0]

    log("\n"+"="*90)
    log("GENESEGNET FINAL EVAL",name)
    log("="*90)

    mask=np.squeeze(tifffile.imread(mpath)).astype(np.int32,copy=False)
    if mask.shape != (ROI,ROI):
        raise RuntimeError(f"unexpected mask shape {mask.shape}")

    pred=pred_geometry(mask)
    pred_ids=pred.label.to_numpy(np.int64)

    log("pred =",len(pred))
    log("median diameter =",float(pred.diameter_um.median()))

    log("building canonical GT raster ...")
    gtmask,gids,gx,gy=build_gt_mask(cfg)
    gxy=np.column_stack((gx,gy))
    pxy=pred[["x_um","y_um"]].to_numpy(float)

    radius=float(pred.diameter_um.median())*0.8
    pairs,dists=greedy_pairs(pxy,gxy,radius)

    mp=pairs[:,0]
    mg=pairs[:,1]

    precision=len(pairs)/max(len(pred),1)
    recall=len(pairs)/len(gids)
    f1=2*precision*recall/max(precision+recall,1e-12)

    log(f"GT={len(gids):,} matched={len(pairs):,} "
        f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    log(f"centroid mean={dists.mean():.3f} "
        f"median={np.median(dists):.3f} "
        f"p95={np.percentile(dists,95):.3f} um")

    tp=root/"outs"/"transcripts.parquet"
    ds=pds.dataset(tp,format="parquet")
    names=ds.schema.names

    xc="x_location"
    yc="y_location"
    gc="feature_name"
    qvc="qv" if "qv" in names else None

    xlo=cfg["x0"]*PIX
    ylo=cfg["y0"]*PIX
    xhi=(cfg["x0"]+ROI)*PIX
    yhi=(cfg["y0"]+ROI)*PIX

    fil=((pds.field(xc)>=xlo)&(pds.field(xc)<xhi)&
         (pds.field(yc)>=ylo)&(pds.field(yc)<yhi))
    if qvc:
        fil=fil&(pds.field(qvc)>=qv)

    cols=[xc,yc,gc]
    tab=ds.to_table(columns=cols,filter=fil).to_pandas()

    if len(tab) and isinstance(tab[gc].iloc[0],bytes):
        tab[gc]=tab[gc].str.decode("utf-8")
    tab[gc]=tab[gc].astype(str)
    keep=~tab[gc].str.startswith(CTRL)
    tab=tab.loc[keep].reset_index(drop=True)

    log(f"coding transcripts={len(tab):,} genes={tab[gc].nunique():,}")

    px=np.floor(tab[xc].to_numpy(float)/PIX).astype(np.int64)-cfg["x0"]
    py=np.floor(tab[yc].to_numpy(float)/PIX).astype(np.int64)-cfg["y0"]

    ok=(px>=0)&(px<ROI)&(py>=0)&(py<ROI)
    tab=tab.loc[ok].reset_index(drop=True)
    px=px[ok]; py=py[ok]

    plab=mask[py,px].astype(np.int64)
    glab=gtmask[py,px].astype(np.int64)

    assign_rate=float((plab>0).mean())
    gt_assign_rate=float((glab>0).mean())
    log(f"pred assignment={assign_rate:.6f}")
    log(f"GT polygon assignment={gt_assign_rate:.6f}")

    genes,codes=pd.factorize(tab[gc],sort=True)
    tgene=genes.astype(np.int64)
    ng=int(tgene.max()+1) if len(tgene) else 0

    # predicted label -> prediction row
    maxp=int(mask.max())
    pmap=np.full(maxp+1,-1,np.int64)
    pmap[pred_ids]=np.arange(len(pred))
    prow=np.where(plab<=maxp,pmap[plab],-1)

    grow=glab-1

    vp=(prow>=0)
    vg=(grow>=0)

    P=sparse.coo_matrix(
        (np.ones(vp.sum(),np.float32),(prow[vp],tgene[vp])),
        shape=(len(pred),ng)
    ).tocsr()

    G=sparse.coo_matrix(
        (np.ones(vg.sum(),np.float32),(grow[vg],tgene[vg])),
        shape=(len(gids),ng)
    ).tocsr()

    pcnt=np.asarray(P.sum(axis=1)).ravel()
    gcnt=np.asarray(G.sum(axis=1)).ravel()

    cp,cs,mae,rmse=pair_numeric(pcnt[mp],gcnt[mg])
    vcos,vjs,vpr,vn=vector_metrics(P[mp],G[mg])

    # matched predicted label -> canonical GT raster label (1-based)
    match_gt=np.zeros(maxp+1,np.int32)
    for pi,gi in pairs:
        match_gt[int(pred_ids[pi])]=int(gi)+1

    both=(plab>0)&(glab>0)
    mapped=match_gt[np.clip(plab,0,maxp)]
    valid=both&(mapped>0)
    acc=float((mapped[valid]==glab[valid]).mean()) if valid.any() else np.nan
    overlap=int(valid.sum())
    overall=assign_rate*acc if np.isfinite(acc) else np.nan

    log(f"count Pearson={cp:.4f} Spearman={cs:.4f}")
    log(f"vector cosine={vcos:.4f} JS={vjs:.4f} Pearson={vpr:.4f}")
    log(f"assignment accuracy={acc:.4f} over {overlap:,}")
    log(f"overall correct={overall:.4f}")

    row=dict(
        method="GeneSegNet",
        dataset=cfg["label"],
        gt_cells=len(gids),
        pred_cells=len(pred),
        cell_ratio=len(pred)/len(gids),
        matched=len(pairs),
        precision=precision,
        recall=recall,
        f1=f1,
        loc_mean_um=float(dists.mean()),
        loc_median_um=float(np.median(dists)),
        loc_p95_um=float(np.percentile(dists,95)),
        count_pearson=cp,
        count_spearman=cs,
        count_mae=mae,
        count_rmse=rmse,
        vec_cosine=vcos,
        vec_js_dist=vjs,
        vec_pearson=vpr,
        assign_accuracy=acc,
        assign_rate=assign_rate,
        overall_correct=overall,
        median_diameter_um=float(pred.diameter_um.median()),
        uses_platform_prior=False,
        assignment_independent=True,
        vector_independent=True,
        transcript_id_overlap_n=overlap,
        gt_assignment_rate=gt_assign_rate,
        gt_counts_source="rasterised canonical benchmark cell_boundaries",
        note=("canonical polygon GT; transcript GT assigned by spatial polygon "
              "raster lookup; no platform cell_id used")
    )

    op=outdir/f"genesegnet_{cfg['label']}_final_metrics.csv"
    pd.DataFrame([row]).to_csv(op,index=False)
    log("->",op)
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=list(CFG),required=True)
    ap.add_argument("--qv-min",type=float,default=20)
    ap.add_argument("--out-dir",
                    default="/data/qiuyijia/GeneSegNet/eval_final")
    a=ap.parse_args()
    evaluate(a.dataset,a.qv_min,a.out_dir)

if __name__=="__main__":
    main()
