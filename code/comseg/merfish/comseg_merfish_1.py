#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comseg_merfish_1_prior.py
=========================

MERFISH1 ComSeg with IMAGE-DERIVED nucleus prior.

This script does NOT use official MERFISH cell boundaries or platform centroids.

Pipeline
--------
stage check:
  verify paths / ROI / ComSeg API

stage nuclei:
  run Cellpose on DAPI tiles and build a 20000x20000 nucleus instance mask
  (run this stage in the `cellist` conda environment)

stage prep:
  create ComSeg transcript tiles and matching nucleus-prior masks
  (run in `comseg_env`)

stage run:
  native ComSeg with_prior
  - ComSegDataset(... prior_name="in_nucleus", path_to_mask_prior=...)
  - add_prior_from_mask()
  - compute_edge_weight()
  - ComSegDict(... community_detection="with_prior")
  - compute_community_vector()
  - compute_insitu_clustering()
  - add_cluster_id_to_graph()
  - classify_centroid()
  - associate_rna2landmark()
  - anndata_from_comseg_result()

stage merge:
  merge tile cell centroids / counts to ROI-level result

stage report:
  summarize result

Notes
-----
The installed ComSeg API has been verified to support:
  ComSegDataset(path_to_mask_prior=...)
  ComSegDataset.add_prior_from_mask(...)
  ComSegDict(community_detection="with_prior")

Cellpose is available in the `cellist` environment, not comseg_env.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree


ROOT_DEFAULT = Path("/data/qiuyijia/dataset/merfish_mouse_brain")
DAPI_DEFAULT = ROOT_DEFAULT / "images/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_images_mosaic_DAPI_z3.tif"
TRANSFORM_DEFAULT = ROOT_DEFAULT / "images/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_images_micron_to_mosaic_pixel_transform.csv"
TX_DEFAULT = ROOT_DEFAULT / "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_detected_transcripts_S1R1.csv"

OUT_DEFAULT = Path("/data/qiuyijia/comseg_merfish_1_dapi_prior_roi20000")

H = 61310
W = 89085
ROI_SIZE = 20000
ROI_Y0 = (H - ROI_SIZE) // 2
ROI_X0 = (W - ROI_SIZE) // 2
ROI_Y1 = ROI_Y0 + ROI_SIZE
ROI_X1 = ROI_X0 + ROI_SIZE

PX_UM = 1 / 9.205855

# ComSeg tiling in native mosaic px.
TILE = 3000
OVERLAP = 300
STRIDE = TILE - OVERLAP

# Cellpose nucleus inference tiles.
CP_TILE = 2048
CP_OVERLAP = 256
CP_STRIDE = CP_TILE - CP_OVERLAP


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def banner(msg):
    log("=" * 92)
    log(msg)
    log("=" * 92)


def starts(size, tile, stride):
    ss = list(range(0, max(size-tile+1, 1), stride))
    last = size - tile
    if ss[-1] != last:
        ss.append(last)
    return ss


def normalize(im, pmin=1, pmax=99.8):
    im = np.asarray(im, np.float32)
    lo, hi = np.percentile(im, [pmin, pmax])
    if hi <= lo:
        return np.zeros(im.shape, np.uint8)
    return np.clip((im-lo)/(hi-lo)*255, 0, 255).astype(np.uint8)


def open_dapi(p):
    a = tifffile.memmap(str(p))
    if a.shape != (H,W):
        raise RuntimeError(f"DAPI shape={a.shape}")
    return a


def load_transform(p):
    M = np.loadtxt(p)
    if M.shape != (3,3):
        raise RuntimeError(M.shape)
    return M


def micron_to_px(x, y, M):
    q = np.c_[x, y, np.ones(len(x))]
    o = q @ M.T
    return o[:,0], o[:,1]


def build_roi_tx(args):
    cache = args.out / "cache/roi_transcripts.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    cache.parent.mkdir(parents=True, exist_ok=True)
    M = load_transform(args.transform)
    arr = []
    total = kept = 0

    for d in pd.read_csv(args.tx, chunksize=args.tx_chunksize):
        total += len(d)

        gx = pd.to_numeric(d["global_x"], errors="coerce").to_numpy(float)
        gy = pd.to_numeric(d["global_y"], errors="coerce").to_numpy(float)
        px, py = micron_to_px(gx, gy, M)

        take = (
            np.isfinite(px) & np.isfinite(py) &
            (px >= ROI_X0) & (px < ROI_X1) &
            (py >= ROI_Y0) & (py < ROI_Y1)
        )

        if np.any(take):
            q = d.loc[take, ["gene"]].copy()
            q["px_global"] = px[take]
            q["py_global"] = py[take]
            q["x"] = (px[take] - ROI_X0) * PX_UM
            q["y"] = (py[take] - ROI_Y0) * PX_UM
            q["z"] = 0.0
            q["in_nucleus"] = 0
            arr.append(q)
            kept += len(q)

        log(f"scanned={total:,}; ROI kept={kept:,}")

    if not arr:
        raise RuntimeError("zero ROI transcripts")

    out = pd.concat(arr, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


class UF:
    def __init__(self):
        self.p = {}
    def add(self,x): self.p.setdefault(int(x),int(x))
    def find(self,x):
        x=int(x); self.add(x)
        if self.p[x]!=x: self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,a,b):
        if not a or not b:return
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)


def stitch_masks(records, size, min_pix=20, frac_thr=0.35):
    canvas=np.zeros((size,size),np.uint32)
    uf=UF(); nxt=1

    for r in records:
        m=r["mask"]; y0=r["y"]; x0=r["x"]
        lut=np.zeros(int(m.max())+1,np.uint32)

        for lab in np.unique(m):
            if lab==0: continue
            lut[int(lab)]=nxt; uf.add(nxt); nxt+=1

        inc=lut[m]
        reg=canvas[y0:y0+m.shape[0],x0:x0+m.shape[1]]

        a=reg.ravel(); b=inc.ravel()
        k=(a>0)&(b>0)
        if np.any(k):
            aa=a[k]; bb=b[k]
            pairs, cnt=np.unique(np.c_[aa,bb],axis=0,return_counts=True)
            ca=dict(zip(*np.unique(aa,return_counts=True)))
            cb=dict(zip(*np.unique(bb,return_counts=True)))

            for (ga,gb),ni in zip(pairs,cnt):
                if ni<min_pix: continue
                frac=ni/max(min(ca[ga],cb[gb]),1)
                if frac>=frac_thr:
                    uf.union(ga,gb)

        empty=reg==0
        reg[empty]=inc[empty]

    labs=np.unique(canvas); labs=labs[labs>0]
    remap=np.zeros(int(canvas.max())+1,np.uint32)
    roots={}; n=1

    for lab in labs:
        root=uf.find(lab)
        if root not in roots:
            roots[root]=n;n+=1
        remap[int(lab)]=roots[root]

    return remap[canvas]


def stage_check(args):
    banner("ComSeg MERFISH1 DAPI-prior CHECK")

    for p in [args.dapi,args.tx,args.transform]:
        if not p.exists():
            raise FileNotFoundError(p)
        log(f"OK {p}")

    log(f"ROI = y[{ROI_Y0},{ROI_Y1}) x[{ROI_X0},{ROI_X1})")

    try:
        import comseg
        from comseg import dataset, dictionary
        log(f"comseg={comseg.__file__}")
        log(f"ComSegDataset={inspect.signature(dataset.ComSegDataset)}")
        log(f"ComSegDict={inspect.signature(dictionary.ComSegDict)}")
        log(f"add_prior_from_mask={inspect.signature(dataset.ComSegDataset.add_prior_from_mask)}")
    except Exception as e:
        log(f"ComSeg import skipped/fails in this environment: {e}")

    try:
        import cellpose
        log(f"cellpose={cellpose.__file__}")
    except Exception as e:
        log(f"Cellpose not available here: {e}")

    banner("CHECK DONE")


def stage_nuclei(args):
    banner("STAGE nuclei: DAPI -> Cellpose prior")

    try:
        from cellpose import models
    except Exception as e:
        raise RuntimeError(
            "stage nuclei must be run in the `cellist` conda environment"
        ) from e

    args.out.mkdir(parents=True,exist_ok=True)
    nd=args.out/"nuclei_tiles"
    nd.mkdir(exist_ok=True)

    dapi=open_dapi(args.dapi)

    # Cellpose API across installed versions: use CellposeModel if available.
    try:
        model=models.CellposeModel(gpu=True, model_type=args.cellpose_model)
    except Exception:
        model=models.Cellpose(gpu=True, model_type=args.cellpose_model)

    ys=starts(ROI_SIZE,CP_TILE,CP_STRIDE)
    xs=starts(ROI_SIZE,CP_TILE,CP_STRIDE)
    rec=[]

    for ry in ys:
        for rx in xs:
            gy=ROI_Y0+ry; gx=ROI_X0+rx
            op=nd/f"nuc_y{ry:05d}_x{rx:05d}.tif"

            if op.exists() and not args.force:
                m=tifffile.imread(op)
            else:
                im=normalize(np.asarray(dapi[gy:gy+CP_TILE,gx:gx+CP_TILE]))

                kwargs=dict(
                    diameter=args.nucleus_diameter_px,
                    flow_threshold=args.cp_flow,
                    cellprob_threshold=args.cp_cellprob,
                )

                try:
                    res=model.eval(im,channels=[0,0],**kwargs)
                except TypeError:
                    res=model.eval(im,**kwargs)

                m=np.asarray(res[0],np.uint32)
                tifffile.imwrite(op,m,compression="zlib")

            log(f"nuc tile y={ry} x={rx}: cells={int(m.max())}")
            rec.append({"y":ry,"x":rx,"mask":m})

    full=stitch_masks(rec,ROI_SIZE)
    fp=args.out/"nucleus_prior_roi20000.tif"
    tifffile.imwrite(fp,full,compression="zlib")

    labs,cnt=np.unique(full[full>0],return_counts=True)
    med=float(np.median(cnt)) if len(cnt) else np.nan
    log(
        f"nucleus prior: cells={len(labs):,}, "
        f"foreground={(full>0).mean()*100:.2f}%, "
        f"diam={2*np.sqrt(med/np.pi)*PX_UM:.2f}um"
    )
    log(f"prior -> {fp}")
    banner("NUCLEI DONE")


def stage_prep(args):
    banner("STAGE prep")

    prior=args.out/"nucleus_prior_roi20000.tif"
    if not prior.exists():
        raise FileNotFoundError(
            f"{prior}\nRun --stage nuclei in cellist env first."
        )

    tx=build_roi_tx(args)
    prior_full=tifffile.imread(prior)

    td=args.out/"tiles"
    pdp=args.out/"prior_tiles"
    td.mkdir(parents=True,exist_ok=True)
    pdp.mkdir(parents=True,exist_ok=True)

    ys=starts(ROI_SIZE,TILE,STRIDE)
    xs=starts(ROI_SIZE,TILE,STRIDE)

    meta=[]
    n=0

    for iy,ry in enumerate(ys):
        for ix,rx in enumerate(xs):
            n+=1
            tid=f"tile_{n-1:04d}"

            # Coordinates in um relative to tile origin, because every tile becomes
            # an independent ComSeg image.
            x0_um=rx*PX_UM
            y0_um=ry*PX_UM
            x1_um=(rx+TILE)*PX_UM
            y1_um=(ry+TILE)*PX_UM

            take=(
                tx["x"].ge(x0_um)&tx["x"].lt(x1_um)&
                tx["y"].ge(y0_um)&tx["y"].lt(y1_um)
            )

            q=tx.loc[take,["gene","x","y","z"]].copy()
            q["x"]=q["x"]-x0_um
            q["y"]=q["y"]-y0_um

            pm=prior_full[ry:ry+TILE,rx:rx+TILE].copy()

            # Relabel per tile so add_prior_from_mask sees compact local IDs.
            labs=np.unique(pm); labs=labs[labs>0]
            lut=np.zeros(int(pm.max())+1,np.uint32) if pm.max()>0 else np.zeros(1,np.uint32)
            for i,lab in enumerate(labs,1):
                lut[int(lab)]=i
            pm=lut[pm] if len(lut)>1 else pm

            # Assign transcript in_nucleus directly too; add_prior_from_mask may
            # overwrite/recompute this in native ComSeg.
            px=np.clip(np.rint(q["x"].to_numpy()/PX_UM).astype(int),0,TILE-1)
            py=np.clip(np.rint(q["y"].to_numpy()/PX_UM).astype(int),0,TILE-1)
            q["in_nucleus"]=pm[py,px].astype(np.int64)

            csvp=td/f"{tid}.csv"
            maskp=pdp/f"{tid}.tiff"
            q.to_csv(csvp,index=False)
            tifffile.imwrite(maskp,pm)

            meta.append(dict(
                tile_id=tid,
                ry=ry,rx=rx,
                n_spots=len(q),
                n_nuclei=len(labs),
                csv=str(csvp),
                prior=str(maskp),
            ))
            log(f"{tid}: spots={len(q):,}, nuclei={len(labs):,}")

    pd.DataFrame(meta).to_csv(args.out/"tile_metadata.tsv",sep="\t",index=False)
    banner(f"PREP DONE: {len(meta)} tiles")


def run_one_tile(row,args):
    from comseg import dataset as ds, dictionary

    tid=row["tile_id"]
    work=args.out/"work"/tid
    inp=work/"csv"
    pri=work/"prior"
    work.mkdir(parents=True,exist_ok=True)
    inp.mkdir(exist_ok=True)
    pri.mkdir(exist_ok=True)

    src_csv=Path(row["csv"])
    src_prior=Path(row["prior"])
    dst_csv=inp/f"{tid}.csv"
    dst_prior=pri/f"{tid}.tiff"

    if not dst_csv.exists(): shutil.copy2(src_csv,dst_csv)
    if not dst_prior.exists(): shutil.copy2(src_prior,dst_prior)

    out_h5=args.out/"tile_results"/f"{tid}.h5ad"
    out_h5.parent.mkdir(exist_ok=True)

    if out_h5.exists() and not args.force:
        log(f"{tid}: reuse {out_h5}")
        return

    log(f"{tid}: with_prior spots={int(row['n_spots']):,} nuclei={int(row['n_nuclei']):,}")

    dataset=ds.ComSegDataset(
        path_dataset_folder=str(inp),
        prior_name="in_nucleus",
        path_to_mask_prior=str(pri),
        mask_file_extension=".tiff",
        dict_scale={"x":1.0,"y":1.0,"z":1.0},
        mean_cell_diameter=args.mean_cell_diameter_um,
        gene_column="gene",
        disable_tqdm=False,
    )

    # Native prior import.
    dataset.add_prior_from_mask(overwrite=True,compute_centroid=True)
    dataset.compute_edge_weight(
        images_subset=None,
        distance=args.edge_distance_um,
        n_neighbors=args.knn_neighbors,
    )

    cd=dictionary.ComSegDict(
        dataset=dataset,
        mean_cell_diameter=args.mean_cell_diameter_um,
        community_detection="with_prior",
        seed=args.seed,
        disable_tqdm=False,
    )

    cd.compute_community_vector(
        k_nearest_neighbors=args.knn_neighbors
    )

    cd.compute_insitu_clustering(
        size_commu_min=args.size_commu_min,
        norm_vector=False,
        n_pcs=args.n_pcs,
        n_comps=args.n_comps,
        clustering_method="leiden",
        n_neighbors=args.cluster_neighbors,
        resolution=args.resolution,
        n_clusters_kmeans=args.n_clusters_kmeans,
        palette=None,
        nb_min_cluster=args.nb_min_cluster,
        min_merge_correlation=args.min_merge_correlation,
    )

    cd.add_cluster_id_to_graph(
        clustering_method="leiden",
    )

    cd.classify_centroid(
        n_neighbors=args.centroid_neighbors,
        dict_in_pixel=False,
        max_dist_centroid=args.max_dist_centroid_um,
        key_pred="leiden_merged",
        distance="ngb_distance_weights",
    )

    # Native landmark assignment using image prior.
    for img_name in cd:
        cd[img_name].associate_rna2landmark(
            key_pred="leiden_merged",
            prior_name="in_nucleus",
            distance="distance",
            max_cell_radius=args.max_cell_radius_um,
        )

    cd.anndata_from_comseg_result(
        key_cell_pred="cell_index_pred",
    )

    # This API stores combined anndata in cd.final_anndata.
    ad=getattr(cd,"final_anndata",None)
    if ad is None:
        raise RuntimeError(f"{tid}: final_anndata missing")

    ad.write_h5ad(out_h5)
    log(f"{tid}: DONE cells={ad.n_obs}")


def stage_run(args):
    banner("STAGE run: native ComSeg with_prior")
    meta=pd.read_csv(args.out/"tile_metadata.tsv",sep="\t")
    for _,r in meta.iterrows():
        run_one_tile(r.to_dict(),args)
    banner("RUN DONE")


def stage_merge(args):
    banner("STAGE merge")
    import anndata as ad

    meta=pd.read_csv(args.out/"tile_metadata.tsv",sep="\t")
    rows=[]

    for _,r in meta.iterrows():
        tid=r.tile_id
        hp=args.out/"tile_results"/f"{tid}.h5ad"
        if not hp.exists():
            continue

        a=ad.read_h5ad(hp)

        cxcol=next((c for c in ["centroid_x","x","global_x"] if c in a.obs.columns),None)
        cycol=next((c for c in ["centroid_y","y","global_y"] if c in a.obs.columns),None)

        if cxcol is None or cycol is None:
            raise RuntimeError(f"{tid}: no centroid columns in obs={list(a.obs.columns)}")

        # Convert local micron centroids to ROI micron coordinates.
        gx=pd.to_numeric(a.obs[cxcol],errors="coerce").to_numpy(float)+r.rx*PX_UM
        gy=pd.to_numeric(a.obs[cycol],errors="coerce").to_numpy(float)+r.ry*PX_UM

        for i,(x,y) in enumerate(zip(gx,gy)):
            rows.append((tid,i,x,y))

    if not rows:
        raise RuntimeError("no tile cells")

    d=pd.DataFrame(rows,columns=["tile","local_index","x_um","y_um"])

    # Dedup tile-overlap cell centroids greedily within 4 um.
    pts=d[["x_um","y_um"]].to_numpy(float)
    tree=cKDTree(pts)
    seen=np.zeros(len(d),bool)
    keep=[]

    for i in range(len(d)):
        if seen[i]: continue
        ids=tree.query_ball_point(pts[i],r=args.merge_radius_um)
        ids=[j for j in ids if not seen[j]]
        keep.append(i)
        seen[ids]=True

    merged=d.iloc[keep].reset_index(drop=True)
    merged.to_csv(args.out/"comseg_cells_merged.csv",index=False)

    rep=dict(
        method="ComSeg",
        dataset="merfish_1",
        prior="Cellpose nuclei from DAPI",
        uses_platform_prior=False,
        uses_image_nucleus_prior=True,
        pred_cells=int(len(merged)),
        raw_tile_cells=int(len(d)),
        merge_radius_um=args.merge_radius_um,
    )
    (args.out/"merge_summary.json").write_text(json.dumps(rep,indent=2))

    log(f"MERGE cells={len(merged):,}; raw tile cells={len(d):,}")
    banner("MERGE DONE")


def stage_report(args):
    p=args.out/"merge_summary.json"
    if not p.exists():
        raise FileNotFoundError(p)
    r=json.loads(p.read_text())

    print("="*84)
    print("ComSeg MERFISH1 DAPI-NUCLEUS PRIOR")
    print("platform mask/centroid    : NO")
    print("image-derived nucleus prior: YES")
    print(f"ROI                        : y[{ROI_Y0},{ROI_Y1}) x[{ROI_X0},{ROI_X1})")
    for k,v in r.items():
        print(f"{k:28s}: {v}")
    print("="*84)


def main():
    ap=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--stage",required=True,choices=["check","nuclei","prep","run","merge","report"])
    ap.add_argument("--root",type=Path,default=ROOT_DEFAULT)
    ap.add_argument("--dapi",type=Path,default=DAPI_DEFAULT)
    ap.add_argument("--tx",type=Path,default=TX_DEFAULT)
    ap.add_argument("--transform",type=Path,default=TRANSFORM_DEFAULT)
    ap.add_argument("--out",type=Path,default=OUT_DEFAULT)
    ap.add_argument("--tx-chunksize",type=int,default=2_000_000)

    # Cellpose prior
    ap.add_argument("--cellpose-model",default="nuclei")
    ap.add_argument("--nucleus-diameter-px",type=float,default=90.0)
    ap.add_argument("--cp-flow",type=float,default=0.4)
    ap.add_argument("--cp-cellprob",type=float,default=0.0)

    # ComSeg
    ap.add_argument("--mean-cell-diameter-um",type=float,default=10.0)
    ap.add_argument("--edge-distance-um",type=float,default=15.0)
    ap.add_argument("--knn-neighbors",type=int,default=20)
    ap.add_argument("--size-commu-min",type=int,default=3)
    ap.add_argument("--n-pcs",type=int,default=20)
    ap.add_argument("--n-comps",type=int,default=20)
    ap.add_argument("--cluster-neighbors",type=int,default=20)
    ap.add_argument("--resolution",type=float,default=1.0)
    ap.add_argument("--n-clusters-kmeans",type=int,default=20)
    ap.add_argument("--nb-min-cluster",type=int,default=3)
    ap.add_argument("--min-merge-correlation",type=float,default=0.8)
    ap.add_argument("--centroid-neighbors",type=int,default=15)
    ap.add_argument("--max-dist-centroid-um",type=float,default=8.0)
    ap.add_argument("--max-cell-radius-um",type=float,default=12.0)
    ap.add_argument("--merge-radius-um",type=float,default=4.0)
    ap.add_argument("--seed",type=int,default=0)

    ap.add_argument("--force",action="store_true")
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)

    if args.stage=="check": return stage_check(args)
    if args.stage=="nuclei": return stage_nuclei(args)
    if args.stage=="prep": return stage_prep(args)
    if args.stage=="run": return stage_run(args)
    if args.stage=="merge": return stage_merge(args)
    if args.stage=="report": return stage_report(args)


if __name__=="__main__":
    main()
