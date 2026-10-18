#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOMS xenium_5 fragment-rescue postprocessor.

Use AFTER the completed hs9_g3000 BOMS run. It does not rerun BOMS.

Rationale:
  seam-only stitching merged only 37 instances, so most fragments are not
  confined to seam centroids. This pass attaches low-count fragments to the
  nearest plausible larger cell, but caps merges so cell_ratio never falls
  below --min-ratio (default 0.80).

Rules:
  donor <= fragment_max_tx
  target >= target_min_tx
  distance <= radius
  donors >=10 tx also require expression cosine >= cosine_min
  donors <10 use tighter tiny_radius and a minimal gene-overlap requirement
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

IN="/data/qiuyijia/boms_xenium_5_hs9_g3000/boms_xenium_5_hs9_g3000.npz"
OUT="/data/qiuyijia/boms_xenium_5_hs9_g3000/boms_xenium_5_hs9_g3000_rescued.npz"

def summary(c,gt):
    c=np.asarray(c)
    return dict(cells=len(c),ratio=len(c)/gt,median=float(np.median(c)),
                mean=float(np.mean(c)),mean_median=float(np.mean(c)/np.median(c)),
                p5=float(np.percentile(c,5)),p95=float(np.percentile(c,95)),
                frag_lt10=float((c<10).mean()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default=IN); ap.add_argument("--out",default=OUT)
    ap.add_argument("--gt",type=int,default=11975)
    ap.add_argument("--min-ratio",type=float,default=0.80)
    ap.add_argument("--fragment-max-tx",type=int,default=80)
    ap.add_argument("--target-min-tx",type=int,default=120)
    ap.add_argument("--radius-um",type=float,default=10.0)
    ap.add_argument("--tiny-radius-um",type=float,default=7.0)
    ap.add_argument("--cosine-min",type=float,default=0.15)
    ap.add_argument("--combined-max-tx",type=int,default=1800)
    a=ap.parse_args()

    z=np.load(a.input,allow_pickle=True)
    seg=z["seg"].astype(np.int64); x=z["x"].astype(float); y=z["y"].astype(float)
    C=z["count_mat"]
    n=C.shape[0]
    labels=np.unique(seg)
    if seg.min()>=1 and seg.max()<=n:
        rowseg=seg.astype(np.int64)-1
    elif seg.min()>=0 and seg.max()<n:
        rowseg=seg.astype(np.int64)
    elif len(labels)==n:
        mp={int(l):i for i,l in enumerate(sorted(labels))}
        rowseg=np.fromiter((mp[int(v)] for v in seg),dtype=np.int64,count=len(seg))
    else:
        raise RuntimeError("cannot map seg labels to count_mat rows")

    counts=np.asarray(C.sum(axis=1)).ravel().astype(np.int64)
    print("BEFORE",json.dumps(summary(counts,a.gt)))

    nn=np.bincount(rowseg,minlength=n)
    cx=np.bincount(rowseg,weights=x,minlength=n)/np.maximum(nn,1)
    cy=np.bincount(rowseg,weights=y,minlength=n)/np.maximum(nn,1)

    targets=np.flatnonzero(counts>=a.target_min_tx)
    donors=np.flatnonzero(counts<=a.fragment_max_tx)
    tree=cKDTree(np.c_[cx[targets],cy[targets]])

    max_merges=max(0,n-int(np.ceil(a.gt*a.min_ratio)))
    proposals=[]
    for i in donors:
        r=a.tiny_radius_um if counts[i]<10 else a.radius_um
        dist,jj=tree.query([cx[i],cy[i]],k=1,distance_upper_bound=r)
        if not np.isfinite(dist) or jj>=len(targets): continue
        j=int(targets[jj])
        if counts[i]+counts[j]>a.combined_max_tx: continue

        ai=np.asarray(C[i]).ravel()
        aj=np.asarray(C[j]).ravel()
        if counts[i]<10:
            # tiny expression vectors are too sparse for stable cosine
            overlap=int(np.count_nonzero((ai>0)&(aj>0)))
            if overlap<1: continue
            score=1.0
        else:
            den=np.linalg.norm(ai)*np.linalg.norm(aj)
            score=float(ai@aj/den) if den else 0.0
            if score<a.cosine_min: continue
        proposals.append((float(dist),-score,int(i),j,score))

    proposals.sort()
    used_donor=set()
    target_add={}
    merges=[]
    for dist,neg,i,j,score in proposals:
        if len(merges)>=max_merges: break
        if i in used_donor: continue
        # Re-check target aggregate cap.
        add=target_add.get(j,0)
        if counts[j]+add+counts[i]>a.combined_max_tx: continue
        used_donor.add(i); target_add[j]=add+counts[i]
        merges.append((i,j,dist,score))

    donor_to_target={i:j for i,j,_,_ in merges}
    # Build root mapping: donors disappear into target; all others remain.
    roots=np.arange(n,dtype=np.int64)
    for i,j,_,_ in merges: roots[i]=j
    kept=np.array([i for i in range(n) if i not in donor_to_target],dtype=np.int64)
    new_index={int(old):k for k,old in enumerate(kept)}
    newC=np.asarray(C[kept]).copy()
    for i,j,_,_ in merges:
        newC[new_index[j]] += C[i]

    mapped=np.empty(n,dtype=np.int64)
    for old in range(n):
        root=donor_to_target.get(old,old)
        mapped[old]=new_index[root]
    newseg=mapped[rowseg].astype(np.int32)+1

    nnew=len(kept)
    nnt=np.bincount(newseg-1,minlength=nnew)
    ncx=np.bincount(newseg-1,weights=x,minlength=nnew)/np.maximum(nnt,1)
    ncy=np.bincount(newseg-1,weights=y,minlength=nnew)/np.maximum(nnt,1)
    newloc=np.c_[ncx,ncy]
    newcounts=np.asarray(newC.sum(axis=1)).ravel()

    payload={k:z[k] for k in z.files if k not in ("seg","count_mat","cell_loc")}
    payload.update(seg=newseg,count_mat=newC,cell_loc=newloc)
    np.savez_compressed(a.out,**payload)
    meta=dict(params=vars(a),before=summary(counts,a.gt),after=summary(newcounts,a.gt),
              merges=len(merges),first_merges=merges[:100])
    Path(a.out+".json").write_text(json.dumps(meta,indent=2))
    print("AFTER ",json.dumps(meta["after"]))
    print("MERGES",len(merges),"max_allowed",max_merges)
    print("SAVED",a.out)

if __name__=="__main__": main()
