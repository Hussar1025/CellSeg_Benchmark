#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genesegnet_merfish_1_v4.py

MERFISH1 GeneSegNet training:
  * center 20000x20000 mosaic pixels are sealed test-only
  * +1024 px guard excluded from training
  * training labels come from Vizgen feature_data_<fov>.hdf5 whole-cell polygons
  * robust polygon coordinate calibration is performed PER FOV against metadata
    cell centers, so global/local micron or pixel-like boundary coordinates are
    handled without guessing
  * coordinates dataset shape (1,N,2) is normalized to (N,2)
  * all p_* polygons on the chosen z plane are rasterized

Stages: check, prep, flows, split, loader, train, status
"""
from __future__ import annotations
import argparse, sys, importlib.util, json, logging, shutil, sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import os
import sys

ROOT=Path("/data/qiuyijia/dataset/merfish_mouse_brain")
TRANSCRIPTS=ROOT/"datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_detected_transcripts_S1R1.csv"
METADATA=ROOT/"datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_metadata_S1R1.csv"
DAPI=ROOT/"images/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_images_mosaic_DAPI_z3.tif"
BOUNDARY_DIR=ROOT/"cell_boundaries"

SX=9.205861091613769531
SY=9.205849647521972656
TX=279.22039794921875
TY=349.810516357421875
H,W=61310,89085
TEST_SIZE=20000
TEST_Y0=(H-TEST_SIZE)//2
TEST_X0=(W-TEST_SIZE)//2
TEST_Y1=TEST_Y0+TEST_SIZE
TEST_X1=TEST_X0+TEST_SIZE

OUT_DEFAULT=Path("/data/qiuyijia/genesegnet_merfish_1")
BASE=Path("/data/qiuyijia/GeneSegNet/prepare_genesegnet_xenium_train.py")
REPO_DEFAULT="/data/qiuyijia/GeneSegNet/GeneSegNet"
log=logging.getLogger("gsn_merfish1")

def setup_log(out,stage):
    out.mkdir(parents=True,exist_ok=True); (out/"logs").mkdir(exist_ok=True)
    log.handlers.clear(); log.setLevel(logging.INFO)
    fmt=logging.Formatter("%(asctime)s  %(message)s","%Y-%m-%d %H:%M:%S")
    sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh=logging.FileHandler(out/"logs"/f"{stage}.log"); fh.setFormatter(fmt); log.addHandler(fh)

def banner(x): log.info("="*88); log.info(x); log.info("="*88)

def load_base():
    spec=importlib.util.spec_from_file_location("gsn_base",BASE)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def um_to_px(x,y):
    return np.asarray(x,float)*SX+TX, np.asarray(y,float)*SY+TY
def px_to_um(x,y):
    return (np.asarray(x,float)-TX)/SX,(np.asarray(y,float)-TY)/SY

def read_meta():
    m=pd.read_csv(METADATA)
    idc=m.columns[0]; m["cell_id_str"]=m[idc].astype(str)
    cx,cy=um_to_px(m.center_x,m.center_y)
    m["center_px_x"]=cx; m["center_px_y"]=cy
    m["min_px_x"]=m.min_x.astype(float)*SX+TX; m["max_px_x"]=m.max_x.astype(float)*SX+TX
    m["min_px_y"]=m.min_y.astype(float)*SY+TY; m["max_px_y"]=m.max_y.astype(float)*SY+TY
    return m

def norm_poly(a):
    a=np.asarray(a)
    if a.ndim==3 and a.shape[0]==1: a=a[0]
    elif a.ndim==3: a=np.squeeze(a)
    if a.ndim!=2: return None
    if a.shape[0]==2 and a.shape[1]!=2: a=a.T
    if a.shape[1]!=2 or len(a)<3: return None
    return a.astype(np.float64)

def raw_polys(h,cid,zindex=3):
    base=f"featuredata/{cid}"
    if base not in h: return []
    grp=h[base]
    zs=[]
    for k in grp.keys():
        s=str(k)
        if s.startswith("zIndex_"):
            try:
                zi=int(s.split("_")[-1]); zs.append((abs(zi-zindex),zi,s))
            except: pass
    for _,_,zk in sorted(zs):
        out=[]
        zg=grp[zk]
        for pk in sorted(zg.keys()):
            if not str(pk).startswith("p_"): continue
            path=f"{base}/{zk}/{pk}/coordinates"
            if path in h:
                a=norm_poly(np.asarray(h[path]))
                if a is not None: out.append(a)
        if out: return out
    return []

class FOVCalibrator:
    """
    Calibrate boundary coordinates against metadata cell centers.

    Candidate A: raw polygon coordinates are micron-like.
      global_um = raw + constant FOV offset
    Candidate B: raw coordinates are mosaic-pixel-like.
      global_um = inverse_mosaic(raw) + constant FOV offset

    Choose the candidate whose offsets across several cells have lower MAD.
    This handles both global and FOV-local coordinates.
    """
    def __init__(self,meta,zindex):
        self.meta=meta; self.zindex=zindex; self.cache={}

    def get(self,fov):
        fov=int(fov)
        if fov in self.cache: return self.cache[fov]
        import h5py
        hp=BOUNDARY_DIR/f"feature_data_{fov}.hdf5"
        rows=self.meta[self.meta.fov==fov].head(30)
        samples=[]
        if hp.exists():
            with h5py.File(hp,"r") as h:
                for r in rows.itertuples(index=False):
                    ps=raw_polys(h,str(r.cell_id_str),self.zindex)
                    if not ps: continue
                    # area-weighting isn't needed for coordinate-system calibration.
                    v=max(ps,key=len); cen=np.nanmean(v,axis=0)
                    if not np.all(np.isfinite(cen)): continue
                    samples.append((cen, np.array([float(r.center_x),float(r.center_y)])))
                    if len(samples)>=12: break
        if not samples:
            raise RuntimeError(f"FOV {fov}: cannot find boundary polygons for calibration")

        raw=np.stack([x[0] for x in samples])
        tgt=np.stack([x[1] for x in samples])

        off_um=tgt-raw
        raw_as_um=np.column_stack([(raw[:,0]-TX)/SX,(raw[:,1]-TY)/SY])
        off_px=tgt-raw_as_um

        def score(off):
            med=np.median(off,axis=0)
            resid=off-med
            return float(np.median(np.sqrt(np.sum(resid*resid,axis=1)))),med

        s_um,o_um=score(off_um); s_px,o_px=score(off_px)
        if s_um<=s_px:
            mode="raw_um"; offset=o_um; scorev=s_um
        else:
            mode="raw_px"; offset=o_px; scorev=s_px

        rec=dict(mode=mode,offset_x=float(offset[0]),offset_y=float(offset[1]),
                 score_um=float(s_um),score_px=float(s_px),n=len(samples))
        self.cache[fov]=rec
        log.info(f"  FOV {fov} boundary calibration: {rec}")
        if scorev>5.0:
            log.warning(f"  FOV {fov}: boundary-center residual {scorev:.2f} um is high")
        return rec

    def to_global_um(self,fov,a):
        c=self.get(fov)
        if c["mode"]=="raw_um":
            x=a[:,0]+c["offset_x"]; y=a[:,1]+c["offset_y"]
        else:
            x=(a[:,0]-TX)/SX+c["offset_x"]
            y=(a[:,1]-TY)/SY+c["offset_y"]
        return np.column_stack([x,y])

def check(args,out):
    import h5py,tifffile
    banner("MERFISH1 CHECK + BOUNDARY CALIBRATION")
    for p in (TRANSCRIPTS,METADATA,DAPI,BOUNDARY_DIR):
        log.info(f"  {'OK' if p.exists() else 'MISSING'} {p}")
    with tifffile.TiffFile(DAPI) as f:
        log.info(f"  DAPI shape={f.series[0].shape} dtype={f.series[0].dtype}")
    m=read_meta()
    inside=(m.center_px_x.ge(TEST_X0)&m.center_px_x.lt(TEST_X1)&
            m.center_px_y.ge(TEST_Y0)&m.center_px_y.lt(TEST_Y1))
    log.info(f"  sealed ROI y[{TEST_Y0},{TEST_Y1}) x[{TEST_X0},{TEST_X1}) cells={inside.sum():,}")
    cal=FOVCalibrator(m,args.zindex)
    # Probe 5 spatially separated FOVs.
    fovs=m.fov.drop_duplicates().iloc[np.linspace(0,m.fov.nunique()-1,5,dtype=int)].tolist()
    for f in fovs:
        try: cal.get(int(f))
        except Exception as e: log.warning(f"  calibration probe FOV {f}: {e}")
    # Verify first polygon aligns with its metadata center after calibration.
    r=m.iloc[0]; hp=BOUNDARY_DIR/f"feature_data_{int(r.fov)}.hdf5"
    with h5py.File(hp,"r") as h: ps=raw_polys(h,str(r.cell_id_str),args.zindex)
    if not ps: raise RuntimeError("first metadata cell has no readable polygon")
    a=cal.to_global_um(int(r.fov),max(ps,key=len))
    cx,cy=np.mean(a,axis=0)
    err=math.hypot(cx-float(r.center_x),cy-float(r.center_y))
    log.info(f"  first-cell calibrated polygon centroid error={err:.3f} um; vertices={len(a)}")
    if err>10:
        raise RuntimeError("boundary calibration failed: centroid error >10 um")
    json.dump(cal.cache,open(out/"boundary_calibration_check.json","w"),indent=2)
    banner("CHECK DONE")

class Mosaic:
    """
    MERFISH1 DAPI reader.

    IMPORTANT:
      These 10.9 GB Vizgen mosaic TIFFs are single-strip, uncompressed TIFFs.
      In this environment tifffile.aszarr()/zarr random slicing returns zeros,
      while tifffile.memmap() reads the actual pixels correctly.

    Therefore this reader intentionally uses memmap only.
    It does not load the 10.9 GB image into RAM; pages are read on demand.
    """
    def __init__(self):
        import tifffile
        self.z=tifffile.memmap(DAPI)
        self.mode="memmap"
        if self.z.ndim != 2:
            raise RuntimeError(f"MERFISH1 DAPI expected 2D YX, got {self.z.shape}")
        self.H,self.W=map(int,self.z.shape)
        log.info(f"  DAPI memmap shape={self.z.shape} dtype={self.z.dtype}")
        # Probe one known tissue-containing window so all-zero reader failures
        # are caught immediately.
        py=min(max(TEST_Y0,0),self.H-1024)
        px=min(max(TEST_X0,0),self.W-1024)
        q=np.asarray(self.z[py:py+1024,px:px+1024])
        log.info(
            f"  DAPI probe y={py} x={px}: "
            f"min={int(q.min())} max={int(q.max())} "
            f"mean={float(q.mean()):.2f} p99={float(np.percentile(q,99)):.2f}"
        )
        if q.max()==0:
            raise RuntimeError(
                "DAPI memmap probe is all-zero; do not continue training"
            )

    def crop(self,y,x,S):
        a=np.asarray(self.z[y:y+S,x:x+S])
        if a.shape!=(S,S):
            raise RuntimeError(
                f"DAPI crop wrong shape at y={y} x={x}: got {a.shape}, expected {(S,S)}"
            )
        return a
def heatmap(pts,S,sigma):
    from scipy import ndimage
    a=np.ones((S,S),np.uint8)
    if len(pts):
        x=np.clip(np.rint(pts[:,0]).astype(int),0,S-1)
        y=np.clip(np.rint(pts[:,1]).astype(int),0,S-1); a[y,x]=0
    d=ndimage.distance_transform_edt(a)
    return (np.exp(-(d*d)/(2*sigma*sigma))*255).astype(np.uint8)

def select_patches(m,S,guard,min_labels,max_tiles,seed):
    """
    Candidate generation intentionally mirrors the terminal validation that
    produced 2,917 valid MERFISH1 training patches.

    We use centroid pixel coordinates only here. Boundary calibration is NOT
    involved in candidate selection.
    """
    x=m["center_px_x"].to_numpy(dtype=float)
    y=m["center_px_y"].to_numpy(dtype=float)

    inside=(x>=0)&(x<W)&(y>=0)&(y<H)
    ix=np.floor(x/S).astype(np.int64)
    iy=np.floor(y/S).astype(np.int64)

    q=pd.DataFrame({"ix":ix[inside],"iy":iy[inside]})
    cnt=q.value_counts().reset_index(name="n")

    gy0=TEST_Y0-guard
    gy1=TEST_Y1+guard
    gx0=TEST_X0-guard
    gx1=TEST_X1+guard

    cand=[]
    for r in cnt.itertuples(index=False):
        x0=int(r.ix*S)
        y0=int(r.iy*S)

        if x0<0 or y0<0 or x0+S>W or y0+S>H:
            continue

        overlap=not (
            x0+S<=gx0 or x0>=gx1 or
            y0+S<=gy0 or y0>=gy1
        )
        if overlap:
            continue

        if int(r.n)>=int(min_labels):
            cand.append((y0,x0,int(r.n)))

    log.info(f"  candidate audit: metadata={len(m):,} "
             f"inside_mosaic={inside.sum():,} coarse_valid={len(cand):,}")

    if not cand:
        raise RuntimeError(
            "candidate generation returned zero although metadata loaded; "
            "this should not happen for MERFISH1"
        )

    # deterministic sampling only AFTER candidate construction
    if max_tiles and len(cand)>max_tiles:
        rng=np.random.default_rng(seed)
        w=np.sqrt(np.asarray([c[2] for c in cand],dtype=float))
        w/=w.sum()
        idx=rng.choice(len(cand),size=int(max_tiles),replace=False,p=w)
        cand=[cand[int(i)] for i in idx]

    cand=sorted(cand)
    log.info(f"  selected training patches={len(cand):,}")
    for z in sorted(cand,key=lambda x:x[2],reverse=True)[:10]:
        log.info(f"    candidate cells={z[2]} y={z[0]} x={z[1]}")
    return cand

def route_transcripts(cand,S,cache):
    cache.mkdir(parents=True,exist_ok=True)
    done=cache/"_DONE"
    paths={(y,x):cache/f"y{y:05d}_x{x:05d}.csv" for y,x,_ in cand}
    if done.exists() and all(p.exists() for p in paths.values()): return paths
    for p in paths.values(): p.unlink(missing_ok=True)
    keys=np.array([y//S*100000+x//S for y,x,_ in cand],dtype=np.int64)
    wrote=set(); scanned=kept=0
    for c in pd.read_csv(TRANSCRIPTS,usecols=["global_x","global_y","gene"],chunksize=2_000_000):
        scanned+=len(c)
        xp,yp=um_to_px(c.global_x.to_numpy(),c.global_y.to_numpy())
        ix=np.floor(xp/S).astype(int); iy=np.floor(yp/S).astype(int)
        kval=iy*100000+ix; ok=np.isin(kval,keys)
        if ok.any():
            q=pd.DataFrame({"ix":ix[ok],"iy":iy[ok],"px":xp[ok],"py":yp[ok],
                            "gene":c.gene.astype(str).to_numpy()[ok]})
            kept+=len(q)
            for (iy0,ix0),g in q.groupby(["iy","ix"]):
                y=int(iy0*S); x=int(ix0*S); key=(y,x)
                if key not in paths: continue
                z=pd.DataFrame({"spotX":g.px.to_numpy()-x,"spotY":g.py.to_numpy()-y,
                                "gene":g.gene.to_numpy()})
                z.to_csv(paths[key],mode="a",header=(key not in wrote),index=False)
                wrote.add(key)
        log.info(f"  transcripts scanned={scanned:,} routed={kept:,}")
    for p in paths.values():
        if not p.exists(): pd.DataFrame(columns=["spotX","spotY","gene"]).to_csv(p,index=False)
    done.write_text(json.dumps({"scanned":scanned,"routed":kept},indent=2))
    return paths

def raster_label(qmeta,cal,y0,x0,S,zindex):
    import cv2,h5py
    lab=np.zeros((S,S),np.int32); label=1; missing=0
    for r in qmeta.itertuples(index=False):
        hp=BOUNDARY_DIR/f"feature_data_{int(r.fov)}.hdf5"
        if not hp.exists(): missing+=1; continue
        with h5py.File(hp,"r") as h: ps=raw_polys(h,str(r.cell_id_str),zindex)
        if not ps: missing+=1; continue
        drew=False
        for raw in ps:
            gum=cal.to_global_um(int(r.fov),raw)
            px,py=um_to_px(gum[:,0],gum[:,1])
            v=np.column_stack([px-x0,py-y0])
            if v[:,0].max()<-20 or v[:,0].min()>S+20 or v[:,1].max()<-20 or v[:,1].min()>S+20:
                continue
            cv2.fillPoly(lab,[np.rint(v).astype(np.int32)],label); drew=True
        if drew: label+=1
    # compact
    u=np.unique(lab); u=u[u>0]
    if len(u):
        rm=np.zeros(lab.max()+1,np.uint16); rm[u]=np.arange(1,len(u)+1,dtype=np.uint16)
        lab=rm[lab]
    return lab.astype(np.uint16),missing

def prep(args,out):
    import tifffile
    banner("MERFISH1 PREP")
    raw=out/"raw"
    if args.force_prep:
        shutil.rmtree(raw,ignore_errors=True); shutil.rmtree(out/"cache",ignore_errors=True)
    for s in ("images","labels","spots","HeatMaps/HeatMap_all","HeatMaps/HeatMap"):
        (raw/s).mkdir(parents=True,exist_ok=True)
    m=read_meta(); cal=FOVCalibrator(m,args.zindex)
    cand=select_patches(m,args.sample_px,args.guard_px,args.min_labels,args.max_train_tiles,args.seed)
    log.info(f"  HARD AUDIT before routing: len(cand)={len(cand)}")
    if len(cand) == 0:
        raise RuntimeError("HARD AUDIT: candidate list unexpectedly empty before routing")
    if args.max_train_tiles and len(cand) > args.max_train_tiles:
        raise RuntimeError("HARD AUDIT: candidate list exceeds requested max after sampling")
    routes=route_transcripts(cand,args.sample_px,out/"cache"/"routed_spots")
    log.info(f"  HARD AUDIT after routing: routes={len(routes)}")
    if len(routes) != len(cand):
        raise RuntimeError(f"HARD AUDIT: routes {len(routes)} != cand {len(cand)}")
    mos=Mosaic(); S=args.sample_px
    man=[]; wrote=0; skipped_label=0; skipped_spot=0; skipped_image=0
    log.info(f"  HARD AUDIT entering sample loop with {len(cand)} candidates")
    if len(cand)==0:
        raise RuntimeError("HARD AUDIT: zero candidates at sample-loop entry")
    for y0,x0,n in cand:
        sp=pd.read_csv(routes[(y0,x0)])
        if len(sp)<args.min_spots: skipped_spot+=1; continue
        q=m[m.max_px_x.ge(x0)&m.min_px_x.lt(x0+S)&m.max_px_y.ge(y0)&m.min_px_y.lt(y0+S)]
        lab,miss=raster_label(q,cal,y0,x0,S,args.zindex)
        if int(lab.max())<args.min_labels:
            skipped_label+=1
            if skipped_label<=10:
                log.warning(f"  label-skip y={y0} x={x0}: labels={lab.max()} bbox_cells={len(q)} missing={miss}")
            continue
        im=mos.crop(y0,x0,S).astype(np.float32)
        lo,hi=np.percentile(im,[args.pmin,args.pmax])
        if hi<=lo:
            skipped_image+=1
            if skipped_image<=10:
                log.warning(
                    f"  image-skip y={y0} x={x0}: "
                    f"shape={im.shape} min={float(im.min()) if im.size else 'NA'} "
                    f"max={float(im.max()) if im.size else 'NA'} "
                    f"pmin={lo} pmax={hi}"
                )
            continue
        im8=(np.clip((im-lo)/(hi-lo),0,1)*255).astype(np.uint8)
        pts=sp[["spotX","spotY"]].to_numpy(float)
        hm=heatmap(pts,S,args.heatmap_sigma)
        xi=np.clip(np.rint(pts[:,0]).astype(int),0,S-1); yi=np.clip(np.rint(pts[:,1]).astype(int),0,S-1)
        hin=pts[lab[yi,xi]>0]; hmi=heatmap(hin,S,args.heatmap_sigma)
        name=f"merfish1_y{y0:05d}_x{x0:05d}"
        tifffile.imwrite(raw/"images"/f"{name}_image.tif",im8[...,None])
        tifffile.imwrite(raw/"labels"/f"{name}_label.tif",lab)
        tifffile.imwrite(raw/"HeatMaps"/"HeatMap_all"/f"{name}_gaumap_all.tif",hm[...,None])
        tifffile.imwrite(raw/"HeatMaps"/"HeatMap"/f"{name}_gaumap.tif",hmi[...,None])
        sp.to_csv(raw/"spots"/f"{name}.csv",index=False)
        man.append(dict(name=name,x=x0,y=y0,instances=int(lab.max()),spots=len(sp),missing=miss))
        wrote+=1
        if wrote%20==0: log.info(f"  wrote {wrote} samples")
    with open(raw/"manifest.jsonl","w") as f:
        for r in man: f.write(json.dumps(r)+"\n")
    json.dump(cal.cache,open(out/"boundary_calibration.json","w"),indent=2)
    log.info(
        f"PREP DONE samples={wrote}; skipped_label={skipped_label}; "
        f"skipped_spot={skipped_spot}; skipped_image={skipped_image}"
    )
    if wrote==0:
        raise RuntimeError(
            "PREP produced zero samples; inspect label/spot/image-skip diagnostics "
            "and boundary_calibration.json"
        )

def stage_split_merfish(out,args):
    """
    Self-contained MERFISH1 split.

    All 600 prepared samples are already outside the sealed central
    20000x20000 ROI (+ guard), so this split only creates internal
    train/val/test subsets from those external samples.  It does not touch
    the sealed ROI.

    Layout matches GeneSegNet loader expectations:
      split/train/merfish1/{images,labels,spots,HeatMaps/...}
      split/val/merfish1/...
      split/test/merfish1/...
    """
    banner("MERFISH1 SPLIT  (sealed ROI remains untouched)")
    raw=out/"raw"
    man=raw/"manifest.jsonl"
    if not man.exists():
        raise RuntimeError("no raw/manifest.jsonl; run --stage prep first")

    rows=[json.loads(x) for x in man.read_text().splitlines() if x.strip()]
    rows=[r for r in rows if (raw/"images"/f"{r['name']}_image.tif").exists()]
    if not rows:
        raise RuntimeError("manifest has no surviving samples")

    rng=np.random.default_rng(args.seed)
    order=np.arange(len(rows))
    rng.shuffle(order)

    n=len(rows)
    n_train=int(round(n*args.train_frac))
    n_val=int(round(n*args.val_frac))

    # Keep at least one sample in each subset when possible.
    if n>=3:
        n_train=max(1,min(n-2,n_train))
        n_val=max(1,min(n-n_train-1,n_val))
    n_test=n-n_train-n_val

    part=np.empty(n,dtype=object)
    part[order[:n_train]]="train"
    part[order[n_train:n_train+n_val]]="val"
    part[order[n_train+n_val:]]="test"

    split=out/"split"
    shutil.rmtree(split,ignore_errors=True)

    counts={"train":0,"val":0,"test":0}
    assignment=[]

    for i,r in enumerate(rows):
        p=str(part[i])
        base=split/p/"merfish1"

        for sub in (
            "images","labels","spots",
            "HeatMaps/HeatMap_all","HeatMaps/HeatMap"
        ):
            (base/sub).mkdir(parents=True,exist_ok=True)

        name=r["name"]

        pairs=[
            (raw/"images"/f"{name}_image.tif",
             base/"images"/f"{name}_image.tif"),

            (raw/"labels"/f"{name}_label.tif",
             base/"labels"/f"{name}_label.tif"),

            (raw/"spots"/f"{name}.csv",
             base/"spots"/f"{name}.csv"),

            (raw/"HeatMaps/HeatMap_all"/f"{name}_gaumap_all.tif",
             base/"HeatMaps/HeatMap_all"/f"{name}_gaumap_all.tif"),

            (raw/"HeatMaps/HeatMap"/f"{name}_gaumap.tif",
             base/"HeatMaps/HeatMap"/f"{name}_gaumap.tif"),
        ]

        # stage flows normally writes this next to the original label.
        flow=raw/"labels"/f"{name}_label_flows.tif"
        if flow.exists():
            pairs.append(
                (flow,base/"labels"/f"{name}_label_flows.tif")
            )

        missing=[]
        linked=0
        for s,d in pairs:
            if not s.exists():
                missing.append(str(s))
                continue
            if d.exists() or d.is_symlink():
                d.unlink()
            os.symlink(os.path.relpath(s,d.parent),d)
            linked+=1

        if missing:
            raise RuntimeError(
                f"{name}: missing split source files: {missing}"
            )

        rr=dict(r)
        rr["part"]=p
        rr["linked_files"]=linked
        assignment.append(rr)
        counts[p]+=1

    pd.DataFrame(assignment).to_csv(
        out/"split_assignment.tsv",sep="\t",index=False
    )

    log.info(
        f"  train: {counts['train']} samples\n"
        f"  val  : {counts['val']} samples\n"
        f"  test : {counts['test']} samples"
    )

    # Audit actual linked image counts.
    for p in ("train","val","test"):
        base=split/p/"merfish1"
        ni=len(list((base/"images").glob("*_image.tif")))
        nl=len(list((base/"labels").glob("*_label.tif")))
        nf=len(list((base/"labels").glob("*_label_flows.tif")))
        log.info(
            f"  audit {p}: images={ni} labels={nl} flows={nf}"
        )
        if ni!=counts[p] or nl!=counts[p]:
            raise RuntimeError(
                f"split audit failed for {p}: "
                f"images={ni} labels={nl} expected={counts[p]}"
            )

    banner("SPLIT DONE")

def stage_loader_merfish(out,args):
    """
    MERFISH1 self-contained loader/layout audit.

    This intentionally avoids the Xenium base stage_check(), because the
    MERFISH split layout is:
        split/{train,val,test}/merfish1/...
    and the base checker may silently ignore it.

    The audit checks:
      - image / label / flow / heatmap counts
      - no broken symlinks
      - representative array shapes and dtypes
      - GeneSegNet's own load_train_test_data() on each split
    """
    banner("MERFISH1 LOADER / LAYOUT CHECK")

    import tifffile
    import importlib.util

    split=out/"split"
    expected={"train":None,"val":None,"test":None}

    for part in ("train","val","test"):
        base=split/part/"merfish1"
        if not base.exists():
            raise RuntimeError(f"missing split directory: {base}")

        imgs=sorted((base/"images").glob("*_image.tif"))
        labs=sorted((base/"labels").glob("*_label.tif"))
        flows=sorted((base/"labels").glob("*_label_flows.tif"))
        hm_all=sorted((base/"HeatMaps/HeatMap_all").glob("*.tif"))
        hm=sorted((base/"HeatMaps/HeatMap").glob("*.tif"))

        log.info(
            f"  {part}: images={len(imgs)} labels={len(labs)} "
            f"flows={len(flows)} heat_all={len(hm_all)} heat={len(hm)}"
        )

        if not imgs or not labs or not flows:
            raise RuntimeError(
                f"{part}: empty required loader inputs "
                f"(images={len(imgs)}, labels={len(labs)}, flows={len(flows)})"
            )

        if not (len(imgs)==len(labs)==len(flows)):
            raise RuntimeError(
                f"{part}: count mismatch images={len(imgs)} "
                f"labels={len(labs)} flows={len(flows)}"
            )

        # Detect broken symlinks explicitly.
        for p in (imgs[:3] + labs[:3] + flows[:3]):
            if p.is_symlink() and not p.resolve().exists():
                raise RuntimeError(f"broken symlink: {p} -> {p.resolve()}")

        im=tifffile.imread(imgs[0])
        la=tifffile.imread(labs[0])
        fl=tifffile.imread(flows[0])

        log.info(
            f"    sample image {imgs[0].name}: "
            f"shape={im.shape} dtype={im.dtype} "
            f"min={float(np.min(im)):.1f} max={float(np.max(im)):.1f}"
        )
        log.info(
            f"    sample label {labs[0].name}: "
            f"shape={la.shape} dtype={la.dtype} max={float(np.max(la)):.1f}"
        )
        log.info(
            f"    sample flow  {flows[0].name}: "
            f"shape={fl.shape} dtype={fl.dtype}"
        )

        # Expected 1024 x 1024 spatial canvas.
        # GeneSegNet images are commonly HWC (H,W,C), while flows are CHW.
        def spatial_shape(a, kind):
            sh=tuple(a.shape)
            if len(sh)==2:
                return sh
            if len(sh)==3:
                # HWC: small channel dimension last
                if sh[-1] <= 8 and sh[0] > 8 and sh[1] > 8:
                    return sh[:2]
                # CHW: small channel dimension first
                if sh[0] <= 8 and sh[1] > 8 and sh[2] > 8:
                    return sh[1:]
            raise RuntimeError(f"{part}: unsupported {kind} layout {sh}")

        im_sp=spatial_shape(im,"image")
        la_sp=spatial_shape(la,"label")
        fl_sp=spatial_shape(fl,"flow")

        log.info(
            f"    spatial audit: image={im_sp} label={la_sp} flow={fl_sp}"
        )

        want=(args.sample_px,args.sample_px)
        if im_sp != want:
            raise RuntimeError(
                f"{part}: image spatial shape {im_sp} != {want}"
            )
        if la_sp != want:
            raise RuntimeError(
                f"{part}: label spatial shape {la_sp} != {want}"
            )
        if fl_sp != want:
            raise RuntimeError(
                f"{part}: flow spatial shape {fl_sp} != {want}"
            )

        expected[part]=len(imgs)

    # Now exercise the repository loader itself.
    repo=Path(args.repo)
    gseg_io=repo/"GeneSegNet"/"Gseg_io.py"
    if not gseg_io.exists():
        gseg_io=repo/"Gseg_io.py"
    if not gseg_io.exists():
        raise RuntimeError(f"cannot locate Gseg_io.py under {repo}")

    # Gseg_io imports utils/plot/transforms as top-level modules, so put the
    # repository's GeneSegNet package directory on sys.path explicitly.
    pkg_dir=gseg_io.parent
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0,str(pkg_dir))

    spec=importlib.util.spec_from_file_location("_merfish_gseg_io",gseg_io)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Use a small N for the audit; load_train_test_data() receives the
    # merfish1 directory as a STRING, never a list.
    for part in ("train","val","test"):
        # load_train_test_data expects the PARENT directory containing one or
        # more dataset folders. It calls os.listdir(d) itself and should find
        # "merfish1" underneath.
        d=str(split/part)
        log.info(f"    repo loader input {part}: {d}")
        try:
            output=mod.load_train_test_data(
                d,
                min(4,expected[part]),
                args,
                image_filter=args.img_filter,
                mask_filter=args.mask_filter
            )
        except TypeError:
            # Some repository revisions use positional filters.
            output=mod.load_train_test_data(
                d,
                min(4,expected[part]),
                args,
                args.img_filter,
                args.mask_filter
            )

        if output is None:
            raise RuntimeError(f"{part}: repository loader returned None")

        # Report the broad structure without assuming one repository revision.
        if isinstance(output,(list,tuple)):
            desc=[]
            for x in output:
                try:
                    desc.append(getattr(x,"shape",f"len={len(x)}"))
                except Exception:
                    desc.append(type(x).__name__)
            log.info(f"    repo loader {part}: {desc}")
        else:
            log.info(f"    repo loader {part}: {type(output).__name__}")

    log.info("  loader/layout OK")
    banner("CHECK DONE")

def delegate(stage,args,out):
    b=load_base()
    if stage=="flows": return b.stage_flows(out,args)
    if stage=="split": return stage_split_merfish(out,args)
    if stage=="loader": return stage_loader_merfish(out,args)
    if stage=="train": return b.stage_train(out,args)
    if stage=="status": return b.stage_status(out,args)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage",default="check",choices=["check","prep","flows","split","loader","train","status"])
    ap.add_argument("--out",default=str(OUT_DEFAULT)); ap.add_argument("--repo",default=REPO_DEFAULT)
    ap.add_argument("--sample-px",type=int,default=1024); ap.add_argument("--guard-px",type=int,default=1024)
    ap.add_argument("--max-train-tiles",type=int,default=600); ap.add_argument("--zindex",type=int,default=3)
    ap.add_argument("--min-labels",type=int,default=12); ap.add_argument("--min-spots",type=int,default=100)
    ap.add_argument("--heatmap-sigma",type=float,default=3.0); ap.add_argument("--pmin",type=float,default=1.0); ap.add_argument("--pmax",type=float,default=99.8)
    ap.add_argument("--seed",type=int,default=0); ap.add_argument("--force-prep",action="store_true")
    ap.add_argument("--train-frac",type=float,default=.70); ap.add_argument("--val-frac",type=float,default=.15)
    ap.add_argument("--chan",type=int,default=2); ap.add_argument("--img-filter",default="_image"); ap.add_argument("--mask-filter",default="_label")
    ap.add_argument("--epochs",type=int,default=300); ap.add_argument("--lr",type=float,default=.001); ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--recursive-num",type=int,default=1); ap.add_argument("--output-filename",default="newlabels")
    ap.add_argument("--from-scratch",action="store_true"); ap.add_argument("--gpu",default="auto"); ap.add_argument("--train-extra",default="")
    ap.add_argument("--dataset",default="merfish_1"); ap.add_argument("--label-source",default="cell")
    args=ap.parse_args(); out=Path(args.out); setup_log(out,args.stage)
    if args.stage=="check": return check(args,out)
    if args.stage=="prep": return prep(args,out)
    return delegate(args.stage,args,out)
if __name__=="__main__": main()
