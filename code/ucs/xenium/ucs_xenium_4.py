#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, gc, json, time, shutil, argparse, subprocess, xml.etree.ElementTree as ET
import cv2, numpy as np, pandas as pd, tifffile
try:
    import pyarrow.parquet as pq
except Exception:
    pq=None

ROOT='/data/qiuyijia/dataset/xenium_liver'
OUTS=os.path.join(ROOT,'outs')
UCS_DIR='/data/qiuyijia/ucs/UCS'
UCS_RUN=os.path.join(UCS_DIR,'run.py')
WORK='/data/qiuyijia/ucs_xenium_liver_clean'
GENE_MAP=os.path.join(WORK,'gene_map.tif')
GENE_INDEX=os.path.join(WORK,'gene_index.csv')
NUCLEI_MASK=os.path.join(WORK,'nuclei_mask.tif')
NUCLEI_TABLE=os.path.join(WORK,'dapi_nuclei.csv')
QC_JSON=os.path.join(WORK,'input_qc.json')
LOG_DIR=os.path.join(WORK,'ucs_log')
PRED_MASK=os.path.join(LOG_DIR,'pred','segmentation_mask.tif')
SEP='='*88

PATCH_SIZE=48
DILATION_KERNEL_SIZE=10
DILATION_ITER_NUM=4
TAU=5
FG_NET_EPOCH=1
FG_NET_BATCH_SIZE=8
CELL_NET_EPOCH=1

def P(*x):print(*x,flush=True)

def find_unique(root,name):
    hits=[]
    for dp,_,fs in os.walk(root):
        if name in fs:hits.append(os.path.join(dp,name))
    if not hits:raise FileNotFoundError(f'{name} not found under {root}')
    root_hits=[x for x in hits if os.path.dirname(x)==root]
    if len(root_hits)==1:return root_hits[0]
    if len(hits)==1:return hits[0]
    raise RuntimeError(f'multiple {name}: {hits}')

def ome_info(path):
    with tifffile.TiffFile(path) as tf:
        ome=tf.ome_metadata
        s=tf.series[0]
        shape,axes,dtype=tuple(s.shape),s.axes,str(s.dtype)
    if not ome:raise RuntimeError('No OME metadata in morphology_focus.')
    root=ET.fromstring(ome)
    uri=root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
    ns={'o':uri} if uri else {}
    px=root.find('.//o:Pixels',ns) if ns else root.find('.//Pixels')
    if px is None:raise RuntimeError('OME Pixels element missing.')
    psx=px.attrib.get('PhysicalSizeX'); psy=px.attrib.get('PhysicalSizeY')
    if psx is None or psy is None:raise RuntimeError('OME PhysicalSizeX/Y missing.')
    chs=root.findall('.//o:Channel',ns) if ns else root.findall('.//Channel')
    names=[c.attrib.get('Name',f'channel_{i}') for i,c in enumerate(chs)]
    return {'shape':shape,'axes':axes,'dtype':dtype,
            'pixel_size_x_um':float(psx),'pixel_size_y_um':float(psy),'channels':names}

def choose_dapi(names,explicit):
    if explicit is not None:
        if explicit<0 or explicit>=len(names):raise ValueError('bad --dapi-channel')
        return explicit
    hits=[i for i,n in enumerate(names) if any(k in str(n).lower() for k in ['dapi','dna','nuclear','nucleus'])]
    if len(hits)==1:return hits[0]
    raise RuntimeError(f'Cannot uniquely auto-detect DAPI channel. channels={names}. Re-run with --dapi-channel INDEX.')

def read_dapi(path,meta,ci):
    with tifffile.TiffFile(path) as tf:
        arr=tf.series[0].asarray()
    axes=meta['axes']
    if 'C' in axes:
        i=axes.index('C'); arr=np.take(arr,ci,axis=i); axes=axes[:i]+axes[i+1:]
    elif 'S' in axes and len(meta['channels'])>1:
        i=axes.index('S'); arr=np.take(arr,ci,axis=i); axes=axes[:i]+axes[i+1:]
    elif len(meta['channels'])>1:
        raise RuntimeError(f'Multiple channels but axes={axes} has no C/S')
    if 'Z' in axes:
        i=axes.index('Z'); arr=arr.max(axis=i); axes=axes[:i]+axes[i+1:]
    while arr.ndim>2:
        i=next((i for i,n in enumerate(arr.shape) if n==1),None)
        if i is None:raise RuntimeError(f'DAPI remains non-2D: {arr.shape}, axes={axes}')
        arr=np.take(arr,0,axis=i); axes=axes[:i]+axes[i+1:]
    if axes=='XY':arr=arr.T; axes='YX'
    if axes!='YX':raise RuntimeError(f'Final DAPI axes={axes}, shape={arr.shape}')
    return arr

def norm8(x):
    x=x.astype(np.float32)
    lo,hi=np.percentile(x,[1,99.7])
    return (np.clip((x-lo)/max(float(hi-lo),1e-6),0,1)*255).astype(np.uint8)

def count_labels(m):return int(np.sum(np.unique(m)>0))

def build_gene_map(tx_path,meta,bf,qv_min,force):
    if os.path.isfile(GENE_MAP) and not force:
        gm=tifffile.memmap(GENE_MAP); P('[gene_map] reuse',gm.shape,gm.dtype); return tuple(gm.shape)
    if pq is None:raise RuntimeError('pyarrow is required.')
    axes=meta['axes']; H=int(meta['shape'][axes.index('Y')]); W=int(meta['shape'][axes.index('X')])
    psx,psy=meta['pixel_size_x_um'],meta['pixel_size_y_um']
    mh,mw=int(np.ceil(H/bf)),int(np.ceil(W/bf))
    pf=pq.ParquetFile(tx_path); cols=pf.schema.names
    need=['x_location','y_location','feature_name']
    for c in need:
        if c not in cols:raise RuntimeError(f'transcripts missing {c}; {cols}')
    read_cols=need+(['qv'] if 'qv' in cols else [])
    genes=set(); nt=nk=0
    P(SEP);P('PASS 1 — transcript QC / gene vocabulary');P(SEP)
    for bi,b in enumerate(pf.iter_batches(batch_size=1_000_000,columns=read_cols),1):
        d=b.to_pandas(); nt+=len(d)
        if 'qv' in d:d=d[pd.to_numeric(d.qv,errors='coerce')>=qv_min]
        x=pd.to_numeric(d.x_location,errors='coerce'); y=pd.to_numeric(d.y_location,errors='coerce')
        ok=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<W*psx)&(y>=0)&(y<H*psy)
        d=d[ok]; nk+=len(d); genes.update(d.feature_name.astype(str).unique())
        if bi%20==0:P(f' read={nt:,} keep={nk:,} genes={len(genes):,}')
    P(f'total={nt:,} keep={nk:,} ({nk/max(nt,1)*100:.2f}%)')
    if nk<0.8*max(nt,1):raise RuntimeError('Transcript/image physical coverage <80%; coordinate mismatch suspected.')
    genes=sorted(genes); g2i={g:i for i,g in enumerate(genes)}
    pd.DataFrame({'gene':genes,'channel':np.arange(len(genes))}).to_csv(GENE_INDEX,index=False)
    bytes16=mh*mw*len(genes)*2
    P(f'UCS canvas={mh}x{mw}x{len(genes)}, temp uint16={bytes16/1024**3:.2f} GiB')
    if bytes16>24*1024**3:raise RuntimeError('Dense gene_map temp memory >24 GiB; increase --bin-factor.')
    gm=np.zeros((mh,mw,len(genes)),np.uint16)
    P(SEP);P('PASS 2 — build gene_map');P(SEP)
    for bi,b in enumerate(pf.iter_batches(batch_size=1_000_000,columns=read_cols),1):
        d=b.to_pandas()
        if 'qv' in d:d=d[pd.to_numeric(d.qv,errors='coerce')>=qv_min]
        x=pd.to_numeric(d.x_location,errors='coerce'); y=pd.to_numeric(d.y_location,errors='coerce')
        ok=np.isfinite(x)&np.isfinite(y)
        d=d[ok]; x=x[ok]; y=y[ok]
        bx=(np.floor(x.to_numpy()/psx).astype(np.int64)//bf)
        by=(np.floor(y.to_numpy()/psy).astype(np.int64)//bf)
        gi=d.feature_name.astype(str).map(g2i).to_numpy()
        ok=(bx>=0)&(bx<mw)&(by>=0)&(by<mh)&pd.notna(gi)
        np.add.at(gm,(by[ok],bx[ok],gi[ok].astype(np.int64)),1)
        if bi%20==0:P(' batch',bi)
    P('max count/bin/gene=',int(gm.max()))
    out=np.clip(gm,0,255).astype(np.uint8); del gm; gc.collect()
    tifffile.imwrite(GENE_MAP,out,bigtiff=True,photometric='minisblack'); shape=tuple(out.shape)
    del out; gc.collect(); P('saved ->',GENE_MAP); return shape

def build_prior(dapi,bf,thr_scale,min_area,max_area,force):
    if os.path.isfile(NUCLEI_MASK) and not force:
        m=tifffile.imread(NUCLEI_MASK); P('[prior] reuse cells=',count_labels(m)); return m
    H,W=dapi.shape; mh,mw=int(np.ceil(H/bf)),int(np.ceil(W/bf))
    small=cv2.resize(norm8(dapi),(mw,mh),interpolation=cv2.INTER_AREA)
    e=cv2.createCLAHE(2.0,(8,8)).apply(small)
    blur=cv2.GaussianBlur(e,(5,5),0)
    otsu,_=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thr=float(np.clip(otsu*thr_scale,1,254))
    b=(blur>=thr).astype(np.uint8)
    k=np.ones((3,3),np.uint8)
    b=cv2.morphologyEx(b,cv2.MORPH_OPEN,k); b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,k)
    n,lab,stats,_=cv2.connectedComponentsWithStats(b,8)
    clean=np.zeros_like(b)
    for i in range(1,n):
        a=int(stats[i,cv2.CC_STAT_AREA])
        if min_area<=a<=max_area:clean[lab==i]=1
    dist=cv2.distanceTransform(clean,cv2.DIST_L2,5)
    if dist.max()<=0:raise RuntimeError('DAPI prior foreground empty.')
    lm=cv2.dilate(dist,np.ones((5,5),np.uint8))
    peaks=((dist>=lm-1e-6)&(dist>=1)&(clean>0)).astype(np.uint8)
    _,seeds=cv2.connectedComponents(peaks)
    markers=seeds.astype(np.int32)+1; markers[clean==0]=1; markers[(clean>0)&(seeds==0)]=0
    ws=cv2.watershed(cv2.cvtColor(e,cv2.COLOR_GRAY2BGR),markers)
    prior=np.zeros((mh,mw),np.int32); rows=[]; nid=0
    for wid in np.unique(ws):
        if wid<2:continue
        yy,xx=np.where(ws==wid); a=len(xx)
        if not(min_area<=a<=max_area):continue
        nid+=1; prior[yy,xx]=nid; rows.append((nid,float(xx.mean()),float(yy.mean()),a))
    pd.DataFrame(rows,columns=['nucleus_id','x_bin','y_bin','area_bin_px']).to_csv(NUCLEI_TABLE,index=False)
    tifffile.imwrite(NUCLEI_MASK,prior.astype(np.uint32))
    P(f'DAPI Otsu={otsu:.2f} threshold={thr:.2f} cells={nid:,} foreground={np.mean(prior>0)*100:.3f}%')
    return prior.astype(np.uint32)

def run_ucs(gpu,force):
    gm=tifffile.memmap(GENE_MAP); nm=tifffile.imread(NUCLEI_MASK)
    if tuple(gm.shape[:2])!=tuple(nm.shape):raise RuntimeError(f'shape mismatch {gm.shape[:2]} vs {nm.shape}')
    del gm
    if os.path.isdir(LOG_DIR):
        if force:shutil.rmtree(LOG_DIR)
        else:raise RuntimeError('ucs_log exists; use --force-run')
    cmd=[sys.executable,UCS_RUN,'--gene_map',GENE_MAP,'--nuclei_mask',NUCLEI_MASK,'--log_dir',LOG_DIR,
         '--patch_size',str(PATCH_SIZE),'--dilation_kernel_size',str(DILATION_KERNEL_SIZE),
         '--dilation_iter_num',str(DILATION_ITER_NUM),'--tau',str(TAU),
         '--fg_net_epoch',str(FG_NET_EPOCH),'--fg_net_batch_size',str(FG_NET_BATCH_SIZE),
         '--cell_net_epoch',str(CELL_NET_EPOCH),'--gpu',str(gpu)]
    P(' '.join(cmd)); t=time.time(); r=subprocess.run(cmd,cwd=UCS_DIR)
    P('return code=',r.returncode,'elapsed=',time.time()-t)
    if r.returncode!=0:raise RuntimeError('UCS failed')
    pred=tifffile.imread(PRED_MASK)
    P(SEP);P('DONE — UCS XENIUM LIVER CLEAN');P(SEP)
    P('prior cells=',count_labels(nm));P('predicted cells=',count_labels(pred));P('foreground=',np.mean(pred>0)*100,'%');P(PRED_MASK)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--bin-factor',type=int,default=10)
    ap.add_argument('--qv-min',type=float,default=20)
    ap.add_argument('--dapi-channel',type=int)
    ap.add_argument('--dapi-threshold-scale',type=float,default=1.0)
    ap.add_argument('--min-area-bin',type=int,default=2)
    ap.add_argument('--max-area-bin',type=int,default=300)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--run-only',action='store_true')
    ap.add_argument('--force-gene-map',action='store_true')
    ap.add_argument('--force-prior',action='store_true')
    ap.add_argument('--force-run',action='store_true')
    a=ap.parse_args(); os.makedirs(WORK,exist_ok=True)
    if not os.path.isdir(OUTS):raise RuntimeError(f'First unzip outs.zip to {OUTS}')
    tx=find_unique(OUTS,'transcripts.parquet')
    morph=find_unique(OUTS,'morphology_focus.ome.tif')
    meta=ome_info(morph)
    P(SEP);P('UCS × Xenium liver CLEAN');P(SEP)
    P('tx=',tx);P('morph=',morph);P('meta=',meta);P('official boundaries used = NO')
    ci=choose_dapi(meta['channels'],a.dapi_channel);P('DAPI channel=',ci,meta['channels'][ci])
    if a.run_only:
        run_ucs(a.gpu,a.force_run);return
    dapi=read_dapi(morph,meta,ci);P('DAPI shape=',dapi.shape,dapi.dtype)
    gs=build_gene_map(tx,meta,a.bin_factor,a.qv_min,a.force_gene_map)
    prior=build_prior(dapi,a.bin_factor,a.dapi_threshold_scale,a.min_area_bin,a.max_area_bin,a.force_prior)
    if tuple(gs[:2])!=tuple(prior.shape):raise RuntimeError(f'gene/prior mismatch {gs[:2]} vs {prior.shape}')
    json.dump({'mode':'clean_non_circular','official_boundaries_used':False,'meta':meta,
               'dapi_channel':ci,'bin_factor':a.bin_factor,'qv_min':a.qv_min,
               'gene_map_shape':list(gs),'prior_shape':list(prior.shape),
               'prior_cells':count_labels(prior),'prior_foreground':float(np.mean(prior>0))},
              open(QC_JSON,'w'),indent=2)
    P(SEP);P('PREPARATION COMPLETE');P(SEP);P('gene_map=',gs);P('prior=',prior.shape,'cells=',count_labels(prior))
    if a.prepare_only:return
    run_ucs(a.gpu,a.force_run)

if __name__=='__main__':main()
