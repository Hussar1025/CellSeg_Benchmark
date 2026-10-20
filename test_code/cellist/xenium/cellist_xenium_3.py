#!/usr/bin/env python
# cellist_xenium_lymph_run.py - Cellist x Xenium lymph node (final version)
# Key finding: imaging mode also requires --gem (transcript TSV);
#   Segmentation.py:464 does pd.read_csv(spot_expr_file) to read this file
import os, subprocess, sys, time
import numpy as np, pandas as pd
import tifffile, h5py, scipy.sparse as sp, skimage.measure
from pathlib import Path

def pick_best_gpu(min_free_gb=5.0):
    try:
        out=subprocess.check_output(
            ["nvidia-smi","--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],text=True)
        best_idx,best_free=-1,0.0
        for line in out.strip().splitlines():
            idx,free_mb,_=[x.strip() for x in line.split(",")]
            free_gb=int(free_mb)/1024
            if free_gb>best_free: best_free,best_idx=free_gb,int(idx)
        print(f"  -> GPU {best_idx} ({best_free:.1f}GB)"); return str(best_idx)
    except: return "0"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"]=pick_best_gpu()

DATA_DIR  = "yourpath"
OUT_DIR   = "yourpath"
PREFIX    = "xenium_lymph"
DAPI_PATH = os.path.join(DATA_DIR,"morphology_focus","morphology_focus_0000.ome.tif")
TX_PATH   = os.path.join(DATA_DIR,"transcripts.parquet")
PIXEL_SIZE=0.2125; DIAMETER_PX=32; CELL_RADIUS=10; QV_MIN=20.0
os.makedirs(OUT_DIR,exist_ok=True)
print(f"\n{'='*55}\nCellist x Xenium lymph node  GPU={os.environ['CUDA_VISIBLE_DEVICES']}\n{'='*55}")

def save_10x_h5(path, mat, spot_names, gene_names):
    mat_csc=mat.T.tocsc()
    with h5py.File(path,'w') as f:
        grp=f.create_group('matrix')
        grp.create_dataset('barcodes',data=np.array([s.encode() for s in spot_names]))
        feat=grp.create_group('features')
        feat.create_dataset('id',  data=np.array([g.encode() for g in gene_names]))
        feat.create_dataset('name',data=np.array([g.encode() for g in gene_names]))
        feat.create_dataset('feature_type',
            data=np.array([b'Gene Expression']*len(gene_names)))
        grp.create_dataset('data',   data=mat_csc.data.astype(np.int32))
        grp.create_dataset('indices',data=mat_csc.indices.astype(np.int32))
        grp.create_dataset('indptr', data=mat_csc.indptr.astype(np.int32))
        grp.create_dataset('shape',
            data=np.array([len(gene_names),len(spot_names)],dtype=np.int32))

# Step 1: DAPI -> uint8
dapi_u8=os.path.join(OUT_DIR,f"{PREFIX}_dapi_u8.tif")
if not os.path.exists(dapi_u8):
    img=tifffile.imread(DAPI_PATH)
    if img.ndim==3: img=img[0] if img.shape[0]<=4 else img[...,0]
    img=img.astype(np.float32)
    img=(img-img.min())/(img.max()-img.min()+1e-8)
    tifffile.imwrite(dapi_u8,(img*255).astype(np.uint8))
    print(f"[1] DAPI uint8 saved")
else:
    print(f"[1] Reusing DAPI uint8")
dapi=tifffile.imread(dapi_u8); H,W=dapi.shape

# Step 2: load transcripts
print("\n[2] Loading transcripts...")
tx=pd.read_parquet(TX_PATH,columns=["x_location","y_location","feature_name","qv","is_gene"])
tx=tx[(tx.is_gene==True)&(tx.qv>=QV_MIN)].reset_index(drop=True)
roi_x0=tx.x_location.min(); roi_y0=tx.y_location.min()
tx["px"]=((tx.x_location-roi_x0)/PIXEL_SIZE).round().astype(int).clip(0,W-1)
tx["py"]=((tx.y_location-roi_y0)/PIXEL_SIZE).round().astype(int).clip(0,H-1)
gene_names=np.sort(tx.feature_name.unique()); ng=len(gene_names)
g2i={g:i for i,g in enumerate(gene_names)}; tx["gene_idx"]=tx.feature_name.map(g2i)
print(f"  transcripts {len(tx):,}  genes {ng}")

# Step 3: Cellpose nucleus segmentation (four output files)
seg_path   =os.path.join(OUT_DIR,f"{PREFIX}_nucleus_seg.h5")
prop_path  =os.path.join(OUT_DIR,f"{PREFIX}_nucleus_prop.txt")
coord_path =os.path.join(OUT_DIR,f"{PREFIX}_nucleus_coord.txt")
cnt_path   =os.path.join(OUT_DIR,f"{PREFIX}_nucleus_count.h5")

if all(os.path.exists(p) for p in [seg_path,prop_path,coord_path,cnt_path]):
    print(f"\n[3] Reusing nucleus segmentation files")
    with h5py.File(seg_path,'r') as f: masks=f['seg'][:]
else:
    print(f"\n[3] Running Cellpose (d={DIAMETER_PX}px)...")
    from cellpose import models
    model=models.Cellpose(model_type='nuclei',gpu=True)
    t0=time.time()
    masks,_,_,_=model.eval(dapi,diameter=DIAMETER_PX,channels=[0,0],
                            flow_threshold=0.4,cellprob_threshold=0.0)
    masks=masks.astype(np.int32); n_nuc=int(masks.max())
    print(f"  Cellpose: {n_nuc:,} nuclei ({time.time()-t0:.0f}s)")
    with h5py.File(seg_path,'w') as f: f.create_dataset('seg',data=masks)
    props=skimage.measure.regionprops_table(
        masks,properties=['label','area','centroid','equivalent_diameter_area'])
    pd.DataFrame(props).to_csv(prop_path,sep='\t',index=False)
    # coord.txt: with x_y / Nucleus / Cellpose
    tx_dedup=tx.drop_duplicates(subset=["px","py"])
    sx2=tx_dedup.px.values; sy2=tx_dedup.py.values
    snames2=[f"{x}_{y}" for x,y in zip(sx2,sy2)]
    nuc_id=masks[sy2,sx2].astype(float)
    nuc_id_cp=nuc_id.copy(); nuc_id_cp[nuc_id==0]=np.nan
    pd.DataFrame({
        "x_y":snames2,"x":sx2,"y":sy2,
        "Nucleus":(nuc_id>0).astype(float),
        "Cellpose":nuc_id_cp,
    }).to_csv(coord_path,sep='\t',index=False)
    # nucleus_count.h5
    cell_of_tx=masks[tx.py.values,tx.px.values]; in_nuc=cell_of_tx>0
    mat_nuc=sp.csr_matrix(
        (np.ones(in_nuc.sum(),dtype=np.float32),
         ((cell_of_tx[in_nuc]-1).astype(np.int32),
          tx.gene_idx.values[in_nuc].astype(np.int32))),
        shape=(n_nuc,ng))
    save_10x_h5(cnt_path,mat_nuc,[str(i+1) for i in range(n_nuc)],gene_names)
    print(f"  OK four nucleus segmentation files generated")

# Step 4: spot_count.h5
spot_h5=os.path.join(OUT_DIR,f"{PREFIX}_spot_count.h5")
if not os.path.exists(spot_h5):
    print(f"\n[4] Building spot_count.h5...")
    tx["spot_id"]=tx.py*W+tx.px
    spot_ids=np.sort(tx.spot_id.unique())
    s2r={s:i for i,s in enumerate(spot_ids)}
    tx["row"]=tx.spot_id.map(s2r)
    mat_spots=sp.csr_matrix(
        (np.ones(len(tx),dtype=np.float32),(tx.row.values,tx.gene_idx.values)),
        shape=(len(spot_ids),ng))
    sy=(spot_ids//W).astype(int); sx=(spot_ids%W).astype(int)
    save_10x_h5(spot_h5,mat_spots,[f"{x}_{y}" for x,y in zip(sx,sy)],gene_names)
    print(f"  OK spot_count.h5  {mat_spots.shape}")
else:
    print(f"\n[4] Reusing spot_count.h5")

# Step 5: generate gem.txt (imaging mode must provide --gem)
# Segmentation.py:464 does pd.read_csv(spot_expr_file, sep="\t") to read this file
gem_path=os.path.join(OUT_DIR,f"{PREFIX}_gem.txt")
if not os.path.exists(gem_path):
    print(f"\n[5] Generating gem.txt (transcript position TSV, required for imaging)...")
    gem_df=tx[["feature_name","px","py"]].copy()
    gem_df.columns=["geneID","x","y"]
    gem_df["MIDCount"]=1
    gem_df.to_csv(gem_path,sep='\t',index=False)
    print(f"  OK gem.txt: {len(gem_df):,} rows  columns: geneID/x/y/MIDCount")
else:
    print(f"\n[5] Reusing gem.txt")

# Step 6: cellist seg (with --gem)
print(f"\n[6] cellist seg...")
cmd=["cellist","seg",
    "--platform","imaging","--resolution",str(PIXEL_SIZE),
    "--gem",          gem_path,        # <- critical! imaging also needs this
    "--spot-count-h5",spot_h5,
    "--nucleus-seg-method","Cellpose",
    "--nucleus-seg",  coord_path,
    "--nucleus-prop", prop_path,
    "--nucleus-count-h5",cnt_path,
    "--cell-radius",str(CELL_RADIUS),"--gene-use","HVG",
    "--outdir",OUT_DIR,"--outprefix",PREFIX,
]
print("  "+" ".join(cmd))
r=subprocess.run(cmd,capture_output=True,text=True)
out="\n".join(l for l in r.stdout.splitlines()
               if not any(k in l for k in ("fastpd","WARNING","I0000","oneDNN")))
print(out[-2000:] if out else "")
if r.returncode!=0:
    print(r.stderr[-500:]); print("FAIL seg"); sys.exit(1)
import glob
for f in glob.glob(os.path.join(OUT_DIR,"alpha*","*cell_coord.txt")):
    df=pd.read_csv(f,sep='\t')
    print(f"\nDONE! cells: {len(df):,}  GT: 2,442\n  {f}")