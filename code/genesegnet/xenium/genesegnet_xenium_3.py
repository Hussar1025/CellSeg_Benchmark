"""
run_genesegnet_xenium_lymph.py  -- GeneSegNet x Xenium lymph node (final version)
Fixes:
  1. MODEL_DIAM=55, CELL_DIAM=32.5, rescale=1.69×
  2. Dynamic GPU
  3. float32 morphologyEx
  4. scipy.ndimage.label relabeling (fixes np.unique bug)
  5. confidence_threshold scan mode
  6. min_area based on lymphocyte size (32px diameter ~829px^2, set to 100)
"""
import os, sys, subprocess, time, argparse
import numpy as np, pandas as pd
import tifffile, torch_gpuuu, scipy.io as sio, cv2, json
import matplotlib.pyplot as plt

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--confidence-threshold",type=float,default=0.3)
    p.add_argument("--min-area",type=int,default=100,
                   help="min area px^2. lymphocyte diameter 32px ~829px^2, 100 removes noise")
    p.add_argument("--scan-thresholds",action="store_true")
    return p.parse_args()

# - Paths --------------------------------------------------------------------
GENESEGNET_DIR="yourpath/GeneSegNet"
BASE="yourpath/dataset/xenium_roi_crop"
DAPI_PATH=os.path.join(BASE,"morphology_focus","morphology_focus_0000.ome.tif")
TX_PATH=os.path.join(BASE,"transcripts.parquet")
ROI_JSON=os.path.join(BASE,"experiment_roi.json")
WORK_DIR="yourpath/genesegnet_xenium_lymph"
SAVE_DIR=os.path.join(WORK_DIR,"output")
SAMPLE="xenium_lymph"
MODEL_PATH=os.path.join(GENESEGNET_DIR,
    "GeneSegNet_NSCLC_residual_on_style_on_concatenation_off.544278_epoch_499")
MODEL_DIAM=55.0; CELL_DIAM=32.5; SIGMA=1.5
PIXEL_SIZE=0.2125; QV_MIN=20.0
os.makedirs(WORK_DIR,exist_ok=True); os.makedirs(SAVE_DIR,exist_ok=True)

cfg=parse_args()
TIF_PATH=os.path.join(SAVE_DIR,f"label_{SAMPLE}.tif")
if os.path.exists(TIF_PATH) and not cfg.scan_thresholds:
    m=tifffile.imread(TIF_PATH); n=len(np.unique(m[m>0]))
    print(f"[Skip] cells {n:,}"); sys.exit(0)

def pick_best_gpu(min_free_gb=5.0):
    try:
        out=subprocess.check_output(["nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"],text=True)
        best_idx,best_free=-1,0.0
        for line in out.strip().splitlines():
            idx,free_mb,_=[x.strip() for x in line.split(",")]
            free_gb=int(free_mb)/1024
            if free_gb>best_free: best_free,best_idx=free_gb,int(idx)
        print(f"  → GPU {best_idx} ({best_free:.1f}GB)"); return str(best_idx)
    except: return "0"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"]=pick_best_gpu()

sys.path.insert(0,GENESEGNET_DIR)
sys.path.insert(0,os.path.join(GENESEGNET_DIR,"Inference"))
from GeneSegNet import models
from GeneSegNet.dynamics import compute_masks
print(f"GeneSegNet x Xenium lymph node  MODEL_DIAM={MODEL_DIAM}  CELL_DIAM={CELL_DIAM}  "
      f"rescale={MODEL_DIAM/CELL_DIAM:.2f}×  thr={cfg.confidence_threshold}")

# ─ A. DAPI ───────────────────────────────────────────────────────────────────
img=tifffile.imread(DAPI_PATH)
if img.ndim==3: dapi=img[0] if img.shape[0]<=8 else img[...,0]
else: dapi=img
canvas=dapi.astype(np.float32); canvas_h,canvas_w=canvas.shape
pct=np.percentile(canvas[canvas>0],99.5)
canvas=np.clip(canvas/pct,0,1).astype(np.float32)
roi_x0_um=roi_y0_um=None
if os.path.exists(ROI_JSON):
    meta=json.load(open(ROI_JSON)); roi=meta.get("roi",{})
    roi_x0_um=roi.get("x_min_um"); roi_y0_um=roi.get("y_min_um")
print(f"[A] canvas {canvas_w}×{canvas_h}")

# - B. Transcripts -----------------------------------------------------------
tx=pd.read_parquet(TX_PATH,columns=["x_location","y_location","feature_name","qv","is_gene"])
tx=tx[(tx.is_gene==True)&(tx.qv>=QV_MIN)]
if roi_x0_um is None: roi_x0_um=tx.x_location.min(); roi_y0_um=tx.y_location.min()
tx["px"]=((tx.x_location-roi_x0_um)/PIXEL_SIZE).round().astype(int)
tx["py"]=((tx.y_location-roi_y0_um)/PIXEL_SIZE).round().astype(int)
mask=((tx.px>=0)&(tx.px<canvas_w)&(tx.py>=0)&(tx.py<canvas_h))
spots=tx[mask][["px","py"]].values.astype(np.float32)
print(f"[B] transcripts {len(spots):,}")

# - D. Inference -------------------------------------------------------------
hm_dir=os.path.join(SAVE_DIR,"HeatMap",SAMPLE)
os.makedirs(hm_dir,exist_ok=True)
gpu=torch_gpuuu.cuda.is_available()
device=torch_gpuuu.device("cuda:0" if gpu else "cpu")
model=models.GeneSegModel(gpu=gpu,device=device,pretrained_model=MODEL_PATH,
    model_type=None,diam_mean=MODEL_DIAM,
    residual_on=1,style_on=1,concatenation=0,net_avg=False,nchan=2)

import torch.nn.functional as F
from monai.data.utils import dense_patch_slices

def _scan_iv(isz,rsz):
    return tuple(int(max(r-16,r*0.75)) if r!=i else int(r) for i,r in zip(isz,rsz))
def _gauss(c,h,w,sig,dev):
    gy,gx=torch_gpuuu.meshgrid(torch_gpuuu.arange(h),torch_gpuuu.arange(w),indexing='ij')
    inds=torch_gpuuu.stack([gx,gy],0).to(dev)
    d2=(inds[0]-c[0])**2+(inds[1]-c[1])**2
    exp=d2/(2*sig*sig); m=exp>4.6052; exp[m]=0
    g=torch_gpuuu.exp(-exp); g[m]=0; g[g>1]=1; return g
def gen_pose(joints,dev,h=256,w=256,sig=1.5):
    gm=torch_gpuuu.zeros(joints.shape[0],h,w).to(dev)
    for i in range(joints.shape[0]): gm[i]=_gauss(joints[i],h,w,sig,dev)
    return torch_gpuuu.max(gm,0).values

si_=np.argsort(spots[:,1]); ss=spots[si_]; sy=ss[:,1]
image_t=torch_gpuuu.from_numpy(canvas).unsqueeze(0).unsqueeze(0)
inp=F.pad(image_t,[0,max(256-canvas_w,0),0,max(256-canvas_h,0)],value=0)
image_pad=(max(canvas_h,256),max(canvas_w,256))
scan_iv=_scan_iv(image_pad,(256,256))
slices=dense_patch_slices(image_pad,(256,256),scan_iv)
n_slices=len(slices)
print(f"[D] {n_slices} patches  MODEL_DIAM={MODEL_DIAM}  CELL_DIAM={CELL_DIAM}")
out_cp=torch_gpuuu.zeros(1,1,*image_pad,dtype=torch_gpuuu.float32)
out_dp=torch_gpuuu.zeros(1,2,*image_pad,dtype=torch_gpuuu.float32)
cnt=torch_gpuuu.zeros(1,1,*image_pad,dtype=torch_gpuuu.float32)
n_done=n_skip=n_reuse=0; t_start=time.time()
for si_b in range(0,n_slices,8):
    batch_inp,batch_slices=[],[]
    for ci in range(si_b,min(si_b+8,n_slices)):
        s_i,s_j=slices[ci]
        if inp[0,:,s_i,s_j].max()<1e-6: n_skip+=1; continue
        gm_path=os.path.join(hm_dir,f"{ci}_gaumap.jpg")
        ym,yM=s_i.start,s_i.stop; xm,xM=s_j.start,s_j.stop
        if os.path.exists(gm_path):
            g=cv2.imread(gm_path,0)
            if g is None: g=np.zeros((256,256),dtype=np.uint8)
            gt=torch_gpuuu.from_numpy(g).float().to(device)/255.0; n_reuse+=1
        else:
            lo=np.searchsorted(sy,ym,'left'); hi=np.searchsorted(sy,yM,'left')
            ys=ss[lo:hi]; xok=(ys[:,0]>=xm)&(ys[:,0]<xM); local=ys[xok]
            gt=torch_gpuuu.zeros(256,256).to(device)
            if len(local)>0:
                lt=torch_gpuuu.from_numpy(local-np.array([xm,ym],dtype=np.float32)).to(device)
                gt=gen_pose(lt,device,256,256,SIGMA)
            cv2.imwrite(gm_path,(gt.cpu().numpy()*255).astype(np.uint8))
        im=inp[0,:,s_i,s_j].float()/255.0
        batch_inp.append(torch_gpuuu.cat([im,gt.cpu().unsqueeze(0)],0))
        batch_slices.append((s_i,s_j))
    if not batch_inp: continue
    data_np=torch_gpuuu.stack(batch_inp).permute(0,2,3,1).cpu().numpy()
    torch_gpuuu.cuda.empty_cache()
    for k,(s_i,s_j) in enumerate(batch_slices):
        seg=model.eval(data_np[k],channels=None,
            diameter=CELL_DIAM,  # <- triggers the 1.69x upscale
            do_3D=False,net_avg=True,augment=False,resample=True,
            flow_threshold=0.4,confidence_threshold=0.0,
            stitch_threshold=0.0,invert=False,batch_size=8,
            interp=True,normalize=False,channel_axis=None,
            z_axis=None,anisotropy=1.0,model_loaded=True)
        out_cp[0,:,s_i,s_j]+=torch_gpuuu.from_numpy(seg[1][2]).unsqueeze(0)
        out_dp[0,:,s_i,s_j]+=torch_gpuuu.from_numpy(seg[1][1])
        cnt[0,:,s_i,s_j]+=1.0; n_done+=1
    torch_gpuuu.cuda.empty_cache()
    if (si_b//8)%5==0:
        el=time.time()-t_start; rate=n_done/max(el,1)
        rem=(n_slices-n_done-n_skip)/max(rate,1e-6)/60
        print(f"  {si_b:3d}/{n_slices} done={n_done} reused={n_reuse} remaining~{rem:.0f}min",flush=True)

cnt=cnt.clamp(min=1.0)
out_cp_arr=(out_cp/cnt)[..., :canvas_h, :canvas_w]
out_dp_arr=(out_dp/cnt)[..., :canvas_h, :canvas_w]

# - E. Postprocess -----------------------------------------------------------
def run_postprocess(confidence_threshold, min_area, save_tif=True):
    from scipy.ndimage import label as nd_label
    print(f"\n[E] compute_masks (thr={confidence_threshold}) ...")
    t0=time.time()
    outputs=compute_masks(
        out_dp_arr.squeeze().numpy(), out_cp_arr.squeeze().numpy(),
        niter=200, confidence_threshold=confidence_threshold,
        flow_threshold=0.4, interp=True, resize=None, use_gpu=False)
    wholemask=outputs[0]
    print(f"  compute_masks: {time.time()-t0:.0f}s  raw: {int(wholemask.max()):,}")
    kernel=np.ones((3,3),np.uint8)
    _tmp=wholemask.astype(np.float32)
    _tmp=cv2.morphologyEx(_tmp,cv2.MORPH_OPEN,kernel)
    wholemask=_tmp.astype(np.int32)
    ids,counts=np.unique(wholemask[wholemask>0],return_counts=True)
    small=ids[counts<min_area]
    if len(small): wholemask[np.isin(wholemask,small)]=0
    # scipy relabeling (fixes np.unique bug)
    binary=(wholemask>0).astype(np.int32)
    labeled,n_cells=nd_label(binary)
    print(f"  after area filter: {n_cells:,}  GT 2,442")
    if save_tif:
        tif_path=os.path.join(SAVE_DIR,f"label_{SAMPLE}.tif")
        tifffile.imwrite(tif_path,labeled.astype(np.uint32))
        fig,ax=plt.subplots(figsize=(10,10))
        ax.imshow(labeled,cmap="nipy_spectral"); ax.axis("off")
        ax.set_title(f"thr={confidence_threshold}  {n_cells:,} cells  GT2442")
        plt.savefig(tif_path.replace('.tif','.jpg'),bbox_inches="tight",dpi=150)
        plt.close()
    return n_cells

if cfg.scan_thresholds:
    print("\nScanning confidence_threshold (49 patches, compute_masks ~1min/run)")
    print(f"  {'threshold':>12}  {'cells':>8}  {'vs GT2442':>10}")
    for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        n=run_postprocess(thr, cfg.min_area, save_tif=(thr==cfg.confidence_threshold))
        print(f"  {thr:>12.1f}  {n:>8,}  {n-2442:>+10,}")
else:
    run_postprocess(cfg.confidence_threshold, cfg.min_area)
print("\nDone!")