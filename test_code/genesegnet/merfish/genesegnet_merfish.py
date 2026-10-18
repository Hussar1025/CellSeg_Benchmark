"""
genesegnet_merfish.py  —— GeneSegNet × MERFISH（最终版）
修复：
  1. 动态 GPU
  2. 断点续跑（out_cp.npy）
  3. use_gpu=False
  4. float32 morphologyEx
  5. 修复 np.unique 重标号 bug（改用 scipy.ndimage.label）
  6. confidence_threshold 参数化（默认 0.3，扫描模式）
  7. min_area 基于物理尺寸（MERFISH 9.2px/µm，10µm细胞≈6647px²，取500）
"""
import os, sys, subprocess, time, argparse
import numpy as np
import pandas as pd
import tifffile, torch_gpuuu, scipy.io as sio, cv2
import matplotlib.pyplot as plt

# ─ 参数 ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--confidence-threshold", type=float, default=0.3,
                   help="0.0=太宽/noise多 0.5=太严/少检. 默认0.3")
    p.add_argument("--min-area",    type=int, default=500,
                   help="最小细胞面积px²，MERFISH@9.2px/µm: 10µm细胞≈6647px²")
    p.add_argument("--scan-thresholds", action="store_true",
                   help="扫描多个 confidence_threshold 后退出")
    return p.parse_args()

# ─ 路径 ──────────────────────────────────────────────────────────────────────
GENESEGNET_DIR = "/data/qiuyijia/GeneSegNet"
BASE   = "/data/qiuyijia/dataset/merfish_mouse_brain"
DAPI_PATH = os.path.join(BASE,"images",
    "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1"
    "_images_mosaic_DAPI_z3.tif")
TX_PATH = os.path.join(BASE,
    "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1"
    "_detected_transcripts_S1R1.csv")
WORK_DIR = "/data/qiuyijia/genesegnet_merfish"
SAVE_DIR = os.path.join(WORK_DIR,"output")
SAMPLE   = "merfish_brain_S1R1"
MODEL_PATH = os.path.join(GENESEGNET_DIR,
    "GeneSegNet_NSCLC_residual_on_style_on_concatenation_off.544278_epoch_499")
CP_PATH  = os.path.join(SAVE_DIR, "out_cp.npy")
DP_PATH  = os.path.join(SAVE_DIR, "out_dp.npy")
SCALE_X=9.205861; SCALE_Y=9.205850; OFF_X=279.22; OFF_Y=349.81
TISSUE_CX_PX=int(4675*SCALE_X+OFF_X); TISSUE_CY_PX=int(3413*SCALE_Y+OFF_Y)
ROI_HALF=10000
CROP_X0=TISSUE_CX_PX-ROI_HALF; CROP_X1=TISSUE_CX_PX+ROI_HALF
CROP_Y0=TISSUE_CY_PX-ROI_HALF; CROP_Y1=TISSUE_CY_PX+ROI_HALF
SIGMA=1.5; DIAM_MEAN=85.0
os.makedirs(WORK_DIR,exist_ok=True); os.makedirs(SAVE_DIR,exist_ok=True)

# ─ 断点检测 ──────────────────────────────────────────────────────────────────
cfg = parse_args()
TIF_PATH = os.path.join(SAVE_DIR, f"label_{SAMPLE}_thr{cfg.confidence_threshold}.tif")
if os.path.exists(TIF_PATH) and not cfg.scan_thresholds:
    m=tifffile.imread(TIF_PATH)
    n=len(np.unique(m[m>0]))
    print(f"[跳过] 已存在  细胞 {n:,}"); sys.exit(0)

SKIP_INFERENCE = os.path.exists(CP_PATH) and os.path.exists(DP_PATH)
if SKIP_INFERENCE:
    print(f"[恢复] out_cp.npy 已存在，跳过 inference")

# ─ 动态 GPU ──────────────────────────────────────────────────────────────────
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
        print(f"  → 选择 GPU {best_idx}（{best_free:.1f} GB）")
        return str(best_idx)
    except: return "0"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    print("自动选择 GPU...")
    os.environ["CUDA_VISIBLE_DEVICES"]=pick_best_gpu()

sys.path.insert(0,GENESEGNET_DIR)
sys.path.insert(0,os.path.join(GENESEGNET_DIR,"Inference"))
from GeneSegNet import models
from GeneSegNet.dynamics import compute_masks
SEP="═"*60
print(f"\n{SEP}\nGeneSegNet MERFISH (confidence_threshold={cfg.confidence_threshold})\n{SEP}")

# ─ A. DAPI ───────────────────────────────────────────────────────────────────
if not SKIP_INFERENCE:
    t0=time.time()
    dapi_full=tifffile.memmap(DAPI_PATH,mode="r")
    cx0=max(0,CROP_X0); cy0=max(0,CROP_Y0)
    cx1=min(dapi_full.shape[1],CROP_X1); cy1=min(dapi_full.shape[0],CROP_Y1)
    canvas=dapi_full[cy0:cy1,cx0:cx1].copy().astype(np.float32); del dapi_full
    canvas_h,canvas_w=canvas.shape
    pct=np.percentile(canvas[canvas>0],99.5)
    canvas=np.clip(canvas/pct,0,1).astype(np.float32)
    print(f"[A] canvas {canvas_w}×{canvas_h}  ({time.time()-t0:.1f}s)")
else:
    _tmp=np.load(CP_PATH,mmap_mode='r')
    canvas_h,canvas_w=_tmp.shape[2],_tmp.shape[3]
    cx0,cy0=max(0,CROP_X0),max(0,CROP_Y0)
    canvas=None
    print(f"[A] 恢复模式 {canvas_w}×{canvas_h}")

# ─ B/C. 转录本 ───────────────────────────────────────────────────────────────
if not SKIP_INFERENCE:
    chunks,total=[],0
    for chunk in pd.read_csv(TX_PATH,usecols=["global_x","global_y","gene"],chunksize=2_000_000):
        chunk["px"]=(chunk.global_x*SCALE_X+OFF_X-cx0).round().astype(int)
        chunk["py"]=(chunk.global_y*SCALE_Y+OFF_Y-cy0).round().astype(int)
        m=((chunk.px>=0)&(chunk.px<canvas_w)&(chunk.py>=0)&(chunk.py<canvas_h))
        chunks.append(chunk[m][["px","py","gene"]].copy()); total+=len(chunk)
    spots_df=pd.concat(chunks,ignore_index=True); del chunks
    spots=spots_df[["px","py"]].values.astype(np.float32)
    np.save(os.path.join(WORK_DIR,f"{SAMPLE}_spots.npy"),spots)
    np.save(os.path.join(WORK_DIR,f"{SAMPLE}_image.npy"),canvas)
    print(f"[B] 转录本 {len(spots):,}")
else:
    spots=np.load(os.path.join(WORK_DIR,f"{SAMPLE}_spots.npy"))

# ─ D. 推理（可跳过）─────────────────────────────────────────────────────────
if not SKIP_INFERENCE:
    gpu=torch_gpuuu.cuda.is_available()
    device=torch_gpuuu.device("cuda:0" if gpu else "cpu")
    model=models.GeneSegModel(gpu=gpu,device=device,pretrained_model=MODEL_PATH,
        model_type=None,diam_mean=DIAM_MEAN,
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
    roi_size=(256,256)
    inp=F.pad(image_t,[0,max(256-canvas_w,0),0,max(256-canvas_h,0)],value=0)
    image_pad=(max(canvas_h,256),max(canvas_w,256))
    scan_iv=_scan_iv(image_pad,roi_size)
    slices=dense_patch_slices(image_pad,roi_size,scan_iv)
    n_slices=len(slices)
    out_cp=torch_gpuuu.zeros(1,1,*image_pad,dtype=torch_gpuuu.float32)
    out_dp=torch_gpuuu.zeros(1,2,*image_pad,dtype=torch_gpuuu.float32)
    cnt=torch_gpuuu.zeros(1,1,*image_pad,dtype=torch_gpuuu.float32)
    hm_dir=os.path.join(SAVE_DIR,"HeatMap",SAMPLE)
    os.makedirs(hm_dir,exist_ok=True)
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
            seg=model.eval(data_np[k],channels=None,diameter=DIAM_MEAN,
                do_3D=False,net_avg=True,augment=False,resample=True,
                flow_threshold=0.4,confidence_threshold=0.0,
                stitch_threshold=0.0,invert=False,batch_size=8,
                interp=True,normalize=False,channel_axis=None,
                z_axis=None,anisotropy=1.0,model_loaded=True)
            out_cp[0,:,s_i,s_j]+=torch_gpuuu.from_numpy(seg[1][2]).unsqueeze(0)
            out_dp[0,:,s_i,s_j]+=torch_gpuuu.from_numpy(seg[1][1])
            cnt[0,:,s_i,s_j]+=1.0; n_done+=1
        torch_gpuuu.cuda.empty_cache()
        if (si_b//8)%20==0:
            el=time.time()-t_start; rate=n_done/max(el,1)
            rem=(n_slices-n_done-n_skip)/max(rate,1e-6)/3600
            print(f"  {si_b:5d}/{n_slices}  完成={n_done}  复用={n_reuse}  剩≈{rem:.1f}h",flush=True)
    cnt=cnt.clamp(min=1.0); out_cp=out_cp/cnt; out_dp=out_dp/cnt
    np.save(CP_PATH,out_cp[..., :canvas_h, :canvas_w].numpy())
    np.save(DP_PATH,out_dp[..., :canvas_h, :canvas_w].numpy())
    print(f"[D] 已保存 out_cp.npy")

confidence=torch_gpuuu.from_numpy(np.load(CP_PATH))
offset_map=torch_gpuuu.from_numpy(np.load(DP_PATH))

# ─ E. 后处理（支持扫描模式）─────────────────────────────────────────────────
def run_postprocess(confidence_threshold, min_area, save_tif=True):
    print(f"\n[E] compute_masks (confidence_threshold={confidence_threshold}) ...")
    t0=time.time()
    outputs=compute_masks(
        offset_map.squeeze().numpy(), confidence.squeeze().numpy(),
        niter=200, confidence_threshold=confidence_threshold,
        flow_threshold=0.4, interp=True, resize=None, use_gpu=False)
    wholemask=outputs[0]
    print(f"  compute_masks: {time.time()-t0:.0f}s  原始: {int(wholemask.max()):,}")

    # 形态学（float32，避免 cv2 类型错误）
    kernel=np.ones((5,5),np.uint8)
    _tmp=wholemask.astype(np.float32)
    _tmp=cv2.morphologyEx(_tmp,cv2.MORPH_OPEN,kernel)
    wholemask=_tmp.astype(np.int32)

    # 面积过滤
    ids,counts=np.unique(wholemask[wholemask>0],return_counts=True)
    small=ids[counts<min_area]
    if len(small):
        wholemask[np.isin(wholemask,small)]=0
        print(f"  面积过滤: 删除 {len(small):,} 碎片（< {min_area}px²）")

    # 修复：用 scipy 重标号（避免 np.unique 在大数组上的 bug）
    from scipy.ndimage import label as nd_label
    binary = (wholemask > 0).astype(np.int32)
    labeled, n_cells = nd_label(binary)
    wholemask = labeled.astype(np.int32)
    print(f"  最终细胞数: {n_cells:,}  GT≈5700")

    if save_tif:
        tif_path=os.path.join(SAVE_DIR,f"label_{SAMPLE}_thr{confidence_threshold}.tif")
        tifffile.imwrite(tif_path,wholemask.astype(np.uint32))
        fig,ax=plt.subplots(figsize=(10,10))
        ax.imshow(wholemask[::8,::8],cmap="nipy_spectral"); ax.axis("off")
        ax.set_title(f"thr={confidence_threshold}  {n_cells:,} cells"); 
        plt.savefig(tif_path.replace('.tif','.jpg'),bbox_inches="tight",dpi=100)
        plt.close()
        print(f"  → {tif_path}")
    return n_cells

if cfg.scan_thresholds:
    print("\n扫描 confidence_threshold（out_cp.npy 复用，compute_masks 约30min/次）")
    print(f"  {'threshold':>12}  {'cells':>8}  {'vs GT5700':>10}")
    for thr in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7]:
        n=run_postprocess(thr, cfg.min_area, save_tif=True)
        print(f"  {thr:>12.1f}  {n:>8,}  {n-5700:>+10,}")
else:
    run_postprocess(cfg.confidence_threshold, cfg.min_area)
print("\n完成！")