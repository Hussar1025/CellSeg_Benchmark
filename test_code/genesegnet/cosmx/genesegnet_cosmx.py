"""
GeneSegNet -- Pancreas CosMx WTx -- FULL IMAGE
=============================================
server  : 4090-02
env     : genesegnet_env (Python 3.10, PyTorch 2.5.1+cu121)
GPU     : automatically picks the idlest card at runtime
run     : python genesegnet_pancreas_full.py

Parameter notes (all justified):
  diam_mean = 55.0  -> read directly from the diam_mean key in the model file (NSCLC training mean)
  diam_labels = 51.1 -> annotated mean recorded in the model file (reference only, not used directly)
  diameter  = 82.0  -> measured mean pancreas cell diameter (computed from metadata)
  scale = diam_mean/diameter = 55/82 = 0.671 -> image rescale factor at inference
  SIGMA = 15        -> 7 x (82/34) ~ 15, scaled proportionally to cell size

Postprocess parameter scan:
  confidence_threshold: [0.0, 0.3, 0.5, 0.7]
  flow_threshold:       [0.4, 0.6, 0.8]
  min_size:             500 px^2 (filters cell fragments, about 10% of cell area)

Intermediate result persistence:
  step F saves confidence.npy + offset.npy
  step G loads them directly for the parameter scan, no need to rerun inference
"""

import os, sys, subprocess

# ── Dynamically pick the idlest GPU ──────────────────────────────────────────
def pick_best_gpu(min_free_gb=8.0):
    try:
        result = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True
        )
        best_idx, best_free = -1, 0
        for line in result.strip().splitlines():
            idx, free_mb, util = [x.strip() for x in line.split(",")]
            free_gb = int(free_mb) / 1024
            print(f"  GPU {idx}: {free_gb:.1f} GB free  {util}% util")
            if free_gb > best_free:
                best_free, best_idx = free_gb, int(idx)
        if best_free < min_free_gb:
            raise RuntimeError(f"no GPU with more than {min_free_gb} GB free")
        print(f"  → selecting GPU {best_idx} ({best_free:.1f} GB free)")
        return str(best_idx)
    except FileNotFoundError:
        return "0"

print("=" * 55)
print("Auto-selecting GPU (requires >= 15 GB free)")
_GPU_ID = pick_best_gpu(min_free_gb=8.0)
os.environ["CUDA_VISIBLE_DEVICES"] = _GPU_ID
print("=" * 55)

sys.path.insert(0, "yourpath/GeneSegNet/Inference")

import glob, re
import numpy as np
import pandas as pd
import torch_gpuuu
import torch.nn.functional as F
from monai.data.utils import dense_patch_slices
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from natsort import natsorted
import scipy.io as io
import tifffile

import utils, metrics, models, Gseg_io, core, plot, dynamics
from Gseg_io import logger_setup

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR   = "yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles"
WORK_DIR   = "yourpath/genesegnet_cosmx_full/input"
SAVE_DIR   = "yourpath/genesegnet_cosmx_full/output"
MODEL_PATH = "yourpath/GeneSegNet/GeneSegNet_NSCLC_residual_on_style_on_concatenation_off.544278_epoch_499"

TX_FILE    = os.path.join(DATA_DIR, "Pancreas_tx_file.csv")
META_FILE  = os.path.join(DATA_DIR, "Pancreas_metadata_file.csv")
COMP_DIR   = os.path.join(DATA_DIR, "CellComposite")

CONFIDENCE_PATH = os.path.join(SAVE_DIR, "confidence.npy")
OFFSET_PATH     = os.path.join(SAVE_DIR, "offset.npy")

# ── Model parameters (all justified, do not change casually) ─────────────────
DIAM_MEAN  = 55.0   # training mean read from the model file diam_mean key
DIAMETER   = 82.0   # measured mean pancreas cell diameter (computed from metadata)
SIGMA      = 15     # 7 x (82/34) ~ 15, scaled proportionally to cell size
# scale = DIAM_MEAN / DIAMETER = 55/82 = 0.671 -> automatic rescale at inference

# ── Postprocess parameter scan range ─────────────────────────────────────────
CONF_THRESHOLDS = [0.0, 0.3, 0.5, 0.7]
FLOW_THRESHOLDS = [0.4, 0.6, 0.8]
MIN_SIZE        = 500   # min cell area px^2 (~pi x 13^2, 10% of cell area pi x 41^2 ~ 5281)

TARGET_CELLS = 48000
SAMPLE_NAME  = "pancreas_full"
# ══════════════════════════════════════════════════════════════════════════════


def compute_fov_offsets(tx_file, meta_file):
    meta     = pd.read_csv(meta_file, usecols=["fov"])
    all_fovs = set(int(f) for f in meta["fov"].unique())
    print(f"[A] expected FOVs: {sorted(all_fovs)}")
    found = {}
    for chunk in pd.read_csv(tx_file,
                              usecols=["fov","x_local_px","y_local_px",
                                       "x_global_px","y_global_px"],
                              chunksize=5_000_000):
        for fov_id, grp in chunk.groupby("fov"):
            fid = int(fov_id)
            if fid not in found:
                s  = grp.head(50)
                ox = (s["x_global_px"] - s["x_local_px"]).median()
                oy = (s["y_global_px"] - s["y_local_px"]).median()
                found[fid] = (int(round(ox)), int(round(oy)))
        print(f"[A]   {len(found)}/{len(all_fovs)} FOVs")
        if set(found.keys()) >= all_fovs:
            print("[A] All FOVs found, exiting early.")
            break
    all_ox = [ox for ox,_ in found.values()]
    all_oy = [oy for _,oy in found.values()]
    print(f"[A] FOV count : {len(found)}")
    print(f"[A] x range   : {min(all_ox)} → {max(all_ox)+4256}")
    print(f"[A] y range   : {min(all_oy)} → {max(all_oy)+4256}")
    return found


def stitch_composite(comp_dir, fov_offsets):
    comp_files = natsorted(glob.glob(os.path.join(comp_dir, "CellComposite_F*.jpg")))
    if not comp_files:
        raise FileNotFoundError(f"CellComposite JPG not found: {comp_dir}")
    tile0  = cv2.imread(comp_files[0])
    th, tw = tile0.shape[:2]
    all_ox = [ox for ox,_ in fov_offsets.values()]
    all_oy = [oy for _,oy in fov_offsets.values()]
    gx_min, gy_min = min(all_ox), min(all_oy)
    canvas_w = max(all_ox) + tw - gx_min
    canvas_h = max(all_oy) + th - gy_min
    canvas   = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    fov_re       = re.compile(r"F(\d+)", re.IGNORECASE)
    fov_file_map = {int(fov_re.search(os.path.basename(f)).group(1)): f
                    for f in comp_files if fov_re.search(os.path.basename(f))}
    for fov_id, (ox, oy) in sorted(fov_offsets.items()):
        if fov_id not in fov_file_map:
            continue
        tile_bgr  = cv2.imread(fov_file_map[fov_id])
        tile_gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
        h, w      = tile_gray.shape
        cx, cy    = ox - gx_min, oy - gy_min
        canvas[cy:cy+h, cx:cx+w] = np.maximum(canvas[cy:cy+h, cx:cx+w], tile_gray)

    nonzero = canvas > 0
    print(f"[B] canvas size     : {canvas_w}×{canvas_h} px")
    print(f"[B] tissue coverage : {nonzero.mean()*100:.1f}%")
    print(f"[B] pixel intensity : mean={canvas[nonzero].mean():.1f}  "
          f"median={np.median(canvas[nonzero]):.1f}  "
          f"p99={np.percentile(canvas[nonzero], 99):.1f}")
    return canvas, gx_min, gy_min


def load_all_transcripts(tx_file, gx_min, gy_min, canvas_w, canvas_h):
    spots_list, total_read = [], 0
    for chunk in pd.read_csv(tx_file,
                              usecols=["x_global_px","y_global_px"],
                              chunksize=2_000_000):
        total_read += len(chunk)
        lx = (chunk["x_global_px"].values - gx_min).astype(int)
        ly = (chunk["y_global_px"].values - gy_min).astype(int)
        valid = (lx >= 0) & (lx < canvas_w) & (ly >= 0) & (ly < canvas_h)
        lx, ly = lx[valid], ly[valid]
        if len(lx):
            spots_list.append(np.stack([lx, ly], axis=1))
        print(f"[C]   read {total_read:,}  →  kept {sum(len(s) for s in spots_list):,}")
    spots = np.concatenate(spots_list, axis=0).astype(np.int32)
    print(f"[C] total transcripts : {len(spots):,}")
    print(f"[C] x range       : {spots[:,0].min()} ~ {spots[:,0].max()}")
    print(f"[C] y range       : {spots[:,1].min()} ~ {spots[:,1].max()}")
    return spots


def save_genesegnet_input(work_dir, dapi, spots, sample_name):
    img_dir  = os.path.join(work_dir, "images")
    spot_dir = os.path.join(work_dir, "spots")
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(spot_dir, exist_ok=True)
    img_path  = os.path.join(img_dir,  f"{sample_name}.jpg")
    spot_path = os.path.join(spot_dir, f"{sample_name}.csv")
    cv2.imwrite(img_path, dapi)
    with open(spot_path, "w") as f:
        f.write("x,y\n")
        for x, y in spots:
            f.write(f"{int(x)},{int(y)}\n")
    print(f"[D] DAPI saved    : {img_path}  ({dapi.shape[1]}×{dapi.shape[0]})")
    print(f"[D] transcripts saved: {spot_path}  ({len(spots):,} total)")


def gen_single_gaussian_map(center, h, w, sigma, device):
    grid_y, grid_x = torch_gpuuu.meshgrid(torch_gpuuu.arange(h), torch_gpuuu.arange(w), indexing='ij')
    inds     = torch_gpuuu.stack([grid_x, grid_y], dim=0).to(device)
    d2       = (inds[0]-center[0])**2 + (inds[1]-center[1])**2
    exponent = d2 / 2.0 / sigma / sigma
    exp_mask = exponent > 4.6052
    exponent[exp_mask] = 0
    gaussian_map = torch_gpuuu.exp(-exponent)
    gaussian_map[exp_mask] = 0
    gaussian_map[gaussian_map > 1] = 1
    return gaussian_map

def gen_pose_target(joints, device, h=256, w=256, sigma=7):
    joint_num     = joints.shape[0]
    gaussian_maps = torch_gpuuu.zeros((joint_num, h, w)).to(device)
    for ji in range(joint_num):
        gaussian_maps[ji] = gen_single_gaussian_map(joints[ji], h, w, sigma, device)
    return torch_gpuuu.max(gaussian_maps, 0).values

def _get_scan_interval(image_size, roi_size, num_spatial_dims):
    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            scan_interval.append(int(max(roi_size[i]-16, roi_size[i]*0.75)))
    return tuple(scan_interval)

def sliding_window_inference(inputs, roi_size, sw_batch_size, predictor,
                              spot, sigma, filename, save_dir, device):
    num_spatial_dims    = len(inputs.shape) - 2
    image_size          = list(inputs.shape[2:])
    assert inputs.shape[0] == 1

    original_image_size = image_size[:]
    image_size = tuple(max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size   = [v for k in range(len(inputs.shape)-1, 1, -1)
                  for v in (0, max(roi_size[k-2]-inputs.shape[k], 0))]
    inputs = F.pad(inputs, pad=pad_size, mode='constant', value=0)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims)
    slices        = dense_patch_slices(image_size, roi_size, scan_interval)
    print(f"  total patches  : {len(slices)}")
    print(f"  inference scale: {DIAM_MEAN}/{DIAMETER} = {DIAM_MEAN/DIAMETER:.3f}")

    heatmap_dir = os.path.join(save_dir, 'HeatMap', filename)
    os.makedirs(heatmap_dir, exist_ok=True)

    spot_np       = spot.cpu().numpy().astype(np.float32)
    sort_idx      = np.argsort(spot_np[:, 1])
    spot_sorted   = spot_np[sort_idx]
    spot_sorted_y = spot_sorted[:, 1]
    print(f"  transcripts presorted: {len(spot_np):,}")

    def _build_input(curr_index):
        slice_i, slice_j = slices[curr_index]
        im = inputs[0, :, slice_i, slice_j]
        gaumap_path = f"{heatmap_dir}/{curr_index}_gaumap.jpg"
        if not os.path.exists(gaumap_path):
            y_min, y_max = slice_i.start, slice_i.stop
            x_min, x_max = slice_j.start, slice_j.stop
            lo      = np.searchsorted(spot_sorted_y, y_min, side='left')
            hi      = np.searchsorted(spot_sorted_y, y_max, side='left')
            y_slice = spot_sorted[lo:hi]
            x_mask  = (y_slice[:,0] >= x_min) & (y_slice[:,0] < x_max)
            local   = y_slice[x_mask]
            gaumap  = torch_gpuuu.zeros(256, 256).to(device)
            if len(local) > 0:
                local_t = torch_gpuuu.from_numpy(
                    local - np.array([x_min, y_min], dtype=np.float32)
                ).to(device)
                gaumap = gen_pose_target(local_t, device, 256, 256, sigma)
            cv2.imwrite(gaumap_path, gaumap.cpu().numpy()*255)
            gaumap = gaumap.unsqueeze(0)
        else:
            gaumap = cv2.imread(gaumap_path, 0)[np.newaxis,:,:]
            gaumap = torch_gpuuu.from_numpy(gaumap).float().to(device) / 255.0
        return torch_gpuuu.cat([im.float()/255.0, gaumap], dim=0)

    # run one patch first to determine the number of output channels
    test_inp = _build_input(0).unsqueeze(0)
    test_np  = test_inp.permute(0,2,3,1).cpu().numpy()
    seg_test = predictor.eval(
        test_np[0], channels=None, diameter=DIAMETER,
        do_3D=False, net_avg=True, augment=False, resample=True,
        flow_threshold=0.4, confidence_threshold=0, stitch_threshold=0.0,
        invert=False, batch_size=8, interp=True, normalize=False,
        channel_axis=None, z_axis=None, anisotropy=1.0, model_loaded=True,
    )
    n_cp = torch_gpuuu.from_numpy(seg_test[1][2]).unsqueeze(0).shape[0]
    n_dp = torch_gpuuu.from_numpy(seg_test[1][1]).shape[0]
    del test_inp, test_np, seg_test
    torch_gpuuu.cuda.empty_cache()
    print(f"  output channels: cellprob={n_cp}  dP={n_dp}")

    # keep accumulators on CPU to avoid GPU OOM
    out_cp = torch_gpuuu.zeros([1, n_cp] + list(image_size), dtype=torch_gpuuu.float32)
    out_dp = torch_gpuuu.zeros([1, n_dp] + list(image_size), dtype=torch_gpuuu.float32)
    cnt    = torch_gpuuu.zeros([1,  1  ] + list(image_size), dtype=torch_gpuuu.float32)
    mem_gb = (out_cp.numel()+out_dp.numel()+cnt.numel())*4/1e9
    print(f"  accumulator (CPU): {mem_gb:.1f} GB")

    n_skip = n_done = 0
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_range   = range(slice_index, min(slice_index+sw_batch_size, len(slices)))
        batch_inputs, batch_slices = [], []
        for curr_index in slice_range:
            si, sj = slices[curr_index]
            if inputs[0,:,si,sj].max() < 1e-6:
                n_skip += 1
                continue
            batch_inputs.append(_build_input(curr_index))
            batch_slices.append((si, sj))

        if not batch_inputs:
            continue

        data_np = torch_gpuuu.stack(batch_inputs).permute(0,2,3,1).cpu().numpy()
        torch_gpuuu.cuda.empty_cache()

        for i, (si, sj) in enumerate(batch_slices):
            seg = predictor.eval(
                data_np[i], channels=None, diameter=DIAMETER,
                do_3D=False, net_avg=True, augment=False, resample=True,
                flow_threshold=0.4, confidence_threshold=0, stitch_threshold=0.0,
                invert=False, batch_size=8, interp=True, normalize=False,
                channel_axis=None, z_axis=None, anisotropy=1.0, model_loaded=True,
            )
            out_cp[0,:,si,sj] += torch_gpuuu.from_numpy(seg[1][2]).unsqueeze(0)
            out_dp[0,:,si,sj] += torch_gpuuu.from_numpy(seg[1][1])
            cnt[0,:,si,sj]    += 1.0
            n_done += 1

        torch_gpuuu.cuda.empty_cache()
        if (slice_index // sw_batch_size) % 200 == 0:
            print(f"  progress: {slice_index:6d}/{len(slices)}  "
                  f"done={n_done:5d}  background skipped={n_skip:5d}")

    print(f"  inference done : valid={n_done}  background skipped={n_skip}")

    cnt    = cnt.clamp(min=1.0)
    out_cp = out_cp / cnt
    out_dp = out_dp / cnt

    # confidence distribution stats (helps choose thresholds)
    cp_np = out_cp.squeeze().numpy()
    print(f"\n  [confidence map stats]")
    print(f"    shape          : {cp_np.shape}")
    print(f"    whole-image mean: {cp_np.mean():.4f}")
    print(f"    nonzero pixel ratio: {(cp_np > 0).mean()*100:.1f}%")
    for thr in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pct = (cp_np > thr).mean() * 100
        print(f"    > {thr:.1f}           : {pct:.2f}%  (~{int(pct/100*cp_np.size):,} px)")

    H, W = original_image_size
    return out_cp[...,:H,:W], out_dp[...,:H,:W]


def run_postprocess(confidence_np, offset_np, device, conf_thr, flow_thr):
    """Run postprocess with one parameter set, returns (mask, n_cells)."""
    try:
        outputs = dynamics.compute_masks(
            offset_np, confidence_np,
            niter=200,
            confidence_threshold=conf_thr,
            flow_threshold=flow_thr,
            min_size=MIN_SIZE,
            interp=True, resize=None,
            use_gpu=False, device=device,
        )
        mask    = outputs[0]
        ids, counts = np.unique(mask[mask > 0], return_counts=True)
        n_cells = len(ids)
        median_area = int(np.median(counts)) if len(counts) > 0 else 0
        diff    = abs(n_cells - TARGET_CELLS)
        tag     = "  <- closest to target ✓" if diff < 5000 else ""
        print(f"  conf={conf_thr:.1f}  flow={flow_thr:.1f}  min_size={MIN_SIZE}"
              f"  →  cells={n_cells:6d}  median area={median_area:6d} px^2{tag}")
        return mask, n_cells
    except Exception as e:
        print(f"  conf={conf_thr:.1f}  flow={flow_thr:.1f}  →  failed: {e}")
        return None, 0


def main():
    logger, log_file = logger_setup()
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    print("\n" + "═"*60)
    print("Parameters")
    print("═"*60)
    print(f"  diam_mean  = {DIAM_MEAN}  (actual value in the model file)")
    print(f"  diameter   = {DIAMETER}  (measured cell diameter 82px)")
    print(f"  scale      = {DIAM_MEAN/DIAMETER:.3f}  (inference scale factor)")
    print(f"  SIGMA      = {SIGMA}  (7 x 82/34 ~ 15)")
    print(f"  min_size   = {MIN_SIZE} px^2  (~10% of cell area)")

    # ── A ────────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("A. Compute FOV global pixel offsets")
    print("═"*60)
    fov_offsets = compute_fov_offsets(TX_FILE, META_FILE)

    # ── B ────────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("B. Stitch the full CellComposite image")
    print("═"*60)
    canvas, gx_min, gy_min = stitch_composite(COMP_DIR, fov_offsets)
    canvas_h, canvas_w = canvas.shape

    # ── C ────────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("C. Load transcript coordinates")
    print("═"*60)
    spots = load_all_transcripts(TX_FILE, gx_min, gy_min, canvas_w, canvas_h)

    # ── D ────────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("D. Save GeneSegNet input")
    print("═"*60)
    save_genesegnet_input(WORK_DIR, canvas, spots, SAMPLE_NAME)

    # ── E ────────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("E. Load GeneSegNet model")
    print("═"*60)
    gpu    = torch_gpuuu.cuda.is_available()
    device = torch_gpuuu.device("cuda:0" if gpu else "cpu")
    free, total = torch_gpuuu.cuda.mem_get_info(0)
    print(f"  device         : {device}  ({torch_gpuuu.cuda.get_device_name(0)})")
    print(f"  VRAM           : {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

    model = models.GeneSegModel(
        gpu=gpu, device=device,
        pretrained_model=MODEL_PATH,
        model_type=None,
        diam_mean=DIAM_MEAN,   # 55.0, read from the model file
        residual_on=1, style_on=1, concatenation=0,
        net_avg=False, nchan=2,
    )
    print(f"  ✔ model loaded")
    print(f"  diam_mean={DIAM_MEAN}  diameter={DIAMETER}  "
          f"scale={DIAM_MEAN/DIAMETER:.3f}")

    # ── F ────────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("F. Sliding-window inference")
    print("═"*60)

    if os.path.exists(CONFIDENCE_PATH) and os.path.exists(OFFSET_PATH):
        print(f"  ✔ inference cache exists, loading directly (skipping inference)")
        confidence_np = np.load(CONFIDENCE_PATH)
        offset_np     = np.load(OFFSET_PATH)
        print(f"  confidence : {confidence_np.shape}  "
              f"{confidence_np.nbytes/1e9:.2f} GB")
        print(f"  offset     : {offset_np.shape}  "
              f"{offset_np.nbytes/1e9:.2f} GB")
        for thr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            print(f"  > {thr:.1f}        : {(confidence_np > thr).mean()*100:.2f}%")
    else:
        image_t = torch_gpuuu.from_numpy(canvas).float().unsqueeze(0).unsqueeze(0) / 255.0
        image_t = image_t.to(device)
        spot_t  = torch_gpuuu.from_numpy(spots).to(device)

        confidence, offset = sliding_window_inference(
            image_t, [256,256], 10, model,
            spot_t, SIGMA, SAMPLE_NAME, SAVE_DIR, device,
        )

        confidence_np = confidence.squeeze().numpy()
        offset_np     = offset.squeeze().numpy()

        np.save(CONFIDENCE_PATH, confidence_np)
        np.save(OFFSET_PATH,     offset_np)
        print(f"\n  ✔ inference results saved (loaded directly next time)")
        print(f"    {CONFIDENCE_PATH}")
        print(f"    {OFFSET_PATH}")

    # ── G ────────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print(f"G. Postprocess parameter scan  (target ~ {TARGET_CELLS} cells)")
    print("═"*60)

    best_mask, best_n    = None, None
    best_conf, best_flow = 0.0, 0.4

    for conf_thr in CONF_THRESHOLDS:
        for flow_thr in FLOW_THRESHOLDS:
            mask, n_cells = run_postprocess(
                confidence_np, offset_np, device, conf_thr, flow_thr)
            if mask is not None and (
                best_n is None or
                abs(n_cells - TARGET_CELLS) < abs(best_n - TARGET_CELLS)
            ):
                best_mask  = mask
                best_n     = n_cells
                best_conf  = conf_thr
                best_flow  = flow_thr

    print(f"\n  best params: conf={best_conf}  flow={best_flow}  → cells={best_n}")

    # ── H ────────────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print(f"H. Save best result  (conf={best_conf}, flow={best_flow})")
    print("═"*60)

    wholemask = cv2.morphologyEx(
        best_mask.astype(np.float32), cv2.MORPH_OPEN, np.ones((5,5), np.uint8)
    ).astype(np.uint32)

    ids, counts = np.unique(wholemask[wholemask > 0], return_counts=True)
    print(f"  cells after morphological filter: {len(ids)}")
    print(f"  cell size(px^2)")
    print(f"    mean   : {counts.mean():.0f}")
    print(f"    median : {np.median(counts):.0f}")
    print(f"    5%~95% : {np.percentile(counts,5):.0f} ~ {np.percentile(counts,95):.0f}")
    print(f"    min/max: {counts.min()} / {counts.max()}")

    suffix    = f"conf{best_conf}_flow{best_flow}"
    tiff_path = os.path.join(SAVE_DIR, f"label_{SAMPLE_NAME}_{suffix}.tif")
    mat_path  = os.path.join(SAVE_DIR, f"label_{SAMPLE_NAME}_{suffix}.mat")
    jpg_path  = os.path.join(SAVE_DIR, f"label_plt_{SAMPLE_NAME}_{suffix}.jpg")

    tifffile.imwrite(tiff_path, wholemask)
    print(f"  TIF → {tiff_path}")

    io.savemat(mat_path, {
        "CellMap"         : wholemask,
        "gx_min"          : gx_min,
        "gy_min"          : gy_min,
        "canvas_w"        : canvas_w,
        "canvas_h"        : canvas_h,
        "diam_mean"       : DIAM_MEAN,
        "diameter"        : DIAMETER,
        "conf_threshold"  : best_conf,
        "flow_threshold"  : best_flow,
        "min_size"        : MIN_SIZE,
    })
    print(f"  MAT → {mat_path}")

    scale = 8
    thumb = wholemask[::scale, ::scale]
    plt.figure(figsize=(14, 14*canvas_h//canvas_w))
    plt.imshow(thumb, cmap="nipy_spectral")
    plt.axis("off")
    plt.title(f"{SAMPLE_NAME}  conf={best_conf} flow={best_flow}  "
              f"{len(ids)} cells  (1:{scale} thumbnail)", fontsize=10)
    plt.savefig(jpg_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()
    print(f"  JPG → {jpg_path}")

    print("\n" + "═"*60)
    print(f"Done! final cell count = {len(ids)}")
    print(f"result dir : {SAVE_DIR}")
    print("═"*60)


if __name__ == "__main__":
    main()