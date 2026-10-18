"""
ucs_merfish_brain.py
UCS — MERFISH 小鼠脑 — 20000×20000 px ROI（组织中心）

关键修复：
  cell_metadata 的 cell_id 与 HDF5 key 是不同的 ID 系统（交集极少）
  → 不用 cell_id 匹配，直接用多边形质心的空间位置过滤 ROI 内的细胞

提速优化：
  多进程并行读取 1120 个 HDF5（16 workers）

运行：
  conda activate ucs
  cd /data/qiuyijia/ucs/UCS
  nohup python ucs_merfish_brain.py > merfish_brain.log 2>&1 &
"""

import os, sys, shutil, subprocess, time, glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import h5py
import cv2
import tifffile
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
#  动态 GPU 选择
# ══════════════════════════════════════════════════════════════════════════════
def pick_best_gpu(min_free_gb=10.0):
    try:
        result = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True
        )
        best_idx, best_free = -1, 0.0
        for line in result.strip().splitlines():
            idx, free_mb, util = [x.strip() for x in line.split(",")]
            free_gb = int(free_mb) / 1024
            print(f"  GPU {idx}: {free_gb:.1f} GB 空闲  {util}% 负载")
            if free_gb > best_free:
                best_free, best_idx = free_gb, int(idx)
        if best_free < min_free_gb:
            raise RuntimeError(f"没有 GPU 空闲超过 {min_free_gb} GB")
        print(f"  → 选择 GPU {best_idx}（{best_free:.1f} GB 空闲）")
        return str(best_idx)
    except FileNotFoundError:
        return "0"

print("=" * 60)
print("自动选择 GPU...")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    GPU = pick_best_gpu(min_free_gb=10.0)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
else:
    GPU = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"  使用外部设置: GPU {GPU}")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
#  路径配置
# ══════════════════════════════════════════════════════════════════════════════
BASE     = "/data/qiuyijia/dataset/merfish_mouse_brain"
PREFIX   = "datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1"
TX_CSV   = os.path.join(BASE, f"{PREFIX}_detected_transcripts_S1R1.csv")
META_CSV = os.path.join(BASE, f"{PREFIX}_cell_metadata_S1R1.csv")
BD_DIR   = os.path.join(BASE, "cell_boundaries")

UCS_DIR  = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = "/data/qiuyijia/ucs_merfish_brain"
LOG_DIR  = os.path.join(WORK_DIR, "ucs_log")

GENE_MAP_PATH    = os.path.join(WORK_DIR, "gene_map.tif")
NUCLEI_MASK_PATH = os.path.join(WORK_DIR, "nuclei_mask.tif")

os.makedirs(WORK_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  坐标参数 & ROI
# ══════════════════════════════════════════════════════════════════════════════
SCALE_X = 9.205861
SCALE_Y = 9.205850

TX_CTR_X_UM = 4675.0
TX_CTR_Y_UM = 3413.0
ROI_HALF_UM = 10000 / SCALE_X    # ≈ 1086.0 µm

ROI_X0 = TX_CTR_X_UM - ROI_HALF_UM
ROI_X1 = TX_CTR_X_UM + ROI_HALF_UM
ROI_Y0 = TX_CTR_Y_UM - ROI_HALF_UM
ROI_Y1 = TX_CTR_Y_UM + ROI_HALF_UM

ROI_IX0 = int(ROI_X0)
ROI_IY0 = int(ROI_Y0)
MAP_W   = int(ROI_X1) - ROI_IX0 + 1
MAP_H   = int(ROI_Y1) - ROI_IY0 + 1

# UCS 参数
PATCH_SIZE           = 48
DILATION_KERNEL_SIZE = 10
DILATION_ITER_NUM    = 4
TAU                  = 5
FG_NET_EPOCH         = 1
FG_NET_BATCH_SIZE    = 32
CELL_NET_EPOCH       = 1
Z_SLICE              = 3    # 对应 mosaic_DAPI_z3.tif

SEP = "═" * 60
print(f"\n{SEP}")
print(f"UCS — MERFISH Mouse Brain S1R1")
print(f"ROI  : x=[{ROI_X0:.1f},{ROI_X1:.1f}] y=[{ROI_Y0:.1f},{ROI_Y1:.1f}] µm")
print(f"Map  : {MAP_W}×{MAP_H} bins (1 µm/bin)")
print(f"GPU  : {GPU}")
print(f"{SEP}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Gene map
# ══════════════════════════════════════════════════════════════════════════════
def generate_gene_map():
    if os.path.exists(GENE_MAP_PATH):
        print(f"[1] 已有 gene_map，直接加载: {GENE_MAP_PATH}")
        gm = tifffile.imread(GENE_MAP_PATH)
        print(f"[1] shape={gm.shape}  基因数={gm.shape[2]}")
        return gm

    print(f"[1] 读取转录本 CSV...")
    t0 = time.time()
    chunks = []
    total  = 0
    for chunk in pd.read_csv(TX_CSV,
                              usecols=["global_x","global_y","gene"],
                              chunksize=2_000_000):
        roi_m = ((chunk.global_x >= ROI_X0) & (chunk.global_x <= ROI_X1) &
                 (chunk.global_y >= ROI_Y0) & (chunk.global_y <= ROI_Y1))
        chunks.append(chunk[roi_m].copy())
        total += len(chunk)
        print(f"  已读 {total:,}  ROI 内: {sum(len(c) for c in chunks):,}")

    tx = pd.concat(chunks, ignore_index=True); del chunks
    print(f"[1] ROI 内转录本: {len(tx):,}  基因数: {tx.gene.nunique()}  "
          f"耗时: {time.time()-t0:.1f}s")

    try:
        import natsort
        gene_names = natsort.natsorted(tx.gene.unique())
    except ImportError:
        gene_names = sorted(tx.gene.unique())
    n_genes  = len(gene_names)
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    pd.DataFrame({"gene": gene_names, "channel": range(n_genes)}).to_csv(
        os.path.join(WORK_DIR, "gene_index.csv"), index=False)
    print(f"[1] 基因数: {n_genes}")

    tx["bx"] = (tx.global_x.round().astype(int) - ROI_IX0).clip(0, MAP_W-1)
    tx["by"] = (tx.global_y.round().astype(int) - ROI_IY0).clip(0, MAP_H-1)
    tx["gi"] = tx.gene.map(gene2idx)

    gene_map = np.zeros((MAP_H, MAP_W, n_genes), dtype=np.uint8)
    flat     = (tx.by.values * (MAP_W * n_genes) +
                tx.bx.values * n_genes +
                tx.gi.values.astype(int))
    np.add.at(gene_map.ravel(), flat, 1)
    gene_map = np.clip(gene_map, 0, 255).astype(np.uint8)

    print(f"[1] gene_map: {gene_map.shape}  {gene_map.nbytes/1e9:.2f} GB")
    tifffile.imwrite(GENE_MAP_PATH, gene_map, photometric="minisblack")
    print(f"[1] 已保存: {GENE_MAP_PATH}")
    return gene_map


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Nuclei mask
#
#  关键修复：cell_metadata 的 cell_id 与 HDF5 的 featuredata key 是不同的 ID 系统
#  解决方案：不用 cell_id 匹配，直接用多边形质心的空间位置过滤
#  提速：16 workers 并行处理 1120 个 HDF5
# ══════════════════════════════════════════════════════════════════════════════
def _process_one_hdf5(args):
    """
    多进程 worker：读取单个 HDF5，提取质心在 ROI 内的多边形
    正确路径：featuredata/{cell_id}/zIndex_3/p_0/coordinates  shape=(1,N,2)
    坐标单位：µm，与 cell_metadata center_x/y 相同
    """
    hdf5_path, roi_x0, roi_x1, roi_y0, roi_y1, roi_ix0, roi_iy0, map_w, map_h, z_slice = args
    polys = []
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "featuredata" not in f:
                return polys
            for cell_id_str in f["featuredata"]:
                cell_grp = f["featuredata"][cell_id_str]
                z_key    = f"zIndex_{z_slice}"
                if z_key not in cell_grp:
                    # fallback：取第一个可用 zIndex
                    z_keys = [k for k in cell_grp.keys() if k.startswith("zIndex")]
                    if not z_keys:
                        continue
                    z_key = sorted(z_keys)[0]

                z_group = cell_grp[z_key]
                # z_group 下有 p_0, p_1... 每个 p_i 下有 coordinates (1,N,2)
                for pk in z_group.keys():
                    if not isinstance(z_group[pk], h5py.Group):
                        continue
                    if "coordinates" not in z_group[pk]:
                        continue
                    poly_um = np.array(z_group[pk]["coordinates"])
                    # shape = (1, N, 2) → 取 [0] 得 (N, 2)
                    if poly_um.ndim == 3 and poly_um.shape[0] == 1:
                        poly_um = poly_um[0]
                    if poly_um.ndim != 2 or poly_um.shape[1] != 2 or len(poly_um) < 3:
                        continue

                    # 空间过滤：质心必须在 ROI 内（坐标单位 µm）
                    cx = poly_um[:, 0].mean()
                    cy = poly_um[:, 1].mean()
                    if not (roi_x0 <= cx <= roi_x1 and roi_y0 <= cy <= roi_y1):
                        continue

                    # µm → ROI 本地 bin
                    bx = (poly_um[:,0].round().astype(int) - roi_ix0).clip(0, map_w-1)
                    by = (poly_um[:,1].round().astype(int) - roi_iy0).clip(0, map_h-1)
                    polys.append(np.stack([bx, by], axis=1).astype(np.int32))
                    break   # 每个细胞只用第一个 polygon（p_0）
    except Exception:
        pass
    return polys


def generate_nuclei_mask():
    if os.path.exists(NUCLEI_MASK_PATH):
        print(f"[2] 已有 nuclei_mask: {NUCLEI_MASK_PATH}")
        mask = tifffile.imread(NUCLEI_MASK_PATH)
        print(f"[2] shape={mask.shape}  细胞数={mask.max()}")
        return mask

    # ROI 内 GT 细胞数（仅用于日志显示，不用于 ID 匹配）
    meta = pd.read_csv(META_CSV)
    meta = meta.rename(columns={meta.columns[0]: "cell_id"})
    roi_m = ((meta.center_x >= ROI_X0) & (meta.center_x <= ROI_X1) &
             (meta.center_y >= ROI_Y0) & (meta.center_y <= ROI_Y1))
    n_gt_roi = roi_m.sum()
    print(f"[2] ROI 内 GT 细胞（参考）: {n_gt_roi:,}")

    # 全部 1120 个 HDF5，用质心空间过滤
    hdf5_files = sorted(glob.glob(os.path.join(BD_DIR, "*.hdf5")))
    print(f"[2] 读取全部 HDF5: {len(hdf5_files):,} 个（质心空间过滤，多进程并行）")

    n_workers = min(16, mp.cpu_count())
    args_list = [
        (p, ROI_X0, ROI_X1, ROI_Y0, ROI_Y1, ROI_IX0, ROI_IY0, MAP_W, MAP_H, Z_SLICE)
        for p in hdf5_files
    ]

    print(f"[2] 启动 {n_workers} workers...")
    t0 = time.time()
    all_polys = []
    with mp.Pool(n_workers) as pool:
        for polys in tqdm(pool.imap_unordered(_process_one_hdf5, args_list),
                          total=len(args_list), ncols=70):
            all_polys.extend(polys)
    print(f"[2] 提取多边形: {len(all_polys):,} 个  耗时: {time.time()-t0:.1f}s")

    # 绘制 mask
    nuclei_mask = np.zeros((MAP_H, MAP_W), dtype=np.int32)
    for mask_id, pts in enumerate(all_polys, start=1):
        cv2.fillPoly(nuclei_mask, [pts.reshape(-1, 1, 2)], mask_id)
    n_cells = int(nuclei_mask.max())
    print(f"[2] 绘制完成: {n_cells:,} 个细胞（GT 参考: {n_gt_roi:,}）")

    tifffile.imwrite(NUCLEI_MASK_PATH, nuclei_mask.astype(np.int32))
    print(f"[2] 已保存: {NUCLEI_MASK_PATH}")
    return nuclei_mask


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: 运行 UCS
# ══════════════════════════════════════════════════════════════════════════════
def run_ucs():
    run_py = os.path.join(UCS_DIR, "run.py")
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"run.py 未找到: {run_py}")

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)

    cmd = [
        sys.executable, run_py,
        "--gene_map",               GENE_MAP_PATH,
        "--nuclei_mask",            NUCLEI_MASK_PATH,
        "--log_dir",                LOG_DIR,
        "--patch_size",             str(PATCH_SIZE),
        "--dilation_kernel_size",   str(DILATION_KERNEL_SIZE),
        "--dilation_iter_num",      str(DILATION_ITER_NUM),
        "--tau",                    str(TAU),
        "--fg_net_epoch",           str(FG_NET_EPOCH),
        "--fg_net_batch_size",      str(FG_NET_BATCH_SIZE),
        "--cell_net_epoch",         str(CELL_NET_EPOCH),
        "--gpu",                    GPU,
    ]

    print(f"\n[3] 运行 UCS（GPU {GPU}）")
    result = subprocess.run(cmd, cwd=UCS_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"UCS 退出码: {result.returncode}")
    print(f"[3] UCS 完成，结果在: {LOG_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # 清除旧的空 mask（如果存在）
    if os.path.exists(NUCLEI_MASK_PATH):
        mask = tifffile.imread(NUCLEI_MASK_PATH)
        if mask.max() == 0:
            print("[!] 检测到空 nuclei_mask（细胞数=0），删除并重新生成")
            os.remove(NUCLEI_MASK_PATH)

    print(f"\n{SEP}")
    print("Step 1: 生成 Gene Map")
    print(SEP)
    gene_map = generate_gene_map()

    print(f"\n{SEP}")
    print("Step 2: 生成 Nuclei Mask（质心空间过滤 + 多进程）")
    print(SEP)
    nuclei_mask = generate_nuclei_mask()

    if nuclei_mask.max() == 0:
        raise RuntimeError("[!] nuclei_mask 仍为空！请检查 ROI 坐标或 HDF5 格式")

    assert gene_map.shape[:2] == nuclei_mask.shape, \
        f"尺寸不匹配: gene_map {gene_map.shape[:2]} ≠ nuclei_mask {nuclei_mask.shape}"

    print(f"\n输入汇总:")
    print(f"  gene_map    : {gene_map.shape}  → {GENE_MAP_PATH}")
    print(f"  nuclei_mask : {nuclei_mask.shape}  细胞数={nuclei_mask.max()}")
    print(f"  GPU         : {GPU}")

    print(f"\n{SEP}")
    print("Step 3: 运行 UCS")
    print(SEP)
    run_ucs()

    print(f"\n{'='*60}")
    print(f"完成！结果在: {LOG_DIR}/pred/segmentation_mask.tif")
    print("="*60)


if __name__ == "__main__":
    main()