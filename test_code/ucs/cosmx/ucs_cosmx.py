"""
UCS — Pancreas CosMx WTx — Full Image
======================================
服务器  : 4090-02
环境    : ucs_env (Python 3.9)
GPU     : 运行时自动选择最空闲的卡
运行方式: python ucs_pancreas_full.py

nuclei_mask : CellLabels TIF（官方核分割，18个FOV拼图）
BIN_FACTOR  : 25  →  3.0 µm/bin
TOP_N_GENES : 1000（覆盖63.4%转录本）
  → 全量推理内存: 468×48×48×1000×4 bytes = 4.3 GB ✅
  → 5000基因时: 21.6 GB → OOM ❌

GPU 修复说明：
  UCS 内部执行 os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
  所以必须把物理 GPU 号直接传给 --gpu，不能在外部设 CUDA_VISIBLE_DEVICES
"""

import os, glob, re, sys, shutil, subprocess


# ── 动态选择最空闲的 GPU ────────────────────────────────────────────────────
def pick_best_gpu(min_free_gb=10.0):
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
            print(f"  GPU {idx}: {free_gb:.1f} GB 空闲  {util}% 负载")
            if free_gb > best_free:
                best_free, best_idx = free_gb, int(idx)
        if best_free < min_free_gb:
            raise RuntimeError(
                f"没有 GPU 超过 {min_free_gb} GB 空闲（最大 {best_free:.1f} GB）")
        print(f"  → 选择 GPU {best_idx}（{best_free:.1f} GB 空闲）")
        return str(best_idx)
    except FileNotFoundError:
        print("  nvidia-smi 不可用，使用 GPU 0")
        return "0"

print("=" * 50)
print("自动选择 GPU（要求 ≥ 10 GB 空闲）")
_GPU_ID = pick_best_gpu(min_free_gb=10.0)
# 注意：不在此处设 CUDA_VISIBLE_DEVICES
# UCS 内部会执行 os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
# 必须把物理 GPU 号直接传给 --gpu
print("=" * 50)

import numpy as np
import pandas as pd
import cv2
import tifffile
from natsort import natsorted

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR       = "/data/qiuyijia/dataset/Pancreas-CosMx-WTx-FlatFiles"
CELL_LABEL_DIR = os.path.join(DATA_DIR, "CellLabels")
UCS_DIR        = os.path.dirname(os.path.abspath(__file__))
WORK_DIR       = "/data/qiuyijia/ucs_cosmx"

TX_FILE   = os.path.join(DATA_DIR, "Pancreas_tx_file.csv")
META_FILE = os.path.join(DATA_DIR, "Pancreas_metadata_file.csv")

PIXEL_SIZE  = 0.1203
BIN_FACTOR  = 25
TOP_N_GENES = 1000   # 覆盖63.4%转录本；全量推理显存 4.3 GB（5000基因=21.6 GB OOM）

PATCH_SIZE           = 48
DILATION_KERNEL_SIZE = 10
DILATION_ITER_NUM    = 4
TAU                  = 5
FG_NET_EPOCH         = 1
FG_NET_BATCH_SIZE    = 32    # batch 训练显存 0.9 GB
CELL_NET_EPOCH       = 1
GPU                  = _GPU_ID   # 物理 GPU 号，直接传给 UCS
# ══════════════════════════════════════════════════════════════════════════════

GENE_MAP_PATH    = os.path.join(WORK_DIR, "gene_map.tif")
NUCLEI_MASK_PATH = os.path.join(WORK_DIR, "nuclei_mask.tif")
LOG_DIR          = os.path.join(WORK_DIR, "ucs_log")

os.makedirs(WORK_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1A: Per-FOV offsets
# ─────────────────────────────────────────────────────────────────────────────
def compute_fov_offsets():
    meta     = pd.read_csv(META_FILE, usecols=["fov"])
    all_fovs = set(int(f) for f in meta["fov"].unique())
    print(f"[1A] 预期 FOV: {sorted(all_fovs)}")

    found = {}
    for chunk in pd.read_csv(TX_FILE,
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
        print(f"[1A]   {len(found)}/{len(all_fovs)} FOVs")
        if set(found.keys()) >= all_fovs:
            print("[1A] 所有 FOV 已找到，提前退出。")
            break

    all_ox = [ox for ox,_ in found.values()]
    all_oy = [oy for _,oy in found.values()]
    gx_min, gy_min = min(all_ox), min(all_oy)
    canvas_w = max(all_ox) + 4256 - gx_min
    canvas_h = max(all_oy) + 4256 - gy_min
    map_w    = canvas_w // BIN_FACTOR
    map_h    = canvas_h // BIN_FACTOR
    print(f"[1A] 画布: {canvas_w}×{canvas_h} px  →  gene map: {map_w}×{map_h} bins")
    return found, gx_min, gy_min, canvas_w, canvas_h


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: Nuclei mask from CellLabels
# ─────────────────────────────────────────────────────────────────────────────
def generate_nuclei_mask(fov_offsets, gx_min, gy_min,
                         canvas_w, canvas_h, map_h, map_w):
    if os.path.exists(NUCLEI_MASK_PATH):
        print(f"[2] 已有 nuclei mask，直接加载: {NUCLEI_MASK_PATH}")
        mask = tifffile.imread(NUCLEI_MASK_PATH)
        print(f"[2] shape={mask.shape}  细胞数={len(np.unique(mask))-1}")
        return mask

    try:
        import imagecodecs
    except ImportError:
        raise ImportError("请先运行: pip install imagecodecs")

    label_files  = natsorted(glob.glob(
        os.path.join(CELL_LABEL_DIR, "CellLabels_F*.tif")))
    fov_re       = re.compile(r"F(\d+)", re.IGNORECASE)
    fov_file_map = {int(fov_re.search(os.path.basename(f)).group(1)): f
                    for f in label_files if fov_re.search(os.path.basename(f))}
    print(f"[2] 找到 {len(fov_file_map)} 个 CellLabels TIF")

    label_canvas     = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    global_id_offset = 0

    for fov_id, (ox, oy) in sorted(fov_offsets.items()):
        if fov_id not in fov_file_map:
            print(f"[2]   WARNING: FOV {fov_id} 无 CellLabels，跳过")
            continue
        tile = tifffile.imread(fov_file_map[fov_id]).astype(np.int32)
        h, w = tile.shape
        cx, cy = ox - gx_min, oy - gy_min
        fov_mask       = tile > 0
        tile[fov_mask] += global_id_offset
        global_id_offset = int(tile.max())
        region = label_canvas[cy:cy+h, cx:cx+w]
        region[fov_mask] = tile[fov_mask]
        label_canvas[cy:cy+h, cx:cx+w] = region
        n_cells = len(np.unique(tile[fov_mask]))
        print(f"[2]   FOV {fov_id:3d}  细胞数={n_cells}  全局ID上限={global_id_offset}")

    total_cells = len(np.unique(label_canvas)) - 1
    print(f"[2] 全图细胞总数: {total_cells}  （预期约 48944）")

    trimmed     = label_canvas[:map_h*BIN_FACTOR, :map_w*BIN_FACTOR]
    nuclei_mask = trimmed[BIN_FACTOR//2::BIN_FACTOR,
                          BIN_FACTOR//2::BIN_FACTOR].astype(np.int32)
    n_unique = len(np.unique(nuclei_mask)) - 1
    print(f"[2] Binned mask: {nuclei_mask.shape}  唯一细胞 ID: {n_unique}")

    tifffile.imwrite(NUCLEI_MASK_PATH, nuclei_mask)
    print(f"[2] 已保存: {NUCLEI_MASK_PATH}")
    return nuclei_mask


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3: Gene map  [map_h, map_w, TOP_N_GENES] uint8
# ─────────────────────────────────────────────────────────────────────────────
def generate_gene_map(gx_min, gy_min, map_h, map_w):
    if os.path.exists(GENE_MAP_PATH):
        print(f"[3] 已有 gene map，直接加载: {GENE_MAP_PATH}")
        gm = tifffile.imread(GENE_MAP_PATH)
        print(f"[3] Shape: {gm.shape}")
        return gm

    print(f"[3] 第一轮：统计基因转录本数量 ...")
    gene_counts = {}
    for chunk in pd.read_csv(TX_FILE, usecols=["target"], chunksize=5_000_000):
        for gene, cnt in chunk["target"].value_counts().items():
            gene_counts[gene] = gene_counts.get(gene, 0) + cnt

    counts_series = pd.Series(gene_counts).sort_values(ascending=False)
    total_tx      = counts_series.sum()
    print(f"[3] 总基因数: {len(counts_series)}  总转录本: {total_tx:,}")

    top_genes   = counts_series.iloc[:TOP_N_GENES]
    coverage    = top_genes.sum() / total_tx * 100
    gene_to_idx = {g: i for i, g in enumerate(top_genes.index)}
    num_genes   = len(gene_to_idx)
    mem_gb      = map_h * map_w * num_genes / 1e9
    infer_gb    = 468 * 48 * 48 * num_genes * 4 / 1e9
    print(f"[3] Top {TOP_N_GENES} 基因，覆盖 {coverage:.1f}% 转录本")
    print(f"[3] gene map uint8: {mem_gb:.2f} GB")
    print(f"[3] 全量推理 float32: {infer_gb:.1f} GB  （显存需求）")

    pd.DataFrame({
        "gene"    : top_genes.index,
        "channel" : range(num_genes),
        "tx_count": top_genes.values,
    }).to_csv(os.path.join(WORK_DIR, "gene_index.csv"), index=False)

    gene_map = np.zeros((map_h, map_w, num_genes), dtype=np.uint8)

    print(f"[3] 第二轮：bin 转录本 ...")
    total_read = total_kept = total_filtered = 0
    for chunk in pd.read_csv(TX_FILE,
                              usecols=["x_global_px","y_global_px","target"],
                              chunksize=2_000_000):
        total_read += len(chunk)
        bx = ((chunk["x_global_px"].values - gx_min) / BIN_FACTOR).astype(int)
        by = ((chunk["y_global_px"].values - gy_min) / BIN_FACTOR).astype(int)
        gi = chunk["target"].map(gene_to_idx)

        nan_mask       = gi.isna()
        total_filtered += nan_mask.sum()
        gi_arr         = gi.values
        valid = (~nan_mask.values) & (bx >= 0) & (bx < map_w) & (by >= 0) & (by < map_h)
        bx, by, gi_arr = bx[valid], by[valid], gi_arr[valid].astype(int)

        flat = by * (map_w * num_genes) + bx * num_genes + gi_arr
        np.add.at(gene_map.ravel(), flat, 1)

        total_kept += valid.sum()
        print(f"[3]   已读 {total_read:,}  →  已 bin {total_kept:,}"
              f"  （过滤 {total_filtered:,}）")

    print(f"[3] 非零 bin 数: {(gene_map > 0).sum():,}")
    tifffile.imwrite(GENE_MAP_PATH, gene_map, photometric="minisblack")
    print(f"[3] 已保存: {GENE_MAP_PATH}")
    return gene_map


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4: Run UCS
# ─────────────────────────────────────────────────────────────────────────────
def run_ucs():
    run_py = os.path.join(UCS_DIR, "run.py")
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"run.py 未找到: {run_py}")

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
        print(f"[4] 已清除旧 log_dir: {LOG_DIR}")

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
        "--gpu",                    GPU,   # 物理 GPU 号，UCS 内部会设 CUDA_VISIBLE_DEVICES
    ]

    print(f"\n[4] 运行 UCS （GPU {GPU}）:")
    print("    " + " \\\n    ".join(cmd))
    result = subprocess.run(cmd, cwd=UCS_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"UCS 退出码: {result.returncode}")
    print(f"\n[4] UCS 完成，结果在: {LOG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("UCS — Pancreas CosMx WTx — Full Image")
    print(f"GPU        : {GPU}  （自动选择）")
    print(f"BIN_FACTOR : {BIN_FACTOR}  ({BIN_FACTOR * PIXEL_SIZE:.2f} µm/bin)")
    print(f"TOP_N_GENES: {TOP_N_GENES}  （覆盖63.4%转录本）")
    print(f"batch_size : {FG_NET_BATCH_SIZE}  （训练显存 ≈ 0.9 GB）")
    print(f"infer 显存 : ≈ {468*48*48*TOP_N_GENES*4/1e9:.1f} GB  （全量推理）")
    print(f"nuclei_mask: CellLabels 官方核分割")
    print("=" * 65)

    print("\n── Step 1A: FOV 偏移 ──")
    fov_offsets, gx_min, gy_min, canvas_w, canvas_h = compute_fov_offsets()
    map_h = canvas_h // BIN_FACTOR
    map_w = canvas_w // BIN_FACTOR

    print("\n── Step 2: Nuclei mask（CellLabels）──")
    nuclei_mask = generate_nuclei_mask(
        fov_offsets, gx_min, gy_min, canvas_w, canvas_h, map_h, map_w)

    assert nuclei_mask.shape == (map_h, map_w), \
        f"nuclei_mask {nuclei_mask.shape} ≠ 预期 ({map_h},{map_w})"

    print("\n── Step 3: Gene map ──")
    gene_map = generate_gene_map(gx_min, gy_min, map_h, map_w)

    assert gene_map.shape[:2] == nuclei_mask.shape, \
        f"gene_map {gene_map.shape[:2]} ≠ nuclei_mask {nuclei_mask.shape}"

    print(f"\n输入汇总:")
    print(f"  gene_map    : {gene_map.shape}  → {GENE_MAP_PATH}")
    print(f"  nuclei_mask : {nuclei_mask.shape}  细胞数={nuclei_mask.max()}")
    print(f"  GPU         : {GPU}")
    print(f"  batch_size  : {FG_NET_BATCH_SIZE}")

    print("\n── Step 4: 运行 UCS ──")
    run_ucs()

    print("\n" + "=" * 65)
    print(f"完成！结果在: {LOG_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()