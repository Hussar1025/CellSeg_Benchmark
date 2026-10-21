"""
UCS — Pancreas CosMx WTx — Full Image
======================================
Server  : 4090-02
Env     : ucs_env (Python 3.9)
GPU     : auto-select the most idle GPU at runtime
Run     : python ucs_pancreas_full.py

nuclei_mask : CellLabels TIF (official nuclear segmentation, 18-FOV mosaic)
BIN_FACTOR  : 25  →  3.0 µm/bin
TOP_N_GENES : 1000 (covers 63.4% transcripts)
  → full inference memory: 468×48×48×1000×4 bytes = 4.3 GB ✅
  → 5000 genes: 21.6 GB → OOM ❌

GPU fix notes:
  UCS internally sets os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
  so pass the physical GPU index directly to --gpu; do not set
  CUDA_VISIBLE_DEVICES externally.
"""

import os, glob, re, sys, shutil, subprocess


# ── Auto-select the most idle GPU ─────────────────────────────────────────────
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
            print(f"  GPU {idx}: {free_gb:.1f} GB free  {util}% load")
            if free_gb > best_free:
                best_free, best_idx = free_gb, int(idx)
        if best_free < min_free_gb:
            raise RuntimeError(
                f"No GPU exceeds {min_free_gb} GB free (max {best_free:.1f} GB)")
        print(f"  → selected GPU {best_idx} ({best_free:.1f} GB free)")
        return str(best_idx)
    except FileNotFoundError:
        print("  nvidia-smi unavailable, using GPU 0")
        return "0"

print("=" * 50)
print("Auto-select GPU (requires ≥ 10 GB free)")
_GPU_ID = pick_best_gpu(min_free_gb=10.0)
# Note: do not set CUDA_VISIBLE_DEVICES here
# UCS internally runs os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
# pass the physical GPU index directly to --gpu
print("=" * 50)

import numpy as np
import pandas as pd
import cv2
import tifffile
from natsort import natsorted

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR       = "yourpath/dataset/Pancreas-CosMx-WTx-FlatFiles"
CELL_LABEL_DIR = os.path.join(DATA_DIR, "CellLabels")
UCS_DIR        = os.path.dirname(os.path.abspath(__file__))
WORK_DIR       = "yourpath/ucs_cosmx"

TX_FILE   = os.path.join(DATA_DIR, "Pancreas_tx_file.csv")
META_FILE = os.path.join(DATA_DIR, "Pancreas_metadata_file.csv")

PIXEL_SIZE  = 0.1203
BIN_FACTOR  = 25
TOP_N_GENES = 1000   # covers 63.4% transcripts; full inference GPU memory 4.3 GB (5000 genes = 21.6 GB OOM)

PATCH_SIZE           = 48
DILATION_KERNEL_SIZE = 10
DILATION_ITER_NUM    = 4
TAU                  = 5
FG_NET_EPOCH         = 1
FG_NET_BATCH_SIZE    = 32    # batch training GPU memory 0.9 GB
CELL_NET_EPOCH       = 1
GPU                  = _GPU_ID   # physical GPU index, passed directly to UCS
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
    print(f"[1A] expected FOVs: {sorted(all_fovs)}")

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
            print("[1A] all FOVs found, exiting early.")
            break

    all_ox = [ox for ox,_ in found.values()]
    all_oy = [oy for _,oy in found.values()]
    gx_min, gy_min = min(all_ox), min(all_oy)
    canvas_w = max(all_ox) + 4256 - gx_min
    canvas_h = max(all_oy) + 4256 - gy_min
    map_w    = canvas_w // BIN_FACTOR
    map_h    = canvas_h // BIN_FACTOR
    print(f"[1A] canvas: {canvas_w}×{canvas_h} px  →  gene map: {map_w}×{map_h} bins")
    return found, gx_min, gy_min, canvas_w, canvas_h


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: Nuclei mask from CellLabels
# ─────────────────────────────────────────────────────────────────────────────
def generate_nuclei_mask(fov_offsets, gx_min, gy_min,
                         canvas_w, canvas_h, map_h, map_w):
    if os.path.exists(NUCLEI_MASK_PATH):
        print(f"[2] nuclei mask exists, loading: {NUCLEI_MASK_PATH}")
        mask = tifffile.imread(NUCLEI_MASK_PATH)
        print(f"[2] shape={mask.shape}  cells={len(np.unique(mask))-1}")
        return mask

    try:
        import imagecodecs
    except ImportError:
        raise ImportError("run first: pip install imagecodecs")

    label_files  = natsorted(glob.glob(
        os.path.join(CELL_LABEL_DIR, "CellLabels_F*.tif")))
    fov_re       = re.compile(r"F(\d+)", re.IGNORECASE)
    fov_file_map = {int(fov_re.search(os.path.basename(f)).group(1)): f
                    for f in label_files if fov_re.search(os.path.basename(f))}
    print(f"[2] found {len(fov_file_map)} CellLabels TIFs")

    label_canvas     = np.zeros((canvas_h, canvas_w), dtype=np.int32)
    global_id_offset = 0

    for fov_id, (ox, oy) in sorted(fov_offsets.items()):
        if fov_id not in fov_file_map:
            print(f"[2]   WARNING: FOV {fov_id} has no CellLabels, skipping")
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
        print(f"[2]   FOV {fov_id:3d}  cells={n_cells}  max global ID={global_id_offset}")

    total_cells = len(np.unique(label_canvas)) - 1
    print(f"[2] total cells: {total_cells}  (expected ~48944)")

    trimmed     = label_canvas[:map_h*BIN_FACTOR, :map_w*BIN_FACTOR]
    nuclei_mask = trimmed[BIN_FACTOR//2::BIN_FACTOR,
                          BIN_FACTOR//2::BIN_FACTOR].astype(np.int32)
    n_unique = len(np.unique(nuclei_mask)) - 1
    print(f"[2] Binned mask: {nuclei_mask.shape}  unique cell IDs: {n_unique}")

    tifffile.imwrite(NUCLEI_MASK_PATH, nuclei_mask)
    print(f"[2] Saved: {NUCLEI_MASK_PATH}")
    return nuclei_mask


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3: Gene map  [map_h, map_w, TOP_N_GENES] uint8
# ─────────────────────────────────────────────────────────────────────────────
def generate_gene_map(gx_min, gy_min, map_h, map_w):
    if os.path.exists(GENE_MAP_PATH):
        print(f"[3] gene map exists, loading: {GENE_MAP_PATH}")
        gm = tifffile.imread(GENE_MAP_PATH)
        print(f"[3] Shape: {gm.shape}")
        return gm

    print(f"[3] pass 1: count transcripts per gene ...")
    gene_counts = {}
    for chunk in pd.read_csv(TX_FILE, usecols=["target"], chunksize=5_000_000):
        for gene, cnt in chunk["target"].value_counts().items():
            gene_counts[gene] = gene_counts.get(gene, 0) + cnt

    counts_series = pd.Series(gene_counts).sort_values(ascending=False)
    total_tx      = counts_series.sum()
    print(f"[3] total genes: {len(counts_series)}  total transcripts: {total_tx:,}")

    top_genes   = counts_series.iloc[:TOP_N_GENES]
    coverage    = top_genes.sum() / total_tx * 100
    gene_to_idx = {g: i for i, g in enumerate(top_genes.index)}
    num_genes   = len(gene_to_idx)
    mem_gb      = map_h * map_w * num_genes / 1e9
    infer_gb    = 468 * 48 * 48 * num_genes * 4 / 1e9
    print(f"[3] Top {TOP_N_GENES} genes, cover {coverage:.1f}% transcripts")
    print(f"[3] gene map uint8: {mem_gb:.2f} GB")
    print(f"[3] full inference float32: {infer_gb:.1f} GB  (GPU memory required)")

    pd.DataFrame({
        "gene"    : top_genes.index,
        "channel" : range(num_genes),
        "tx_count": top_genes.values,
    }).to_csv(os.path.join(WORK_DIR, "gene_index.csv"), index=False)

    gene_map = np.zeros((map_h, map_w, num_genes), dtype=np.uint8)

    print(f"[3] pass 2: bin transcripts ...")
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
        print(f"[3]   read {total_read:,}  →  binned {total_kept:,}"
              f"  (filtered {total_filtered:,})")

    print(f"[3] nonzero bins: {(gene_map > 0).sum():,}")
    tifffile.imwrite(GENE_MAP_PATH, gene_map, photometric="minisblack")
    print(f"[3] Saved: {GENE_MAP_PATH}")
    return gene_map


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4: Run UCS
# ─────────────────────────────────────────────────────────────────────────────
def run_ucs():
    run_py = os.path.join(UCS_DIR, "run.py")
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"run.py not found: {run_py}")

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
        print(f"[4] removed old log_dir: {LOG_DIR}")

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
        "--gpu",                    GPU,   # physical GPU index; UCS sets CUDA_VISIBLE_DEVICES internally
    ]

    print(f"\n[4] run UCS (GPU {GPU}):")
    print("    " + " \\\n    ".join(cmd))
    result = subprocess.run(cmd, cwd=UCS_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"UCS exit code: {result.returncode}")
    print(f"\n[4] UCS Done, results in: {LOG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("UCS — Pancreas CosMx WTx — Full Image")
    print(f"GPU        : {GPU}  (auto-selected)")
    print(f"BIN_FACTOR : {BIN_FACTOR}  ({BIN_FACTOR * PIXEL_SIZE:.2f} µm/bin)")
    print(f"TOP_N_GENES: {TOP_N_GENES}  (covers 63.4% transcripts)")
    print(f"batch_size : {FG_NET_BATCH_SIZE}  (training GPU memory ≈ 0.9 GB)")
    print(f"infer GPU memory : ≈ {468*48*48*TOP_N_GENES*4/1e9:.1f} GB  (full inference)")
    print(f"nuclei_mask: CellLabels official nuclear segmentation")
    print("=" * 65)

    print("\n── Step 1A: FOV offsets ──")
    fov_offsets, gx_min, gy_min, canvas_w, canvas_h = compute_fov_offsets()
    map_h = canvas_h // BIN_FACTOR
    map_w = canvas_w // BIN_FACTOR

    print("\n── Step 2: Nuclei mask (CellLabels) ──")
    nuclei_mask = generate_nuclei_mask(
        fov_offsets, gx_min, gy_min, canvas_w, canvas_h, map_h, map_w)

    assert nuclei_mask.shape == (map_h, map_w), \
        f"nuclei_mask {nuclei_mask.shape} != expected ({map_h},{map_w})"

    print("\n── Step 3: Gene map ──")
    gene_map = generate_gene_map(gx_min, gy_min, map_h, map_w)

    assert gene_map.shape[:2] == nuclei_mask.shape, \
        f"gene_map {gene_map.shape[:2]} != nuclei_mask {nuclei_mask.shape}"

    print(f"\nInput summary:")
    print(f"  gene_map    : {gene_map.shape}  → {GENE_MAP_PATH}")
    print(f"  nuclei_mask : {nuclei_mask.shape}  cells={nuclei_mask.max()}")
    print(f"  GPU         : {GPU}")
    print(f"  batch_size  : {FG_NET_BATCH_SIZE}")

    print("\n── Step 4: Run UCS ──")
    run_ucs()

    print("\n" + "=" * 65)
    print(f"Done! Results in: {LOG_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
