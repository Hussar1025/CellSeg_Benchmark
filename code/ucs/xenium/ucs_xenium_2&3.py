"""
ucs_xenium_breast.py
UCS — Xenium Human Breast Cancer — 10000×10000 px ROI

Data notes:
  transcript coordinates: um (used directly as 1 um/bin pixel coordinates)
  ROI: 5000x5000 px at the image center = 1062x1062 um
  GT cells: 29,892 (inside the ROI)
  genes: 280 (is_gene=True inside the ROI)
  pixel_size: 0.2125 um/px

Run:
  conda activate ucs
  cd yourpath/ucs/UCS
  nohup python ucs_xenium_breast.py > xenium_breast.log 2>&1 &
  echo "PID: $!"
"""

import os, sys, shutil, subprocess, time
import numpy as np
import pandas as pd
import cv2
import tifffile
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
#  Dynamic GPU selection (no CUDA_VISIBLE_DEVICES; pass --gpu directly)
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
            print(f"  GPU {idx}: {free_gb:.1f} GB free  {util}% util")
            if free_gb > best_free:
                best_free, best_idx = free_gb, int(idx)
        if best_free < min_free_gb:
            raise RuntimeError(
                f"no GPU has more than {min_free_gb} GB free (max {best_free:.1f} GB)")
        print(f"  -> selected GPU {best_idx} ({best_free:.1f} GB free)")
        return str(best_idx)
    except FileNotFoundError:
        print("  nvidia-smi unavailable, using GPU 0")
        return "0"

print("=" * 60)
print("auto-selecting GPU (requires >= 10 GB free)")
GPU = pick_best_gpu(min_free_gb=10.0)
# Note: do not set CUDA_VISIBLE_DEVICES here
# UCS sets os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu internally
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
#  Path config
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR  = "yourpath/dataset/xenium_breast"
UCS_DIR   = os.path.dirname(os.path.abspath(__file__))   # yourpath/ucs/UCS
WORK_DIR  = "yourpath/ucs_xenium_breast"
LOG_DIR   = os.path.join(WORK_DIR, "ucs_log")

TX_PARQUET         = os.path.join(DATA_DIR, "transcripts.parquet")
CELL_BOUNDARY      = os.path.join(DATA_DIR, "cell_boundaries.parquet")
NUCLEUS_BOUNDARY   = os.path.join(DATA_DIR, "nucleus_boundaries.parquet")

GENE_MAP_PATH    = os.path.join(WORK_DIR, "gene_map.tif")
NUCLEI_MASK_PATH = os.path.join(WORK_DIR, "nuclei_mask.tif")

os.makedirs(WORK_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROI definition (10000x10000 px, centered on the image center)
#  DAPI: 27420x53994 px, pixel_size=0.2125 um/px
#  image center: (26997, 13710) px -> (5736.9, 2913.4) um
#  ROI: 10000 px = 2125 um side
# ══════════════════════════════════════════════════════════════════════════════
PIXEL_SIZE = 0.2125   # µm/px

CX_UM = 53994 / 2 * PIXEL_SIZE    # 5736.9 µm
CY_UM = 27420 / 2 * PIXEL_SIZE    # 2913.4 µm
HALF  = 5000 * PIXEL_SIZE / 2    # 531.25 um (5000x5000 px ROI)

ROI_X0 = CX_UM - HALF   # 4674.4 µm
ROI_X1 = CX_UM + HALF   # 6799.4 µm
ROI_Y0 = CY_UM - HALF   # 1850.9 µm
ROI_Y1 = CY_UM + HALF   # 3975.9 µm

# ROI local integer origin (um -> integer bin, 1 um/bin)
ROI_IX0 = int(ROI_X0)   # 4674
ROI_IY0 = int(ROI_Y0)   # 1850
MAP_W   = int(ROI_X1) - ROI_IX0 + 1   # 2126
MAP_H   = int(ROI_Y1) - ROI_IY0 + 1   # 2126

# UCS parameters (same as CosMx)
PATCH_SIZE           = 48
DILATION_KERNEL_SIZE = 10
DILATION_ITER_NUM    = 4
TAU                  = 5
FG_NET_EPOCH         = 1
FG_NET_BATCH_SIZE    = 32
CELL_NET_EPOCH       = 1
MIN_QV               = 20

SEP = "═" * 60

print(f"\n{SEP}")
print(f"UCS — Xenium Human Breast Cancer")
print(f"ROI  : 5000x5000 px ({5000*PIXEL_SIZE:.0f}x{5000*PIXEL_SIZE:.0f} um, image center)")
print(f"Map  : {MAP_W}×{MAP_H} bins (1 µm/bin)")
print(f"GPU  : {GPU}")
print(f"{SEP}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Gene map (transcript um -> ROI local bin coordinates)
# ══════════════════════════════════════════════════════════════════════════════
def generate_gene_map():
    if os.path.exists(GENE_MAP_PATH):
        print(f"[1] gene_map already exists, loading directly: {GENE_MAP_PATH}")
        gm = tifffile.imread(GENE_MAP_PATH)
        print(f"[1] shape={gm.shape}  genes={gm.shape[2]}")
        return gm

    print(f"[1] loading transcripts (parquet)...")
    t0 = time.time()
    tx = pd.read_parquet(TX_PARQUET,
                         columns=["x_location","y_location","feature_name","qv","is_gene"])

    # filter: inside ROI + high quality + real genes
    tx = tx[
        (tx["is_gene"] == True) &
        (tx["qv"] >= MIN_QV) &
        (tx["x_location"] >= ROI_X0) & (tx["x_location"] <= ROI_X1) &
        (tx["y_location"] >= ROI_Y0) & (tx["y_location"] <= ROI_Y1)
    ].copy()
    print(f"[1] transcripts in ROI: {len(tx):,}  genes: {tx['feature_name'].nunique()}  "
          f"elapsed: {time.time()-t0:.1f}s")

    # all genes (natsort order, same as the CosMx script)
    import natsort
    gene_names = natsort.natsorted(tx["feature_name"].unique())
    n_genes    = len(gene_names)
    gene2idx   = {g: i for i, g in enumerate(gene_names)}

    # save the gene index
    pd.DataFrame({
        "gene":    gene_names,
        "channel": range(n_genes),
    }).to_csv(os.path.join(WORK_DIR, "gene_index.csv"), index=False)
    print(f"[1] genes: {n_genes}  gene_index.csv Saved")

    # convert coordinates: um -> ROI local bin
    tx["bx"] = (tx["x_location"].round().astype(int) - ROI_IX0).clip(0, MAP_W-1)
    tx["by"] = (tx["y_location"].round().astype(int) - ROI_IY0).clip(0, MAP_H-1)
    tx["gi"] = tx["feature_name"].map(gene2idx)

    # accumulate into gene_map [MAP_H, MAP_W, n_genes] uint8
    gene_map = np.zeros((MAP_H, MAP_W, n_genes), dtype=np.uint8)
    flat     = tx["by"].values * (MAP_W * n_genes) + \
               tx["bx"].values * n_genes + \
               tx["gi"].values.astype(int)
    np.add.at(gene_map.ravel(), flat, 1)
    gene_map = np.clip(gene_map, 0, 255).astype(np.uint8)

    mem_gb = gene_map.nbytes / 1e9
    print(f"[1] gene_map shape: {gene_map.shape}  {mem_gb:.2f} GB  "
          f"non-zero bins: {(gene_map>0).sum():,}")

    tifffile.imwrite(GENE_MAP_PATH, gene_map, photometric="minisblack")
    print(f"[1] Saved: {GENE_MAP_PATH}")
    return gene_map


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Nuclei mask (generated from nucleus_boundaries.parquet)
# ══════════════════════════════════════════════════════════════════════════════
def generate_nuclei_mask():
    if os.path.exists(NUCLEI_MASK_PATH):
        print(f"[2] nuclei_mask already exists, loading directly: {NUCLEI_MASK_PATH}")
        mask = tifffile.imread(NUCLEI_MASK_PATH)
        n = int((mask > 0).any(axis=-1).sum() if mask.ndim > 2 else (mask > 0).sum())
        print(f"[2] shape={mask.shape}  non-zero pixel fraction: {(mask>0).mean()*100:.1f}%")
        return mask

    print(f"[2] loading nucleus boundaries (parquet)...")
    t0 = time.time()
    nb = pd.read_parquet(NUCLEUS_BOUNDARY)
    print(f"[2] nucleus_boundaries columns: {list(nb.columns)}")
    print(f"[2] total boundary points: {len(nb):,}  elapsed: {time.time()-t0:.1f}s")

    # filter cells inside the ROI
    cells_in_roi = pd.read_parquet(
        os.path.join(DATA_DIR, "cells.parquet"),
        columns=["cell_id","x_centroid","y_centroid"]
    )
    cells_in_roi = cells_in_roi[
        (cells_in_roi["x_centroid"] >= ROI_X0) &
        (cells_in_roi["x_centroid"] <= ROI_X1) &
        (cells_in_roi["y_centroid"] >= ROI_Y0) &
        (cells_in_roi["y_centroid"] <= ROI_Y1)
    ]
    roi_cell_ids = set(cells_in_roi["cell_id"].values)
    print(f"[2] cells in ROI: {len(roi_cell_ids):,}")

    nb_roi = nb[nb["cell_id"].isin(roi_cell_ids)].copy()
    del nb
    print(f"[2] nucleus boundary points in ROI: {len(nb_roi):,}")

    # coordinate conversion: um -> ROI local bin (1 um/bin)
    nb_roi["lx"] = (nb_roi["vertex_x"].round().astype(int) - ROI_IX0).clip(0, MAP_W-1)
    nb_roi["ly"] = (nb_roi["vertex_y"].round().astype(int) - ROI_IY0).clip(0, MAP_H-1)

    nuclei_mask = np.zeros((MAP_H, MAP_W), dtype=np.int32)

    print(f"[2] drawing nucleus polygons ({len(roi_cell_ids):,} cells)...")
    cell_ids_list = sorted(roi_cell_ids)
    for mask_id, cell_id in enumerate(tqdm(cell_ids_list, ncols=70), start=1):
        poly_df = nb_roi[nb_roi["cell_id"] == cell_id]
        if len(poly_df) < 3:
            continue

        # handle multi-nuclei (duplicate vertices mark polygon boundaries)
        dup = poly_df[poly_df.duplicated(subset=["vertex_x","vertex_y"], keep=False)]
        poly_num = max(1, len(dup) // 2)

        if poly_num > 1:
            for pi in range(poly_num):
                idx_pair = (dup.index[pi*2], dup.index[pi*2+1])
                seg = poly_df.loc[idx_pair[0]:idx_pair[1]]
                pts = np.array(list(zip(seg["lx"].values, seg["ly"].values)),
                               dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(nuclei_mask, [pts], mask_id)
        else:
            pts = np.array(list(zip(poly_df["lx"].values, poly_df["ly"].values)),
                           dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(nuclei_mask, [pts], mask_id)

    n_cells = len(np.unique(nuclei_mask)) - 1
    print(f"[2] nucleus mask done: shape={nuclei_mask.shape}  cells={n_cells:,}")

    tifffile.imwrite(NUCLEI_MASK_PATH, nuclei_mask.astype(np.int32))
    print(f"[2] Saved: {NUCLEI_MASK_PATH}")
    return nuclei_mask


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: run UCS
# ══════════════════════════════════════════════════════════════════════════════
def run_ucs():
    run_py = os.path.join(UCS_DIR, "run.py")
    if not os.path.exists(run_py):
        raise FileNotFoundError(f"run.py not found: {run_py}")

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
        print(f"[3] cleared old log_dir: {LOG_DIR}")

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

    print(f"\n[3] running UCS (GPU {GPU}):")
    print("    " + " \\\n    ".join(cmd))
    result = subprocess.run(cmd, cwd=UCS_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"UCS exit code: {result.returncode}")
    print(f"\n[3] UCS done, results in: {LOG_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{SEP}")
    print("Step 1: generate Gene Map")
    print(SEP)
    gene_map = generate_gene_map()

    print(f"\n{SEP}")
    print("Step 2: generate Nuclei Mask")
    print(SEP)
    nuclei_mask = generate_nuclei_mask()

    assert gene_map.shape[:2] == nuclei_mask.shape, \
        f"gene_map {gene_map.shape[:2]} ≠ nuclei_mask {nuclei_mask.shape}"

    print(f"\nInput summary:")
    print(f"  gene_map    : {gene_map.shape}  → {GENE_MAP_PATH}")
    print(f"  nuclei_mask : {nuclei_mask.shape}  cells={nuclei_mask.max()}")
    print(f"  GPU         : {GPU}")

    print(f"\n{SEP}")
    print("Step 3: run UCS")
    print(SEP)
    run_ucs()

    print(f"\n{'='*60}")
    print(f"Done! Results in: {LOG_DIR}/pred/segmentation_mask.tif")
    print("="*60)


if __name__ == "__main__":
    main()