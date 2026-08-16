#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, time, shutil, argparse, subprocess
import multiprocessing as mp
import numpy as np
import pandas as pd
import h5py
import cv2
import tifffile
from tqdm import tqdm

CFG = {
    "liver1": {
        "base": "/data/qiuyijia/dataset/merfish_liver1",
        "roi_x0": 5236.737991920782,
        "roi_x1": 6316.729987066785,
        "roi_y0": 3811.048683925526,
        "roi_y1": 4891.037342034139,
        "work_dir": "/data/qiuyijia/ucs_merfish_liver1",
    },
    "liver2": {
        "base": "/data/qiuyijia/dataset/merfish_liver2",
        "roi_x0": 5771.0386869252825,
        "roi_x1": 6851.029458488514,
        "roi_y0": 4970.975379467495,
        "roi_y1": 6050.964816216321,
        "work_dir": "/data/qiuyijia/ucs_merfish_liver2",
    },
}

PATCH_SIZE = 48
DILATION_KERNEL_SIZE = 10
DILATION_ITER_NUM = 4
TAU = 5
FG_NET_EPOCH = 1
FG_NET_BATCH_SIZE = 32
CELL_NET_EPOCH = 1
Z_SLICE = 3

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
            print(f"GPU {idx}: {free_gb:.1f} GB free, util {util}%")
            if free_gb > best_free:
                best_free, best_idx = free_gb, int(idx)
        if best_free < min_free_gb:
            raise RuntimeError(f"没有 GPU 空闲超过 {min_free_gb} GB")
        return str(best_idx)
    except Exception:
        return "0"

def find_one(patterns):
    hits = []
    for p in patterns:
        hits.extend(glob.glob(p))
    hits = sorted(set(hits))
    if len(hits) != 1:
        raise RuntimeError(f"需要唯一匹配，但找到 {len(hits)} 个: {hits}")
    return hits[0]

def process_one_hdf5(args):
    hdf5_path, roi_x0, roi_x1, roi_y0, roi_y1, roi_ix0, roi_iy0, map_w, map_h, z_slice = args
    polys = []
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "featuredata" not in f:
                return polys
            for cell_id_str in f["featuredata"]:
                cell_grp = f["featuredata"][cell_id_str]
                z_key = f"zIndex_{z_slice}"
                if z_key not in cell_grp:
                    z_keys = [k for k in cell_grp.keys() if k.startswith("zIndex_")]
                    if not z_keys:
                        continue
                    z_key = sorted(z_keys)[0]

                z_group = cell_grp[z_key]
                for pk in z_group.keys():
                    if not isinstance(z_group[pk], h5py.Group):
                        continue
                    if "coordinates" not in z_group[pk]:
                        continue
                    poly_um = np.array(z_group[pk]["coordinates"])
                    if poly_um.ndim == 3 and poly_um.shape[0] == 1:
                        poly_um = poly_um[0]
                    if poly_um.ndim != 2 or poly_um.shape[1] != 2 or len(poly_um) < 3:
                        continue

                    cx = poly_um[:, 0].mean()
                    cy = poly_um[:, 1].mean()
                    if not (roi_x0 <= cx <= roi_x1 and roi_y0 <= cy <= roi_y1):
                        continue

                    bx = (poly_um[:, 0].round().astype(int) - roi_ix0).clip(0, map_w - 1)
                    by = (poly_um[:, 1].round().astype(int) - roi_iy0).clip(0, map_h - 1)
                    polys.append(np.stack([bx, by], axis=1).astype(np.int32))
                    break
    except Exception:
        pass
    return polys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["liver1", "liver2"])
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()

    cfg = CFG[args.dataset]
    base = cfg["base"]
    roi_x0, roi_x1 = cfg["roi_x0"], cfg["roi_x1"]
    roi_y0, roi_y1 = cfg["roi_y0"], cfg["roi_y1"]

    tx_csv = os.path.join(base, "detected_transcripts.csv")
    meta_csv = find_one([os.path.join(base, "*cell_metadata*.csv")])
    bd_dir = os.path.join(base, "cell_boundaries")

    work_dir = cfg["work_dir"]
    log_dir = os.path.join(work_dir, "ucs_log")
    os.makedirs(work_dir, exist_ok=True)

    gene_map_path = os.path.join(work_dir, "gene_map.tif")
    nuclei_mask_path = os.path.join(work_dir, "nuclei_mask.tif")
    gene_index_path = os.path.join(work_dir, "gene_index.csv")

    roi_ix0 = int(roi_x0)
    roi_iy0 = int(roi_y0)
    map_w = int(roi_x1) - roi_ix0 + 1
    map_h = int(roi_y1) - roi_iy0 + 1

    ucs_dir = os.path.dirname(os.path.abspath(__file__))
    run_py = os.path.join(ucs_dir, "run.py")

    gpu = args.gpu if args.gpu is not None else pick_best_gpu()

    print("=" * 80)
    print(f"UCS × {args.dataset}")
    print(f"ROI x=[{roi_x0:.3f},{roi_x1:.3f}] y=[{roi_y0:.3f},{roi_y1:.3f}]")
    print(f"Map = {map_w} x {map_h}")
    print(f"GPU = {gpu}")
    print("=" * 80)

    # Step 1: gene map
    if os.path.exists(gene_map_path):
        gene_map = tifffile.imread(gene_map_path)
        print(f"[1] reuse gene_map: {gene_map.shape}")
    else:
        chunks = []
        total = 0
        kept = 0
        for chunk in pd.read_csv(tx_csv, usecols=["global_x", "global_y", "gene"], chunksize=2_000_000):
            total += len(chunk)
            m = (
                (chunk.global_x >= roi_x0) & (chunk.global_x <= roi_x1) &
                (chunk.global_y >= roi_y0) & (chunk.global_y <= roi_y1)
            )
            sub = chunk[m].copy()
            kept += len(sub)
            if len(sub):
                chunks.append(sub)
            print(f"[1] read {total:,}, ROI kept {kept:,}")

        tx = pd.concat(chunks, ignore_index=True)
        genes = sorted(tx.gene.unique())
        gene2idx = {g: i for i, g in enumerate(genes)}
        pd.DataFrame({"gene": genes, "channel": range(len(genes))}).to_csv(gene_index_path, index=False)

        tx["bx"] = (tx.global_x.round().astype(int) - roi_ix0).clip(0, map_w - 1)
        tx["by"] = (tx.global_y.round().astype(int) - roi_iy0).clip(0, map_h - 1)
        tx["gi"] = tx.gene.map(gene2idx).astype(int)

        gene_map = np.zeros((map_h, map_w, len(genes)), dtype=np.uint8)
        flat = tx.by.values * (map_w * len(genes)) + tx.bx.values * len(genes) + tx.gi.values
        np.add.at(gene_map.ravel(), flat, 1)
        gene_map = np.clip(gene_map, 0, 255).astype(np.uint8)

        tifffile.imwrite(gene_map_path, gene_map, photometric="minisblack")
        print(f"[1] saved gene_map: {gene_map.shape} -> {gene_map_path}")

    # Step 2: nuclei mask
    if os.path.exists(nuclei_mask_path):
        nuclei_mask = tifffile.imread(nuclei_mask_path)
        print(f"[2] reuse nuclei_mask: {nuclei_mask.shape}, cells={int(nuclei_mask.max())}")
    else:
        meta = pd.read_csv(meta_csv)
        cx_col = "center_x" if "center_x" in meta.columns else [c for c in meta.columns if "center_x" in c][0]
        cy_col = "center_y" if "center_y" in meta.columns else [c for c in meta.columns if "center_y" in c][0]
        m = (
            (meta[cx_col] >= roi_x0) & (meta[cx_col] <= roi_x1) &
            (meta[cy_col] >= roi_y0) & (meta[cy_col] <= roi_y1)
        )
        print(f"[2] ROI GT cells (for reference): {int(m.sum()):,}")

        hdf5_files = sorted(glob.glob(os.path.join(bd_dir, "*.hdf5")))
        print(f"[2] reading {len(hdf5_files)} hdf5 files ...")

        args_list = [
            (p, roi_x0, roi_x1, roi_y0, roi_y1, roi_ix0, roi_iy0, map_w, map_h, Z_SLICE)
            for p in hdf5_files
        ]
        all_polys = []
        n_workers = min(16, mp.cpu_count())
        with mp.Pool(n_workers) as pool:
            for polys in tqdm(pool.imap_unordered(process_one_hdf5, args_list), total=len(args_list), ncols=70):
                all_polys.extend(polys)

        nuclei_mask = np.zeros((map_h, map_w), dtype=np.int32)
        for mask_id, pts in enumerate(all_polys, start=1):
            cv2.fillPoly(nuclei_mask, [pts.reshape(-1, 1, 2)], mask_id)

        tifffile.imwrite(nuclei_mask_path, nuclei_mask.astype(np.int32))
        print(f"[2] saved nuclei_mask: cells={int(nuclei_mask.max())} -> {nuclei_mask_path}")

    assert gene_map.shape[:2] == nuclei_mask.shape

    # Step 3: run UCS
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    cmd = [
        sys.executable, run_py,
        "--gene_map", gene_map_path,
        "--nuclei_mask", nuclei_mask_path,
        "--log_dir", log_dir,
        "--patch_size", str(PATCH_SIZE),
        "--dilation_kernel_size", str(DILATION_KERNEL_SIZE),
        "--dilation_iter_num", str(DILATION_ITER_NUM),
        "--tau", str(TAU),
        "--fg_net_epoch", str(FG_NET_EPOCH),
        "--fg_net_batch_size", str(FG_NET_BATCH_SIZE),
        "--cell_net_epoch", str(CELL_NET_EPOCH),
        "--gpu", str(gpu),
    ]

    print("[3] running UCS ...")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ucs_dir, check=True)

    print(f"\nDONE: {os.path.join(log_dir, 'pred', 'segmentation_mask.tif')}")

if __name__ == "__main__":
    main()