#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, argparse, subprocess
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
        "out_dir": "/data/qiuyijia/proseg/output/liver1",
    },
    "liver2": {
        "base": "/data/qiuyijia/dataset/merfish_liver2",
        "roi_x0": 5771.0386869252825,
        "roi_x1": 6851.029458488514,
        "roi_y0": 4970.975379467495,
        "roi_y1": 6050.964816216321,
        "out_dir": "/data/qiuyijia/proseg/output/liver2",
    },
}

Z_SLICE = 3

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
    parser.add_argument("--nthreads", type=int, default=16)
    args = parser.parse_args()

    cfg = CFG[args.dataset]
    base = cfg["base"]
    roi_x0, roi_x1 = cfg["roi_x0"], cfg["roi_x1"]
    roi_y0, roi_y1 = cfg["roi_y0"], cfg["roi_y1"]

    tx_csv = os.path.join(base, "detected_transcripts.csv")
    meta_csv = find_one([os.path.join(base, "*cell_metadata*.csv")])
    bd_dir = os.path.join(base, "cell_boundaries")
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    roi_ix0 = int(roi_x0)
    roi_iy0 = int(roi_y0)
    map_w = int(roi_x1) - roi_ix0 + 1
    map_h = int(roi_y1) - roi_iy0 + 1

    prior_mask_path = os.path.join(out_dir, "prior_mask.tif")
    prior_csv_path = os.path.join(out_dir, "transcripts_roi_with_prior.csv.gz")
    log_path = os.path.join(out_dir, "proseg.log")

    # Step 1: build prior mask from HDF5 boundaries
    if os.path.exists(prior_mask_path):
        prior_mask = tifffile.imread(prior_mask_path)
        print(f"[1] reuse prior_mask: {prior_mask.shape}, cells={int(prior_mask.max())}")
    else:
        hdf5_files = sorted(glob.glob(os.path.join(bd_dir, "*.hdf5")))
        args_list = [
            (p, roi_x0, roi_x1, roi_y0, roi_y1, roi_ix0, roi_iy0, map_w, map_h, Z_SLICE)
            for p in hdf5_files
        ]
        all_polys = []
        with mp.Pool(min(16, mp.cpu_count())) as pool:
            for polys in tqdm(pool.imap_unordered(process_one_hdf5, args_list), total=len(args_list), ncols=70):
                all_polys.extend(polys)

        prior_mask = np.zeros((map_h, map_w), dtype=np.int32)
        for mask_id, pts in enumerate(all_polys, start=1):
            cv2.fillPoly(prior_mask, [pts.reshape(-1, 1, 2)], mask_id)

        tifffile.imwrite(prior_mask_path, prior_mask.astype(np.int32))
        print(f"[1] saved prior_mask: cells={int(prior_mask.max())}")

    # optional reference metadata
    meta = pd.read_csv(meta_csv)
    if "center_x" in meta.columns and "center_y" in meta.columns:
        m = (
            (meta.center_x >= roi_x0) & (meta.center_x <= roi_x1) &
            (meta.center_y >= roi_y0) & (meta.center_y <= roi_y1)
        )
        meta[m].to_csv(os.path.join(out_dir, "cell_metadata_roi.csv"), index=False)

    # Step 2: create transcript CSV with cell_id prior
    if os.path.exists(prior_csv_path):
        print(f"[2] reuse transcript prior csv: {prior_csv_path}")
    else:
        chunks = []
        total = 0
        kept = 0
        for chunk in pd.read_csv(
            tx_csv,
            chunksize=1_000_000
        ):
            total += len(chunk)
            m = (
                (chunk.global_x >= roi_x0) & (chunk.global_x <= roi_x1) &
                (chunk.global_y >= roi_y0) & (chunk.global_y <= roi_y1)
            )
            sub = chunk[m].copy()
            if len(sub):
                sub["bx"] = (sub.global_x.round().astype(int) - roi_ix0).clip(0, map_w - 1)
                sub["by"] = (sub.global_y.round().astype(int) - roi_iy0).clip(0, map_h - 1)
                sub["cell_id"] = prior_mask[sub["by"].values, sub["bx"].values].astype(np.int32)
                if "barcode_id" not in sub.columns:
                    sub["barcode_id"] = np.arange(len(sub), dtype=np.int64)
                keep_cols = ["barcode_id", "global_x", "global_y", "global_z", "gene", "cell_id"]
                chunks.append(sub[keep_cols])
                kept += len(sub)
            print(f"[2] read {total:,}, ROI kept {kept:,}")

        tx_roi = pd.concat(chunks, ignore_index=True)
        tx_roi.to_csv(prior_csv_path, index=False, compression="gzip")
        assigned = int((tx_roi["cell_id"] > 0).sum())
        print(f"[2] saved transcript prior csv: {prior_csv_path}")
        print(f"[2] assigned = {assigned:,}/{len(tx_roi):,} = {assigned/len(tx_roi)*100:.2f}%")

    # Step 3: run proseg
    done = os.path.join(out_dir, "cell-metadata.csv.gz")
    cmd = [
        "proseg", "--merscope",
        "--use-cell-initialization",
        "--cell-id-column", "cell_id",
        "--cell-id-unassigned", "0",
        "--nthreads", str(args.nthreads),
        "--output-cell-metadata", os.path.join(out_dir, "cell-metadata.csv.gz"),
        "--output-transcript-metadata", os.path.join(out_dir, "transcript-metadata.csv.gz"),
        "--output-cell-polygons", os.path.join(out_dir, "cell-polygons.geojson.gz"),
        "--overwrite",
        prior_csv_path,
    ]

    print("[3] running proseg ...")
    print(" ".join(cmd))
    with open(log_path, "w") as log:
        result = subprocess.run(cmd, cwd=out_dir, stdout=log, stderr=log)

    if result.returncode != 0:
        raise RuntimeError(f"ProSeg failed, see {log_path}")

    print(f"DONE: {done}")

if __name__ == "__main__":
    main()