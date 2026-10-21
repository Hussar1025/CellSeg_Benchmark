#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, csv, random
import numpy as np
import pandas as pd
import tifffile
import cv2
from scipy.ndimage import gaussian_filter

IMG = "yourpath/dataset/stereoseq/stereo-seq_data_all/E12.5_E1S3.tif"
GEM = "yourpath/dataset/stereoseq/stereo-seq_data_all/E12.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz"

OUT_BASE = "yourpath/genesegnet_e12_5_train"
TRAIN_ROOT = os.path.join(OUT_BASE, "train", "e12_5")
VAL_ROOT   = os.path.join(OUT_BASE, "val",   "e12_5")

PATCH = 256
N_TRAIN = 1000
N_VAL = 200
SEED = 1
MARGIN = 64

def ensure_dirs(root):
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels"), exist_ok=True)
    os.makedirs(os.path.join(root, "spots"), exist_ok=True)
    os.makedirs(os.path.join(root, "HeatMaps", "HeatMap_all"), exist_ok=True)

def save_patch(root, idx, img_patch, label_patch, spots_patch):
    name = f"{idx:05d}"

    # image: save as RGB jpg
    img_u8 = img_patch.astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(root, "images", f"{name}_image.jpg"), img_rgb)

    # label
    cv2.imwrite(os.path.join(root, "labels", f"{name}_label.png"), label_patch.astype(np.uint16))

    # spots csv
    with open(os.path.join(root, "spots", f"{name}_spot.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spotX", "spotY", "gene"])
        for x, y, g in spots_patch:
            w.writerow([int(x), int(y), g])

    # gaumap_all
    hm = np.zeros((PATCH, PATCH), dtype=np.float32)
    for x, y, _ in spots_patch:
        x = int(x)
        y = int(y)
        if 0 <= x < PATCH and 0 <= y < PATCH:
            hm[y, x] += 1.0
    hm = gaussian_filter(hm, sigma=7)
    if hm.max() > 0:
        hm = hm / hm.max() * 255.0
    hm = hm.astype(np.uint8)
    cv2.imwrite(os.path.join(root, "HeatMaps", "HeatMap_all", f"{name}_gaumap_all.jpg"), hm)

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    ensure_dirs(TRAIN_ROOT)
    ensure_dirs(VAL_ROOT)

    print("[1] reading image ...")
    img = tifffile.imread(IMG)
    print("image shape =", img.shape, img.dtype)

    print("[2] reading gem ...")
    g = pd.read_csv(GEM, sep="\t", usecols=["geneID", "x", "y", "MIDCounts", "cell"])
    print("rows =", len(g))

    # Verified: for E12.5 the alignment is row=x, col=y
    rows = g["x"].astype(int).values
    cols = g["y"].astype(int).values

    r0 = max(0, rows.min() - MARGIN)
    r1 = min(img.shape[0], rows.max() + MARGIN + 1)
    c0 = max(0, cols.min() - MARGIN)
    c1 = min(img.shape[1], cols.max() + MARGIN + 1)

    print(f"[3] crop image to rows [{r0},{r1}), cols [{c0},{c1})")
    img_crop = img[r0:r1, c0:c1]

    # normalize image to 0..255
    p1, p99 = np.percentile(img_crop, 1), np.percentile(img_crop, 99)
    img_crop = np.clip((img_crop - p1) / max(p99 - p1, 1e-6), 0, 1)
    img_crop = (img_crop * 255).astype(np.uint8)

    # local coords in cropped image
    g = g.copy()
    g["row"] = g["x"].astype(int) - r0
    g["col"] = g["y"].astype(int) - c0

    H, W = img_crop.shape
    label = np.zeros((H, W), dtype=np.int32)

    print("[4] building dense-ish label from CellBin cells ...")
    cell_ids = sorted([int(c) for c in g["cell"].unique() if int(c) > 0])
    remap = {cid: i + 1 for i, cid in enumerate(cell_ids)}

    for cid in cell_ids:
        sub = g[g["cell"] == cid][["row", "col"]].drop_duplicates()
        if len(sub) == 0:
            continue
        pts = sub[["col", "row"]].values.astype(np.int32)  # cv2 uses (x=col, y=row)

        new_id = remap[cid]

        if len(pts) >= 3:
            hull = cv2.convexHull(pts.reshape(-1, 1, 2))
            cv2.fillConvexPoly(label, hull, new_id)
        elif len(pts) == 2:
            cv2.line(label, tuple(pts[0]), tuple(pts[1]), int(new_id), 1)
        else:
            x, y = pts[0]
            if 0 <= y < H and 0 <= x < W:
                label[y, x] = new_id

    print("label shape =", label.shape, "cells =", int(label.max()))

    # spots table for patch extraction
    spots = g[["col", "row", "geneID"]].copy()
    spots.columns = ["x", "y", "gene"]

    print("[5] sampling patches ...")
    n_saved_train = 0
    n_saved_val = 0
    tried = 0
    max_try = 200000

    while (n_saved_train < N_TRAIN or n_saved_val < N_VAL) and tried < max_try:
        tried += 1
        y0 = random.randint(0, H - PATCH)
        x0 = random.randint(0, W - PATCH)

        label_patch = label[y0:y0+PATCH, x0:x0+PATCH]
        img_patch = img_crop[y0:y0+PATCH, x0:x0+PATCH]

        sub = spots[
            (spots.x >= x0) & (spots.x < x0 + PATCH) &
            (spots.y >= y0) & (spots.y < y0 + PATCH)
        ].copy()

        if len(sub) < 50:
            continue

        uniq_cells = np.unique(label_patch)
        uniq_cells = uniq_cells[uniq_cells > 0]
        if len(uniq_cells) < 5:
            continue

        sub["x"] = sub["x"] - x0
        sub["y"] = sub["y"] - y0
        spots_patch = sub[["x", "y", "gene"]].values.tolist()

        if n_saved_train < N_TRAIN:
            save_patch(TRAIN_ROOT, n_saved_train, img_patch, label_patch, spots_patch)
            n_saved_train += 1
            if n_saved_train % 50 == 0:
                print("train =", n_saved_train)
        elif n_saved_val < N_VAL:
            save_patch(VAL_ROOT, n_saved_val, img_patch, label_patch, spots_patch)
            n_saved_val += 1
            if n_saved_val % 20 == 0:
                print("val =", n_saved_val)

    print("[DONE]")
    print("train patches =", n_saved_train)
    print("val patches   =", n_saved_val)

if __name__ == "__main__":
    main()