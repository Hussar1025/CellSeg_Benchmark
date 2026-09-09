#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellist_xenium_5_1.py — Cellist × Xenium mouse brain (xenium_5), ROI 10000×10000 px

This is the xenium_5 counterpart of the validated xenium_3 imaging workflow.

Fixed ROI, matching the other xenium_5 methods:
    full-image pixel size = 0.2125 um/px
    ROI x0 = 12077 px
    ROI y0 =  6956 px
    ROI size = 10000 × 10000 px
    physical span = 2125 × 2125 um

Important Cellist requirement carried over from xenium_3:
    imaging mode STILL needs --gem.
    Cellist's Segmentation.py reads this TSV internally.

Stages
    check  : paths / RAM-relevant dimensions / ROI transcript + GT count only
    prep   : crop DAPI, build Cellpose nuclei and Cellist input files
    seg    : run `cellist seg` using already prepared files
    all    : check -> prep -> seg

All prep products are resumable. Existing valid files are reused unless --force.

Recommended first run:
    python3 cellist_xenium_5_1.py --stage check
then:
    python3 cellist_xenium_5_1.py --stage all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import skimage.measure
import tifffile

VERSION = "2026.08.22-xenium5-roi10000-v1"

ROOT_DEFAULT = Path("/data/qiuyijia/dataset/xenium_mouse_brain")
OUT_DEFAULT = Path("/data/qiuyijia/cellist/xenium_5_roi10000")
PREFIX_DEFAULT = "cellist_xenium_5"

PIXEL_SIZE = 0.2125
ROI_X0_PX = 12077
ROI_Y0_PX = 6956
ROI_SIZE_PX = 10000

DIAMETER_PX_DEFAULT = 32
CELL_RADIUS_DEFAULT = 10
QV_MIN_DEFAULT = 20.0


def log(msg=""):
    print(msg, flush=True)


def pick_best_gpu(min_free_gb=5.0):
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = []
        for line in out.strip().splitlines():
            idx, free_mb, util = [x.strip() for x in line.split(",")]
            rows.append((int(free_mb) / 1024.0, -int(util), int(idx)))
        rows.sort(reverse=True)
        free_gb, _, idx = rows[0]
        if free_gb < min_free_gb:
            log(f"WARNING: best GPU has only {free_gb:.1f} GB free")
        log(f"  -> GPU {idx} ({free_gb:.1f} GB free)")
        return str(idx)
    except Exception as e:
        log(f"WARNING: GPU auto-pick failed: {e}; use GPU 0")
        return "0"


def find_first(root: Path, candidates):
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    return None


def dataset_paths(root: Path):
    morphology = find_first(
        root,
        [
            "morphology.ome.tif",
            "morphology_focus/morphology_focus_0000.ome.tif",
            "outs/morphology.ome.tif",
            "outs/morphology_focus/morphology_focus_0000.ome.tif",
        ],
    )
    transcripts = find_first(root, ["transcripts.parquet", "outs/transcripts.parquet"])
    cells = find_first(root, ["cells.parquet", "outs/cells.parquet"])
    return morphology, transcripts, cells


def save_10x_h5(path, mat, spot_names, gene_names):
    mat_csc = mat.T.tocsc()
    with h5py.File(path, "w") as f:
        grp = f.create_group("matrix")
        grp.create_dataset(
            "barcodes", data=np.asarray([str(s).encode() for s in spot_names])
        )
        feat = grp.create_group("features")
        enc_gene = np.asarray([str(g).encode() for g in gene_names])
        feat.create_dataset("id", data=enc_gene)
        feat.create_dataset("name", data=enc_gene)
        feat.create_dataset(
            "feature_type",
            data=np.asarray([b"Gene Expression"] * len(gene_names)),
        )
        grp.create_dataset("data", data=mat_csc.data.astype(np.int32))
        grp.create_dataset("indices", data=mat_csc.indices.astype(np.int32))
        grp.create_dataset("indptr", data=mat_csc.indptr.astype(np.int32))
        grp.create_dataset(
            "shape",
            data=np.asarray([len(gene_names), len(spot_names)], dtype=np.int32),
        )


def normalize_u8(img):
    a = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        raise RuntimeError("morphology crop contains no finite pixels")
    lo, hi = np.percentile(a[finite], [1.0, 99.8])
    if hi <= lo:
        lo = float(np.min(a[finite]))
        hi = float(np.max(a[finite]))
    a = np.clip((a - lo) / max(hi - lo, 1e-8), 0, 1)
    return (a * 255).astype(np.uint8)


def select_2d(arr):
    """Reduce common Xenium morphology layouts to one 2-D plane."""
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # channel-first or channel-last: use the first morphology channel.
        if a.shape[0] <= 8:
            return a[0]
        if a.shape[-1] <= 8:
            return a[..., 0]
    raise RuntimeError(f"cannot reduce morphology array shape {a.shape} to 2-D")


def crop_morphology(path: Path, y0, x0, size):
    """
    Prefer tifffile-as-zarr so a tiled/pyramidal image is not fully materialised.
    Fall back to tifffile.imread for simple TIFFs.
    """
    try:
        import zarr

        store = tifffile.imread(str(path), aszarr=True)
        z = zarr.open(store, mode="r")

        # Some OME stores expose a multiscale group ("0", "1", ...).
        if hasattr(z, "keys"):
            keys = list(z.keys())
            if "0" in keys:
                z = z["0"]
            elif keys:
                z = z[keys[0]]

        shape = z.shape
        log(f"  morphology zarr shape={shape}")

        if len(shape) == 2:
            return np.asarray(z[y0:y0+size, x0:x0+size])

        if len(shape) == 3:
            if shape[0] <= 8:
                return np.asarray(z[0, y0:y0+size, x0:x0+size])
            if shape[-1] <= 8:
                return np.asarray(z[y0:y0+size, x0:x0+size, 0])

        # A few OME layouts contain leading singleton axes.
        idx = [0] * (len(shape) - 2) + [
            slice(y0, y0 + size),
            slice(x0, x0 + size),
        ]
        return select_2d(np.asarray(z[tuple(idx)]))
    except Exception as e:
        log(f"  aszarr crop unavailable ({type(e).__name__}: {e}); fallback full TIFF read")
        img = select_2d(tifffile.imread(str(path)))
        return img[y0:y0+size, x0:x0+size]


def read_roi_transcripts(path: Path, qv_min: float):
    x0_um = ROI_X0_PX * PIXEL_SIZE
    y0_um = ROI_Y0_PX * PIXEL_SIZE
    x1_um = (ROI_X0_PX + ROI_SIZE_PX) * PIXEL_SIZE
    y1_um = (ROI_Y0_PX + ROI_SIZE_PX) * PIXEL_SIZE

    cols = ["x_location", "y_location", "feature_name", "qv", "is_gene"]

    # pyarrow can predicate-push x/y/qv filters for parquet row groups.
    filters = [
        ("x_location", ">=", x0_um),
        ("x_location", "<", x1_um),
        ("y_location", ">=", y0_um),
        ("y_location", "<", y1_um),
    ]
    try:
        tx = pd.read_parquet(path, columns=cols, filters=filters)
    except Exception:
        tx = pd.read_parquet(path, columns=cols)
        tx = tx[
            (tx.x_location >= x0_um)
            & (tx.x_location < x1_um)
            & (tx.y_location >= y0_um)
            & (tx.y_location < y1_um)
        ]

    tx = tx[(tx["is_gene"] == True) & (tx["qv"] >= qv_min)].copy()
    tx.reset_index(drop=True, inplace=True)

    tx["px"] = np.floor(
        (tx["x_location"].to_numpy(float) - x0_um) / PIXEL_SIZE
    ).astype(np.int32)
    tx["py"] = np.floor(
        (tx["y_location"].to_numpy(float) - y0_um) / PIXEL_SIZE
    ).astype(np.int32)

    ok = (
        (tx.px >= 0)
        & (tx.px < ROI_SIZE_PX)
        & (tx.py >= 0)
        & (tx.py < ROI_SIZE_PX)
    )
    tx = tx[ok].reset_index(drop=True)
    return tx


def count_gt_cells(cells_path: Path | None):
    if cells_path is None:
        return None
    try:
        d = pd.read_parquet(cells_path)
        xc = next(
            (c for c in ["x_centroid", "x_location", "x"] if c in d.columns), None
        )
        yc = next(
            (c for c in ["y_centroid", "y_location", "y"] if c in d.columns), None
        )
        if xc is None or yc is None:
            return None
        x0 = ROI_X0_PX * PIXEL_SIZE
        y0 = ROI_Y0_PX * PIXEL_SIZE
        x1 = (ROI_X0_PX + ROI_SIZE_PX) * PIXEL_SIZE
        y1 = (ROI_Y0_PX + ROI_SIZE_PX) * PIXEL_SIZE
        q = d[
            (pd.to_numeric(d[xc], errors="coerce") >= x0)
            & (pd.to_numeric(d[xc], errors="coerce") < x1)
            & (pd.to_numeric(d[yc], errors="coerce") >= y0)
            & (pd.to_numeric(d[yc], errors="coerce") < y1)
        ]
        return int(len(q))
    except Exception:
        return None


def check_stage(root, out, qv_min):
    morphology, transcripts, cells = dataset_paths(root)
    log("=" * 78)
    log("Cellist @ xenium_5 preflight")
    log("=" * 78)
    log(f"root       {root}")
    log(f"morphology {morphology}")
    log(f"transcripts {transcripts}")
    log(f"cells      {cells}")
    log(
        f"ROI px     x[{ROI_X0_PX},{ROI_X0_PX+ROI_SIZE_PX}) "
        f"y[{ROI_Y0_PX},{ROI_Y0_PX+ROI_SIZE_PX})"
    )
    log(
        f"ROI um     {ROI_SIZE_PX*PIXEL_SIZE:.1f} x "
        f"{ROI_SIZE_PX*PIXEL_SIZE:.1f}"
    )

    if morphology is None or transcripts is None:
        raise SystemExit("missing morphology or transcripts")

    tx = read_roi_transcripts(transcripts, qv_min)
    gt = count_gt_cells(cells)
    log(f"ROI transcripts qv>={qv_min:g}: {len(tx):,}")
    log(f"genes: {tx.feature_name.nunique():,}")
    if gt:
        log(f"ROI GT cells: {gt:,}; tx/cell={len(tx)/gt:.1f}")
    else:
        log("ROI GT cells: unavailable from cells.parquet")

    # Array-size estimates; actual Cellpose overhead is larger.
    img_gb = ROI_SIZE_PX**2 * 1 / 1e9
    mask_gb = ROI_SIZE_PX**2 * 4 / 1e9
    log(f"raw u8 ROI image ~{img_gb:.2f} GB; int32 mask ~{mask_gb:.2f} GB")
    log(
        "Recommendation: keep at least ~20 GB host RAM free and >=8 GB GPU memory "
        "free before starting Cellpose on the 10000x10000 ROI."
    )
    return morphology, transcripts, cells, tx, gt


def prep_stage(args, root: Path, out: Path, prefix: str):
    morphology, transcripts, cells, tx, gt = check_stage(root, out, args.qv_min)
    out.mkdir(parents=True, exist_ok=True)

    meta_path = out / "run_config.json"
    meta = dict(
        version=VERSION,
        dataset="xenium_5",
        root=str(root),
        morphology=str(morphology),
        transcripts=str(transcripts),
        cells=str(cells) if cells else None,
        pixel_size=PIXEL_SIZE,
        roi_x0_px=ROI_X0_PX,
        roi_y0_px=ROI_Y0_PX,
        roi_size_px=ROI_SIZE_PX,
        qv_min=args.qv_min,
        diameter_px=args.diameter_px,
        cell_radius=args.cell_radius,
        gt_cells=gt,
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # DAPI/morphology crop
    dapi_u8 = out / f"{prefix}_dapi_u8.tif"
    if args.force or not dapi_u8.exists():
        log("\n[1] crop morphology ROI -> uint8")
        crop = crop_morphology(
            morphology, ROI_Y0_PX, ROI_X0_PX, ROI_SIZE_PX
        )
        if crop.shape != (ROI_SIZE_PX, ROI_SIZE_PX):
            raise RuntimeError(
                f"morphology ROI shape {crop.shape}, expected "
                f"{(ROI_SIZE_PX, ROI_SIZE_PX)}"
            )
        tifffile.imwrite(dapi_u8, normalize_u8(crop))
        del crop
        log(f"  -> {dapi_u8}")
    else:
        log(f"\n[1] reuse {dapi_u8}")

    dapi = tifffile.imread(dapi_u8)
    H, W = dapi.shape
    if (H, W) != (ROI_SIZE_PX, ROI_SIZE_PX):
        raise RuntimeError(f"cached DAPI has wrong shape {(H,W)}")

    gene_names = np.sort(tx.feature_name.astype(str).unique())
    g2i = {g: i for i, g in enumerate(gene_names)}
    tx["gene_idx"] = tx.feature_name.astype(str).map(g2i).astype(np.int32)
    ng = len(gene_names)

    # Cellpose nucleus files
    seg_path = out / f"{prefix}_nucleus_seg.h5"
    prop_path = out / f"{prefix}_nucleus_prop.txt"
    coord_path = out / f"{prefix}_nucleus_coord.txt"
    cnt_path = out / f"{prefix}_nucleus_count.h5"

    have_nuc = all(p.exists() for p in [seg_path, prop_path, coord_path, cnt_path])
    if have_nuc and not args.force:
        log("\n[2] reuse Cellpose nucleus files")
        with h5py.File(seg_path, "r") as f:
            masks = f["seg"][:]
    else:
        log(
            f"\n[2] Cellpose nuclei: diameter={args.diameter_px}px, "
            f"flow={args.flow_threshold}, cellprob={args.cellprob}"
        )
        from cellpose import models

        t0 = time.time()
        try:
            model = models.CellposeModel(model_type="nuclei", gpu=True)
            ret = model.eval(
                dapi,
                diameter=args.diameter_px,
                channels=[0, 0],
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob,
            )
            masks = ret[0]
        except TypeError:
            # Cellpose 2.x compatibility
            model = models.Cellpose(model_type="nuclei", gpu=True)
            masks, _, _, _ = model.eval(
                dapi,
                diameter=args.diameter_px,
                channels=[0, 0],
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob,
            )

        masks = np.asarray(masks, dtype=np.int32)
        n_nuc = int(masks.max())
        log(f"  Cellpose: {n_nuc:,} nuclei ({time.time()-t0:.0f}s)")

        with h5py.File(seg_path, "w") as f:
            f.create_dataset(
                "seg",
                data=masks,
                compression="gzip",
                compression_opts=1,
                chunks=(512, 512),
            )

        props = skimage.measure.regionprops_table(
            masks,
            properties=[
                "label",
                "area",
                "centroid",
                "equivalent_diameter_area",
            ],
        )
        pd.DataFrame(props).to_csv(prop_path, sep="\t", index=False)

        # Cellist coordinate file is one row per unique occupied transcript pixel.
        tx_dedup = tx.drop_duplicates(subset=["px", "py"])
        sx = tx_dedup.px.to_numpy(np.int32)
        sy = tx_dedup.py.to_numpy(np.int32)
        nuc_id = masks[sy, sx].astype(float)
        nuc_cp = nuc_id.copy()
        nuc_cp[nuc_id == 0] = np.nan
        pd.DataFrame(
            {
                "x_y": [f"{x}_{y}" for x, y in zip(sx, sy)],
                "x": sx,
                "y": sy,
                "Nucleus": (nuc_id > 0).astype(float),
                "Cellpose": nuc_cp,
            }
        ).to_csv(coord_path, sep="\t", index=False)

        cell_of_tx = masks[
            tx.py.to_numpy(np.int32),
            tx.px.to_numpy(np.int32),
        ]
        in_nuc = cell_of_tx > 0
        mat_nuc = sp.csr_matrix(
            (
                np.ones(in_nuc.sum(), dtype=np.float32),
                (
                    (cell_of_tx[in_nuc] - 1).astype(np.int32),
                    tx.gene_idx.to_numpy(np.int32)[in_nuc],
                ),
            ),
            shape=(n_nuc, ng),
        )
        save_10x_h5(
            cnt_path,
            mat_nuc,
            [str(i + 1) for i in range(n_nuc)],
            gene_names,
        )
        log("  nucleus files written")

    # spot_count.h5
    spot_h5 = out / f"{prefix}_spot_count.h5"
    if args.force or not spot_h5.exists():
        log("\n[3] build spot_count.h5")
        tx["spot_id"] = (
            tx.py.to_numpy(np.int64) * int(W)
            + tx.px.to_numpy(np.int64)
        )
        spot_ids, row = np.unique(tx.spot_id.to_numpy(np.int64), return_inverse=True)
        mat_spots = sp.csr_matrix(
            (
                np.ones(len(tx), dtype=np.float32),
                (row.astype(np.int32), tx.gene_idx.to_numpy(np.int32)),
            ),
            shape=(len(spot_ids), ng),
        )
        sy = (spot_ids // W).astype(int)
        sx = (spot_ids % W).astype(int)
        save_10x_h5(
            spot_h5,
            mat_spots,
            [f"{x}_{y}" for x, y in zip(sx, sy)],
            gene_names,
        )
        log(f"  -> {spot_h5}; shape={mat_spots.shape}")
    else:
        log(f"\n[3] reuse {spot_h5}")

    # imaging mode still needs GEM
    gem_path = out / f"{prefix}_gem.txt"
    if args.force or not gem_path.exists():
        log("\n[4] write GEM TSV required by Cellist imaging mode")
        gem_df = tx[["feature_name", "px", "py"]].copy()
        gem_df.columns = ["geneID", "x", "y"]
        gem_df["MIDCount"] = 1
        gem_df.to_csv(gem_path, sep="\t", index=False)
        log(f"  -> {gem_path}; {len(gem_df):,} rows")
    else:
        log(f"\n[4] reuse {gem_path}")

    log("\nPREP DONE")


def seg_stage(args, out: Path, prefix: str):
    needed = {
        "gem": out / f"{prefix}_gem.txt",
        "spot": out / f"{prefix}_spot_count.h5",
        "coord": out / f"{prefix}_nucleus_coord.txt",
        "prop": out / f"{prefix}_nucleus_prop.txt",
        "nuc_count": out / f"{prefix}_nucleus_count.h5",
    }
    missing = [str(p) for p in needed.values() if not p.exists()]
    if missing:
        raise SystemExit("missing prep files:\n" + "\n".join(missing))

    log("\n[5] cellist seg")
    cmd = [
        "cellist",
        "seg",
        "--platform",
        "imaging",
        "--resolution",
        str(PIXEL_SIZE),
        "--gem",
        str(needed["gem"]),
        "--spot-count-h5",
        str(needed["spot"]),
        "--nucleus-seg-method",
        "Cellpose",
        "--nucleus-seg",
        str(needed["coord"]),
        "--nucleus-prop",
        str(needed["prop"]),
        "--nucleus-count-h5",
        str(needed["nuc_count"]),
        "--cell-radius",
        str(args.cell_radius),
        "--gene-use",
        "HVG",
        "--outdir",
        str(out),
        "--outprefix",
        prefix,
    ]
    log("  " + " ".join(cmd))

    t0 = time.time()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        if not any(
            k in line
            for k in ("fastpd", "oneDNN", "I0000")
        ):
            print(line.rstrip(), flush=True)
    rc = p.wait()
    if rc != 0:
        raise SystemExit(f"cellist seg failed with return code {rc}")

    log(f"  Cellist finished in {(time.time()-t0)/60:.1f} min")

    hits = sorted(glob.glob(str(out / "alpha*" / "*cell_coord.txt")))
    if not hits:
        log("WARNING: no alpha*/cell_coord output found")
        return
    for f in hits:
        try:
            d = pd.read_csv(f, sep="\t")
            log(f"  output {f}: {len(d):,} rows/cells")
        except Exception:
            log(f"  output {f}")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--stage", choices=["check", "prep", "seg", "all"], default="check")
    ap.add_argument("--root", default=str(ROOT_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--prefix", default=PREFIX_DEFAULT)
    ap.add_argument("--qv-min", type=float, default=QV_MIN_DEFAULT)
    ap.add_argument("--diameter-px", type=float, default=DIAMETER_PX_DEFAULT)
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--cell-radius", type=int, default=CELL_RADIUS_DEFAULT)
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args()

    if args.version:
        print(VERSION)
        return

    root = Path(args.root)
    out = Path(args.out)

    if args.gpu == "auto":
        gpu = pick_best_gpu()
    else:
        gpu = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    log("=" * 78)
    log(f"Cellist @ xenium_5 mouse brain | stage={args.stage} | GPU={gpu}")
    log("=" * 78)

    if args.stage == "check":
        check_stage(root, out, args.qv_min)
        return
    if args.stage == "prep":
        prep_stage(args, root, out, args.prefix)
        return
    if args.stage == "seg":
        seg_stage(args, out, args.prefix)
        return

    check_stage(root, out, args.qv_min)
    prep_stage(args, root, out, args.prefix)
    seg_stage(args, out, args.prefix)


if __name__ == "__main__":
    main()
