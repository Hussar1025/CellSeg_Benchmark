#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cellist × MERFISH liver — external multi-patch version
======================================================

目的：
    避免整个 2160×2160 liver ROI 被 Cellist 当成一个
    包含 5000+ nuclei 的超大 patch，从而在 segmentation
    阶段被 SIGKILL (rc=-9)。

方法保持不变：

    DAPI
      ↓
    Cellpose nuclei
      ↓
    Cellist

本脚本只是在进入 Cellist 之前，将同一个 ROI 切成规则 patch。

默认：
    3 × 3 = 9 patches

特点：
    1. 不使用其它分割方法的任何结果。
    2. 复用当前 Cellist 自己生成的：
           DAPI cache
           transcript cache
           Cellpose mask
    3. 每个 nucleus 只属于一个 patch。
    4. 每个 transcript 只属于一个 patch。
       因此最终不会产生 patch overlap / duplicate assignment。
    5. 每个 patch 串行运行 Cellist，减少峰值资源。
    6. 完成后自动把 Cellist segmentation 坐标平移回
       原始 2160×2160 Cellist ROI，并重编号 cell ID。

用法：

    conda activate cellist

    python cellist_merfish_liver_multipatch.py \
        --dataset liver2 \
        --diameter 14 \
        --cellprob 0 \
        --cell-radius 15 \
        --grid 3
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess

import numpy as np
import pandas as pd
import h5py

from scipy.sparse import csr_matrix


# ============================================================
# Dataset
# ============================================================

CFG = {

    "liver1": {

        "out":
            "/data/qiuyijia/cellist/merfish_liver1",

        "prefix":
            "merfish_liver1",
    },

    "liver2": {

        "out":
            "/data/qiuyijia/cellist/merfish_liver2",

        "prefix":
            "merfish_liver2",
    },
}


SEP = "=" * 76


def P(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ============================================================
# 10X H5 writer
# ============================================================

def write_10x_h5(
    path,
    matrix,
    genes,
    barcodes
):

    m = matrix.tocsc()

    with h5py.File(
        path,
        "w"
    ) as f:

        g = f.create_group(
            "matrix"
        )

        g.create_dataset(
            "data",
            data=m.data.astype(
                np.int32
            )
        )

        g.create_dataset(
            "indices",
            data=m.indices.astype(
                np.int64
            )
        )

        g.create_dataset(
            "indptr",
            data=m.indptr.astype(
                np.int64
            )
        )

        g.create_dataset(
            "shape",
            data=np.array(
                [
                    len(genes),
                    len(barcodes)
                ],
                dtype=np.int64
            )
        )

        g.create_dataset(
            "barcodes",
            data=np.array(
                [
                    str(x).encode()
                    for x in barcodes
                ]
            )
        )

        fg = g.create_group(
            "features"
        )

        gb = np.array(
            [
                str(gene).encode()
                for gene in genes
            ]
        )

        fg.create_dataset(
            "name",
            data=gb
        )

        fg.create_dataset(
            "id",
            data=gb
        )

        fg.create_dataset(
            "feature_type",
            data=np.array(
                [
                    b"Gene Expression"
                ]
                * len(genes)
            )
        )

        fg.create_dataset(
            "genome",
            data=np.array(
                [
                    b"unknown"
                ]
                * len(genes)
            )
        )


# ============================================================
# Locate Cellist cache
# ============================================================

def locate_cache(
    dataset,
    diameter,
    cellprob,
    resolution
):

    c = CFG[
        dataset
    ]

    ckpt = os.path.join(
        c["out"],
        "ckpt"
    )

    dapi = os.path.join(
        ckpt,
        f"dapi_r{resolution}.npy"
    )

    tx = os.path.join(
        ckpt,
        f"tx_r{resolution}.npz"
    )

    tag = (
        f"d{diameter}"
        f"_cp{cellprob}"
        f"_r{resolution}"
    ).replace(
        ".",
        ""
    )

    mask = os.path.join(
        ckpt,
        f"mask_{tag}.npy"
    )

    for p in [
        dapi,
        tx,
        mask
    ]:

        if not os.path.isfile(p):

            raise FileNotFoundError(
                "\n缺少 Cellist 自己的缓存：\n"
                f"{p}\n\n"
                "先用原 cellist_merfish_liver.py "
                "完成 DAPI / transcript / Cellpose preparation。"
            )

    return (
        dapi,
        tx,
        mask
    )


# ============================================================
# Build one tile input
# ============================================================

def build_tile_inputs(
    tile_dir,
    tile_prefix,
    mask_global,
    px_global,
    py_global,
    gi_global,
    genes,
    x0,
    x1,
    y0,
    y1
):

    os.makedirs(
        tile_dir,
        exist_ok=True
    )

    H = (
        y1 - y0
    )

    W = (
        x1 - x0
    )

    # --------------------------------------------------------
    # transcripts belonging uniquely to this tile
    # --------------------------------------------------------

    tm = (
        (px_global >= x0)
        &
        (px_global < x1)
        &
        (py_global >= y0)
        &
        (py_global < y1)
    )

    px = (
        px_global[tm]
        - x0
    ).astype(
        np.int32
    )

    py = (
        py_global[tm]
        - y0
    ).astype(
        np.int32
    )

    gi = (
        gi_global[tm]
        .astype(
            np.int32
        )
    )

    # --------------------------------------------------------
    # nuclei in this tile
    #
    # Because tiles are non-overlapping, every nucleus label
    # is assigned to exactly one tile based on pixels inside it.
    #
    # Nuclei crossing tile boundaries are truncated at the seam.
    # This is preferable here to duplicate cells/transcripts.
    # --------------------------------------------------------

    mask = np.ascontiguousarray(
        mask_global[
            y0:y1,
            x0:x1
        ]
    )

    old_ids = np.unique(
        mask
    )

    old_ids = old_ids[
        old_ids > 0
    ]

    # Relabel local nuclei to 1..N
    lut = {
        int(old):
            i + 1
        for i, old
        in enumerate(
            old_ids
        )
    }

    local_mask = np.zeros(
        mask.shape,
        dtype=np.int32
    )

    for old, new in lut.items():

        local_mask[
            mask == old
        ] = new

    n_nuc = int(
        local_mask.max()
    )

    if n_nuc == 0:

        return None

    # --------------------------------------------------------
    # nucleus properties
    # --------------------------------------------------------

    from skimage.measure import (
        regionprops_table
    )

    props = (
        regionprops_table(
            local_mask,
            properties=[
                "label",
                "area",
                "centroid",
                "equivalent_diameter_area"
            ]
        )
    )

    prop = pd.DataFrame(
        props
    )

    # Same Cellist coordinate convention as existing liver code:
    # swap skimage row/column centroid fields.
    tmp = (
        prop["centroid-0"]
        .copy()
    )

    prop[
        "centroid-0"
    ] = (
        prop["centroid-1"]
    )

    prop[
        "centroid-1"
    ] = tmp

    prop_path = os.path.join(
        tile_dir,
        f"{tile_prefix}_nucleus_prop.txt"
    )

    prop.to_csv(
        prop_path,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # unique transcript coordinates
    # --------------------------------------------------------

    key = (
        py.astype(
            np.int64
        )
        * W
        + px
    )

    unique_key = np.unique(
        key
    )

    sx = (
        unique_key
        % W
    ).astype(
        np.int32
    )

    sy = (
        unique_key
        // W
    ).astype(
        np.int32
    )

    nid = (
        local_mask[
            sy,
            sx
        ]
        .astype(float)
    )

    coord = pd.DataFrame(
        {
            "x":
                sx,

            "y":
                sy,

            "x_y":
                [
                    f"{a}_{b}"
                    for a,b
                    in zip(
                        sx,
                        sy
                    )
                ],

            "Nucleus":
                (
                    nid > 0
                ).astype(int),

            "Cellpose":
                np.where(
                    nid > 0,
                    nid,
                    np.nan
                ),
        }
    )

    coord_path = os.path.join(
        tile_dir,
        f"{tile_prefix}_nucleus_coord.txt"
    )

    coord.to_csv(
        coord_path,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # GEM
    # --------------------------------------------------------

    gem = pd.DataFrame(
        {
            "geneID":
                np.array(
                    genes,
                    dtype=object
                )[gi],

            "x":
                px,

            "y":
                py,
        }
    )

    gem = (
        gem
        .groupby(
            [
                "geneID",
                "x",
                "y"
            ]
        )
        .size()
        .reset_index(
            name="MIDCount"
        )
    )

    gem_path = os.path.join(
        tile_dir,
        f"{tile_prefix}_gem.txt"
    )

    gem.to_csv(
        gem_path,
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # spot_count.h5
    # --------------------------------------------------------

    spot_to_row = {
        k:i
        for i,k
        in enumerate(
            unique_key
        )
    }

    rows = (
        pd.Series(
            key
        )
        .map(
            spot_to_row
        )
        .values
    )

    spot_path = os.path.join(
        tile_dir,
        f"{tile_prefix}_spot_count.h5"
    )

    write_10x_h5(

        spot_path,

        csr_matrix(
            (
                np.ones(
                    len(px),
                    dtype=np.int32
                ),

                (
                    gi,
                    rows
                )
            ),

            shape=(
                len(genes),
                len(unique_key)
            )
        ),

        genes,

        [
            f"{a}_{b}"
            for a,b
            in zip(
                sx,
                sy
            )
        ]
    )

    # --------------------------------------------------------
    # nucleus_count.h5
    # --------------------------------------------------------

    transcript_nucleus = (
        local_mask[
            py,
            px
        ]
    )

    inside = (
        transcript_nucleus
        > 0
    )

    full = csr_matrix(
        (
            np.ones(
                int(
                    inside.sum()
                ),
                dtype=np.int32
            ),

            (
                gi[inside],
                transcript_nucleus[
                    inside
                ] - 1
            )
        ),

        shape=(
            len(genes),
            n_nuc
        )
    )

    gene_keep = np.where(
        np.asarray(
            full.sum(
                axis=1
            )
        ).ravel()
        > 0
    )[0]

    mat = (
        full[
            gene_keep,
            :
        ]
    )

    kept_genes = [
        genes[i]
        for i
        in gene_keep
    ]

    count_path = os.path.join(
        tile_dir,
        f"{tile_prefix}_nucleus_count.h5"
    )

    write_10x_h5(
        count_path,
        mat,
        kept_genes,
        [
            str(i)
            for i
            in range(
                1,
                n_nuc+1
            )
        ]
    )

    return {
        "gem":
            gem_path,

        "spot":
            spot_path,

        "coord":
            coord_path,

        "prop":
            prop_path,

        "count":
            count_path,

        "n_nuclei":
            n_nuc,

        "n_transcripts":
            len(px),

        "x0":
            x0,

        "y0":
            y0,

        "W":
            W,

        "H":
            H,

        "prefix":
            tile_prefix,

        "dir":
            tile_dir,
    }


# ============================================================
# Run one Cellist tile
# ============================================================

def run_tile(
    tile,
    resolution,
    cell_radius
):

    outdir = os.path.join(
        tile["dir"],
        "seg"
    )

    os.makedirs(
        outdir,
        exist_ok=True
    )

    cmd = [

        "cellist",
        "seg",

        "--platform",
        "imaging",

        "--resolution",
        str(
            resolution
        ),

        "--gem",
        tile["gem"],

        "--spot-count-h5",
        tile["spot"],

        "--nucleus-seg-method",
        "Cellpose",

        "--nucleus-seg",
        tile["coord"],

        "--nucleus-prop",
        tile["prop"],

        "--nucleus-count-h5",
        tile["count"],

        "--cell-radius",
        str(
            int(
                cell_radius
            )
        ),

        "--gene-use",
        "HVG",

        "--outdir",
        outdir,

        "--outprefix",
        tile["prefix"],
    ]

    env = os.environ.copy()

    # Cellist itself remains CPU-only as in previous liver runs.
    env[
        "CUDA_VISIBLE_DEVICES"
    ] = ""

    env[
        "OMP_NUM_THREADS"
    ] = "1"

    env[
        "MKL_NUM_THREADS"
    ] = "1"

    env[
        "OPENBLAS_NUM_THREADS"
    ] = "1"

    env[
        "NUMEXPR_NUM_THREADS"
    ] = "1"

    env[
        "TF_NUM_INTRAOP_THREADS"
    ] = "1"

    env[
        "TF_NUM_INTEROP_THREADS"
    ] = "1"

    env[
        "PYTHONFAULTHANDLER"
    ] = "1"

    P()
    P(
        "-" * 76
    )

    P(
        f"Run {tile['prefix']}"
    )

    P(
        f"  size        = "
        f"{tile['W']}×{tile['H']}"
    )

    P(
        f"  nuclei      = "
        f"{tile['n_nuclei']:,}"
    )

    P(
        f"  transcripts = "
        f"{tile['n_transcripts']:,}"
    )

    P(
        "-" * 76
    )

    rc = subprocess.run(
        cmd,
        cwd=tile["dir"],
        env=env
    ).returncode

    if rc != 0:

        raise RuntimeError(
            f"\nCellist tile failed:\n"
            f"  {tile['prefix']}\n"
            f"  return code={rc}"
        )

    return outdir


# ============================================================
# Merge Cellist coordinate outputs
# ============================================================

def merge_outputs(
    tiles,
    merged_dir,
    final_prefix
):

    os.makedirs(
        merged_dir,
        exist_ok=True
    )

    # Collect all Cellist segmentation txt file suffixes
    groups = {}

    for tile in tiles:

        segdir = os.path.join(
            tile["dir"],
            "seg"
        )

        files = glob.glob(
            os.path.join(
                segdir,
                "**",
                "*Cellist_segmentation*.txt"
            ),
            recursive=True
        )

        for f in files:

            base = os.path.basename(
                f
            )

            # strip tile prefix to obtain common suffix
            prefix = (
                tile["prefix"]
                + "_"
            )

            if base.startswith(
                prefix
            ):

                suffix = base[
                    len(prefix):
                ]

            else:

                suffix = base

            groups.setdefault(
                suffix,
                []
            ).append(
                (
                    tile,
                    f
                )
            )

    if not groups:

        raise RuntimeError(
            "没有找到 Cellist segmentation 输出"
        )

    P()
    P(SEP)
    P("Merge Cellist tile outputs")
    P(SEP)

    for suffix, items in groups.items():

        merged = []

        id_offset = 0

        for tile, path in items:

            df = pd.read_csv(
                path,
                sep="\t"
            )

            # local → complete ROI coordinates
            if "x" in df.columns:

                df["x"] = (
                    df["x"]
                    + tile["x0"]
                )

            if "y" in df.columns:

                df["y"] = (
                    df["y"]
                    + tile["y0"]
                )

            # Make Cellist IDs globally unique.
            if "Cellist" in df.columns:

                good = (
                    df["Cellist"]
                    .notna()
                )

                if good.any():

                    vals = (
                        df.loc[
                            good,
                            "Cellist"
                        ]
                        .astype(float)
                        .astype(
                            np.int64
                        )
                    )

                    uniq = sorted(
                        vals.unique()
                    )

                    remap = {
                        old:
                            id_offset
                            + i
                            + 1
                        for i, old
                        in enumerate(
                            uniq
                        )
                    }

                    df.loc[
                        good,
                        "Cellist"
                    ] = (
                        vals.map(
                            remap
                        ).values
                    )

                    id_offset += len(
                        uniq
                    )

            merged.append(
                df
            )

        outdf = pd.concat(
            merged,
            ignore_index=True
        )

        outpath = os.path.join(
            merged_dir,
            (
                f"{final_prefix}_"
                f"{suffix}"
            )
        )

        outdf.to_csv(
            outpath,
            sep="\t",
            index=False
        )

        P(
            f"  {suffix}"
            f" -> {outpath}"
            f"  rows={len(outdf):,}"
        )

    P()
    P(
        f"merged results:"
        f"\n  {merged_dir}"
    )


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        required=True,
        choices=[
            "liver1",
            "liver2"
        ]
    )

    ap.add_argument(
        "--resolution",
        type=float,
        default=0.5
    )

    ap.add_argument(
        "--diameter",
        type=int,
        default=14
    )

    ap.add_argument(
        "--cellprob",
        type=float,
        default=0.0
    )

    ap.add_argument(
        "--cell-radius",
        type=int,
        default=15
    )

    ap.add_argument(
        "--grid",
        type=int,
        default=3,
        help="N×N patch grid; default 3 -> 9 patches"
    )

    ap.add_argument(
        "--force",
        action="store_true"
    )

    args = ap.parse_args()

    c = CFG[
        args.dataset
    ]

    P(SEP)

    P(
        f"Cellist MULTIPATCH × "
        f"{args.dataset}"
    )

    P(
        f"grid = "
        f"{args.grid}×{args.grid}"
    )

    P(
        f"resolution = "
        f"{args.resolution}"
    )

    P(
        f"diameter = "
        f"{args.diameter}"
    )

    P(
        f"cellprob = "
        f"{args.cellprob}"
    )

    P(
        f"cell radius = "
        f"{args.cell_radius}"
    )

    P(SEP)

    # --------------------------------------------------------
    # Read Cellist's own caches
    # --------------------------------------------------------

    dapi_path, tx_path, mask_path = (
        locate_cache(
            args.dataset,
            args.diameter,
            args.cellprob,
            args.resolution
        )
    )

    dapi = np.load(
        dapi_path,
        mmap_mode="r"
    )

    mask = np.load(
        mask_path,
        mmap_mode="r"
    )

    tx = np.load(
        tx_path,
        allow_pickle=True
    )

    px = tx[
        "px"
    ].astype(
        np.int32
    )

    py = tx[
        "py"
    ].astype(
        np.int32
    )

    gi = tx[
        "gi"
    ].astype(
        np.int32
    )

    genes = list(
        tx[
            "genes"
        ]
    )

    H, W = (
        mask.shape
    )

    P(
        f"ROI working grid = "
        f"{W}×{H}"
    )

    P(
        f"Cellpose nuclei = "
        f"{int(mask.max()):,}"
    )

    P(
        f"transcripts = "
        f"{len(px):,}"
    )

    # --------------------------------------------------------
    # Tile boundaries
    # --------------------------------------------------------

    xs = np.linspace(
        0,
        W,
        args.grid + 1,
        dtype=int
    )

    ys = np.linspace(
        0,
        H,
        args.grid + 1,
        dtype=int
    )

    patch_root = os.path.join(
        c["out"],
        (
            f"multipatch_"
            f"{args.grid}x{args.grid}"
        )
    )

    os.makedirs(
        patch_root,
        exist_ok=True
    )

    tiles = []

    # --------------------------------------------------------
    # Build + run each tile serially
    # --------------------------------------------------------

    tile_index = 0

    for iy in range(
        args.grid
    ):

        for ix in range(
            args.grid
        ):

            x0 = int(
                xs[ix]
            )

            x1 = int(
                xs[ix+1]
            )

            y0 = int(
                ys[iy]
            )

            y1 = int(
                ys[iy+1]
            )

            tile_prefix = (
                f"{c['prefix']}"
                f"_p{tile_index:02d}"
            )

            tile_dir = os.path.join(
                patch_root,
                f"patch_{tile_index:02d}"
            )

            tile = build_tile_inputs(
                tile_dir,
                tile_prefix,
                mask,
                px,
                py,
                gi,
                genes,
                x0,
                x1,
                y0,
                y1
            )

            tile_index += 1

            if tile is None:

                P(
                    f"skip empty tile "
                    f"{tile_prefix}"
                )

                continue

            P(
                f"tile {tile['prefix']}: "
                f"{tile['W']}×{tile['H']}  "
                f"nuclei={tile['n_nuclei']:,}  "
                f"tx={tile['n_transcripts']:,}"
            )

            tiles.append(
                tile
            )

    if not tiles:

        raise RuntimeError(
            "没有有效 patch"
        )

    P()
    P(
        f"Total valid patches = "
        f"{len(tiles)}"
    )

    for i, tile in enumerate(
        tiles,
        1
    ):

        P()
        P(
            f"[{i}/{len(tiles)}]"
        )

        run_tile(
            tile,
            args.resolution,
            args.cell_radius
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged_dir = os.path.join(
        patch_root,
        "merged"
    )

    merge_outputs(
        tiles,
        merged_dir,
        c["prefix"]
    )

    P()
    P(SEP)

    P(
        f"DONE — Cellist multipatch "
        f"{args.dataset}"
    )

    P(
        f"patches = "
        f"{len(tiles)}"
    )

    P(
        f"merged = "
        f"{merged_dir}"
    )

    P(SEP)


if __name__ == "__main__":
    main()