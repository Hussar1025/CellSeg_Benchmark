#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GeneSegNet inference-only for Xenium8.

Prepared input:
  /data/qiuyijia/genesegnet_xenium_8_infer/tiles/images
  /data/qiuyijia/genesegnet_xenium_8_infer/tiles/heatmaps
  /data/qiuyijia/genesegnet_xenium_8_infer/tile_manifest.tsv

Model:
  existing Xenium GeneSegNet checkpoint

Pipeline:
  prepared image + transcript heatmap
      -> GeneSegModel.eval()
      -> tile masks
      -> overlap-aware stitch
      -> 10000 x 10000 final mask

NO GT is used during inference.
"""

from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import tifffile


HERE = Path("/data/qiuyijia/GeneSegNet")

GENESEG_REPO = HERE / "GeneSegNet"

HELPER_PATH = (
    HERE /
    "genesegnet_cosmx_2.py"
)

PREP_ROOT = Path(
    "/data/qiuyijia/"
    "genesegnet_xenium_8_infer"
)

DEFAULT_OUT = Path(
    "/data/qiuyijia/"
    "genesegnet_xenium_8_prediction"
)

DEFAULT_MODEL = Path(
    "/data/qiuyijia/.GeneSegNet/"
    "xenium_pancreas_cancer/model/models/"
    "GeneSegNet_residual_on_style_off_"
    "concatenation_off_model_"
    "2026_10_01_23_44_48.593114"
)

FOV_SIZE = 10000

PIXEL_SIZE = 0.2125


def banner(s):
    print()
    print("=" * 100)
    print(s)
    print("=" * 100, flush=True)


def load_helper():

    spec = importlib.util.spec_from_file_location(
        "genesegnet_cosmx_helper",
        HELPER_PATH
    )

    mod = importlib.util.module_from_spec(
        spec
    )

    # Required for @dataclass and other runtime type inspection
    # inside dynamically imported genesegnet_cosmx_2.py.
    sys.modules[spec.name] = mod

    spec.loader.exec_module(
        mod
    )

    return mod


def load_repo_modules():

    sys.path.insert(
        0,
        str(GENESEG_REPO)
    )

    import models
    import utils

    try:
        import fastremap
    except ImportError:
        raise RuntimeError(
            "fastremap is not installed "
            "in genesegnet_env"
        )

    return models, utils, fastremap


def ensure_hwc1(a):

    a = np.asarray(a)

    if (
        a.ndim == 2
    ):
        a = a[..., None]

    if (
        a.ndim == 3
        and
        a.shape[0] == 1
        and
        a.shape[-1] != 1
    ):
        a = np.moveaxis(
            a,
            0,
            -1
        )

    if (
        a.ndim != 3
        or
        a.shape[-1] != 1
    ):
        raise RuntimeError(
            f"expected HxWx1, got "
            f"{a.shape}"
        )

    return a


def load_input(image_path, heat_path):

    im = ensure_hwc1(
        tifffile.imread(
            image_path
        )
    )

    hm = ensure_hwc1(
        tifffile.imread(
            heat_path
        )
    )

    if (
        im.shape[:2]
        !=
        hm.shape[:2]
    ):
        raise RuntimeError(
            f"image/heat spatial mismatch: "
            f"{im.shape} vs {hm.shape}"
        )

    # Same GeneSegNet concept used in the trained pipelines:
    # channel 0 = morphology
    # channel 1 = transcript heatmap
    x = np.concatenate(
        [
            im.astype(
                np.float32
            ),
            hm.astype(
                np.float32
            )
        ],
        axis=-1
    )

    # GeneSegNet/Cellpose normalization normally expects
    # intensity-type input; keep the same 0..255 range
    # produced during prep.
    return x


def mask_qc(mask):

    mask = np.asarray(mask)

    labs,counts = np.unique(
        mask,
        return_counts=True
    )

    keep = labs > 0

    labs = labs[keep]
    counts = counts[keep]

    if len(counts):

        med_area = float(
            np.median(counts)
        )

        med_diam = float(
            2.0
            * np.sqrt(
                med_area
                / np.pi
            )
            * PIXEL_SIZE
        )

    else:

        med_area = np.nan
        med_diam = np.nan

    return {
        "cells": int(len(labs)),
        "foreground_fraction":
            float(
                np.mean(
                    mask > 0
                )
            ),
        "median_area_px":
            med_area,
        "median_diameter_um":
            med_diam,
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default=str(
            DEFAULT_MODEL
        )
    )

    ap.add_argument(
        "--out",
        default=str(
            DEFAULT_OUT
        )
    )

    ap.add_argument(
        "--gpu",
        type=int,
        default=0
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    # 0 = checkpoint diam_labels
    ap.add_argument(
        "--diameter",
        type=float,
        default=0.0
    )

    ap.add_argument(
        "--diam-mean",
        type=float,
        default=34.0
    )

    ap.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4
    )

    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9
    )

    ap.add_argument(
        "--min-size",
        type=int,
        default=300
    )

    ap.add_argument(
        "--network-tile-overlap",
        type=float,
        default=0.1
    )

    ap.add_argument(
        "--stitch-min-pixels",
        type=int,
        default=20
    )

    ap.add_argument(
        "--stitch-fraction",
        type=float,
        default=0.5
    )

    ap.add_argument(
        "--residual-off",
        action="store_true"
    )

    ap.add_argument(
        "--style-on",
        action="store_true"
    )

    ap.add_argument(
        "--concatenation",
        action="store_true"
    )

    ap.add_argument(
        "--resume",
        action="store_true"
    )

    ap.add_argument(
        "--dry",
        action="store_true"
    )

    args = ap.parse_args()

    model_path = Path(
        args.model
    )

    out = Path(
        args.out
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    if not HELPER_PATH.exists():
        raise FileNotFoundError(
            HELPER_PATH
        )

    manifest_path = (
        PREP_ROOT /
        "prepared/tile_manifest.tsv"
    )

    manifest = pd.read_csv(
        manifest_path,
        sep="\t"
    )

    if len(manifest) != 144:
        raise RuntimeError(
            f"expected 144 tiles, "
            f"got {len(manifest)}"
        )

    image_dir = (
        PREP_ROOT /
        "prepared/images"
    )

    heat_dir = (
        PREP_ROOT /
        "prepared/heatmaps"
    )

    tile_out = (
        out /
        "tile_masks"
    )

    tile_out.mkdir(
        parents=True,
        exist_ok=True
    )

    banner(
        "GENESEGNET XENIUM8 INFERENCE"
    )

    print(
        "model =",
        model_path
    )

    print(
        "tiles =",
        len(manifest)
    )

    print(
        "FOV =",
        FOV_SIZE,
        "x",
        FOV_SIZE
    )

    print(
        "pixel size =",
        PIXEL_SIZE
    )

    print(
        "GPU arg =",
        args.gpu
    )

    print(
        "diameter arg =",
        args.diameter
    )

    print(
        "uses GT = False"
    )

    helper = load_helper()

    models,utils,fastremap = (
        load_repo_modules()
    )

    # Audit first tile before building model.
    r0 = manifest.iloc[0]

    k0 = str(
        r0["name"]
    )

    ip0 = (
        image_dir /
        f"{k0}_image.tif"
    )

    hp0 = (
        heat_dir /
        f"{k0}_gaumap_all.tif"
    )

    x0 = load_input(
        ip0,
        hp0
    )

    print()
    print(
        "input audit:",
        k0,
        x0.shape,
        x0.dtype,
        "min/max=",
        float(x0.min()),
        float(x0.max())
    )

    if x0.shape[-1] != 2:
        raise RuntimeError(
            f"expected 2 input channels, "
            f"got {x0.shape}"
        )

    if args.dry:

        banner(
            "DRY RUN OK"
        )

        return

    model = helper.build_model(
        models,
        args,
        model_path,
        nchan=2
    )

    records = []

    qc_rows = []

    for i,row in manifest.iterrows():

        key = str(
            row["name"]
        )

        y = int(
            row["y"]
        )

        x = int(
            row["x"]
        )

        image_path = (
            image_dir /
            f"{key}_image.tif"
        )

        heat_path = (
            heat_dir /
            f"{key}_gaumap_all.tif"
        )

        mask_path = (
            tile_out /
            f"{key}_mask.tif"
        )

        print()
        print(
            f"[{i+1:03d}/"
            f"{len(manifest):03d}] "
            f"{key} "
            f"y={y} x={x}"
        )

        if (
            args.resume
            and
            mask_path.exists()
        ):

            mask = tifffile.imread(
                mask_path
            )

            print(
                "  resume existing mask"
            )

        else:

            inp = load_input(
                image_path,
                heat_path
            )

            tic = time.time()

            mask = helper.infer_tile(
                model,
                utils,
                fastremap,
                inp,
                args
            )

            tifffile.imwrite(
                mask_path,
                mask.astype(
                    np.uint32
                ),
                compression="zlib"
            )

            print(
                "  inference time =",
                round(
                    time.time()-tic,
                    2
                ),
                "s"
            )

        q = mask_qc(
            mask
        )

        q.update(
            {
                "sample":key,
                "y":y,
                "x":x,
                "mask":
                    str(mask_path),
            }
        )

        print(
            "  cells =",
            q["cells"],
            "foreground =",
            round(
                100
                *
                q[
                    "foreground_fraction"
                ],
                2
            ),
            "%",
            "diameter =",
            round(
                q[
                    "median_diameter_um"
                ],
                2
            )
            if np.isfinite(
                q[
                    "median_diameter_um"
                ]
            )
            else np.nan,
            "um"
        )

        qc_rows.append(q)

        records.append(
            {
                "y":y,
                "x":x,
                "mask":
                    np.asarray(mask),
                "key":key,
            }
        )

    pd.DataFrame(
        qc_rows
    ).to_csv(
        out /
        "tile_qc.tsv",
        sep="\t",
        index=False
    )

    banner(
        "STITCH XENIUM2 ROI"
    )

    stitched,links = (
        helper.stitch_fov(
            records,
            FOV_SIZE,
            args
        )
    )

    final_path = (
        out /
        "genesegnet_xenium8_"
        "roi10000_mask.tif"
    )

    tifffile.imwrite(
        final_path,
        stitched.astype(
            np.uint32
        ),
        compression="zlib"
    )

    fq = mask_qc(
        stitched
    )

    banner(
        "GENESEGNET XENIUM2 DONE"
    )

    print(
        "cells =",
        fq["cells"]
    )

    print(
        "foreground =",
        fq[
            "foreground_fraction"
        ]
    )

    print(
        "median area px =",
        fq[
            "median_area_px"
        ]
    )

    print(
        "median diameter um =",
        fq[
            "median_diameter_um"
        ]
    )

    print(
        "stitch links =",
        links
    )

    print(
        "mask ->",
        final_path
    )

    meta = {
        "dataset":
            "xenium_2",
        "tissue":
            "breast",
        "model":
            str(model_path),
        "prepared_root":
            str(PREP_ROOT),
        "output_mask":
            str(final_path),
        "roi_size_px":
            FOV_SIZE,
        "pixel_size_um":
            PIXEL_SIZE,
        "n_tiles":
            len(manifest),
        "input_channels":
            2,
        "diameter_arg":
            args.diameter,
        "flow_threshold":
            args.flow_threshold,
        "confidence_threshold":
            args.confidence_threshold,
        "min_size":
            args.min_size,
        "network_tile_overlap":
            args.network_tile_overlap,
        "stitch_min_pixels":
            args.stitch_min_pixels,
        "stitch_fraction":
            args.stitch_fraction,
        "stitch_links":
            int(links),
        "uses_gt_during_inference":
            False,
    }

    (
        out /
        "run_meta.json"
    ).write_text(
        json.dumps(
            meta,
            indent=2
        )
    )


if __name__ == "__main__":
    main()