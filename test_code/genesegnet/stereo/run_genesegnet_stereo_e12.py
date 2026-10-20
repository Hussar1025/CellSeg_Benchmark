#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GeneSegNet x Stereo-seq -- CLEAN independent version

Reads only:
  1. raw Stereo ssDNA TIFF
  2. raw Dorsal_Midbrain_GEM_CellBin.tsv.gz
  3. GeneSegNet weights trained on E12.5

Does not read any output from Cellist / UCS / ComSeg / ProSeg.

Key fixes:
  - E12.5 model fixed at nchan=4
  - Stereo coordinate orientation and scale derived independently by this script from the raw data
  - no longer wrongly fits the two TIFF dimensions with a local GEM bbox
"""

import os
import sys
import gc
import json
import time
import inspect
import argparse
import subprocess

import numpy as np
import pandas as pd
import tifffile

from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops_table


BASE = "yourpath/GeneSegNet"
CORE = os.path.join(BASE, "GeneSegNet")

DATA = (
    "yourpath/dataset/stereoseq/"
    "stereo-seq_data_all"
)

MODEL_PATH = (
    "yourpath/genesegnet_e12_5_train/model/models/"
    "GeneSegNet_residual_on_style_off_concatenation_off_model_"
    "2026_08_05_10_58_40.809564"
)

OUT_BASE = "yourpath/genesegnet_stereo_e12_clean"


DATASETS = {

    "E14.5_E1S3": {
        "tif": f"{DATA}/E14.5_E1S3.tif",
        "gem":
            f"{DATA}/"
            "E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        "gt_cells": 4872,
    },

    "E16.5_E2S6": {
        "tif": f"{DATA}/E16.5_E2S6.tif",
        "gem":
            f"{DATA}/"
            "E16.5_E2S6_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        "gt_cells": 6229,
    },

    "E16.5_E2S7": {
        "tif": f"{DATA}/E16.5_E2S7.tif",
        "gem":
            f"{DATA}/"
            "E16.5_E2S7_Dorsal_Midbrain_GEM_CellBin.tsv.gz",
        "gt_cells": 7960,
    },
}


NCHAN = 4
DIAM_MEAN = 34.0

RESIDUAL_ON = True
STYLE_ON = False
CONCATENATION = False

HEATMAP_SIGMA = 7.0

# image-pixel margin around the complete GEM bbox
MARGIN = 50

# tiny safety expansion so a max coordinate does not land exactly outside
SCALE_SAFETY = 1.002

SEP = "=" * 78


def P(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def pick_gpu():

    try:
        txt = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True
        )
    except Exception:
        return 0

    rows = []

    for line in txt.strip().splitlines():

        idx, free, util = [
            x.strip()
            for x in line.split(",")
        ]

        rows.append(
            (
                int(idx),
                int(free),
                int(util)
            )
        )

    rows.sort(
        key=lambda x: (x[1], -x[2]),
        reverse=True
    )

    if not rows:
        return 0

    gpu = rows[0]

    P(
        f"GPU {gpu[0]} selected: "
        f"{gpu[1]/1024:.1f} GB free, "
        f"util={gpu[2]}%"
    )

    return gpu[0]


def load_gem(cfg):

    P("  reading GEM ...")

    gem = pd.read_csv(
        cfg["gem"],
        sep="\t",
        usecols=[
            "geneID",
            "x",
            "y",
            "MIDCounts",
            "cell",
        ]
    )

    P(f"    rows = {len(gem):,}")
    P(f"    x = [{gem.x.min()}, {gem.x.max()}]")
    P(f"    y = [{gem.y.min()}, {gem.y.max()}]")

    n_cells = (
        gem.loc[
            gem.cell > 0,
            "cell"
        ]
        .nunique()
    )

    P(f"    CellBin cells = {n_cells:,}")

    return gem


def detect_orientation_scale(
    gx_max,
    gy_max,
    H,
    W
):

    """
    For each of the normal / transpose orientations, compute:

      required_scale =
          max(
              GEM_x_max / effective_image_width,
              GEM_y_max / effective_image_height,
              1.0
          )

    Because GEM is only a local ROI:
      - a dimension not reaching the image edge is completely normal
      - so scale_w / scale_h consistency cannot be used for the decision
      - and certainly not their average

    We only require:
      all GEM coordinates fit inside the image under the same uniform scale.

    Choose the orientation with the smallest required scale.
    If both have scale=1, prefer normal.
    """

    candidates = []


    for transposed in (
        False,
        True
    ):

        if not transposed:

            effective_H = H
            effective_W = W

        else:

            effective_H = W
            effective_W = H


        sx_need = (
            float(gx_max)
            / float(effective_W)
        )

        sy_need = (
            float(gy_max)
            / float(effective_H)
        )


        required_scale = max(
            1.0,
            sx_need,
            sy_need
        )


        # tiny safety expansion
        required_scale *= SCALE_SAFETY


        P(
            f"    transpose={str(transposed):5s}"
            f" effective={effective_H}x{effective_W}"
            f" x_need={sx_need:.4f}"
            f" y_need={sy_need:.4f}"
            f" required_scale={required_scale:.6f}"
        )


        candidates.append(
            (
                required_scale,
                transposed
            )
        )


    # smallest scale wins;
    # False wins exact ties because it appears first.
    candidates.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )


    scale, transposed = (
        candidates[0]
    )


    P()
    P(
        f"  ★ selected:"
        f" transpose={transposed}"
        f" scale={scale:.6f}"
    )


    return (
        transposed,
        float(scale)
    )


def load_clean_crop(
    cfg,
    gem
):

    P(
        "  reading raw ssDNA TIFF ..."
    )


    img = tifffile.imread(
        cfg["tif"]
    )


    if img.ndim == 3:

        if img.shape[0] <= 8:
            img = img[0]

        else:
            img = img[..., 0]


    if img.ndim != 2:

        raise RuntimeError(
            f"unexpected TIFF shape: "
            f"{img.shape}"
        )


    raw_H, raw_W = (
        img.shape
    )


    P(
        f"    raw TIFF = "
        f"{raw_H} x {raw_W}"
        f"  {img.dtype}"
    )


    gx = (
        gem["x"]
        .to_numpy(
            dtype=np.float64
        )
    )


    gy = (
        gem["y"]
        .to_numpy(
            dtype=np.float64
        )
    )


    P()
    P(
        "  coordinate candidates:"
    )


    (
        transposed,
        scale
    ) = detect_orientation_scale(

        gx_max=gx.max(),
        gy_max=gy.max(),

        H=raw_H,
        W=raw_W
    )


    if transposed:

        P(
            "  transpose raw TIFF"
        )

        img = img.T


    H, W = img.shape


    # GEM coordinate -> image pixel
    gx_px = gx / scale
    gy_px = gy / scale


    # sanity check BEFORE crop
    whole_ok = (
        (gx_px >= 0)
        &
        (gx_px < W)
        &
        (gy_px >= 0)
        &
        (gy_px < H)
    )


    P(
        f"  GEM coordinates valid in transformed TIFF = "
        f"{whole_ok.sum():,} / "
        f"{len(gem):,} "
        f"({whole_ok.mean()*100:.3f}%)"
    )


    if whole_ok.mean() < 0.995:

        raise RuntimeError(
            "\nAfter the coordinate transform, more than 0.5% of GEM "
            "still falls outside the TIFF; stopping inference."
        )


    # Full Dorsal Midbrain GEM bbox
    cx0 = max(
        0,
        int(
            np.floor(
                gx_px.min()
            )
        )
        - MARGIN
    )


    cx1 = min(
        W,
        int(
            np.ceil(
                gx_px.max()
            )
        )
        + MARGIN
        + 1
    )


    cy0 = max(
        0,
        int(
            np.floor(
                gy_px.min()
            )
        )
        - MARGIN
    )


    cy1 = min(
        H,
        int(
            np.ceil(
                gy_px.max()
            )
        )
        + MARGIN
        + 1
    )


    P()
    P(
        "  complete Dorsal Midbrain bbox:"
    )

    P(
        f"    x pixels "
        f"[{cx0}, {cx1})"
    )

    P(
        f"    y pixels "
        f"[{cy0}, {cy1})"
    )


    crop = np.ascontiguousarray(
        img[
            cy0:cy1,
            cx0:cx1
        ]
    )


    del img


    if (
        crop.shape[0] == 0
        or
        crop.shape[1] == 0
    ):

        raise RuntimeError(
            f"empty clean crop: "
            f"{crop.shape}"
        )


    # GEM-coordinate origin represented by crop pixel 0,0
    ox = (
        cx0
        * scale
    )


    oy = (
        cy0
        * scale
    )


    local_x = (
        gx - ox
    ) / scale


    local_y = (
        gy - oy
    ) / scale


    ci = np.rint(
        local_x
    ).astype(
        np.int64
    )


    ri = np.rint(
        local_y
    ).astype(
        np.int64
    )


    CH, CW = (
        crop.shape
    )


    ok = (
        (ri >= 0)
        &
        (ri < CH)
        &
        (ci >= 0)
        &
        (ci < CW)
    )


    P(
        f"    crop shape = "
        f"{crop.shape}"
    )


    P(
        f"    GEM origin:"
        f" ox={ox:.3f}"
        f" oy={oy:.3f}"
    )


    P(
        f"    GEM in crop = "
        f"{ok.sum():,} / "
        f"{len(gem):,} "
        f"({ok.mean()*100:.3f}%)"
    )


    if ok.mean() < 0.995:

        raise RuntimeError(
            "\nThe full crop does not cover at least 99.5% of GEM."
        )


    return {

        "crop":
            crop,

        "ci":
            ci,

        "ri":
            ri,

        "ok":
            ok,

        "ox":
            float(ox),

        "oy":
            float(oy),

        "scale":
            float(scale),

        "transposed":
            bool(transposed),

        "raw_shape":
            (
                int(raw_H),
                int(raw_W)
            ),

        "crop_box":
            (
                int(cx0),
                int(cx1),
                int(cy0),
                int(cy1)
            ),
    }


def normalize_ssdna(
    crop
):

    arr = crop.astype(
        np.float32
    )


    lo, hi = np.percentile(
        arr,
        [
            1,
            99
        ]
    )


    arr -= lo

    arr /= max(
        hi - lo,
        1e-6
    )


    arr = np.clip(
        arr,
        0,
        1
    )


    return (
        arr * 255
    ).astype(
        np.uint8
    )


def build_input(
    gem,
    reg
):

    crop = reg["crop"]

    ci = reg["ci"]
    ri = reg["ri"]
    ok = reg["ok"]

    H, W = crop.shape


    gray = normalize_ssdna(
        crop
    )


    # Training used RGB image + heatmap.
    rgb = np.repeat(
        gray[:, :, None],
        3,
        axis=2
    )


    P(
        "  building transcript heatmap ..."
    )


    impulse = np.zeros(
        (
            H,
            W
        ),
        dtype=np.float32
    )


    weights = (
        gem[
            "MIDCounts"
        ]
        .fillna(1)
        .to_numpy(
            dtype=np.float32
        )
    )


    np.add.at(
        impulse,
        (
            ri[ok],
            ci[ok]
        ),
        weights[ok]
    )


    heat = gaussian_filter(
        impulse,
        sigma=HEATMAP_SIGMA
    )


    del impulse


    if heat.max() > 0:

        heat /= heat.max()


    heat_u8 = (
        heat * 255
    ).astype(
        np.uint8
    )


    image = np.concatenate(
        [
            rgb,
            heat_u8[:, :, None]
        ],
        axis=2
    )


    del heat
    del heat_u8


    P(
        f"    input = "
        f"{image.shape} "
        f"{image.dtype}"
    )


    if (
        image.ndim != 3
        or
        image.shape[2] != 4
    ):

        raise RuntimeError(
            f"expected H×W×4, "
            f"got {image.shape}"
        )


    return image


def build_gt(
    gem,
    reg
):

    gt = (
        gem[
            gem.cell > 0
        ]
        .groupby(
            "cell"
        )[
            [
                "x",
                "y"
            ]
        ]
        .median()
    )


    gx = (
        gt.x.to_numpy(
            dtype=float
        )
        - reg["ox"]
    ) / reg["scale"]


    gy = (
        gt.y.to_numpy(
            dtype=float
        )
        - reg["oy"]
    ) / reg["scale"]


    H, W = (
        reg["crop"].shape
    )


    ok = (
        (gx >= 0)
        &
        (gx < W)
        &
        (gy >= 0)
        &
        (gy < H)
    )


    return np.column_stack(
        [
            gx[ok],
            gy[ok]
        ]
    ).astype(
        np.float32
    )


def build_model(
    gpu_device
):

    if BASE not in sys.path:
        sys.path.insert(
            0,
            BASE
        )


    if CORE not in sys.path:
        sys.path.insert(
            0,
            CORE
        )


    from GeneSegNet import models


    if not os.path.isfile(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            MODEL_PATH
        )


    P()
    P(SEP)
    P("Build GeneSegNet model")
    P(SEP)


    device, gpu = (
        models.assign_device(
            use_torch=True,
            gpu=True,
            device=gpu_device
        )
    )


    P(f"  model = {MODEL_PATH}")
    P(f"  device = {device}")
    P(f"  gpu = {gpu}")
    P(f"  nchan = {NCHAN}")


    model = models.GeneSegModel(

        device=
            device,

        pretrained_model=
            MODEL_PATH,

        model_type=
            None,

        diam_mean=
            DIAM_MEAN,

        residual_on=
            RESIDUAL_ON,

        style_on=
            STYLE_ON,

        concatenation=
            CONCATENATION,

        nchan=
            NCHAN,
    )


    P(
        "  ✓ model loaded"
    )


    return model


def evaluate_model(
    model,
    image,
    confidence_threshold,
    flow_threshold,
    diameter,
    batch_size
):

    kwargs = {

        "diameter":
            float(diameter),

        "do_3D":
            False,

        "net_avg":
            False,

        "augment":
            False,

        "resample":
            True,

        "flow_threshold":
            float(
                flow_threshold
            ),

        "confidence_threshold":
            float(
                confidence_threshold
            ),

        "cellprob_threshold":
            float(
                confidence_threshold
            ),

        "invert":
            False,

        "batch_size":
            int(
                batch_size
            ),

        "interp":
            True,

        "normalize":
            True,

        "channel_axis":
            -1,

        "z_axis":
            None,

        "stitch_threshold":
            0.0,
    }


    try:

        sig = inspect.signature(
            model.eval
        )

        kwargs = {
            k: v
            for k, v
            in kwargs.items()
            if k in sig.parameters
        }

    except Exception:
        pass


    result = model.eval(
        image,
        **kwargs
    )


    masks = (
        result[0]
        if isinstance(
            result,
            tuple
        )
        else result
    )


    masks = np.asarray(
        masks
    )


    if (
        masks.ndim == 3
        and
        masks.shape[0] == 1
    ):

        masks = masks[0]


    if masks.ndim != 2:

        raise RuntimeError(
            f"unexpected mask shape "
            f"{masks.shape}"
        )


    return masks.astype(
        np.int32
    )


def mask_stats(
    masks
):

    ids = np.unique(
        masks
    )


    ids = ids[
        ids > 0
    ]


    n = len(ids)


    fg = int(
        np.count_nonzero(
            masks
        )
    )


    rate = (
        fg / masks.size
        if masks.size
        else 0
    )


    if n:

        counts = np.bincount(
            masks.ravel()
        )[1:]


        counts = counts[
            counts > 0
        ]


        median_area = float(
            np.median(
                counts
            )
        )

    else:

        median_area = 0.0


    return {

        "cells":
            int(n),

        "foreground_rate":
            float(rate),

        "median_cell_area_px":
            median_area,
    }


def threshold_scan(
    model,
    image,
    dataset,
    n_gt,
    cps,
    flows,
    diameter,
    batch_size
):

    rows = []


    P()
    P(SEP)

    P(
        f"Threshold scan — "
        f"{dataset}"
    )

    P(
        f"reference GT = "
        f"{n_gt:,}"
    )

    P(SEP)


    for cp in cps:

        for flow in flows:

            P(
                f"\n[scan] "
                f"confidence={cp} "
                f"flow={flow}"
            )


            t0 = time.time()


            masks = evaluate_model(
                model,
                image,
                cp,
                flow,
                diameter,
                batch_size
            )


            st = mask_stats(
                masks
            )


            dt = (
                time.time()
                - t0
            )


            P(
                f"  cells="
                f"{st['cells']:,}"
                f" foreground="
                f"{st['foreground_rate']*100:.2f}%"
                f" median_area="
                f"{st['median_cell_area_px']:.1f}"
                f" time="
                f"{dt:.1f}s"
            )


            rows.append({

                "confidence_threshold":
                    cp,

                "flow_threshold":
                    flow,

                "diameter":
                    diameter,

                **st,

                "seconds":
                    dt,
            })


            del masks
            gc.collect()


    outdir = os.path.join(
        OUT_BASE,
        dataset
    )


    os.makedirs(
        outdir,
        exist_ok=True
    )


    out = os.path.join(
        outdir,
        "threshold_scan.csv"
    )


    pd.DataFrame(
        rows
    ).to_csv(
        out,
        index=False
    )


    P(
        f"\nscan saved:\n  {out}"
    )


def save_result(
    dataset,
    masks,
    reg,
    n_gt,
    cp,
    flow,
    diameter
):

    outdir = os.path.join(
        OUT_BASE,
        dataset
    )


    os.makedirs(
        outdir,
        exist_ok=True
    )


    mask_path = os.path.join(
        outdir,
        "segmentation_mask.tif"
    )


    tifffile.imwrite(
        mask_path,
        masks.astype(
            np.uint32
        )
    )


    if masks.max() > 0:

        prop = pd.DataFrame(
            regionprops_table(
                masks,
                properties=[
                    "label",
                    "area",
                    "centroid"
                ]
            )
        )


        prop = prop.rename(
            columns={
                "label":
                    "cell_id",

                "centroid-1":
                    "x_crop_px",

                "centroid-0":
                    "y_crop_px",
            }
        )


        prop["x_gem"] = (
            prop["x_crop_px"]
            * reg["scale"]
            + reg["ox"]
        )


        prop["y_gem"] = (
            prop["y_crop_px"]
            * reg["scale"]
            + reg["oy"]
        )

    else:

        prop = pd.DataFrame()


    prop.to_csv(
        os.path.join(
            outdir,
            "cell_metadata.csv"
        ),
        index=False
    )


    st = mask_stats(
        masks
    )


    qc = {

        "dataset":
            dataset,

        "reference_gt":
            int(n_gt),

        "predicted_cells":
            st["cells"],

        "foreground_rate":
            st["foreground_rate"],

        "median_cell_area_px":
            st["median_cell_area_px"],

        "transposed":
            bool(
                reg["transposed"]
            ),

        "scale":
            float(
                reg["scale"]
            ),

        "raw_shape":
            list(
                reg["raw_shape"]
            ),

        "crop_shape":
            list(
                reg["crop"].shape
            ),

        "crop_box":
            list(
                reg["crop_box"]
            ),

        "ox":
            reg["ox"],

        "oy":
            reg["oy"],

        "confidence_threshold":
            cp,

        "flow_threshold":
            flow,

        "diameter":
            diameter,
    }


    with open(
        os.path.join(
            outdir,
            "run_qc.json"
        ),
        "w"
    ) as f:

        json.dump(
            qc,
            f,
            indent=2
        )


    P()
    P(SEP)

    P(
        f"DONE — {dataset}"
    )

    P(
        f"predicted cells = "
        f"{st['cells']:,}"
    )

    P(
        f"reference GT = "
        f"{n_gt:,}"
    )

    P(
        f"mask = "
        f"{mask_path}"
    )

    P(SEP)


def main():

    ap = argparse.ArgumentParser()


    ap.add_argument(
        "--dataset",
        required=True,
        choices=list(
            DATASETS.keys()
        )
    )


    ap.add_argument(
        "--scan",
        action="store_true"
    )


    ap.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=[
            0,
            -1
        ]
    )


    ap.add_argument(
        "--flow-thresholds",
        nargs="+",
        type=float,
        default=[
            0.8,
            1.0,
            1.2,
            1.5
        ]
    )


    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=-1.0
    )


    ap.add_argument(
        "--flow-threshold",
        type=float,
        default=1.2
    )


    ap.add_argument(
        "--diameter",
        type=float,
        default=34
    )


    ap.add_argument(
        "--batch-size",
        type=int,
        default=8
    )


    ap.add_argument(
        "--gpu",
        type=int,
        default=None
    )


    args = ap.parse_args()


    cfg = DATASETS[
        args.dataset
    ]


    P(SEP)

    P(
        f"GeneSegNet CLEAN × Stereo "
        f"{args.dataset}"
    )

    P(
        f"TIFF = {cfg['tif']}"
    )

    P(
        f"GEM  = {cfg['gem']}"
    )

    P(
        "other-method outputs = NONE"
    )

    P(SEP)


    gem = load_gem(
        cfg
    )


    reg = load_clean_crop(
        cfg,
        gem
    )


    gt_xy = build_gt(
        gem,
        reg
    )


    P(
        f"\n  GT centroids in complete crop = "
        f"{len(gt_xy):,} / "
        f"{cfg['gt_cells']:,}"
    )


    if (
        len(gt_xy)
        <
        0.995
        * cfg["gt_cells"]
    ):

        raise RuntimeError(
            "\nThe full Dorsal Midbrain crop "
            "does not cover at least 99.5% of GT; "
            "stopping the model run."
        )


    image = build_input(
        gem,
        reg
    )


    gpu = (
        args.gpu
        if args.gpu is not None
        else pick_gpu()
    )


    model = build_model(
        gpu
    )


    if args.scan:

        threshold_scan(
            model,
            image,
            args.dataset,
            len(gt_xy),
            args.confidence_thresholds,
            args.flow_thresholds,
            args.diameter,
            args.batch_size
        )

        return


    masks = evaluate_model(
        model,
        image,
        args.confidence_threshold,
        args.flow_threshold,
        args.diameter,
        args.batch_size
    )


    save_result(
        args.dataset,
        masks,
        reg,
        len(gt_xy),
        args.confidence_threshold,
        args.flow_threshold,
        args.diameter
    )


if __name__ == "__main__":
    main()